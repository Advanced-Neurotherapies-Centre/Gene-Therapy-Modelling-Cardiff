import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, wilcoxon
from statsmodels.stats.multitest import multipletests

from common import Common, slab

# -----------------------
# Config
# -----------------------
GROUP = 'sub'                    # LOSO grouping
EPS = 1e-6                       # for safe log transform
SEED = 42
np.random.seed(SEED)

sns.set(style="whitegrid", context="talk")
fontsize = {
  'titles': 20,
  'corr': 16,
  'tick': 18,
  'axlabel': 20,
  'sup': 24
}

# -----------------------
# Preprocessing pipelines
# -----------------------

# --- Full preprocessor: logVV, compute raw BVxTV, standardize all together ---
def build_preprocessor_with_raw_interaction_then_scale(features=('BV', 'VV', 'TV')):  # human-checked
    """
    Pipeline that:
      - computes logVV = log(clip(VV, EPS, None)),
      - computes raw interaction BVxTV = BV * TV (from raw BV, TV),
      - stacks [BV, logVV, TV, BVxTV] in that order,
      - standardizes all four columns together using training-fold stats.

    Final output columns (all standardized): [BV, logVV, TV, BVxTV]
    """
    feats = list(features)
    assert all(f in ('BV', 'VV', 'TV') for f in feats), "features must be subset of ('BV','VV','TV')"

    # ColumnTransformer to produce [BV, logVV, TV] in that order
    transformers = []
    # VV -> logVV
    if 'VV' in feats:
        log_vv = Pipeline([
            ('log', FunctionTransformer(lambda x: np.log(np.clip(x, a_min=EPS, a_max=None)),
                                        validate=False))
        ])
        transformers.append(('logVV', log_vv, ['VV']))
    # Pass-through BV and TV (if present)
    passthru = [c for c in feats if c in ('BV', 'TV')]
    if passthru:
        transformers.append(('pass', 'passthrough', passthru))

    ct = ColumnTransformer(transformers, remainder='drop')

    # Reorder to [BV, logVV, TV]
    def reorder_to_bv_logvv_tv(X):
        """
        X columns come out as: [logVV] + [BV, TV] (if both in passthru).
        We need to reorder to [BV, logVV, TV].
        Handles cases where 'BV' or 'TV' might be absent, though we expect both here.
        """
        # Identify shapes safely
        if 'VV' in feats:
            # X[:, 0] is logVV
            # Passthrough columns follow in the same order as 'passthru'
            # Build mapping
            cols = []
            if 'BV' in feats:
                # BV is at index 1 if both BV and TV present and VV present
                # More robust approach: reconstruct from 'passthru' order
                # Let's compute indices programmatically:
                pass_idx = {name: i for i, name in enumerate(passthru, start=1 if 'VV' in feats else 0)}
                bv_col = X[:, [pass_idx['BV']]] if 'BV' in pass_idx else None
                logvv_col = X[:, [0]]
                tv_col = X[:, [pass_idx['TV']]] if 'TV' in pass_idx else None
                # Stack existing ones in [BV, logVV, TV] order
                cols = [c for c in [bv_col, logvv_col, tv_col] if c is not None]
                return np.column_stack(cols)
            else:
                # Rare case: no BV — then keep [logVV, TV] or [logVV] as-is
                return X
        else:
            # No VV: keep order as given (likely [BV, TV])
            return X

    # Compute raw interaction BVxTV from the *unscaled* BV and TV in the reordered array
    def add_raw_bv_tv_interaction(X):
        # We expect X in order [BV, logVV, TV] when VV is present.
        # If VV missing, X may be [BV, TV] or so; we handle robustly.
        # Find indices by column count and presence:
        if X.shape[1] == 3:
            BV = X[:, 0]
            TV = X[:, 2]
        elif X.shape[1] == 2:
            # Assume [BV, TV]
            BV = X[:, 0]
            TV = X[:, 1]
        else:
            raise ValueError("Unexpected feature layout while building BVxTV interaction.")
        inter = (BV * TV).reshape(-1, 1)  # RAW interaction
        return np.hstack([X, inter])      # [BV, logVV, TV, BVxTV]

    pipe = Pipeline([
        ('prep', ct),
        ('reorder', FunctionTransformer(reorder_to_bv_logvv_tv, validate=False)),
        ('add_raw_bvxtv', FunctionTransformer(add_raw_bv_tv_interaction, validate=False)),
        ('scale_all', StandardScaler(with_mean=True, with_std=True))  # single standardization pass
    ])

    return pipe

# --- Build models using ONLY BV×TV interaction (if requested) ---
def build_models(include_interaction=True): # human-checked
    # (1) TV-only
    m1 = Pipeline([
        ('scale', StandardScaler(with_mean=True, with_std=True)),  # scale TV alone
        ('reg', LinearRegression())
    ])

    # (2) Full linear with optional ONLY BV×TV interaction
    #if include_interaction:
    pre2 = build_preprocessor_with_raw_interaction_then_scale(('BV', 'VV', 'TV'))    

    m2 = Pipeline([('pre', pre2), ('reg', LinearRegression())])

    # (3) VDmax only
    m3 = Pipeline([
        ('scale', StandardScaler(with_mean=True, with_std=True)),
        ('reg', LinearRegression())
    ])

    return {'TV_only': m1, 'Full_linear': m2, 'VDmax_only': m3}

# -----------------------
# LOSO evaluation
# -----------------------
def loso_eval(df): # human-checked
    
    PREDICT = 'coverage'
    assert all(col in df.columns for col in [GROUP, PREDICT, 'BV', 'VV', 'TV', 'VDmax'])
    models = build_models()
    groups = df[GROUP].values
    logo = LeaveOneGroupOut()

    # Storage
    preds_store = {name: [] for name in models}
    obs_store = []

    for train_idx, test_idx in logo.split(df, groups=groups):
        train = df.iloc[train_idx].copy()
        test = df.iloc[test_idx].copy()

        y_tr, y_te = train[PREDICT].values, test[PREDICT].values
        obs_store.append(pd.DataFrame({'sub': test['sub'].values, 'side': test['side'].values, 'y': y_te}))

        for name, pipe in models.items():
            if name == 'TV_only':
                Xtr = train[['TV']]
                Xte = test[['TV']]
            elif name == 'Full_linear':
                Xtr = train[['BV', 'VV', 'TV']]
                Xte = test[['BV', 'VV', 'TV']]
            else:  # VDmax only
                Xtr = train[['VDmax']]
                Xte = test[['VDmax']]

            pipe.fit(Xtr, y_tr)
            yhat = pipe.predict(Xte)

            preds_store[name].append(pd.DataFrame({
                'sub': test['sub'].values,
                'side': test['side'].values,
                'yhat': yhat
            }))

    obs = pd.concat(obs_store, ignore_index=True)
    
    preds_long = []
    for name, parts in preds_store.items(): # model name, yhat-per-sub
        dfp = pd.concat(parts, ignore_index=True)
        dfp['model'] = name
        preds_long.append(dfp)
    preds = pd.concat(preds_long, ignore_index=True) # sub, yhat, model
    
    # Merge held-out observations and predictions
    out = preds.merge(obs, on=['sub', 'side']) # sub, model, y, yhat

    # Per-subject metrics (aggregate across sides first)
    summary_subj = []
    for (sid, name), g in out.groupby(['sub', 'model']):
        y = g['y'].values
        yhat = g['yhat'].values
        mae = mean_absolute_error(y, yhat)                                                # within-subject average MAE
        rmse = np.sqrt(mean_squared_error(y, yhat))                                       # within-subject average RMSE
        summary_subj.append({'sub': sid, 'model': name, 'MAE': mae, 'RMSE': rmse})
    subj_tbl = pd.DataFrame(summary_subj)

    # Overall metrics across all held-out observations (optional)
    overall = out.groupby('model').apply(
        lambda g: pd.Series({
            'MAE_all': mean_absolute_error(g['y'], g['yhat']),
            'RMSE_all': np.sqrt(mean_squared_error(g['y'], g['yhat'])),
            'R_all': pearsonr(g['y'], g['yhat'])[0] if len(np.unique(g['yhat'])) > 1 else np.nan,
            'R2_all': (pearsonr(g['y'], g['yhat'])[0]**2
                       if len(np.unique(g['yhat'])) > 1 else np.nan)
        })
    ).reset_index()

    return out, subj_tbl, overall

# -----------------------
# Paired tests & Holm correction
# -----------------------
def paired_wilcoxon(subj_tbl, model_ref, model_alt, metric='MAE'): # human-checked
    w = subj_tbl.pivot(index='sub', columns='model', values=metric)
    d = (w[model_alt] - w[model_ref]).dropna()  # < 0 favors model_alt
    stat, p = wilcoxon(d)
    # Rank-biserial effect size (for paired data): r_rb = 1 - 2*W / (n*(n+1)/2)
    # Using the Wilcoxon W statistic returned by scipy is non-trivial; instead compute from ranks:
    ranks = d.abs().rank()
    pos = ranks[d > 0].sum()
    neg = ranks[d < 0].sum()
    r_rb = (pos - neg) / (len(d) * (len(d) + 1) / 2.0) if len(d) > 0 else np.nan
    return {
        'comparison': f'{model_alt} vs {model_ref}',
        'n_subjects': len(d),
        'median_delta': float(np.median(d)),
        'wilcoxon_stat': float(stat),
        'p_raw': float(p),
        'rank_biserial_r': float(r_rb)
    }

# Optional: paired bootstrap CI for median ΔMAE across subjects
def paired_bootstrap_ci(subj_tbl, model_ref, model_alt, metric='MAE', B=5000, seed=42): # human-checked
    rng = np.random.default_rng(seed)
    w = subj_tbl.pivot(index='sub', columns='model', values=metric).dropna()
    d = (w[model_alt] - w[model_ref]).values
    n = len(d)
    boot = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        boot[b] = np.median(d[idx])
    return np.percentile(boot, [2.5, 97.5])

# -----------------------
# Visualization
# -----------------------
def plot_violin(df, p, ax, title): # manual
  df = df.copy()
  mapping = {'TV_only': 'TV', 'VDmax_only': 'VDmax', 'Full_linear': 'Full'}
  df['model'] = df['model'].map(mapping)
  vp = sns.violinplot(
    data=df, x='model', y='MAE',
    order=['TV', 'Full', 'VDmax'],
    split=False, inner='box', ax=ax,
    palette=['#4C72B0'] * 3, #'#DD8452']
  )

  for pc in vp.collections:
    r, g, b = pc.get_facecolor()[0][:3]
    pc.set_linewidth(3.)
    pc.set_edgecolor((r, g, b, .8))
    pc.set_facecolor((r, g, b, .6))

  # bar + star
  y_max = df['MAE'].max()  
  ax.plot([0, 2], [y_max * 1.3, y_max * 1.3], color='black')
  ax.text(1., y_max * 1.3, slab(p[0]), ha='center')
  ax.plot([1, 2], [y_max * 1.3 - 1.5, y_max * 1.3 - 1.5], color='black')
  ax.text(1.5, y_max * 1.3 - 1.5, slab(p[1]), ha='center')

  ax.set_title(title, fontsize=fontsize['titles'])
  ax.set_xlabel('')
  ax.set_ylabel('Coverage LOSO MAE (%)', fontsize=fontsize['axlabel'])
  ax.set_ylim([-3., 20.]) #[-.03, .2])

# -----------------------
# Run
# -----------------------

data = Common().get_volume_data(return_long = True)
data = data.rename(columns={'volume': 'TV', 'ventricles': 'VV', 'brain': 'BV', 'vdmax': 'VDmax'})
data['coverage'] = data['VDmax'] / data['TV'] * 100.

m = {'tv': 'TV_only', 'full': 'Full_linear', 'vd': 'VDmax_only'} # models

sts = {}
for target in ['caudate', 'putamen']:
  print(f'LOSO CV for {target}')
  out, subj_tbl, overall = loso_eval(data.loc[data['target'] == target])
  overall = overall.sort_values(
    by='model',
    key=lambda s: s.map({v: i for i, v in enumerate(m.values())})
  )
  print("\nOverall (all held-out observations):\n", overall)
  
  # Identify models
  models = subj_tbl['model'].unique().tolist()
  
  # Paired tests (per-subject MAE), Holm correction
  res_vd_vs_tv = paired_wilcoxon(subj_tbl, m['tv'], m['vd'], metric='MAE')
  res_vd_vs_full = paired_wilcoxon(subj_tbl, m['full'], m['vd'], metric='MAE')
  _, p_adj, _, _ = multipletests(np.array([res_vd_vs_tv['p_raw'], res_vd_vs_full['p_raw']]), alpha=.05, method='holm')
  res_vd_vs_tv['p_holm'] = float(p_adj[0]); res_vd_vs_full['p_holm'] = float(p_adj[1])
  
  # Bootstrap CIs for median ΔMAE
  ci_vd_vs_tv = paired_bootstrap_ci(subj_tbl, m['tv'], m['vd'], metric='MAE', B=10000, seed=SEED)
  ci_vd_vs_full = paired_bootstrap_ci(subj_tbl, m['full'], m['vd'], metric='MAE', B=10000, seed=SEED)
  
  print("\nPaired comparisons (ΔMAE < 0 favors VDmax):\n")
  pprint({**res_vd_vs_tv, 'CI95_median_delta': tuple(ci_vd_vs_tv)})
  print(f'\n')
  pprint({**res_vd_vs_full, 'CI95_median_delta': tuple(ci_vd_vs_full)})
  print(f'\n\n')
  
  sts[target] = {'subj_table': subj_tbl, 'pvals': p_adj}

fig, axes = plt.subplots(1, 2, figsize=(10, 7), sharey=True)
for a, (target, data) in enumerate(sts.items()):
  plot_violin(data['subj_table'], data['pvals'], axes[a], target.capitalize())
plt.suptitle('LOSO MAE Comparison Between Models', fontsize=fontsize['sup'])
fig.text(0.5, -0.04, 'Model', ha='center', fontsize=fontsize['axlabel'])
#fig.subplots_adjust(left=.3, bottom=.3)
plt.tight_layout()
#plt.show()
plt.savefig('LOSO_MAE_plot.svg')
plt.close()
