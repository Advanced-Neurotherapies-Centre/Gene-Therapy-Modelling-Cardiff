import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, ttest_rel
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM


from common import Common, slab

fontsize = {
  'sup': 24,
  'title': 21,
  'stat': 18,
  'tick': 18,
  'xlabel': 16,
  'ylabel': 20
}
sns.set(style="whitegrid", context="talk")
trajectories = Common().get_static('target_regions')

def dot_plot(sole, dual, p, ax = None):

  pv = np.array([sole, dual])

  # Plot lines connecting paired points
  if ax is None:
    plt.figure(figsize=(5, 4)) #(4, 8))
    ax = plt.gca()
    save = True
  else:
    save = False
  
  for row in pv.T:
    _ = ax.plot(["Single trajectory", "Dual trajectory"], row, linewidth=2, zorder=1, alpha=0.2, color='gray') #,marker = 'o'

  # Mean markers (optional)
  ax.scatter(["Dual trajectory"] * pv.shape[1], pv[1], color='#4C72B0', label="Dual trajectory", zorder=2, alpha=0.5, s=80)
  ax.scatter(["Single trajectory"] * pv.shape[1], pv[0], color='#DD8452', label="Single trajectory", zorder=2, alpha=0.5, s=80)

  # Add mean lines or dots (optional)
  ax.plot(["Dual trajectory", "Single trajectory"], [pv[1].mean(), pv[0].mean()],
           color="red", lw=2, marker='o', zorder=3)

  ax.set_title("N Trajectories vs. VDmax", fontsize=fontsize['title'])
  ax.set_xticks(ticks=["Dual trajectory", "Single trajectory"], labels=['AP+PP', 'FP'])
  ax.grid(axis="y", linestyle="--", alpha=0.4)
  #plt.legend()

  # --- Add significance annotation ---
  # Position it above the highest data point
  y_max = 2000 #pv.max()
  offset = (y_max - pv.min()) * .02
  y_sig = y_max - offset

  sig_label = slab(p)

  ax.plot(["Dual trajectory", "Single trajectory"], [y_sig + (y_max * .05), y_sig + (y_max * .05)], color="black", lw=1.5)
  ax.text(0.5, y_sig + (y_max * .05), sig_label, ha="center", va="bottom", fontsize=fontsize['stat'])
  ax.tick_params(axis='both', labelsize=fontsize['tick'])
  ax.set_xlabel('')
  ax.set_ylabel('VDmax (mm³)', fontsize=fontsize['ylabel'])

  ax.set_ylim(0, 2300) #y_sig + (y_max * .1))
  if save:
    plt.tight_layout()
    plt.savefig('put_traj_VDmax_comp.svg', bbox_inches='tight')
    plt.close()

def plot_corr(df, ax, xlim, scale = 1., corr_vol = 'vvol', r_mean_within_sub = True, legend = False):

  df = df.copy()
  df[corr_vol] *= scale
  keys = ['VV', 'TV_caudate', 'TV_putamen', 'BV', 'vdmax_caudate', 'vdmax_putamen']
  vals = ['Ventricular Volume', 'Caudate Volume', 'Putaminal Volume', 'Brain Volume', 'Caudate', 'Putamen']
  mapx = {k: v for k, v in zip(keys, vals)}
  mapt = {k: v for k, v in zip(keys, ['Ventricles', 'Caudate', 'Putamen', 'Brain', None, None])}
  df.rename(columns=mapx, inplace=True)
  if r_mean_within_sub:
    df_r = df.groupby('sub')[vals].mean()
  else:
    df_r = df

  markers = {'left': 'o', 'right': 's'}
  colors = {'Caudate': '#4C72B0', 'Putamen': '#DD8452'}

  for target, color in colors.items():
    
    # plot points (separating left/right)
    for side, m in markers.items():
      sub = df[df['side'] == side]
      ax.scatter(sub[mapx[corr_vol]], sub[target],
               label=f'{target} ({side})',
               alpha=0.5,
               marker=m, s=40, color=color)

    # regression line
    sns.regplot(
      data=df_r, x=mapx[corr_vol], y=target, scatter=False,
      ax=ax, color=color, line_kws={"linewidth":2}
    )

    # correlation
    r, p = pearsonr(df_r[mapx[corr_vol]], df_r[target])
    p_str = f'p = {p:.3d}' if p >= .001 else 'p < 0.001'
    ax.text(0.05, 0.9 - (0.08 if target == 'Caudate' else 0),
            f'r = {r:.2f}, {p_str}',
            transform=ax.transAxes, 
            color=color, fontsize=fontsize['stat'])

  ax.set_title(f'{mapt[corr_vol]}', fontsize=fontsize['title'])
  ax.set_xlabel('')
  ticks = np.linspace(*xlim, 5)
  ax.set_xticks(ticks)
  ax.set_xlim(*xlim)
  dec = {'TV_caudate': 2, 'TV_putamen': 2, 'VV': 1, 'BV': 0}[corr_vol]
  ax.set_xticklabels([f'{(t / scale / 10000):.{dec}f}' for t in ticks])
  ax.set_ylabel('')
  ax.set_ylim(0., 2500.) #bottom = 0)
  if legend:
    ax.legend(frameon=False)

def violinplot_nd(df, ax = None):

  df = df.copy()
  for otn, ntn in zip(trajectories, ['C', 'AP', 'PP', 'FP']):
    df['trajectory'] = df['trajectory'].replace(otn, ntn)
  params = {'data':df, 'x':'trajectory', 'y':'ndep', 'split':False, 'inner':'box', 'palette':['#4C72B0'] * 4}
  if not ax is None:
    sns.violinplot(**params, ax=ax)
    save = False
  else:
    plt.figure(figsize=(4,4))
    ax = sns.violinplot(**params)
    save = True

  for pc in ax.collections:
    r, g, b = pc.get_facecolor()[0][:3]
    pc.set_linewidth(3.)
    pc.set_edgecolor((r, g, b, .8))
    pc.set_facecolor((r, g, b, .6))

  ax.set_ylim(0., 7.)
  ax.set_xlabel('') #Trajectories', fontsize=fontsize['ylabel'])
  ax.set_ylabel('Number of Deposits', fontsize=fontsize['ylabel'])
  ax.set_title('Deposits per Trajectory', fontsize=fontsize['title'])

  if save:
    plt.tight_layout()
    plt.savefig('ndep_box.svg', bbox_inches='tight')
    plt.close()

def forest_plot_fixed_effects(res):
    ci = res.conf_int()
    params = res.fe_params
    # Align names and values
    names = params.index
    est = params.values
    lower = ci.loc[names, 0].values
    upper = ci.loc[names, 1].values
    # Exclude intercept for clarity (optional)
    mask = names != 'const'
    names = names[mask]
    est = est[mask]
    lower = lower[mask]
    upper = upper[mask]

    order = np.argsort(est)
    names_ord = names[order]
    est_ord = est[order]
    lower_ord = lower[order]
    upper_ord = upper[order]

    plt.figure(figsize=(8, 3))
    y = np.arange(len(names_ord))
    plt.hlines(y, lower_ord, upper_ord, color='#4C72B0', lw=5, alpha=.5)
    plt.plot(est_ord, y, 'o', mfc='#4C72B0') #, mec='#4C72B0CC')
    plt.axvline(0, color='red', ls='--', lw=1)
    plt.yticks(y, names_ord, fontsize=fontsize['tick'])
    plt.xlabel('Mean change in ΔVDmax per 1 SD predictor (estimate)', fontsize=fontsize['xlabel'])
    #plt.ylabel('Predictors', fontsize=fontsize['ylabel'])
    title = r'VDmax$_{AP+PP}$ - VDmax$_{FP}$ Fixed Effects (95% CI)'
    plt.title(title, fontsize=fontsize['title'])
    plt.tight_layout()
    plt.savefig('Delta_VDmax_Fixed_Effects.svg')
    plt.close()

### PREPARE DATA ###

subs = Common().get_subs()
proc_path, dep_masks = Common().get_static(['proc_path', 'dep_masks_path'])

# prepare number of deposits
ndf = Common().get_traj_data(drop_empty = False)
ndf['ndep'] = [len(dc) for dc in ndf['deposit radii'].values]
ndf = ndf.drop(columns=['origin', 'entry', 'target', 'deposit centres', 'deposit radii', 'VDmax_unmerged', 'VDmax_merged'])

# prepare volume data
vdf = Common().get_volume_data(return_long = True)
for side in ['left', 'right']:
  for sub in subs:
    
    # row IDs
    subside = (vdf['sub'] == sub) & (vdf['side'] == side)
    putamen_row = subside & (vdf['target'] == 'putamen')
    single_row = subside & (vdf['target'] == 'putamen1')
    double_row = subside & (vdf['target'] == 'putamen2')
    
    # add GKS  
    p1 = nib.load(os.path.join(dep_masks, sub, side + '_putamen1_final.nii.gz')).get_fdata()
    p2 = nib.load(os.path.join(dep_masks, sub, side + '_putamen2_final.nii.gz')).get_fdata()
    tc = np.sum(np.minimum(p1, p2)) / np.sum(np.maximum(p1, p2))
    vdf.loc[putamen_row, 'GDS'] = 2 * tc / (tc + 1)
    
    # add diff
    vdf.loc[putamen_row, 'diff'] = vdf.loc[double_row, 'vdmax'].values - vdf.loc[single_row, 'vdmax'].values

# prepare data for comparison
sub_tgt_vdmax = vdf.groupby(['sub', 'target'])['vdmax'].mean().reset_index()
sub_tgt_vdmax = sub_tgt_vdmax.pivot(index='sub', columns='target', values='vdmax')
vdf = vdf.rename(columns={'brain': 'BV', 'ventricles': 'VV', 'volume': 'TV'})
pdf = vdf.loc[~vdf['target'].isin(['caudate', 'putamen1', 'putamen2'])]

### SIMPLE COMPARISONS (FIG.5B-C) ###

# run t-test between number of traj for put
t_stat, p = ttest_rel(sub_tgt_vdmax['putamen1'].values, sub_tgt_vdmax['putamen2'].values)
print(f't-test for vdmax SOLE vs DUAL trajectory (putamen): t={t_stat:.03f} p={p:.03f}')

# plot the number of deposits and VDmax vs number of trajectores
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
violinplot_nd(ndf, ax = axes[0])
dot_plot(sub_tgt_vdmax['putamen1'].values, sub_tgt_vdmax['putamen2'].values, p, ax = axes[1])
plt.tight_layout()
plt.savefig('ndep_and_ntraj.svg')
plt.close()

### LME ###

# scale predictors
for c in ['VV', 'BV', 'TV', 'diff', 'GDS']:
  if c == 'VV':
    pdf[c] = np.log(pdf['VV'])
  sd = pdf[c].std(ddof=0)
  pdf[c] = (pdf[c] - pdf[c].mean()) / (sd if sd > 0 else 1.)

# encode side
pdf['side_right'] = (pdf['side'] == 'right').astype(int)

# list predictors
base = ['VV', 'BV', 'TV', 'side_right']
X = lambda P: [':'.join(P)]
I = lambda L: [base[l] for l in L]
pred_z = I([0,1,2]) + X(I([1,2]))
for P in pred_z:
  if ':' in P:
    a, b = P.split(':')
    pdf[P] = pdf[a] * pdf[b]
    sd = pdf[P].std(ddof=0)
    pdf[P] = (pdf[P] - pdf[P].mean()) / (sd if sd > 0 else 1.)

# run LME
exog = sm.add_constant(pdf[pred_z])
m = MixedLM(endog=pdf['diff'], exog=exog, groups=pdf['sub'])
res = m.fit(reml=True)
print('LME results')
print(res.summary())

# plot fixed effects
forest_plot_fixed_effects(res)

### CORRELATION PLOTS ###

# this ends up with columns: sub, side, BV, VV, TV_caudate, TV_putamen, vdmax_caudate, vdmax_putamen
vdf = vdf.loc[~vdf['target'].isin(['putamen1', 'putamen2'])]
vdf = vdf.pivot(index=['sub', 'side', 'BV', 'VV'], columns='target', values=['TV', 'vdmax'])
vdf = vdf.reset_index()
vdf.columns = ['_'.join(str(x) for x in col if x != '') for col in vdf.columns]
pd.set_option('display.max_columns', None)

cvars = ['TV_caudate', 'TV_putamen', 'VV', 'BV']
xlims, ranges = [], []
for cvar in cvars:
  x = np.array([np.min(vdf[cvar]), np.max(vdf[cvar])])
  r = x[1] - x[0]
  xlims.append(x)
  ranges.append(r)
scale = np.min(ranges) / np.array(ranges)
xlims = np.array(xlims) * scale[:, None] + [-300., 300.]
ranges = np.diff(xlims, axis = 1)
  
fig = plt.figure(figsize=(16, 6))
gs = fig.add_gridspec(1, len(cvars), width_ratios=ranges)
for i, cvar in enumerate(cvars):
  ax = fig.add_subplot(gs[0, i])
  plot_corr(vdf, ax, xlims[i], scale[i], cvar, legend = (i == len(cvars) - 1))
  if i:
    ax.tick_params(labelleft=False)

plt.suptitle('Volumes vs. VDmax', fontsize=fontsize['sup'])  
fig.text(0.5, -0.04, 'Volume (CL)', ha='center', fontsize=fontsize['ylabel'])
fig.text(-0.04, 0.5, 'VDmax (mm³)', va='center', rotation='vertical', fontsize=fontsize['ylabel'])
plt.subplots_adjust(wspace=0.0)
plt.tight_layout()
plt.savefig('volume_correlation_plot.svg')
plt.close()
