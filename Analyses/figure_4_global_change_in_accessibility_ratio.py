import os, trimesh
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from scipy.stats import yeojohnson

import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM

from common import Common

fontsize = {
  'sup': 24,
  'title': 21,
  'stat': 18,
  'tick': 18,
  'xlabel': 16,
  'ylabel': 20,
  'legend': 24
}
sns.set(style="whitegrid", context="talk")

target_regions, vol_path, surv_path, proc_path = Common().get_static(['target_regions', 'vol_data_path', 'surv_data_path', 'proc_path'])

def _forest_plot_fixed_effects(ax, model, title, colours):
    ci = model.conf_int()
    params = model.fe_params
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

    names_ord = names
    est_ord = est
    lower_ord = lower
    upper_ord = upper
    colours = {k: [c / 255 for c in v] for k, v in colours.items()}
    colours_ord = [colours[n] for n in names]

    y = np.arange(len(names_ord))
    r_height = .5
    
    for fidx, estimate in enumerate(est_ord):
      ax.plot(estimate, y[fidx], 'o', mfc=colours_ord[fidx], mec=colours_ord[fidx], label=names_ord[fidx])

      x0 = lower_ord[fidx]
      y0 = y[fidx] - r_height / 2 
      r_width = upper_ord[fidx] - lower_ord[fidx]

      rect = patches.Rectangle(
        (x0, y0),
        r_width,
        r_height,
        linewidth=2,
        edgecolor=colours_ord[fidx],
        facecolor=(*colours_ord[fidx], .3)
      )
      ax.add_patch(rect)
    
    ax.axvline(0, color=colours['NEU'], ls='--', lw=2)
    ax.get_yaxis().set_visible(False)
    ax.set_xlim(-1., 1.)
    title = f'{title.replace("_", " ").capitalize()}'
    ax.set_title(title, fontsize=fontsize['title'])

    for spine in ax.spines.values():
      spine.set_color('black')
      spine.set_linewidth(1.5)

    ax.grid(True, color='0.8', linewidth=1)

def plot_quad(conditions, colours, output_path = 'output.svg'):

  fig, axes = plt.subplots(
    2, 2,
    figsize=(12, 8),
    sharex=True,
    sharey=True
  )
  axes = axes.ravel()
  
  for ax, cond in zip(axes, conditions):
    _forest_plot_fixed_effects(
      ax=ax,
      model=cond['model'],
      title=cond.get('name'),
      colours=colours
    )

  handles, labels = axes[0].get_legend_handles_labels()
  fig.legend(
    handles, labels,
    loc='lower center',
    ncol=2,
    fontsize=fontsize['legend']
  )

  plt.savefig(output_path)
  plt.close()

dfv = Common().get_volume_data()
sides, drop = {}, []
for side in ['left', 'right']:
  dfb = dfv.copy()
  for c in dfv.columns:
    if side in c:
      dfb[c.replace(side + '_', '')] = dfv[c]
      drop.append(c)
  dfb['side'] = side
  sides[side] = dfb
dfv = pd.concat(sides, ignore_index=True).drop(drop, axis = 1).rename(columns={'brain': 'BV', 'ventricles': 'VV'})
dfv['VV'] = np.log(dfv['VV'])
dfv['side_right'] = (dfv['side'] == 'right').astype(int)

dfs = Common().get_surv_data()
for idx, row in dfs.iterrows():
  glb_path = os.path.join(Common().proc_path, row['sub'], '_'.join([row['sub'], row['side'], row['target']]), row['target'] + '.glb')
  tmesh = trimesh.load(glb_path)
  dfs.at[idx, 'TV'] = sum([v.volume for v in tmesh.geometry.values()])

NEUTRAL = [ 50,  50,  50]
BLUE    = [  0, 114, 176] # VV
GREEN   = [ 85, 168, 104] # BV
ORANGE  = [221, 132,  82] # TV
RED     = [180,   4,  38] # BV:TV
colours = {'NEU': NEUTRAL, 'VV': BLUE, 'BV': GREEN, 'TV': ORANGE, 'BV:TV': RED}

# List predictors
conditions = []
for target_region in target_regions:
  df = pd.merge(dfv, dfs.loc[dfs['target'] == target_region], on=['sub', 'side'])
  base = ['VV', 'BV', 'TV', 'side_right']
  X = lambda P: [':'.join(P)]
  I = lambda L: [base[l] for l in L]
  pred_z = I([0,1,2]) + X(I([1,2]))
  for P in pred_z + ['valid_ratio']:
    if ':' in P:
      a, b = P.split(':')
      df[P] = df[a] * df[b]
    else:
      if P == 'valid_ratio':
        df[P], _ = yeojohnson(df[P])
      sd = df[P].std(ddof=0)
      df[P] = (df[P] - df[P].mean()) / (sd if sd > 0 else 1.0)
  exog = sm.add_constant(df[pred_z])
  m = MixedLM(endog=df['valid_ratio'], exog=exog, groups=df['sub'])
  res = m.fit(reml=True)
  print(f'LME for {target_region}')
  print(res.summary())
  conditions.append({'name': target_region, 'model': res})

plot_quad(conditions, colours, 'forest_plots_combined.svg')
