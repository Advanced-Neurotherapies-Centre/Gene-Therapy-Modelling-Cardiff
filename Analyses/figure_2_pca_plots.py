import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from common import Common

fontsize = {
  'titles': 20,
  'arrowlabel': 18,
  'tick': 18,
  'axlabel': 24,
  'sup': 26
}

vdf = Common().get_volume_data()
vrs = ['ventricles', 'brain', 'caudate', 'putamen']
for c in vrs:
  if c == 'ventricles':
    vdf[c] = np.log1p(vdf['ventricles'])
  elif c in vrs[2:]:
    vdf[c] = (vdf['left_' + c] + vdf['right_' + c]) / 2
  sd = vdf[c].std(ddof=0)
  vdf[c] = (vdf[c] - vdf[c].mean()) / (sd if sd > 0 else 1.0)

fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharex=False, sharey=True)
for tid, target in enumerate(vrs[2:]):
 
  print(f'Performing PCA on {target}') 
  X = np.asarray(vdf[vrs[:2] + [target]])

  # Standardize
  #X_std = StandardScaler().fit_transform(X)

  # PCA
  pca = PCA()
  Z = pca.fit_transform(X)

  # Explained variance
  explained_var = pca.explained_variance_ratio_
  cum_var = np.cumsum(explained_var)

  for i, (ev, cv) in enumerate(zip(explained_var, cum_var), 1):
    print(f"PC{i}: {ev:.3f} (cumulative {cv:.3f})")

  loadings = pca.components_.T  # (features × PCs)

  print(loadings)
  ax = axes[tid]
  ax.scatter(Z[:, 0], Z[:, 1], c ='#4C72B0', alpha=0.5)

  for i, var in enumerate(vrs[:2] + [target]):
    ax.arrow(0, 0,
      loadings[i, 0]*2,
      loadings[i, 1]*2,
      head_width=0.05,
      linewidth=4,
      color='#4C72B0')
    ax.text(loadings[i, 0]*2.2,
      loadings[i, 1]*2.2,
      var, fontsize=fontsize['arrowlabel'])

  #ax.set_xlabel('PC1', fontsize=fontsize['axlabel'])
  #ax.ylabel('PC2')
  ax.set_ylabel('')
  ax.set_xlabel('')
  ax.axhline(0, color='black', alpha=.3)
  ax.axvline(0, color='black', alpha=.3)
  ax.tick_params(axis='both')
  ax.set_xlim(-3, 3) 
  ax.set_ylim(-3, 3)
  ax.set_aspect('equal')
  ax.grid('on')
  ax.set_title(target.capitalize(), fontsize=fontsize['titles'])
  ax.text(-2.8, 2.8, f'PC1: {explained_var[0] * 100:.1f}%\nPC2: {explained_var[1] * 100:.1f}%', va='top', fontsize=fontsize['arrowlabel'])
  ax.set_axisbelow(True)
  ax.tick_params(
    axis='both',
    labelsize=fontsize['tick'],
    #which='both',      # affects both major and minor ticks
    #bottom=False,      # hide bottom ticks
    #left=False,        # hide left ticks
    #top=False,         # hide top ticks
    #right=False,       # hide right ticks
    #labelbottom=False, # hide x-axis labels
    #labelleft=False    # hide y-axis labels
  )

plt.suptitle('PCA Biplot', fontsize=fontsize['sup'])  
#plt.supxlabel('PC1', fontsize=fontsize['axlabel'])
#plt.supylabel('PC2', fontsize=fontsize['axlabel'])
fig.text(.5, 0., 'PC1', ha='center', fontsize=fontsize['axlabel'])
fig.text(0., .5, 'PC2', va='center', rotation='vertical', fontsize=fontsize['axlabel'])
fig.subplots_adjust(left=.3, bottom=.3)
plt.tight_layout()
plt.savefig('PCA_atrophy.svg')
plt.close()
#plt.show()
