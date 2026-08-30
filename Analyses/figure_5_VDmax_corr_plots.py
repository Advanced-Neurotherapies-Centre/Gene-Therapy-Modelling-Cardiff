import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

from common import Common

sns.set(style="whitegrid", context="talk")
fontsize = {
  'titles': 20,
  'corr': 16,
  'tick': 18,
  'axlabel': 20,
  'sup': 24
}

def plot_corr(df, ax, title, r_mean_within_sub = True):

  df = df.copy()

  df['ratio'] *= 100. # %
  df.rename(columns={'tvol': 'TV', 'vdmax': 'VDmax'}, inplace=True)
  if r_mean_within_sub:
    df_r = df.groupby('sub')[['TV', 'VDmax', 'ratio']].mean()
  else:
    df_r = df

  markers = {'left': 'o', 'right': 's'}
  colors = {'TV': '#4C72B0', 'VDmax': '#DD8452'}

  for predictor, color in colors.items():
    
    # plot points (separating left/right)
    for side, m in markers.items():
      sub = df[df['side'] == side]
      ax.scatter(sub[predictor], sub['ratio'],
                 label=f'{predictor} ({side})',
                 alpha=0.5,
                 marker=m, s=40, color=color)

    # regression line
    sns.regplot(
      data=df_r, x=predictor, y='ratio', scatter=False,
      ax=ax, color=color, line_kws={"linewidth":2}
    )

    # correlation
    r, p = pearsonr(df_r[predictor], df_r['ratio'])
    p_str = f'p = {p:.3d}' if p >= .001 else 'p < 0.001'
    ax.text(0.05, 0.9 - (0.08 if predictor == 'TV' else 0),
            f'r = {r:.2f}, {p_str}',
            transform=ax.transAxes, 
            color=color, fontsize=fontsize['corr'])

  ax.set_title(title, fontsize=fontsize['titles'])
  ax.set_xlabel('')
  ax.set_ylabel('')
  #ax.set_xlabel('Volume (mm³)', fontsize=fontsize['axlabel'])
  #ax.set_ylabel('Coverage (%)', fontsize=fontsize['axlabel'])
  #ax.set_xlim(0, 5500) #left = 0)
  #ax.set_ylim(0., 45.) #bottom = 0)
  ax.legend(frameon=False)


data = Common().get_volume_data(return_long = True)
data = data.rename(columns={'volume': 'tvol', 'ventricles': 'vvol'})
data = data.loc[~((data['target'] == 'putamen1') | (data['target'] == 'putamen2'))]
data['ratio'] = data['vdmax'] / data['tvol']

# plot correlation
fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharex=True, sharey=True)
plot_corr(data[data['target']=='caudate'], axes[0], 'Caudate')
plot_corr(data[data['target']=='putamen'], axes[1], 'Putamen')
plt.suptitle('Volumes vs. Coverage', fontsize=fontsize['sup'])  
fig.text(.5, -.04, 'Volume (mm³)', ha='center', fontsize=fontsize['axlabel'])
fig.text(-.04, .5, 'Coverage (%)', va='center', rotation='vertical', fontsize=fontsize['axlabel'])
plt.tight_layout()
#plt.show()
plt.savefig('coverage_correlation_plot.svg')
plt.close()