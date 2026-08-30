import numpy as np
import pandas as pd
from skbio.stats.distance import DistanceMatrix, permanova
import statsmodels.stats.multitest as smm

from common import Common, spherical_mean

report = []
target_regions = Common().get_static('target_regions')
pd.set_option('display.max_columns', None)
def rep(string):
  report.append(string)
  print(string, flush=True)

def sel_treg(select=[True for _ in range(len(target_regions))], return_string = True):
  result = (target_regions[i] for i, s in enumerate(select) if s)
  if return_string:
    result = ', '.join(result)
  return result

def prepare_data(select=[True for _ in range(len(target_regions))]):
  
  rep(f'Preparing data with {sel_treg(select)}')
  df = Common().get_traj_talairach()
  dft = df[df['trajectory'].isin(sel_treg(select, return_string = False))]
  vectors = np.vstack(dft['talairach'].values)
  treg_ids = pd.Categorical(dft['trajectory']).codes
  subject_ids = dft['sub'].values
  
  n_samples = len(vectors)
  n_subjects = len(np.unique(subject_ids))
  
  df_total = n_samples - 1
  df_block = n_subjects - 1
  df_factor = np.sum(select) - 1
  df_residual = df_total - df_block - df_factor

  rep(f'Using {n_samples} non-missing samples')
  
  if all(select):

    rep('Circular statistics weighted by component magnitude, mean +/- SD')

    # using circular statistics here
    def circ_stats(theta, weights):
      W = weights.sum()
      C = np.sum(weights * np.cos(theta))
      S = np.sum(weights * np.sin(theta))
      
      # normalized mean angle
      mean_angle = np.arctan2(S, C) 
      
      # circular std (radians)
      R = np.sqrt(C**2 + S**2) / W
      if R <= 0:
        circstd = np.nan
      else:
        circstd = np.sqrt(-2.0 * np.log(np.clip(R, 1e-20, 1.0)))
      return (np.degrees(mean_angle), np.degrees(circstd))
    
    for tid in range(4):
      v = vectors[treg_ids == tid]
      stats = []
      for d in range(2):
        plane_theta = np.arctan2(v[:, d], v[:, 2]) # 0 degrees being Z direction
        plane_mag = np.linalg.norm(v[:, [d, 2]], axis = 1)
        stats.append(circ_stats(plane_theta, plane_mag))

      rep(f'{target_regions[tid]}: LAT {stats[0][0]:.02f} +/- {stats[0][1]:.02f}, AP {stats[1][0]:.02f} +/- {stats[1][1]:.02f}')
  
  df_ = {
    'total': df_total,
    'block': df_block,
    'factor': df_factor,
    'residual': df_residual
  }
  
  return vectors, treg_ids, subject_ids, df_

# shuffles TIDs within subject while keeping left-right same
def permute_within_subject(treg_ids, subject_ids):
  
  perm_labels = treg_ids.copy()
  
  for sub in np.unique(subject_ids):
    
    idx = np.where(subject_ids == sub)[0]
    present = np.unique(treg_ids[idx])
    shuffled = np.random.permutation(present)
    relabel = dict(zip(present, shuffled))
    perm_labels[idx] = np.array([relabel[t] for t in treg_ids[idx]])
  
  return perm_labels

def perform_permanova(
  vectors,
  treg_ids,
  subject_ids,
  df_,
  select=[True for _ in range(len(target_regions))],
  n_perms=9999
):
  
  rep(f'Running PERMANOVA with {sel_treg(select)}')
  n_samples = vectors.shape[0]
  
  dot = np.clip(vectors @ vectors.T, -1., 1.)
  spherical_dist = np.arccos(dot)
  np.fill_diagonal(spherical_dist, 0.)
  dm = DistanceMatrix(
    spherical_dist,
    ids=[str(i) for i in range(n_samples)]
  )
  
  obs = permanova(dm, treg_ids.tolist(), permutations=0)
  obs_F = float(obs['test statistic'])
  
  rep(f'Observed pseudo-F ({df_["factor"]}, {df_["residual"]}): {obs_F}')
  
  perm_Fs = np.zeros(n_perms)  
  for p in range(n_perms):
    perm_treg_ids = permute_within_subject(treg_ids, subject_ids)
    perm_res = permanova(
        dm,
        perm_treg_ids.tolist(),
        permutations=0
    )
    perm_Fs[p] = float(perm_res['test statistic'])

  pval = (np.sum(perm_Fs >= obs_F) + 1) / (n_perms + 1)
  rep(f'Permutation p-value (subject-blocked, {n_perms} perms): {pval}')
  rep('')
  
  return {
    'target_regions': sel_treg(select),
    'F': obs_F,
    'df_factor': df_['factor'],
    'df_residual': df_['residual'],
    'p': pval,
    'perms': n_perms
  }


def perform_permdisp(
  vectors,
  treg_ids,
  subject_ids,
  df_,
  select=[True for _ in range(len(target_regions))],
  n_perms=9999
):
  
  rep(f'Running PERMDISP with {sel_treg(select)}')
  n_samples = len(vectors)
  
  def compute_distances(labels):
    group_means = {g: spherical_mean(vectors[labels == g]) for g in np.unique(labels)}
    dists = np.zeros(n_samples)
    for i in range(n_samples):
      dot = np.clip(np.dot(vectors[i], group_means[labels[i]]), -1., 1.)
      dists[i] = np.arccos(dot)
    return dists
  
  def compute_F(distances, labels):
    groups = np.unique(labels)
    grand_mean = distances.mean()
    ss_between = sum(np.sum(labels == g) * (distances[labels == g].mean() - grand_mean) ** 2 for g in groups)
    ss_within  = sum(np.sum((distances[labels == g] - distances[labels == g].mean()) ** 2) for g in groups)
    df_between = len(groups) - 1
    df_within  = len(distances) - len(groups)
    ms_between = ss_between / df_between
    ms_within  = ss_within / df_within
    return (
      ms_between / ms_within,
      df_between,
      df_within
    )

  obs_dists = compute_distances(treg_ids)
  obs_F, df_between, df_within = compute_F(obs_dists, treg_ids)
  rep(f'Observed PERMDISP F({df_between}, {df_within}): {obs_F}')

  perm_Fs = np.zeros(n_perms)
  for p in range(n_perms):
    perm_labels = permute_within_subject(treg_ids, subject_ids)
    perm_dists = compute_distances(perm_labels)
    perm_Fs[p], _, _ = compute_F(perm_dists, perm_labels)
  
  pval = (np.sum(perm_Fs >= obs_F) + 1) / (n_perms + 1)
  rep(f'PERMDISP p-value (subject-blocked, {n_perms} perms): {pval}')
  rep('')
  
  return {
    'target_regions': sel_treg(select),
    'F': obs_F,
    'df_factor': df_between,
    'df_residual': df_within,
    'p': pval,
    'perms': n_perms
  }

data = prepare_data()
tests = {'PERMANOVA': perform_permanova, 'PERMDISP': perform_permdisp}
for test, function in tests.items():
  result = function(*data)
  if result['p'] < .05:
    pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    results, pvals = [], []
    for pair in pairs:
      select = [idx in pair for idx in range(4)]
      data = prepare_data(select)
      r = function(*data, select, result['perms'])
      results.append(r)
      pvals.append(r['p'])
    _, pvals_corrected, _, _ = smm.multipletests(pvals, method='fdr_bh')
    for r, pc in zip(results, pvals_corrected):
      rep(f"{test} {r['target_regions']}: pseudo-F({r['df_factor']},{r['df_residual']})={r['F']:.3f}, p={r['p']:.8f}, p_corrected={pc:.8f}")

with open('report.txt', 'w') as f:
  f.write('\n'.join(report))
