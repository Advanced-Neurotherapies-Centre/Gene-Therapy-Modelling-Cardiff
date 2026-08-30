import numpy as np
from statsmodels.stats.multitest import multipletests
from pprint import pprint

from common import Common, unit, spherical_mean, log_map, exp_map

"""
Directional regression on the sphere via tangent-space linearisation with
subject-block Freedman–Lane permutations.

Per condition (tid):
  1) LOCATION (centroid) model on 2D tangent responses:
     - Omnibus (joint predictors) and partial tests (vvol, tvol, bvol)
     - Pillai's trace with permutation p-values
     - Within-tid maxT adjusted p across the three partials
     - Effect sizes: Pillai, partial multivariate R^2, angular shift (deg) per +1 SD,
       and optional 3D endpoint for an arrow visualization.

  2) DISPERSION model on geodesic distance r to mean direction:
     - Omnibus and partial tests via FL with subject-block permutations
     - Within-tid maxT adjusted p
     - Effect sizes: partial R^2, Δ(deg) per +1 SD

Across tids:
  - Holm (FWER) across four omnibus p-values for LOCATION
  - Holm (FWER) across four omnibus p-values for DISPERSION

Inputs:
  - vecs:      (N x 3) unit vectors (left/right per subject; may include NaNs for missing)
  - subj_ids:  (N,) subject identifiers
  - traj_ids:  (N,) integers in tids (e.g., 0..3)
  - vvol:      (N,) subject-level predictor; will be log1p-transformed and z-scored per tid
  - bvol:      (N,) subject-level predictor; z-scored per tid
  - tvol_dict: dict {tid: (N,) array}, each z-scored per tid (condition-specific)
  - tids:      iterable of tids to analyze (default: (0,1,2,3))
"""

target_regions = Common().target_regions
treg_map = {i: treg for i, treg in enumerate(target_regions)}

# =========================
# Utility: data prep & stats
# =========================

def zscore(a):
  a = np.asarray(a, float)
  m = np.nanmean(a)
  s = np.nanstd(a)
  if not np.isfinite(s) or s == 0.:
    return a - m
  return (a - m) / s

def log1p_pos(a):
  a = np.asarray(a, float)
  a = np.maximum(a, 0.)
  return np.log1p(a)

# =========================
# Subject-block permutations
# =========================

# return list of index arrays, one per subject block (two sides per subject typically)
def build_subject_blocks(subj_ids):
  subj_ids = np.asarray(subj_ids)
  blocks = []
  for sid in np.unique(subj_ids):
    idx = np.flatnonzero(subj_ids == sid)
    blocks.append(idx)
  return blocks

# permute rows of 'arr' by shuffling subject blocks
# preserve within-block rows and match block sizes (groups by block length)
def permute_by_blocks(arr, blocks, rng):
  arrp = np.empty_like(arr)
  groups = {}
  for b_idx, rows in enumerate(blocks):
    L = len(rows)
    groups.setdefault(L, []).append(b_idx)
  for L, b_list in groups.items():
    b_list = np.asarray(b_list)
    donor = rng.permutation(b_list)
    for dst, src in zip(b_list, donor):
      arrp[blocks[dst]] = arr[blocks[src]]
  return arrp

# =========================
# Pillai, partial R2 helpers
# =========================

# Pillai's trace = tr( H (H+E)^{-1} ) with numerical safeguard
def pillai_from_H_E(H, E, eps = 1e-10):
  HE = H + E
  try:
    val = float(np.trace(np.linalg.solve(HE, H)))
  except np.linalg.LinAlgError:
    HE_reg = HE + eps * np.eye(HE.shape[0])
    val = float(np.trace(np.linalg.pinv(HE_reg) @ H))
  return val

# sum of squares and cross-products, A^T A
def sscp(A):
  return A.T @ A

# trace(H) / trace(H+E)
def partial_multivariate_R2(H, E):
  num = float(np.trace(H))
  den = float(np.trace(H + E))
  return num / den if den > 0 else np.nan

# =========================
# Freedman–Lane stuff
# =========================

# LOCATION omnibus: test joint effect of columns X[:,1:] vs intercept-only using Pillai
def fl_location_omnibus(Y2d, X, subj_ids, n_perm = 10000, rng = None):
  rng = np.random.default_rng(rng)
  
  # full fit
  Bf, *_ = np.linalg.lstsq(X, Y2d, rcond=None)
  Yhat_f = X @ Bf
  Ef = Y2d - Yhat_f
  E_sscp = sscp(Ef)
  
  # reduced (intercept-only)
  X0 = X[:, [0]]
  B0, *_ = np.linalg.lstsq(X0, Y2d, rcond=None)
  Yhat_0 = X0 @ B0
  H_obs = sscp(Yhat_f - Yhat_0)
  V_obs = pillai_from_H_E(H_obs, E_sscp)

  # permutations on reduced residuals
  R0 = Y2d - Yhat_0
  blocks = build_subject_blocks(subj_ids)
  V_perm = np.zeros(n_perm)
  for b in range(n_perm):
    Rp = permute_by_blocks(R0, blocks, rng)
    Y_star = Yhat_0 + Rp
    Bf_p, *_ = np.linalg.lstsq(X, Y_star, rcond=None)
    Yhat_fp = X @ Bf_p
    Ef_p = Y_star - Yhat_fp
    H_p = sscp(Yhat_fp - Yhat_0)
    V_perm[b] = pillai_from_H_E(H_p, sscp(Ef_p))
  
  p = (np.sum(V_perm >= V_obs) + 1) / (n_perm + 1)
  return V_obs, p, V_perm

# LOCATION partial tests: per group of columns in cols_interest (e.g., [[1],[2],[3]]),
# FL permutations comparing full vs reduced (dropping the group's columns)
def fl_location_partials(Y2d, X, subj_ids, cols_interest, n_perm = 10000, rng = None):
  rng = np.random.default_rng(rng)
  Bf, *_ = np.linalg.lstsq(X, Y2d, rcond=None)
  Yhat_f = X @ Bf
  Ef = Y2d - Yhat_f
  E_sscp = sscp(Ef)
  
  blocks = build_subject_blocks(subj_ids)
  k = len(cols_interest)
  V_obs = np.zeros(k)
  V_perm_mat = np.zeros((k, n_perm))
  
  for t, cols in enumerate(cols_interest):
    keep = np.ones(X.shape[1], bool)
    keep[cols] = False
    Xr = X[:, keep]
    Br, *_ = np.linalg.lstsq(Xr, Y2d, rcond=None)
    Yhat_r = Xr @ Br
    H_obs = sscp(Yhat_f - Yhat_r)
    V_obs[t] = pillai_from_H_E(H_obs, E_sscp)
    
    Rr = Y2d - Yhat_r
    for b in range(n_perm):
      Rp = permute_by_blocks(Rr, blocks, rng)
      Y_star = Yhat_r + Rp
      Bf_p, *_ = np.linalg.lstsq(X, Y_star, rcond=None)
      Yhat_fp = X @ Bf_p
      Ef_p = Y_star - Yhat_fp
      H_p = sscp(Yhat_fp - Yhat_r)
      V_perm_mat[t, b] = pillai_from_H_E(H_p, sscp(Ef_p))
  
  p_nom = (np.sum(V_perm_mat >= V_obs[:, None], axis=1) + 1) / (n_perm + 1)
  max_perm = V_perm_mat.max(axis=0)
  p_maxT = (np.sum(max_perm[None, :] >= V_obs[:, None], axis=1) + 1) / (n_perm + 1)
  return V_obs, p_nom, p_maxT, V_perm_mat

# DISPERSION omnibus: scalar response r (radians), joint test X[:,1:]
# test statistic: extra-SS ratio (SS_H / SS_E) behaves like an F-like monotone stat
def fl_dispersion_omnibus(r, X, subj_ids, n_perm = 10000, rng = None):
  rng = np.random.default_rng(rng)
  
  # full & reduced fits
  Bf, *_ = np.linalg.lstsq(X, r, rcond=None)
  yhat_f = X @ Bf

  X0 = X[:, [0]]
  B0, *_ = np.linalg.lstsq(X0, r, rcond=None)
  yhat_0 = X0 @ B0

  ss = lambda y: float(np.sum(y**2))
  SS_H = ss(yhat_f - yhat_0)
  SS_E = ss(r - yhat_f)
  V_obs = SS_H / (SS_E + 1e-12)

  R0 = r - yhat_0
  blocks = build_subject_blocks(subj_ids)
  V_perm = np.zeros(n_perm)
  for b in range(n_perm):
    Rp = permute_by_blocks(R0, blocks, rng)
    y_star = yhat_0 + Rp
    Bf_p, *_ = np.linalg.lstsq(X, y_star, rcond=None)
    yhf = X @ Bf_p
    SS_Hp = ss(yhf - yhat_0)
    SS_Ep = ss(y_star - yhf)
    V_perm[b] = SS_Hp / (SS_Ep + 1e-12)

  p = (np.sum(V_perm >= V_obs) + 1) / (n_perm + 1)
  return V_obs, p, V_perm

#DISPERSION partial tests over cols_interest
def fl_dispersion_partials(r, X, subj_ids, cols_interest, n_perm = 10000, rng= None):
  rng = np.random.default_rng(rng)
  Bf, *_ = np.linalg.lstsq(X, r, rcond=None)
  yhat_f = X @ Bf
  ss = lambda y: float(np.sum(y**2))

  blocks = build_subject_blocks(subj_ids)
  k = len(cols_interest)
  V_obs = np.zeros(k)
  V_perm_mat = np.zeros((k, n_perm))

  for t, cols in enumerate(cols_interest):
    keep = np.ones(X.shape[1], bool)
    keep[cols] = False
    Xr = X[:, keep]
    Br, *_ = np.linalg.lstsq(Xr, r, rcond=None)
    yhat_r = Xr @ Br

    SS_H = ss(yhat_f - yhat_r)
    SS_E = ss(r - yhat_f)
    V_obs[t] = SS_H / (SS_E + 1e-12)

    Rr = r - yhat_r
    for b in range(n_perm):
      Rp = permute_by_blocks(Rr, blocks, rng)
      y_star = yhat_r + Rp
      Bf_p, *_ = np.linalg.lstsq(X, y_star, rcond=None)
      yhf = X @ Bf_p
      SS_Hp = ss(yhf - yhat_r)
      SS_Ep = ss(y_star - yhf)
      V_perm_mat[t, b] = SS_Hp / (SS_Ep + 1e-12)

  p_nom = (np.sum(V_perm_mat >= V_obs[:, None], axis=1) + 1) / (n_perm + 1)
  max_perm = V_perm_mat.max(axis=0)
  p_maxT = (np.sum(max_perm[None, :] >= V_obs[:, None], axis=1) + 1) / (n_perm + 1)
  return V_obs, p_nom, p_maxT, V_perm_mat

# =========================
# Omnibus direction for plotting
# =========================

# return the leading response-axis (2D) for the omnibus full-vs-intercept test,
# and its 3D direction at mu. Assumes Y2d are 2D tangent coords, and (e1,e2)
# is the orthonormal basis of the tangent plane used to build Y2d
def omnibus_direction(Y2d, X, mu, e1 = None, e2 = None):
    
  # Full fit
  Bf, *_ = np.linalg.lstsq(X, Y2d, rcond=None)
  Yhat_f = X @ Bf
  Ef = Y2d - Yhat_f

  # Reduced (intercept-only)
  X0 = X[:, [0]]
  B0, *_ = np.linalg.lstsq(X0, Y2d, rcond=None)
  Yhat_0 = X0 @ B0

  # SSCPs in outcome space (2x2)
  H = (Yhat_f - Yhat_0).T @ (Yhat_f - Yhat_0)
  E = Ef.T @ Ef

  # Leading generalized eigenvector: E^{-1} H v = λ v
  w, V = np.linalg.eig(np.linalg.solve(E, H))
  v2d = V.real
  v2d /= np.linalg.norm(v2d, axis = 0, keepdims = True)  # unit direction in the tangent plane

  # DoF for unbiased covariance
  n, p = X.shape
  use_unbiased_cov = True
  rankX = np.linalg.matrix_rank(X)
  df_E = max(n - rankX, 1)
  denom = df_E if use_unbiased_cov else n

  # Residual covariance estimate + ridge
  ridge = 1e-10
  SigmaE = E / float(denom)
  SigmaE_r = SigmaE + ridge * np.eye(2)

  # Invert via Cholesky (stable)
  try:
    C = np.linalg.cholesky(SigmaE_r)
  except np.linalg.LinAlgError:
    # Increase ridge if needed
    SigmaE_r = SigmaE + (ridge + 1e-8) * np.eye(2)
    C = np.linalg.cholesky(SigmaE_r)

  # Solve for SigmaE^{-1/2} r_i efficiently: MD_i = || SigmaE^{-1/2} r_i ||
  # For each row r_i, solve C z = r_i^T  => z = C^{-1} r_i^T,
  # then MD_i^2 = z^T z from the second triangular solve.
  # Vectorized approach: solve for all residuals at once (2 x n), then column norms.
  Z = np.linalg.solve(C, Ef.T)          # shape: (2, n)
  d = np.sqrt(np.sum(Z**2, axis=0))    # Euclidean norm per column -> (n,)
  return {'v': v2d, 'w': w, 'd': d}

# =========================
# Orchestration per condition
# =========================

# LOCATION and DISPERSION per individual target
def tid_directions(df, ivars = ['vvol', 'tvol', 'bvol'], n_perm = 50000, rng = np.random.default_rng(42), compute_arrows = True):
  
  cols = ['talairach', 'sub', 'ventricles', 'brain', 'volume']
  U, S, v, b, t = [np.asarray(df[c].values) for c in cols]
  U = unit(np.vstack(U).astype(float))
  
  # remove missing
  ok = (np.isfinite(U).all(axis=1) & np.isfinite(v) & np.isfinite(b) & np.isfinite(t)).ravel()
  U, S, v, b, t = U[ok], S[ok], v[ok], b[ok], t[ok]
  if len(U) < 5:
    return {'note': 'insufficient data after filtering'}

  # spherical mean and tangent responses
  mu = spherical_mean(U)
  Y2d, r = log_map(mu, U)   # Y2d (n x 2), r in radians
  # predictors: log1p(vvol) then z-score all per tid
  v = zscore(log1p_pos(v))
  b = zscore(b)
  t = zscore(t)
  # design matrix: intercept + v, t, b, (b x t)
  cols = [np.ones(len(v)), v, t, b]
  if len(ivars) == 4: # interaction
    cols.append(b * t)
  X = np.column_stack(cols)

  # --- LOCATION ---
  V_loc_obs, p_loc_omni, V_loc_perm = fl_location_omnibus(Y2d, X, S, n_perm=n_perm, rng=rng)
  # partials for columns [1], [2], [3]
  cols_interest = [[i] for i in range(1, len(ivars) + 1)]
  Vp_obs, p_loc_nom, p_loc_maxT, V_loc_perm_mat = fl_location_partials(
    Y2d, X, S, cols_interest, n_perm=n_perm, rng=rng
  )

  # effect sizes for location partials
  # compute B_full once to extract 2D coefficient vectors
  B_full, *_ = np.linalg.lstsq(X, Y2d, rcond=None)  # shape (4 x 2)
  # per predictor j, 2D coefficient vector is B_full[j, :]
  # angular shift per +1 SD ~ ||beta_j|| (radians) -> degrees
  ang_shift_deg = []
  arrow_endpoints = []
  for j in range(1, len(ivars) + 1):
    beta_j = B_full[j, :]  # (2,)
    shift_deg = float(np.linalg.norm(beta_j) * 180. / np.pi)
    ang_shift_deg.append(shift_deg)
    if compute_arrows and shift_deg > 0:
      # exp-map +1 SD step along beta_j from mu
      end3d = exp_map(mu, beta_j, scale=1.)  # +1 SD step endpoint
      arrow_endpoints.append(end3d.tolist())
    else:
      arrow_endpoints.append(None)

  # partial multivariate R^2 for location: trace(H)/trace(H+E)
  # build H_obs per partial (requires reduced fit); reuse from permutation prep
  loc_partial_R2 = []
  for t_idx, cols in enumerate(cols_interest):
    keep = np.ones(X.shape[1], bool)
    keep[cols] = False
    Xr = X[:, keep]
    Br, *_ = np.linalg.lstsq(Xr, Y2d, rcond=None)
    Yhat_r = Xr @ Br
    Yhat_f = X @ B_full
    E_sscp = sscp(Y2d - Yhat_f)
    H_sscp = sscp(Yhat_f - Yhat_r)
    loc_partial_R2.append(float(partial_multivariate_R2(H_sscp, E_sscp)))

  # --- DISPERSION ---
  V_dis_obs, p_dis_omni, V_dis_perm = fl_dispersion_omnibus(r, X, S, n_perm=n_perm, rng=rng)
  Vd_obs, p_dis_nom, p_dis_maxT, V_dis_perm_mat = fl_dispersion_partials(
    r, X, S, cols_interest, n_perm=n_perm, rng=rng
  )

  # effect sizes for dispersion partials
  # Δ(deg) per +1 SD ~ coefficient from OLS of r on predictors; report in degrees
  # refit OLS to get coefficients
  Bd, *_ = np.linalg.lstsq(X, r, rcond=None)   # (4,)
  disp_deg_per_sd = (Bd[1:] * 180. / np.pi).tolist()  # convert rad -> deg

  # partial R^2 for dispersion: SS_H / (SS_H + SS_E)
  disp_partial_R2 = []
  ss = lambda y: float(np.sum(y**2))
  yhat_f = X @ Bd
  SS_E = ss(r - yhat_f)
  for cols in cols_interest:
    keep = np.ones(X.shape[1], bool); keep[cols] = False
    Xr = X[:, keep]
    Br, *_ = np.linalg.lstsq(Xr, r, rcond=None)
    yhat_r = Xr @ Br
    SS_H = ss(yhat_f - yhat_r)
    disp_partial_R2.append(float(SS_H / (SS_H + SS_E + 1e-12)))

  # Package per tid
  return {
    'n_rows': int(len(U)),
    'coefficients': B_full,
    'mu': mu.tolist(),
    'Y2d': Y2d,
    'o': log_map(mu, [[0,0,0]])[0].ravel(),
    'location': {
      'omnibus': {'pillai': float(V_loc_obs), 'p_perm': float(p_loc_omni), 'n_perm': int(n_perm)},
      'partials': [
        {
          'name': name,  # mapping done below when summarizing
          'pillai': float(Vp_obs[i]),
          'p_perm': float(p_loc_nom[i]),
          'p_maxT_within_tid': float(p_loc_maxT[i]),
          'partial_multivariate_R2': float(loc_partial_R2[i]),
          'angular_shift_deg_per_SD': float(ang_shift_deg[i]),
          'arrow_endpoint_3d': arrow_endpoints[i],
        }
        for i, name in enumerate(ivars)
      ],
      'direction': omnibus_direction(Y2d, X, mu)
    },
    'dispersion': {
      'omnibus': {'stat': float(V_dis_obs), 'p_perm': float(p_dis_omni), 'n_perm': int(n_perm)},
      'partials': [
        {
          'name': name,
          'stat': float(Vd_obs[i]),
          'p_perm': float(p_dis_nom[i]),
          'p_maxT_within_tid': float(p_dis_maxT[i]),
          'partial_R2': float(disp_partial_R2[i]),
          'delta_deg_per_SD': float(disp_deg_per_sd[i]),
        }
        for i, name in enumerate(ivars)
      ]
    }
  }

# run LOCATION and DISPERSION analyses per tid with subject-block FL permutations
# returns a dictionary with per-tid results and Holm-adjusted omnibus p-values across tids
# compute_arrows - return exp-mapped +1 SD arrow endpoints
def analyse_directions_per_tid(df, tids = (0, 1, 2, 3), n_perm = 50000, random_state = 42, compute_arrows = True):

  rng = np.random.default_rng(random_state)

  interaction = True  
  ivars = ['vvol', 'tvol', 'bvol']
  if interaction:
    ivars.append('tvol x bvol')

  per_tid = {}
  loc_omni_p, dis_omni_p, omni_treg = [], [], []
  for tid in tids:
    mask = (df['trajectory'].values == treg_map[tid])
    if not np.any(mask): continue
    result = tid_directions(df.iloc[mask], ivars, n_perm, rng, compute_arrows)
    per_tid[treg_map[tid]] = result
    loc_omni_p.append(result['location']['omnibus']['p_perm'])
    dis_omni_p.append(result['dispersion']['omnibus']['p_perm'])
    omni_treg.append(treg_map[tid])

  # across-tid Holm for omnibus tests (two families)
  loc_holm = {treg: adj for treg, adj in zip(omni_treg, multipletests(loc_omni_p, method='holm')[1])} if loc_omni_p else None
  dis_holm = {treg: adj for treg, adj in zip(omni_treg, multipletests(dis_omni_p, method='holm')[1])} if dis_omni_p else None

  predictor_labels = ['intercept', 'vvol_log_z', 'tvol_z', 'bvol_z']
  if interaction: 
    predictor_labels.append('tvol x bvol')

  return {
    'per_tid': per_tid,
    'omnibus_holm': {'location': loc_holm, 'dispersion': dis_holm},
    'settings': {
      'n_perm': int(n_perm),
      'random_state': int(random_state) if isinstance(random_state, int) else None,
      'predictors': predictor_labels,
      'notes': 'FL permutations with subject-blocking; tangent-space location; scalar r for dispersion.'
    }
  }


# =========================
# Set up and execute
# =========================

viz = True

data_df = Common().get_traj_talairach(drop_empty = True)
data_df['target'] = [traj.split('_')[-1] for traj in data_df['trajectory'].values] # map data_df to vol_df - this is target region, not target cds
vol_df = Common().get_volume_data(return_long = True).dropna()
merged_df = vol_df.merge(
  data_df[['sub', 'side', 'target', 'trajectory', 'talairach']],
  on=['sub', 'side', 'target'],
  how='outer'
)

results = analyse_directions_per_tid(merged_df)

Y2d, origin, fitinfo = [], [], []
for treg in target_regions:
  mu = results['per_tid'][treg]['mu']
  Y2d.append(results['per_tid'][treg]['Y2d'])
  origin.append(results['per_tid'][treg]['o'])
  fitinfo.append(results['per_tid'][treg]['location']['direction'])
  x = np.degrees(np.arctan2(mu[0], mu[2])) # LAT (0 degrees being Z direction)
  y = np.degrees(np.arctan2(mu[1], mu[2])) # AP  (0 degrees being Z direction)
  print(f'Fréchet mean for {treg}: LAT {x:.02f}°, AP {y:.02f}°')
  del results['per_tid'][treg]['mu']
  del results['per_tid'][treg]['Y2d']
  del results['per_tid'][treg]['o']
  del results['per_tid'][treg]['location']['direction']

print('\n')
pprint(results)

if viz:

  import matplotlib.pyplot as plt
  from matplotlib.colors import ListedColormap, Normalize
  from matplotlib.cm import ScalarMappable

  fontsize = {
    'suptitle': 18,
    'title': 14,
    'axes': 16,
    'pvals': 12,
    'ticks': 13
  }

  def custom_colours(values):

    values = np.clip(values, 0., 1.)
    colours = np.zeros((len(values), 4))
    colours[:, 0] = np.where(values < .5, 2 * values, 1.)
    colours[values >= .5, 1] = 2 * values[values >= .5] - 1.
    colours[:, -1] = 1.

    return colours #(colours * 255).astype(np.uint8)

  fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6, 6))
  max_d = max([max(fitinfo[tid]['d']) for tid in range(4)])
  for tid, ax in enumerate(axes.ravel()):
    y2d = np.degrees(Y2d[tid].T)
    s = np.argsort(fitinfo[tid]['w'])[::-1]
    eigvec = fitinfo[tid]['v'][:, s] 
    eigval = fitinfo[tid]['w'][s] * 5.
    fiterror = fitinfo[tid]['d']
    oxy = np.degrees(origin[tid])
    p_corr = results['omnibus_holm']['location'][treg_map[tid]]
    ax.scatter(*y2d, s = 5, c = custom_colours(fiterror / max_d), zorder=2)
    ax.arrow(0, 0, *(np.degrees(eigvec[:, 0]) * eigval[0]), head_width = 2., linewidth=5., color='#4C72B099', zorder=1)
    ax.arrow(0, 0, *(np.degrees(eigvec[:, 1]) * eigval[1]), head_width = 2., linewidth=5., color='#DD845299', zorder=1)
    #ax.axvline(oxy[0], color='black', alpha=.3)
    #ax.axhline(oxy[1], color='black', alpha=.3)
    ax.scatter(*oxy, s = 100, c='grey', marker='x', zorder=1)
    p = f'p={p_corr:.3f}' if p_corr >= .001 else 'p<0.001'
    ax.text(
      .93, .93, p,
      transform=ax.transAxes,
      ha='right', va='top',
      fontsize=fontsize['pvals']
    )
    ax.set_title(target_regions[tid].replace('_', ' ').title(), fontsize=fontsize['title'])
    ax.tick_params(axis='both', labelsize=fontsize['ticks'])
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-115., 115.)
    ax.set_ylim(-115., 115.)
    ax.set_xticks([-90, -45, 0, 45, 90])
    ax.set_yticks([-90, -45, 0, 45, 90])
    ax.grid()
    ax.set_axisbelow(True)
  
  fig.supxlabel('Lateral (deg)', fontsize=fontsize['axes'])
  fig.supylabel('Anteroposterior (deg)', fontsize=fontsize['axes'])
  fig.suptitle('Atrophy Effects on Best Trajectory', fontsize=fontsize['suptitle'])
  plt.tight_layout()
  
  cmap = ListedColormap(custom_colours(np.linspace(0, 1, 256)))
  sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
  sm.set_array([])  
  cbar = fig.colorbar(sm, ax=axes, location='right')
  cbar.ax.tick_params(labelsize=fontsize['ticks'])
  cbar.set_ticks(np.linspace(0, 1, 5))
  cbar.set_ticklabels([f'{t:.1f}' for t in np.linspace(0, max_d, 5)])
  cbar.set_label('Mahalanobis residuals'.upper(), fontsize=fontsize['axes'], rotation=270)
  plt.savefig('angle_vvol.svg')
