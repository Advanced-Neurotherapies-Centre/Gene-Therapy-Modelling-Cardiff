import os, json
import numpy as np
import pandas as pd
import nibabel as nib
import trimesh

"""

DATA FILE COLUMNS/HEADERS:

traj_data (up to 8 / subject) [trajectory ~ target region]: 
----------
sub, side, trajectory, origin, entry, target, deposit centres, deposit radii, VDmax_unmerged, VDmax_merged

vol_data (1 / subject): 
----------
sub, ventricles, brain, 
left_caudate, left_putamen, left_caudate_vdmax, left_putamen1_vdmax, left_putamen2_vdmax, left_putamen_vdmax, 
right_caudate, right_putamen, right_caudate_vdmax, right_putamen1_vdmax, right_putamen2_vdmax, right_putamen_vdmax

surv_data (up to 8 / subject):
----------
sub, skull_area, skull_ppmm2, skull_pts, side, target, target_area, target_ppmm2, target_pts, 
n_trajectories_start, n_trajectories_valid, valid_ratio

"""

class Common:
  
  def __init__(self,):
    
    self.fsdir = '' # path to directory with per-subject FreeSurfer outputs
    self.param_path = '' # used to access original meshes (e.g., striatal nuclei and ventricles) and subject's talairach transform
    self.proc_path = '' # used to access target region (cropped) meshes
    self.traj_data_path = '' # best-trajectory data (pkl)
    self.vol_data_path = '' # volumes data (csv)
    self.surv_data_path = '' # csv used to access numbers / ratios of surviving trajectories ('accessibility') (csv)
    self.tgt_masks_path = '' # path to directory containing rasterised striatal nuclei masks
    self.dep_masks_path = '' # path to directory containing rasterised best-volume deposit masks
    self.native_x_path = '' # pre-calculated cortical intersections in native space (bary cds + tri IDs)
    self.fsavg_x_path = '' # pre-calculated cortical intersections in fsavg space (bary cds + tri IDs) from _preparation_transform_VDmax_trajectories.py
    
    self.traj_data = None # as above
    self.vol_data = None # as above, provided ventricles.csv is a curated version of this original
    self.surv_data = None # as above
    
    self.target_regions = ['caudate', 'anterior_putamen', 'posterior_putamen', 'full_putamen']
  
  def _load_traj_data(self, drop_empty = False, reset = False):
    
    if not self.traj_data is None and not reset:
      return
    df = pd.read_pickle(self.traj_data_path).dropna()
    if drop_empty:
      df = df[df['VDmax_merged'] != 0]
    self.traj_data = df
  
  def _load_vol_data(self):
    
    if not self.vol_data is None: return
    self.vol_data = pd.read_csv(self.vol_data_path)
  
  def _load_surv_data(self):
    
    if not self.surv_data is None: return
    self.surv_data = pd.read_csv(self.surv_data_path)
  
  def get_static(self, fields):
    
    if not isinstance(fields, list):
      fields = [fields]
    result = [getattr(self, field) for field in fields]
    if len(fields) == 1:
      result = result[0]
    return result
  
  def get_subs(self):
    
    self._load_traj_data()
    return pd.unique(self.traj_data['sub'])
  
  def get_traj_data(self, drop_empty = False, reset = False):
    
    self._load_traj_data(drop_empty, reset)
    return self.traj_data
  
  def get_talairach_vecs(self):
  
    talairach_vecs = {}
    for sid, sub in enumerate(self.get_subs()):
      inp_path = os.path.join(self.param_path, sub, sub + '_datapaths.json') # location of per-subject file containing paths
      if not os.path.exists(inp_path): continue
      with open(inp_path, 'r') as jf:
        js = json.load(jf)
      talairach = np.array(js['talairach'])
      talairach_vecs[sub] = unit(talairach[:3, :3].T)
    
    return talairach_vecs
  
  def get_traj_talairach(self, drop_empty = True, flip_left = True):
    
    df = self.get_traj_data(drop_empty=drop_empty).copy()
    talbasis = self.get_talairach_vecs()
    trajvecs = []
    for idx, row in df.iterrows():
      entry, target = [np.asarray(row[label]) for label in ['entry', 'target']]
      vec = unit(entry - target) # reversed direction here
      tvec = np.sum(vec * talbasis[row['sub']], axis = 1)
      if flip_left and row['side'] == 'left':
        tvec[0] *= -1
      trajvecs.append(unit(tvec))
    df['talairach'] = trajvecs
    return df
  
  def get_volume_data(self, drop_phen = True, return_long = False):
    
    self._load_vol_data()
    
    if drop_phen:
      data = self.vol_data.drop(['CAG', 'Female?', 'TFC',	'Age', 'TOPF',	'MOCA'], axis = 1)
    else:
      data = self.vol_data.copy()
    
    if return_long:
    
      # will return ['sub', 'ventricles', 'brain', 'side', 'target', 'vdmax', 'volume']
      # target containing caudate, putamen, putamen1, putamen2
      # volume only populated for caudate and putamen (but latter same for putamen1 and putamen2)
      
      dim_names=('side', 'target')

      def parse(col):
        parts = col.split('_')
        if len(parts) < 2:
          return None
        side = parts[0]
        target = parts[1]
        metric = '_'.join(parts[2:]) if len(parts) > 2 else 'volume'
        return side, target, metric

      measure_cols = [c for c in data.columns if parse(c) is not None]
      id_cols = [c for c in data.columns if c not in measure_cols]

      tmp = data.melt(
        id_vars=id_cols,
        value_vars=measure_cols,
        var_name='col',
        value_name='val'
      )

      tmp[[*dim_names, 'metric']] = pd.DataFrame(
        tmp['col'].map(parse).tolist(),
        index=tmp.index
      )

      tmp = tmp.pivot(index=id_cols + list(dim_names), columns='metric', values='val')
      data = tmp.reset_index().rename_axis(columns=None)
        
    return data

  def get_ventricular_volumes(self):
    
    df = self.get_volume_data()
    return dict(zip(df['sub'], df['ventricles']))
  
  def get_surv_data(self):
    
    self._load_surv_data()
    return self.surv_data

# ============
# VECTOR STUFF
# ============

# row-wise unit normalisation (safe for zero-length)
def unit(X):
  X = np.asarray(X, float)
  n = np.linalg.norm(X, axis=-1, keepdims=True)
  n[n == 0] = 1.
  return X / n

# intrinsic (Fréchet / Karcher) mean on S^2
# assumes data are contained in a hemisphere → unique solution
def spherical_mean(U, tol = 1e-10, itmax = 200):

  #row-wise unit normalization (safe for zero-length)
  def unit(X):
    X = np.asarray(X, float)
    n = np.linalg.norm(X, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

  U = unit(U)
  mu = unit(U.mean(axis=0))   # safe initialization under hemisphere condition

  for _ in range(itmax):
    dots = np.clip(U @ mu, -1.0, 1.0)
    theta = np.arccos(dots)

    # Tangent vectors at mu (log map)
    sin_theta = np.sin(theta)
    factor = np.ones_like(theta)
    mask = sin_theta > 1e-12
    factor[mask] = theta[mask] / sin_theta[mask]

    V = U - dots[:, None] * mu
    V *= factor[:, None]

    vbar = V.mean(axis=0)
    step = np.linalg.norm(vbar)

    if step < tol:
      break

    # Exponential map
    mu = np.cos(step) * mu + np.sin(step) * (vbar / step)
    mu = unit(mu)

  return mu

# build an orthonormal 2D basis (e1, e2) for the tangent plane at mu
def tangent_basis(mu): 
  
  mu = np.asarray(mu, float)
  a = np.array([1., 0., 0.], float)
  if abs(mu @ a) > .95: # not the case in our data
    a = np.array([0., 1., 0.], float)
  
  e1 = a - (mu @ a) * mu
  n1 = np.linalg.norm(e1)
  if n1 == 0: # extremely unlikely; choose another direction
    a = np.array([0., 0., 1.], float)
    e1 = a - (mu @ a) * mu
    n1 = np.linalg.norm(e1)
  e1 /= n1
  
  e2 = np.cross(mu, e1)
  n2 = np.linalg.norm(e2) # in case of numerical imprecisions
  if n2 == 0: # extremely unlikely; choose another direction
    a = np.array([0., 1., 0.], float)
    e2 = a - (mu @ a) * mu
    n2 = np.linalg.norm(e2)
  e2 /= n2
  
  if e2 @ np.array([0., 1., 0.]) < 0:
    e2 = -e2
  return e1, e2 # should be LAT, AP

# log map of points U (n x 3) to tangent plane T_mu S^2
# returns (Y2d (n x 2), r (n,)), where r are geodesic distances (radians)
def log_map(mu, U): 
  
  mu = np.asarray(mu, float)
  U = unit(U)
  
  # angles
  cosang = np.clip(U @ mu, -1., 1.)
  r = np.arccos(cosang)  # geodesic distances
  
  # tangent directions [norm(U - outer) is also sinang]
  Tdir = unit(U - np.outer(cosang, mu)) # normalised direction vector
  
  # 2D coordinates in a fixed tangent basis
  e1, e2 = tangent_basis(mu)
  V = (r[:, None]) * Tdir # direction vector of length r
  v1 = V @ e1
  v2 = V @ e2
  Y2d = np.column_stack([v1, v2])
  
  zpts = np.all(U == 0, axis = 1)
  if np.any(zpts):
    o, ro = log_map([0, 0, 0], [-mu])
    Y2d[zpts] = o
    r[zpts] = ro
  
  return Y2d, r

# exponential map to project a 2D tangent vector (per obs) back to S^2 endpoint
# accepts a single 2D vector; returns a 3D unit vector on the sphere
def exp_map(mu, v2d, scale = 1.):
  
  mu = np.asarray(mu, float)
  e1, e2 = tangent_basis(mu)
  v = scale * (v2d[0] * e1 + v2d[1] * e2)  # 3D tangent vector
  nv = np.linalg.norm(v)
  if nv == 0:
    return mu.copy()
  return np.cos(nv) * mu + np.sin(nv) * (v / nv)

# ============
# MESH STUFF
# ============

def read_fs_mesh(mesh_path):
  
  v, t, m = nib.freesurfer.io.read_geometry(mesh_path, read_metadata=True)
  v += m['cras']
  return v, t

def load_fsaverage_mesh(surface_type, hemi = 'lh'):

  surface_path = os.path.join(Common().fsdir, 'fsaverage', 'surf', hemi + '.' + surface_type)
  v, t = read_fs_mesh(surface_path)
  mesh = trimesh.Trimesh(vertices = v, faces = t, process = False)
  _ = mesh.vertex_normals
  return mesh

def prepare_meshes(surface_type = 'inflated', interhemisphere_gap = None):

  lh = load_fsaverage_mesh(surface_type, 'lh')
  rh = load_fsaverage_mesh(surface_type, 'rh')
  if not interhemisphere_gap is None:
    x_span_lh = lh.vertices[:, 0].max() - lh.vertices[:, 0].min()
    x_span_rh = rh.vertices[:, 0].max() - rh.vertices[:, 0].min()  
    x_span = x_span_lh + x_span_rh + interhemisphere_gap
    lh.vertices[:, 0] += -lh.vertices[:, 0].min() - x_span / 2
    rh.vertices[:, 0] += -rh.vertices[:, 0].max() + x_span / 2
  return {'left': lh, 'right': rh}

def map_point_values(pt_tidx, pt_uvs, mesh_tris, nv = None):

  nv = np.max(mesh_tris) if nv is None else nv
  f_vertex = np.zeros(nv)
  for tidx, uvw in zip(pt_tidx, pt_uvs):
    vidx = mesh_tris[tidx]
    for d in range(3):
      f_vertex[vidx[d]] += uvw[d]
  return f_vertex


# ============
# COLOUR STUFF
# ============

def srgb_to_linear(c):
  c = c / 255.
  return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(c):
  c = np.clip(c, 0., 1.)
  return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c**(1/2.4) - 0.055) * 255.

def rgb_to_oklab(rgb):
  rgb = np.asarray(rgb)
  r, g, b = [srgb_to_linear(c) for c in rgb.T]
  l = 0.4122214708*r + 0.5363325363*g + 0.0514459929*b
  m = 0.2119034982*r + 0.6806995451*g + 0.1073969566*b
  s = 0.0883024619*r + 0.2817188376*g + 0.6299787005*b
  l_, m_, s_ = np.sign(l) * np.abs(l)**(1/3), \
               np.sign(m) * np.abs(m)**(1/3), \
               np.sign(s) * np.abs(s)**(1/3)
  return np.stack([
    0.2104542553*l_ + 0.7936177850*m_ - 0.0040720468*s_,
    1.9779984951*l_ - 2.4285922050*m_ + 0.4505937099*s_,
    0.0259040371*l_ + 0.7827717662*m_ - 0.8086757660*s_
  ]).T

def oklab_to_rgb(Lab):
  L, a, b = Lab.T
  l_ = L + 0.3963377774*a + 0.2158037573*b
  m_ = L - 0.1055613458*a - 0.0638541728*b
  s_ = L - 0.0894841775*a - 1.2914855480*b
  l = l_**3
  m = m_**3
  s = s_**3
  r =  4.0767416621*l - 3.3077115913*m + 0.2309699292*s
  g = -1.2684380046*l + 2.6097574011*m - 0.3413193965*s
  b = -0.0041960863*l - 0.7034186147*m + 1.7076147010*s
  return np.clip(np.vstack([linear_to_srgb(c) for c in (r, g, b)]).T, 0., 255.)


# =============
# OTHER
# =============

def slab(p):

  if p < 0.001:
    return "***"
  elif p < 0.01:
    return "**"
  elif p < 0.05:
    return "*"
  else:
    return "ns"
