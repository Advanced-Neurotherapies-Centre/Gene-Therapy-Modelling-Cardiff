import os, trimesh, json
import numpy as np
import nibabel as nib
from multiprocessing import Pool

from common import Common

common = Common()
fsdir, param_path = common.get_static(['fsdir', 'param_path'])
ref_path = lambda sub: os.path.join(fsdir, sub, 'mri', 'T1.mgz')
common.load_traj_data()

# given N deposits, generate a discrete 2D contour of upper half of cross-section
# and optionally merge using pairwise mutual tangents (consecutive pairs only)
def generate_2d_contour(deposit_centres, deposit_radii, xfm = np.eye(4), number_of_bins = 500, merge = True):

  # rotate points, sort, get their absolute coordinates
  rotated = (xfm[:3, :3] @ deposit_centres.T).T + xfm[:3, 3]
  x_coords = rotated[:, 2]
  sort_idx = np.argsort(x_coords)
  x_sorted = x_coords[sort_idx]
  r_sorted = deposit_radii[sort_idx]

  # generate bins
  x_min = x_sorted[0] - r_sorted[0]
  x_max = x_sorted[-1] + r_sorted[-1]
  x_cds = np.linspace(x_min, x_max, number_of_bins)
  y_cds = np.zeros(number_of_bins)
  
  # populate with deposits
  for i, x in enumerate(x_sorted):
    X = x_cds - x
    Y = r_sorted[i] * r_sorted[i] - X * X
    Y[Y < 0] = 0
    Y[Y > 0] = np.sqrt(Y[Y > 0])
    y_cds = np.maximum(y_cds, Y)  

    if merge and i:
      centre_distance = x - x_sorted[i - 1]
      safety = (centre_distance + r_sorted[i - 1]) > r_sorted[i] and (centre_distance + r_sorted[i]) > r_sorted[i - 1]
      if safety: # ensures circles never fully within each other - but should not happen with this code
        tangent_length = (centre_distance ** 2 - (r_sorted[i] - r_sorted[i - 1]) ** 2) ** 0.5
        slope = (r_sorted[i] - r_sorted[i - 1]) / tangent_length
        denom = (1 + slope ** 2) ** 0.5
        # NB! this is only valid given x_last always < x_current
        x_tangent_point_last = x_sorted[i - 1] - r_sorted[i - 1] * slope / denom
        x_tangent_point_current = x - r_sorted[i] * slope / denom
        intercept = r_sorted[i] / denom - x_tangent_point_current * slope
        x_valid = (x_cds >= x_tangent_point_last) & (x_cds <= x_tangent_point_current)
        x_tn = x_cds[x_valid]
        y_tn = x_tn * slope + intercept
        y_cds[x_valid] = np.maximum(y_cds[x_valid], y_tn)
      else:
        ValueError('deposits within each other')

  return np.array([x_cds, y_cds]).T

# using the 2D contour above, create a mesh for the resulting solid of revolution
def revolve_points(points_2d, xfm = np.eye(4), n_theta = 64):
  
  xs, ys = points_2d[:, 0], points_2d[:, 1]
  theta = np.linspace(0, 2 * np.pi, n_theta, endpoint = False)
  
  meshes = []
  start = 0
  tol = 1e-8
  while start < len(xs):
    
    # skip regions where y == 0
    while start < len(xs) and abs(ys[start]) < tol:
      start += 1
    if start >= len(xs):
      break
    end = start
    while end < len(xs) and abs(ys[end]) >= tol:
      end += 1
    
    xs_seg = xs[start:end]
    ys_seg = ys[start:end]
    if len(xs_seg) < 2:
      start = end
      continue
    
    # generate 3D vertices for this segment
    xx, tt = np.meshgrid(xs_seg, theta)
    yy = np.tile(ys_seg, (n_theta, 1))
    verts = np.column_stack([
      (yy * np.cos(tt)).ravel(),
      (yy * np.sin(tt)).ravel(),
      xx.ravel()
    ])

    # create faces (quad → two triangles)
    n_x = len(xs_seg)
    faces = []
    for i in range(n_theta):
      i_next = (i + 1) % n_theta
      for j in range(n_x - 1):
        a = i * n_x + j
        b = i_next * n_x + j
        c = i * n_x + j + 1
        d = i_next * n_x + j + 1
        faces.append([a, b, c])
        faces.append([b, d, c])
    
    # cap from both sides
    caps = []
    if ys_seg[0] > 0:
      cap_start_idx = len(verts)
      verts = np.vstack([verts, [0, 0, xs_seg[0]]])
      for i in range(n_theta):
        i_next = (i + 1) % n_theta
        a = i * n_x
        b = i_next * n_x
        caps.append([cap_start_idx, b, a])
    
    if ys_seg[-1] > 0:
      cap_end_idx = len(verts)
      verts = np.vstack([verts, [0, 0, xs_seg[-1]]])
      for i in range(n_theta):
        i_next = (i + 1) % n_theta
        a = i * n_x + (n_x - 1)
        b = i_next * n_x + (n_x - 1)
        caps.append([cap_end_idx, a, b])
    
    faces.extend(caps)
        
    # return the X-aligned body of revolution back to its original trajectory
    ixfm = np.linalg.inv(xfm)
    verts = (ixfm[:3, :3] @ verts.T).T + ixfm[:3, 3]

    # convert to mesh
    mesh = trimesh.Trimesh(vertices = verts, faces = np.array(faces)[:, [1, 0, 2]], process = True)
    meshes.append(mesh)
    start = end
  
  # combine all connected components
  if not meshes:
    return None
  if len(meshes) == 1:
    return meshes[0]
  return trimesh.util.concatenate(meshes)

# wrapper combining the two functions above with some edge cases
def make_model(deposit_centres, deposit_radii, xfm = np.eye(4), out_path = None, merge = True):

  if len(deposit_radii):
    points_2d = generate_2d_contour(deposit_centres, deposit_radii, xfm = xfm, number_of_bins = 200, merge = merge)
    if np.all(np.abs(points_2d[:, 1]) < 1e-8):
      model = trimesh.Trimesh()
    else:
      model = revolve_points(points_2d, xfm, n_theta = 25)
  else:
    model = trimesh.Trimesh()
  if not out_path is None:
    trimesh.exchange.export.export_mesh(model, file_obj=out_path, include_normals = True)
  return model

# rasterise the 3D mesh into a non-binary mask using reference volume
def rasterise_model(model, ref_path, invert = False):

  ref = nib.load(ref_path)
  affine = ref.affine
  shape = ref.shape

  # model bb in vox cds
  world_to_vox = np.linalg.inv(affine)
  verts_vox = nib.affines.apply_affine(world_to_vox, model.vertices)
  minv = np.floor(verts_vox.min(axis=0)).astype(int)
  maxv = np.ceil(verts_vox.max(axis=0)).astype(int)

  # get voxel centre cds and bb mask
  ijk = np.indices(shape).reshape(3, -1).T
  xyz = nib.affines.apply_affine(affine, ijk)
  vox_mask = np.all((ijk >= (minv - 1)) & (ijk <= (maxv + 1)), axis = 1)
  #print(len(vox_mask), flush = True)

  # signed distance at each voxel center
  dist = model.nearest.signed_distance(xyz[vox_mask])

  # convert to 0–1 occupancy using a soft boundary kernel
  voxel_size = np.mean(ref.header.get_zooms())
  bb_mask_vals = np.clip(0.5 - dist / (2*voxel_size), 0, 1)
  if invert:
    bb_mask_vals = 1. - bb_mask_vals

  # restore original mask
  mask = np.zeros(shape, dtype=np.float32)
  mask[*ijk[vox_mask].T] = bb_mask_vals  
  return nib.Nifti1Image(mask, affine, ref.header)

def make_model_img(row):
  
  # prep trajectory data
  entry = np.array(row['entry'])
  target = np.array(row['target'])
  if np.any(np.isnan(entry) | np.isnan(target)) or np.isnan(row['VDmax_merged']) or row['VDmax_merged'] == 0:
    return
  tvec = target - entry
  tlen = np.linalg.norm(tvec)
  tvec /= tlen
  centres = entry[None, :] * (1 - np.array(row['deposit centres']))[:, None] + target[None, :] * np.array(row['deposit centres'])[:, None]
  radii = np.array(row['deposit radii']) * tlen
  
  # prepare trajectory transform
  zvec = -tvec
  yvec = np.cross([1, 0, 0], zvec)
  yvec /= np.linalg.norm(yvec)
  xvec = np.cross(zvec, yvec)
  xfm_mat = np.eye(4)
  xfm_mat[:3, :3] = np.array([xvec, yvec, zvec]).T
  xfm_mat[:3, 3] = np.mean(centres, axis = 0)
  inv_aff = np.linalg.inv(xfm_mat)
  
  # make model
  model = make_model(centres, radii, xfm = inv_aff)
  
  # make image
  mask = rasterise_model(model, ref_path(row['sub']), invert = True)
  
  # save image
  outpath = f'{row["sub"]}_{row["side"]}_{row["trajectory"]}.nii.gz' # replace as needed
  if not os.path.exists(os.path.dirname(outpath)):
    os.makedirs(os.path.dirname(outpath), exist_ok = True)
  nib.save(mask, outpath)
  print(f'mask for {row["sub"]} {row["side"]} {row["trajectory"]} (deposit) completed', flush=True)
  
def make_target_img(path):

  model = [v for v in trimesh.load(path['path']).geometry.values()]
  model = trimesh.util.concatenate(model)
  mask = rasterise_model(model, ref_path(path['sub']), invert = True)
  
  outpath = f'{path["sub"]}_{path["side"]}_{path["target"]}.nii.gz' # replace as needed
  if not os.path.exists(os.path.dirname(outpath)):
    os.makedirs(os.path.dirname(outpath), exist_ok = True)
  nib.save(mask, outpath)
  print(f'mask for {path["sub"]} {path["side"]} {path["target"]} (target) completed', flush=True)

if __name__ == "__main__":
  
  # deposits first
  dict_list = common.traj_data.dropna().to_dict(orient='records')
  ex = lambda r: False # replace with path checker to exclude missing
  with Pool(4) as p:
    p.map(make_model_img, [r for r in dict_list if not ex(r)])

  # targets next (similar to above for compatibility, as opposed to taking the binary FS masks)
  tgt_mesh_paths = []
  for sub in common.get_subs():
    with open(os.path.join(param_path, sub, sub + '_datapaths.json'), 'r') as jf:
      js = json.load(jf)
    for side in ['left', 'right']:
      for target in ['caudate', 'putamen']:
        tgt_mesh_paths.append({
          'sub': sub,
          'side': side,
          'target': target,
          'path': js['surfaces'][side][target]
        })
  with Pool(4) as p:
    p.map(make_target_img, tgt_mesh_paths)  