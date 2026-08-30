import numpy as np
import os, trimesh
import nibabel as nib
from multiprocessing import Pool

from common import Common

fsdir, target_regions = Common().get_static(['fsdir', 'target_regions'])
native_cortical_intersections, fsavg_cortical_intersections = Common().get_static(['native_x_path', 'fsavg_x_path'])

def points_to_fsaverage_space(inputs):
  
  sub, side = inputs
  
  # load data
  hemi = 'lh' if side == 'left' else 'rh'
  native_sphere_path = os.path.join(fsdir, sub, 'surf', hemi + '.sphere.reg')
  fsavg_sphere_path = os.path.join(fsdir, 'fsaverage', 'surf', hemi + '.sphere')
  
  native_sphere_verts, native_tris = nib.freesurfer.io.read_geometry(native_sphere_path)
  fsavg_sphere_verts, fsavg_tris = nib.freesurfer.io.read_geometry(fsavg_sphere_path)
  fsavg_sphere_mesh = trimesh.Trimesh(vertices=fsavg_sphere_verts, faces=fsavg_tris, process=False)
  
  # accrue all intersection points irrespective of target region
  tris, uvs, tids = [], [], []
  for tid, region in enumerate(target_regions):
    tris.append(np.load(os.path.join(native_cortical_intersections, sub, side, region, 'tris.npy')))
    uvs.append(np.load(os.path.join(native_cortical_intersections, sub, side, region, 'uvs.npy')))
    tids.append(np.full(tris[-1].size, tid)) 
  tris = np.hstack(tris)
  uvs = np.vstack(uvs)
  tids = np.hstack(tids)

  # calculate intersection points in registration sphere coordinates
  # essentially transform to points on the "unit" sphere (actually r~100) that native mesh is mapped to
  v_idx = native_tris[tris]
  pts_on_sphere = (uvs[:, :, None] * native_sphere_verts[v_idx]).sum(axis = 1)

  # perform transformation
  closest_pts, distances, tri_indices = fsavg_sphere_mesh.nearest.on_surface(pts_on_sphere)
  tris_fs = fsavg_tris[tri_indices] # shape (len(tri_indices), 3)
  verts_fs = fsavg_sphere_verts[tris_fs] # shape (len(tri_indices), 3, 3)
  fs_uvs = trimesh.triangles.points_to_barycentric(verts_fs, closest_pts)
  fs_tris = tri_indices.tolist()
  
  # reconstruct per-region and save
  for tid, region in enumerate(target_regions):
    outdir = os.path.join(fsavg_cortical_intersections, region, side, sub)
    if not os.path.exists(outdir):
      os.makedirs(outdir, exist_ok = True)
    np.save(os.path.join(outdir, 'fs_tris.npy'), np.array(fs_tris)[tids == tid])
    np.save(os.path.join(outdir, 'fs_uvs.npy'), fs_uvs[tids == tid])
  
  print(f'{sub} {side} computed successfully')

if __name__ == '__main__':

  inputs = []
  for sub in os.listdir(native_cortical_intersections):
    for side in ['left', 'right']:
      ss_path = os.path.join(native_cortical_intersections, sub, side)
      if os.path.exists(ss_path) and len(os.listdir(ss_path)):
        inputs.append([sub, side])

  with Pool(4) as p:
    p.map(points_to_fsaverage_space, inputs)
