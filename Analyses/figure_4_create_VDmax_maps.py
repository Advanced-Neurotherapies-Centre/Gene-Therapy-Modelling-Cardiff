import numpy as np
import os, trimesh

from common import Common, prepare_meshes, map_point_values

fsdir, target_regions, fsavg_cortical_intersections = Common().get_static(['fsdir', 'target_regions', 'fsavg_x_path'])

def place_points(data, pt_tidx, pt_uvs, mesh_v, mesh_tris, radius = 1., clim = None):

  geometry = trimesh.Trimesh(mesh_v, mesh_tris)
  _ = geometry.vertex_normals
  geometry.visual.vertex_colors = np.tile([128, 128, 128], (len(geometry.vertices), 1))
  geometry = [geometry]

  pt_uvs = np.asarray(pt_uvs)
  cds = mesh_v[mesh_tris[pt_tidx, 0]] * pt_uvs[:, 0][:, None] + \
        mesh_v[mesh_tris[pt_tidx, 1]] * pt_uvs[:, 1][:, None] + \
        mesh_v[mesh_tris[pt_tidx, 2]] * pt_uvs[:, 2][:, None]
  
  if not clim is None:
    data = np.clip(data, *clim)
  data -= np.nanmin(data)
  colours = custom_colours(data / np.nanmax(data))

  for (p, col) in zip(cds, colours):
    sphere = trimesh.creation.icosphere(subdivisions=0, radius=radius)
    sphere.apply_translation(p)
    sphere.visual.vertex_colors = np.tile(col, (len(sphere.vertices), 1))
    geometry.append(sphere)

  return geometry

def custom_colours(values):

  values = np.clip(values, 0., 1.)

  """
  import matplotlib.pyplot as plt
  cmap = plt.get_cmap('hot') 
  rgba_float = cmap(values)
  return (rgba_float * 255).astype(np.uint8)
  """

  colours = np.zeros((len(values), 4))
  colours[:, 0] = np.where(values < .5, 2 * values, 1.)
  colours[values >= .5, 1] = 2 * values[values >= .5] - 1.
  colours[:, -1] = 1.
  return (colours * 255).astype(np.uint8)

if __name__ == '__main__':
  
  # select: traj_prob or best_vvol
  # traj_prob = trajectory probability map
  # best_vvol = best trajectory map + colour code by ventricular volume
  map_type = 'traj_prob' 
  
  meshes = prepare_meshes(surface_type = 'pial')
  
  if map_type == 'best_vvol': 
    val_scale = lambda x: np.hstack(x) 
    vvols = Common().get_ventricular_volumes()
    clim = [0., max(vvols.values())] # use to manually clip if desired
  
  output_path = map_type # amend as appropriate
  if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok = True)
  for target in target_regions:
    scene = trimesh.Scene()
    for side in ['left', 'right']:
      hemi = 'lh' if side == 'left' else 'rh'
      tris, uvs, vdmax = [], [], []
      
      # load cortical entry locations for all subjects for this target & side
      for sub in os.listdir(os.path.join(fsavg_cortical_intersections, target, side)):
        if not all([os.path.exists(os.path.join(fsavg_cortical_intersections, target, side, sub, npy + '.npy')) for npy in ['fs_tris', 'fs_uvs', 'vdmax_vals']]):
          continue
        tris_npy = np.load(os.path.join(fsavg_cortical_intersections, target, side, sub, 'fs_tris.npy'))
        uvs_npy = np.load(os.path.join(fsavg_cortical_intersections, target, side, sub, 'fs_uvs.npy'))
        if not len(tris_npy) or not len(uvs_npy): 
          continue
        tris.append(tris_npy)
        uvs.append(uvs_npy)
        if map_type == 'best_vvol': # collect vdmax per trajectory to then select tri/bary of the best trajectory
          vals = np.load(os.path.join(fsavg_cortical_intersections, target, side, sub, 'vdmax_vals.npy'))
          vdmax.append(vals)
      
      if map_type == 'traj_prob':
        scalars = map_point_values(
          np.hstack(tris),
          np.vstack(uvs), 
          meshes[side].faces, 
          nv = len(meshes[side].vertices)
        )
        #np.save(os.path.join(output_path, f'{hemi}.{target}.npy'), scalars)
        meshes[side].visual.vertex_colors = custom_colours(scalars / np.nanmax(scalars))
        scene.add_geometry(meshes[side])
      
      elif map_type == 'best_vvol':
        d, t, u = [], [], []
        data = [vvols[sub] for sub in os.listdir(os.path.join(fsavg_cortical_intersections, target, side))]
        for i, v in enumerate(vdmax):
          if not len(v): continue
          idx = np.argmax(v)
          t.append(tris[i][idx])
          u.append(uvs[i][idx])
          d.append(data[i])
        geometry = place_points(d, t, u, meshes[side].vertices, meshes[side].faces, clim = clim)
        _ = [scene.add_geometry(m) for m in geometry]
    
    scene.export(os.path.join(output_path, f'{target}.glb'))
