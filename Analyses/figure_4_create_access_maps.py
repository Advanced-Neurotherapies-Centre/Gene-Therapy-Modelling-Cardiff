import numpy as np
import pandas as pd
import os, trimesh
from multiprocessing import Pool
import statsmodels.formula.api as smf

from _access_maps_module_meshes import prepare_laplacian, smooth_function_laplacian
from _access_maps_module_colours import build_colour_bar, apply_vertex_mask, custom_colours
from _access_maps_module_contours import contour_as_trimesh_path

from common import Common, prepare_meshes, map_point_values

# set up
map_type    = 'composite' # composite or individual
thr         = -25 # neg for percentile, 0+ for absolute number of traj

lapl_eqwt   = False
lam_subj    = 0.
lam_cont    = 10.

radius      = .5
sections    = 6
cdiff_scale = 1 # e.g. for 1 the scale will be [-100% to 100%]

A = 26 * 4
BG      = np.array([120, 120, 120, 255], dtype=np.uint8)
NEUTRAL = np.array([ 50,  50,  50,   A], dtype=np.uint8)
BLUE    = np.array([  0, 114, 176,   A], dtype=np.uint8) # VV
GREEN   = np.array([ 85, 168, 104,   A], dtype=np.uint8) # BV
ORANGE  = np.array([221, 132,  82,   A], dtype=np.uint8) # TV
colours = [NEUTRAL, BLUE, GREEN, ORANGE]

meshes = prepare_meshes('pial')
L = prepare_laplacian(meshes, lapl_eqwt)

fsdir, target_regions, fsavg_cortical_intersections = Common().get_static(['fsdir', 'target_regions', 'fsavg_x_path'])

def process_maps(i):
  nv = len(meshes[i['side']].vertices)
  mesh_tris = meshes[i['side']].faces
  f = map_point_values(i['tri'], i['uvs'], mesh_tris, nv)
  return f

def fit_per_vertex(vdf):
  model = smf.ols('val ~ 1 + VV + BV + TV + BV:TV', data = vdf).fit()
  pred_df = pd.DataFrame({
    'VV': [0, 1, 0, 0],
    'BV': [0, 0, 1, 0],
    'TV': [0, 0, 0, 1],
  })
  return model.predict(pred_df)

if __name__ == '__main__':
    
  output_path = 'outputs' # replace as appropriate
  if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok = True)
  
  build_colour_bar(custom_colours(None, 'return_hot2cold'), os.path.join(output_path, 'colour_bar.svg'))
  #import sys; sys.exit()
  
  # just read the volume data (independent variables)
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
  dfv = dfv.drop([col for col in dfv.columns if 'vdmax' in col], axis = 1)
  for col in ['VV', 'BV', 'caudate', 'putamen']:
    sd = dfv[col].std(ddof=0)
    dfv[col] = (dfv[col] - dfv[col].mean()) / (sd if sd > 0 else 1.)
  dfv['BVxCV'] = dfv['BV'] * dfv['caudate']
  dfv['BVxPV'] = dfv['BV'] * dfv['putamen']

  # get point distributions and generate surface maps
  process_inputs = []
  for target in target_regions:
    for side in ['left', 'right']:
      hemi = 'lh' if side == 'left' else 'rh'
      tris, uvs = [], []
      for sub in os.listdir(os.path.join(fsavg_cortical_intersections, target, side)):
        if not all([os.path.exists(os.path.join(fsavg_cortical_intersections, target, side, sub, npy + '.npy')) for npy in ['fs_tris', 'fs_uvs']]):
          continue
        tris_npy = np.load(os.path.join(fsavg_cortical_intersections, target, side, sub, 'fs_tris.npy'))
        uvs_npy = np.load(os.path.join(fsavg_cortical_intersections, target, side, sub, 'fs_uvs.npy'))
        process_inputs.append({
          'target': target,
          'side': side,
          'sub': sub,
          'tri': tris_npy,
          'uvs': uvs_npy
        })
    
  with Pool() as p:
    surface_functions = p.map(process_maps, process_inputs)

  # add surface function per entry
  dfs = pd.DataFrame(process_inputs)
  dfs = dfs.drop(['tri', 'uvs'], axis = 1)
  dfs['function'] = surface_functions
  merged = dfs.merge(dfv, on=['sub', 'side'], how='left')
  merged['TV'] = np.where(
    merged['target'] == 'caudate',
    merged['caudate'],
    merged['putamen']
  )
  merged['TVxBV'] = np.where(
    merged['target'] == 'caudate',
    merged['BVxCV'],
    merged['BVxPV']
  )
  df = merged[
    ['sub', 'side', 'target', 'function', 'VV', 'BV', 'TV', 'TVxBV']
  ]

  for target in target_regions:

    # compute contrasts
    contrasts = []
    for side in ['left', 'right']:
      sdf = df.loc[(df['target'] == target) & (df['side'] == side)]
      function = np.vstack(sdf['function'].values)
      valid_verts = np.where(np.any(function, axis = 0))[0]
      vertex_inputs = []
      for idx in valid_verts:
        vdf = sdf[['VV', 'BV', 'TV', 'TVxBV']].copy()
        vdf['val'] = function[:, idx]
        vertex_inputs.append(vdf)
      with Pool() as p:
        vertex_contrasts = np.vstack(p.map(fit_per_vertex, vertex_inputs))
      contrasts.append({
        'side': side,
        'valid': valid_verts,
        'contrasts': vertex_contrasts
      })
    
    scene = trimesh.Scene()
    
    # compute composite meshes
    if map_type == 'composite':   
    
      for sidx, side in enumerate(['left', 'right']):
        mesh = meshes[side].copy()
        mesh.visual.vertex_colors = np.tile(BG, (function.shape[1], 1))
        boundaries = []
        for c in range(4):
          fun = np.zeros(function.shape[1])
          fun[contrasts[sidx]['valid']] = contrasts[sidx]['contrasts'][:, c]
          fun[fun < 0.] = 0.
          fun = smooth_function_laplacian(L[side], fun, lam_cont) 
          fun = fun >= .5  #0. #5
          
          if c: 
            apply_vertex_mask(mesh, fun, colours[c])
            boundaries.append(contour_as_trimesh_path(mesh.vertices, mesh.faces, fun, (*colours[c][:3], 100), radius = radius, sections = sections))
          else: # do not colour in the neutral
            boundaries.append(contour_as_trimesh_path(mesh.vertices, mesh.faces, fun, (*colours[c][:3], 255), radius = radius, sections = sections))
        scene.add_geometry(mesh)
        _ = [scene.add_geometry(b) for b in boundaries]
      scene.export(os.path.join(output_path, '_'.join([target, 'composite', 'marginal.glb'])))
    
    # compute individual meshes
    elif map_type == 'individual':
      
      c_hemi = [c['contrasts'][:, 1:] - c['contrasts'][:, 0][:, None] for c in contrasts]
      max_c = np.max(np.abs(np.vstack(c_hemi))[:, 0])
      max_c = max(max_c, 1e-8)
      for pidx, pred in enumerate(['VV', 'BV', 'TV']):
        scene = trimesh.Scene()
        for sidx, side in enumerate(['left', 'right']):
          base_colour = np.array(custom_colours(.5, 'hot2cold_return')[1].ravel().tolist() + [255]).astype(int)
          colours = np.tile(base_colour, (function.shape[1], 1))
          colours[contrasts[sidx]['valid']] = custom_colours(c_hemi[sidx][:, pidx] / max_c + .5, 'hot2cold')
          mesh = meshes[side].copy()
          mesh.visual.vertex_colors = colours
          scene.add_geometry(mesh)
        out_path = os.path.join(output_path, '_'.join([target, pred, 'marginal.glb']))
        scene.export(out_path)
        print(f'{out_path} saved')
