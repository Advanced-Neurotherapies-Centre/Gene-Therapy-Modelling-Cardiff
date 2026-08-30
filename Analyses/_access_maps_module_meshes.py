import trimesh
import numpy as np
from scipy.sparse import coo_matrix, identity, diags
from scipy.sparse.linalg import spsolve

def prepare_areas(meshes):

  areas = {}
  for side in ['left', 'right']:
    ver_areas = np.zeros(len(meshes[side].vertices))
    tri_areas = meshes[side].area_faces
    for tri, A in zip(meshes[side].faces, tri_areas):
      ver_areas[tri] += A / 3.
    areas[side] = ver_areas
  return areas

def prepare_mass(meshes):

  areas = prepare_areas(meshes)
  mass = {}
  for side in ['left', 'right']:
    nv = len(areas[side])
    mass[side] = coo_matrix((areas[side], (np.arange(nv), np.arange(nv))), shape=(nv, nv))
  return mass

def prepare_laplacian(meshes, lapl_eqwt = True):

  M = prepare_mass(meshes)
  L = {}
  for side in ['left', 'right']:
    L_stiff = trimesh.smoothing.laplacian_calculation(meshes[side], equal_weight=lapl_eqwt)
    d = np.array(L_stiff.sum(axis=1)).ravel()
    L_stiff = L_stiff - diags(d)
    L[side] = diags(1. / M[side].diagonal()) @ L_stiff
  return L

def smooth_function_laplacian(L, f, lam=.3):

  f = f.astype(float).copy()
  I = identity(L.shape[0], format=L.format, dtype=L.dtype)
  return spsolve(I - lam * L, f)