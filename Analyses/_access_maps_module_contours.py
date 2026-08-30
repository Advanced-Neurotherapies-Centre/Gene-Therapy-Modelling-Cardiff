import numpy as np
import trimesh
from collections import defaultdict

# convert unordered (N,2,3) segments into ordered polylines
def stitch_segments(segments, tol=1e-8):
  
  # helper for hashing points with tolerance
  def key(p):
    return tuple((p / tol).round().astype(int))
  
  adj = defaultdict(list)
  
  for i, (p0, p1) in enumerate(segments):
    k0 = key(p0)
    k1 = key(p1)
    adj[k0].append((i, 0))
    adj[k1].append((i, 1))
  
  used = set()
  polylines = []
  
  for i in range(len(segments)):
    if i in used:
      continue
    
    line = [segments[i, 0], segments[i, 1]]
    used.add(i)
    
    # extend forward
    end = key(line[-1])
    while True:
      nxt = [x for x in adj[end] if x[0] not in used]
      if not nxt:
        break
      j, side = nxt[0]
      used.add(j)
      line.append(segments[j][1 - side])
      end = key(line[-1])
    
    # extend backward
    start = key(line[0])
    while True:
      nxt = [x for x in adj[start] if x[0] not in used]
      if not nxt:
        break
      j, side = nxt[0]
      used.add(j)
      line.insert(0, segments[j][1 - side])
      start = key(line[0])
    
    polylines.append(np.asarray(line))
  
  return polylines

# create a continuous tube mesh from a 3D polyline
def tube_from_polyline(P, radius=1., sections=12):
  
  P = np.asarray(P)
  n = len(P)
  if n < 2:
    return None

  # tangents
  T = np.zeros_like(P)
  T[1:-1] = P[2:] - P[:-2]
  T[0] = P[1] - P[0]
  T[-1] = P[-1] - P[-2]
  T = T / np.linalg.norm(T, axis=1, keepdims=True)
  
  # pick a stable initial normal
  normal = np.array([0, 0, 1.])
  if abs(np.dot(normal, T[0])) > .9:
    normal = np.array([0, 1., 0])
  
  # frames
  N = np.zeros_like(P)
  B = np.zeros_like(P)
  for i in range(n):
    N[i] = np.cross(T[i], normal)
    N[i] /= np.linalg.norm(N[i])
    B[i] = np.cross(T[i], N[i])
    normal = N[i]
  
  # vertices
  angles = np.linspace(0, 2 * np.pi, sections, endpoint=False)
  circle = np.stack([np.cos(angles), np.sin(angles)], axis=1)
  
  vertices = []
  for i in range(n):
    for c in circle:
      vertices.append(
        P[i] + radius * (c[0]*N[i] + c[1]*B[i])
      )
  vertices = np.asarray(vertices)

  # faces
  faces = []
  for i in range(n - 1):
    for j in range(sections):
      a = i*sections + j
      b = i*sections + (j+1) % sections
      c = (i+1)*sections + j
      d = (i+1)*sections + (j+1) % sections
      faces.append([a, c, b])
      faces.append([b, c, d])
  
  return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

def contour_tubes_from_polylines(polylines, radius=1., sections=12):

  meshes = []
  for P in polylines:
    tube = tube_from_polyline(P, radius, sections)
    if tube is None:
      continue
    meshes.append(tube)
  if not meshes:
    return None
  return trimesh.util.concatenate(meshes)

def contour_as_trimesh_path(V, F, vertex_bool, colour=np.array((0, 0, 0, 255), dtype=np.uint8), radius = 1., sections = 6, iso = .5):
  
  vals = vertex_bool.astype(float)
  segments = []
  for f in F:
    idx = f
    p = V[idx]
    v = vals[idx]
    above = v >= iso
    mask = above.astype(int)
    case = mask[0] + 2 * mask[1] + 4 * mask[2]
    if case == 0 or case == 7:
      continue  # no crossing
    pts = []
    
    # helper: interpolate along edge (i, j)
    def interp(i, j):
      vi, vj = v[i], v[j]
      t = (iso - vi) / (vj - vi)
      return p[i] + t * (p[j] - p[i])
    
    # edge cases (standard marching triangles table)
    if case in (1, 6):
      pts.append(interp(0, 1))
      pts.append(interp(0, 2))
    elif case in (2, 5):
      pts.append(interp(1, 0))
      pts.append(interp(1, 2))
    elif case in (3, 4):
      pts.append(interp(2, 0))
      pts.append(interp(2, 1))
    if len(pts) == 2:
      segments.append(pts)
  
  contour = contour_tubes_from_polylines(stitch_segments(np.asarray(segments)), radius, sections)
  contour = trimesh.Trimesh(contour.vertices, contour.faces[:, [1, 0, 2]], process = True)
  _ = contour.vertex_normals
  #contour.visual.face_colors = colour
  contour.visual.vertex_colors = np.tile(colour, (len(contour.vertices), 1))
  return contour