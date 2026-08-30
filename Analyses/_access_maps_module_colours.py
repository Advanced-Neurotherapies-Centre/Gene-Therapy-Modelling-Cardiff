import numpy as np

from common import rgb_to_oklab, oklab_to_rgb

def custom_colours(values, kind='hot'):

  if not values is None and not 'return' in kind:
    values = np.clip(values, 0., 1.)
    colours = np.zeros((len(values), 4))
    colours[:, -1] = 255.
  if kind == 'hot':
    colours[:, 0] = np.where(values < .5, 2 * values, 1.)      * 255
    colours[values >= .5, 1] = (2 * values[values >= .5] - 1.) * 255
  elif 'hotcold' in kind:
    #NEG  = (76, 114, 176)   # blue
    #MID  = (140, 140, 140)  # neutral gray
    #POS  = (196, 78, 82)    # red
    NEG = np.array([[ 59,  76, 192]]) # deeper blue
    #MID = np.array([[245, 245, 245]]) # very light neutral
    MID = np.array([[  0,   0,   0]]) # very light neutral
    POS = np.array([[180,   4,  38]]) # deeper red
    if 'return' in kind:
      return [NEG, MID, POS]
    NEG, MID, POS = [rgb_to_oklab(c) for c in [NEG, MID, POS]]
    values = values[:, None]
    Lab = np.where(
      values <= .5,
      (1. - 2 * values) * NEG + (2 * values) * MID,
      (2. - 2 * values) * MID + (2 * values - 1.) * POS
    )
    colours[:, :-1] = oklab_to_rgb(Lab)
    colours[:, -1] = 255.
  elif 'hot2cold' in kind:
    NEG = np.array([[  0,  18, 192]]) # deeper blue
    MID = np.array([[100, 100, 100]]) 
    POS = np.array([[192,   0,  18]]) # deeper red
    if 'return' in kind:
      return [NEG, MID, POS]
    NEG, MID, POS = [rgb_to_oklab(c) for c in [NEG, MID, POS]]
    values = values[:, None]
    Lab = np.where(
      values <= .5,
      (1. - 2 * values) * NEG + (2 * values) * MID,
      (2. - 2 * values) * MID + (2 * values - 1.) * POS
    )
    colours[:, :-1] = oklab_to_rgb(Lab)
    colours[:, -1] = 255.
    
  return (np.clip(np.round(colours), 0, 255)).astype(np.uint8)

# base, overlay: (..., 4) uint8 RGBA
def blend_vertex_colors(base, overlay):
  base = base.astype(np.float32)
  overlay = overlay.astype(np.float32)
  a = overlay[..., 3:4] / 255.  
  out_rgb = (1 - a) * base[..., :3] + a * overlay[..., :3]
  out_a   = base[..., 3:4]  # keep opaque base
  return np.concatenate([out_rgb, out_a], axis=-1).astype(np.uint8)

def apply_vertex_mask(mesh, mask, color):
  vc = mesh.visual.vertex_colors.copy()
  overlay = np.zeros_like(vc)
  overlay[mask] = color
  mesh.visual.vertex_colors = blend_vertex_colors(vc, overlay)

def build_colour_bar(colours, output):
  NEG_LAB, MID_LAB, POS_LAB = [rgb_to_oklab(c) for c in colours] # NEG, MID, POS

  # diverging interpolation in OKLab space
  def interpolate_oklab(t):
    if t <= .5:
      Lab = NEG_LAB * (1 - 2*t) + MID_LAB * (2*t)
    else:
      Lab = MID_LAB * (2 - 2*t) + POS_LAB * (2*t - 1)
    rgb = oklab_to_rgb(Lab[None, :])[0]
    return np.clip(np.round(rgb), 0, 255).astype(int)
  
  def rgb_hex(rgb):
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
  
  # --- SVG geometry ---
  width  = 400
  height = 40
  n_stops = 101

  # --- build gradient stops ---
  stops = []
  for i in range(n_stops):
    t = i / (n_stops - 1)
    colour = rgb_hex(interpolate_oklab(t))
    stops.append(
      f'<stop offset="{t*100:.1f}%" stop-color="{colour}"/>'
    )

  svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}"
     viewBox="0 0 {width} {height}"
     xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="hotcold_oklab"
                    x1="0%" y1="0%"
                    x2="100%" y2="0%">
      {' '.join(stops)}
    </linearGradient>
  </defs>

  <rect x="0" y="0"
        width="{width}" height="{height}"
        fill="url(#hotcold_oklab)"
        stroke="black" stroke-width="1"/>
</svg>
  """

  with open(output, "w") as f:
    f.write(svg)