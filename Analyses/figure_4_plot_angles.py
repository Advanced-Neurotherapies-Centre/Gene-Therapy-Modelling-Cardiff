import numpy as np
import matplotlib.pyplot as plt
import os

from common import Common, rgb_to_oklab, oklab_to_rgb

def get_settings():

  return {
    'min_length': .1,
    'max_length': 2.,
    'scaling_factor': 1.,
    'colour': 'blue',
    'alpha': .4,
    'width': 2.,
    'radius': 2.
  }

def custom_colours(values, kind = 'hotcold'):

  if not values is None:
    values = np.clip(values, 0., 1.)
    colours = np.zeros((len(values), 4))
    colours[:, -1] = 255.
  if kind == 'hot':
    colours[:, 0] = np.where(values < .5, 2 * values, 1.)      * 255
    colours[values >= .5, 1] = (2 * values[values >= .5] - 1.) * 255
  elif 'hotcold' in kind:
    NEG = np.array([[  0,   0, 255]])
    MID = np.array([[255,   0,   0]])
    POS = np.array([[  0, 255,   0]])
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

def smoothstep(x, gamma=1.6):
    x = np.clip(x, 0, 1)
    return .5 + .5 * np.sign(x - .5) * np.abs(2*x - 1)**gamma

def plot_angles(data, settings, title = '', ax_circ = None, vv_max = None):

  # extract settings
  params = ['min_length', 'max_length', 'scaling_factor', 'radius', 'colour', 'alpha', 'width']
  minlen, maxlen, sf, radius, colour, alpha, width = [settings[param] for param in params]

  angles, magnitudes, ventricle_volumes = data.T
  angles %= 2 * np.pi

  # Normalize ventricle_volumes to a range of line lengths
  vv_max = ventricle_volumes.max() if vv_max is None else vv_max
  if np.any(ventricle_volumes - ventricle_volumes[0]):
    lengths = (ventricle_volumes - ventricle_volumes.min()) / (vv_max - ventricle_volumes.min())
  else:
    lengths = np.ones_like(ventricle_volumes)
  lengths = (lengths * (maxlen - minlen) + minlen) * sf

  # Origin points on the unit circle
  x0 = np.cos(angles) * radius
  y0 = np.sin(angles) * radius

  # Direction vectors (radial, outward)
  dx = np.cos(angles) * lengths
  dy = np.sin(angles) * lengths

  # Plot setup
  if ax_circ is None:
    fig_circ, ax_circ = plt.subplots(figsize=(6, 6))
  ax_circ.set_aspect('equal')
  offset = radius * 1.1 + maxlen * sf
  ax_circ.set_xlim(-offset, offset)
  ax_circ.set_ylim(-offset, offset * 1.1)

  # set up colour
  colour = custom_colours(smoothstep(magnitudes, 1.)) / 255

  # Plot lines
  for x, y, u, v, c in zip(x0, y0, dx, dy, colour):
    ax_circ.plot([x, x + u], [y, y + v], c=c, alpha = alpha, linewidth = width)

  # Do annotations  
  if 'lat' in title.lower():
    aticks = np.linspace(np.pi / 2, -np.pi / 2, 5)
    labels = ['0°', '+45° ', '+90° ', '+135°', '+180°']
  else:
    aticks = np.linspace(np.pi, 0, 5)
    labels = ['-90°', '-45°', '0°', '+45° ', '+90° ']
  for angle, label in zip(aticks, labels):
    x = np.cos(angle) * radius * .75
    y = np.sin(angle) * radius * .75
    ax_circ.text(x, y, label, fontsize=8, ha='center', va='center', color='black')

  # Add anatomical direction labels
  if 'lat' in title.lower():
    text2 = 'Lat'
  else:
    ax_circ.text(-offset, -radius * .1, 'Post', ha='center', va='center', fontsize=10, fontweight='bold')
    text2 = 'Ant'
  ax_circ.text(0, offset + radius * .1, 'Sup', ha='center', va='center', fontsize=10, fontweight='bold')
  ax_circ.text(offset, -radius * .1, text2, ha='center', va='center', fontsize=10, fontweight='bold')

  # Clean up plot
  ax_circ.axis('off')
  ax_circ.set_title(title.replace('_', ' '))

def create_angle_figures(param_path, target_regions, settings):
  
  vvols = Common().get_ventricular_volumes()
  df = Common().get_traj_talairach(drop_empty=True)
  df['vvol'] = [vvols[s] for s in df['sub']]
  ap, lat = [], []
  for _, row in df.iterrows():
    x, y, z = row['talairach']
    lat_rot = np.arctan2(z, x)
    ap_rot = np.arctan2(z, y)
    lat_mag = np.linalg.norm([x, z])
    ap_mag = np.linalg.norm([y, z])
    joint_mag = lat_mag + ap_mag
    ap.append((ap_rot, ap_mag / joint_mag))
    lat.append((lat_rot, lat_mag / joint_mag))
  df['AP'] = ap
  df['lat'] = lat
  
  fig_circ, axes_circ = plt.subplots(2, len(target_regions), figsize=(len(target_regions) * 4, 7)); axes_circ = axes_circ.T.flatten()
  vv_max = np.max([v for v in vvols.values()])
  for tid, treg in enumerate(target_regions):
    dft = df[df['trajectory'] == treg]
    for pid, plane in enumerate(['lat', 'AP']):
      data = dft[plane].values
      vv   = dft['vvol'].values
      data = np.hstack([np.vstack(data), np.asarray(vv)[:, None]])
      plot_angles(data, settings, f'{treg.title()} ({plane})', ax_circ = axes_circ[tid * 2 + pid], vv_max = vv_max)
  
  fig_circ.tight_layout()
  sdir = os.path.dirname(os.path.realpath(__file__))
  fig_circ.savefig(os.path.join(sdir, 'circular_plots.svg'), bbox_inches='tight')

if __name__ == '__main__':
  create_angle_figures(*Common().get_static(['param_path', 'target_regions']), get_settings())
