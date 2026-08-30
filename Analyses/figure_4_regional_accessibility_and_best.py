import os
import pandas as pd
import numpy as np
import nibabel as nib
from multiprocessing import Pool
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.ticker import FuncFormatter

from common import Common, read_fs_mesh

reg_type = 'best' # all or best

subs = Common().get_subs()
fsdir, target_regions, native_cortical_intersections = Common().get_static(['fsdir', 'target_regions', 'native_x_path'])

atlas = 'h.aparc.a2009s.annot' # Destrieux 2009

def process_hemi(p):

  # load data
  v, f = read_fs_mesh(p['surf'])
  labels, _, names = nib.freesurfer.read_annot(p['annot'])
  names = [n.decode('utf-8') for n in names]
  
  v_idx, treg_ids = [], []
  for tid, target in enumerate(target_regions):
    res_path = os.path.join(native_cortical_intersections, p['sub'], p['side'], target, 'tris.npy')
    if os.path.exists(res_path):
      tris = np.load(res_path)
      uvs = np.load(res_path.replace('tris.npy', 'uvs.npy'))
      vdmax = np.load(res_path.replace('tris.npy', 'vdmax_vals.npy'))
      if len(vdmax) and np.any(vdmax > 0.):
        if reg_type == 'best':
          valid = [np.argmax(vdmax)]
        elif reg_type == 'all':
          valid = vdmax > 0.
        else:
          ValueError('Unrecognised type')        
        tris = tris[valid]
        uvs = uvs[valid]
        v_idx.append(f[tris, np.argmax(uvs, axis = 1)])
        treg_ids.append(v_idx[-1] * 0 + tid)
  
  v_idx = np.hstack(v_idx)
  treg_ids = np.hstack(treg_ids)
  
  # map points to labels
  point_labels = labels[v_idx]
  
  # calculate probability of encountering points
  result = []
  for label_id, name in enumerate(names):
    for tid, target in enumerate(target_regions):
      sel = treg_ids == tid
      region_prob = np.sum(point_labels[sel] == label_id) / np.sum(sel) if np.any(sel) else 0.
      result.append({
        'sub': p['sub'], 
        'side': p['side'], 
        'target': target, 
        'region': rename_labels(name), 
        'probability': region_prob
      })
  return result

def load_atlas_names(hemi='lh', subject='fsaverage'):
  
  annot_file = f"{os.environ['SUBJECTS_DIR']}/{subject}/label/{hemi[0]}{atlas}"
  _, _, names = nib.freesurfer.read_annot(annot_file)
  region_dict = {rename_labels(name.decode('utf-8')): 0 for name in names}
  return region_dict

def rename_labels(labels, atlas=atlas):
  
  if atlas == 'h.aparc.a2009s.annot': 
    mapping = {
      "G_and_S_frontomargin": "FrM",
      "G_and_S_occipital_inf": "OccI",
      "G_and_S_paracentral": "ParC",
      "G_and_S_subcentral": "SubC",
      "G_and_S_transv_frontopol": "FPo",
      "G_cingul-Ant": "CgA",
      "G_cingul-Mid-Ant": "CgMA",
      "G_cingul-Mid-Post": "CgMP",
      "G_cingul-Post-dorsal": "CgPD",
      "G_cingul-Post-ventral": "CgPV",
      "G_cuneus": "Cun",
      "G_front_inf-Opercular": "FIO",
      "G_front_inf-Orbital": "FIR",
      "G_front_inf-Triangul": "FIT",
      "G_front_middle": "MFG",
      "G_front_sup": "SFG",
      "G_Ins_lg_and_S_cent_ins": "InsLG",
      "G_insular_short": "InsS",
      "G_and_S_occipital_sup": "OccS",
      "G_occipital_middle": "MOcc",
      "G_occipital_sup": "SOcc",
      "G_oc-temp_lat-fusifor": "Fus",
      "G_oc-temp_med-Lingual": "LinG",
      "G_oc-temp_med-Parahip": "PHG",
      "G_orbital": "OrbG",
      "G_pariet_inf-Angular": "AngG",
      "G_pariet_inf-Supramar": "SMG",
      "G_parietal_sup": "SPG",
      "G_postcentral": "PoCG",
      "G_precentral": "PrCG",
      "G_precuneus": "Pcun",
      "G_rectus": "RecG",
      "G_subcallosal": "SubCal",
      "G_temp_sup-G_T_transv": "HG",
      "G_temp_sup-Lateral": "STGL",
      "G_temp_sup-Plan_polar": "PP",
      "G_temp_sup-Plan_tempo": "PT",
      "G_temp_sup": "STG",
      "G_temporal_inf": "ITG",
      "G_temporal_middle": "MTG",
      "Lat_Fis-ant-Horizont": "LFH",
      "Lat_Fis-ant-Vertical": "LFV",
      "Lat_Fis-post": "LFP",
      "Pole_occipital": "POcc",
      "Pole_temporal": "PTemp",
      "S_calcarine": "CalS",
      "S_central": "CS",
      "S_cingul-Marginalis": "CgM",
      "S_cingul-Post-dorsal": "CgPDs",
      "S_cingul-Post-ventral": "CgPVs",
      "S_circular_insula_ant": "InsCA",
      "S_circular_insula_inf": "InsCI",
      "S_circular_insula_sup": "InsCS",
      "S_collat_transv_ant": "CTA",
      "S_collat_transv_post": "CTP",
      "S_front_inf": "SFIn",
      "S_front_middle": "SFM",
      "S_front_sup": "SFS",
      "S_interm_prim-Jensen": "JenS",
      "S_intrapariet_and_P_trans": "IPS",
      "S_occipital_ant": "OccA",
      "S_occipital_notch": "OccN",
      "S_occipital_sup": "OccS",
      "S_oc-temp_lat": "OTL",
      "S_oc-temp_med_and_Lingual": "OTML",
      "S_orbital_lateral": "OrbL",
      "S_orbital_med-olfact": "OrbMO",
      "S_parieto_occipital": "PaOcc",
      "S_pericallosal": "Peri",
      "S_postcentral": "PoCS",
      "S_precentral-inf-part": "PrCSI",
      "S_precentral-sup-part": "PrCSS",
      "S_suborbital": "SubO",
      "S_subparietal": "SubP",
      "S_temporal_inf": "STSinf",
      "S_temporal_sup": "STSsup",
      "S_temporal_transverse": "TrTS",
      "Unknown": "Unk",
      "G_and_S_cingul-Ant": "CgA",
      "G_and_S_cingul-Mid-Ant": "CgMA",
      "G_and_S_cingul-Mid-Post": "CgMP",
      "Medial_wall": "MedW",
      "S_oc_middle_and_Lunatus": "OML",
      "S_oc_sup_and_transversal": "OST",
      "S_orbital-H_Shaped": "OrbH"
    }
  else:
    ValueError('Unknown atlas')

  mapped = []
  islist = True
  if isinstance(labels, str):
    labels = [labels]
    islist = False
  for label in labels:
    if label in mapping:
      mapped.append(mapping[label])
    else:
      mapped.append(label)
      print(f'{label} is absent from the mapping')
  return mapped if islist else mapped[0]

def plot_bilateral(mean_lh, sem_lh, mean_rh, sem_rh, title = None, n_entries = 5, output_path = f'{reg_type}.svg'):

  fontsize = {
    'title': 24,
    'xlab': 22,
    'ticks': 20,
    'legend': 20,
    'regions': 22
  }

  lh_filtered = [(k, v) for k, v in mean_lh.items() if not (v == sem_lh[k] == 0)]
  rh_filtered = [(k, v) for k, v in mean_rh.items() if not (v == sem_rh[k] == 0)]

  lh_sorted = sorted(lh_filtered, key=lambda x: x[1], reverse=True)[:n_entries]
  rh_sorted = sorted(rh_filtered, key=lambda x: x[1], reverse=True)[:n_entries]

  lh_labels = [k for k, _ in lh_sorted]
  lh_means  = np.array([mean_lh[k] for k in lh_labels]) * 1e2
  lh_sems   = np.array([sem_lh[k]  for k in lh_labels]) * 1e2

  rh_labels = [k for k, _ in rh_sorted]
  rh_means  = np.array([mean_rh[k] for k in rh_labels]) * 1e2
  rh_sems   = np.array([sem_rh[k]  for k in rh_labels]) * 1e2

  print('\n')
  for entry in range(n_entries):
    if entry < len(lh_labels):
      print(f'{title}\'s {entry + 1} region on the left: {lh_labels[entry]} {lh_means[entry]:.2f} +/- {lh_sems[entry]:.2f}')
  for entry in range(n_entries):
    if entry < len(rh_labels):
      print(f'{title}\'s {entry + 1} region on the right: {rh_labels[entry]} {rh_means[entry]:.2f} +/- {rh_sems[entry]:.2f}')

  y_pos = np.arange(max(len(lh_labels), len(rh_labels)))[::-1]
  fig, ax = plt.subplots(figsize=(6, 4))
  max_x = max(lh_means.max() + lh_sems.max(),
              rh_means.max() + rh_sems.max())
  
  colors = {"left": "#4C72B0", "right": "#DD8452"}
  fc = lambda h: (*to_rgba(colors[h])[:3], .6)
  ec = lambda h: (*to_rgba(colors[h])[:3], .8)

  ax.barh(
    y_pos[:len(lh_means)],
    -lh_means,
    xerr=lh_sems,
    facecolor=fc('left'),
    edgecolor=ec('left'),
    linewidth=3.,
    label='Left'
  )
  
  for i, (lab, mean, sem) in enumerate(zip(lh_labels, lh_means, lh_sems)):
    ax.text(
      -(mean + sem + 0.02*max_x),
      n_entries - i - 1,
      lab,
      ha='right',
      va='center',
      fontsize=fontsize['regions']
    )
  
  ax.barh(
    y_pos[:len(rh_means)],
    rh_means,
    xerr=rh_sems,
    facecolor=fc('right'),
    edgecolor=ec('right'),
    linewidth=3.,  
    label='Right'
  )
  
  for i, (lab, mean, sem) in enumerate(zip(rh_labels, rh_means, rh_sems)):
    ax.text(
      mean + sem + 0.02*max_x,
      n_entries - i - 1,
      lab,
      ha='left',
      va='center',
      fontsize=fontsize['regions']
    )
  
  ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{abs(x):.1f}'))
  ax.tick_params(axis='x', labelsize=fontsize['ticks']) 
  ax.set_yticks(y_pos)
  ax.set_yticklabels([""] * len(y_pos))
  ax.axvline(0, color='black', linewidth=1)
  if not title is None:
    ax.set_title(title, fontsize=fontsize['title'])
  ax.legend(loc='lower right', fontsize=fontsize['legend'])
  ax.set_xlabel('Trajectory probability (%)', fontsize=fontsize['xlab'])

  plt.tight_layout()
  plt.savefig(output_path)
  plt.close()

def _plot_single_bilateral(ax, mean_lh, sem_lh, mean_rh, sem_rh, title, n_entries, fontsize):

  lh_filtered = [(k, v) for k, v in mean_lh.items() if not (v == sem_lh[k] == 0)]
  rh_filtered = [(k, v) for k, v in mean_rh.items() if not (v == sem_rh[k] == 0)]

  lh_sorted = sorted(lh_filtered, key=lambda x: x[1], reverse=True)[:n_entries]
  rh_sorted = sorted(rh_filtered, key=lambda x: x[1], reverse=True)[:n_entries]

  lh_labels = [k for k, _ in lh_sorted]
  lh_means  = np.array([mean_lh[k] for k in lh_labels]) * 1e2
  lh_sems   = np.array([sem_lh[k]  for k in lh_labels]) * 1e2

  rh_labels = [k for k, _ in rh_sorted]
  rh_means  = np.array([mean_rh[k] for k in rh_labels]) * 1e2
  rh_sems   = np.array([sem_rh[k]  for k in rh_labels]) * 1e2

  print('\n')
  for entry in range(n_entries):
    if entry < len(lh_labels):
      print(f'{title}\'s {entry + 1} region on the left: {lh_labels[entry]} {lh_means[entry]:.2f} +/- {lh_sems[entry]:.2f}')
  for entry in range(n_entries):
    if entry < len(rh_labels):
      print(f'{title}\'s {entry + 1} region on the right: {rh_labels[entry]} {rh_means[entry]:.2f} +/- {rh_sems[entry]:.2f}')

  y_pos = np.arange(max(len(lh_labels), len(rh_labels)))[::-1]
  max_x = max(lh_means.max() + lh_sems.max(),
              rh_means.max() + rh_sems.max())
  
  colors = {"left": "#4C72B0", "right": "#DD8452"}
  fc = lambda h: (*to_rgba(colors[h])[:3], .6)
  ec = lambda h: (*to_rgba(colors[h])[:3], .8)

  ax.barh(
    y_pos[:len(lh_means)],
    -lh_means,
    xerr=lh_sems,
    facecolor=fc('left'),
    edgecolor=ec('left'),
    linewidth=3.,
    label='Left'
  )
  
  for i, (lab, mean, sem) in enumerate(zip(lh_labels, lh_means, lh_sems)):
    ax.text(
      -(mean + sem + 0.02*max_x),
      n_entries - i - 1,
      lab,
      ha='right',
      va='center',
      fontsize=fontsize['regions']
    )
  
  ax.barh(
    y_pos[:len(rh_means)],
    rh_means,
    xerr=rh_sems,
    facecolor=fc('right'),
    edgecolor=ec('right'),
    linewidth=3.,  
    label='Right'
  )
  
  for i, (lab, mean, sem) in enumerate(zip(rh_labels, rh_means, rh_sems)):
    ax.text(
      mean + sem + 0.02*max_x,
      n_entries - i - 1,
      lab,
      ha='left',
      va='center',
      fontsize=fontsize['regions']
    )

  ax.set_xlim(-100, 100)  
  ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{abs(x):.0f}'))
  ax.tick_params(axis='x', labelsize=fontsize['ticks']) 
  ax.set_yticks(y_pos)
  ax.set_yticklabels([""] * len(y_pos))
  ax.get_yaxis().set_visible(False)
  ax.axvline(0, color='black', linewidth=1)
  ax.set_title(title, fontsize=fontsize['title'])


def plot_quad(conditions, n_entries = 5, output_path = f'{reg_type}.svg'):

  fontsize = {
    'title': 24,
    'xlab': 22,
    'ticks': 20,
    'legend': 20,
    'regions': 22
  }

  fig, axes = plt.subplots(
    2, 2,
    figsize=(12, 8),
    sharex=True,
    sharey=True
  )
  axes = axes.ravel()
  
  for ax, cond in zip(axes, conditions):
    _plot_single_bilateral(
      ax=ax,
      mean_lh=cond['mean_lh'],
      sem_lh=cond['sem_lh'],
      mean_rh=cond['mean_rh'],
      sem_rh=cond['sem_rh'],
      title=cond.get('title'),
      n_entries=n_entries,
      fontsize=fontsize
    )

  for ax in axes[2:]:
    ax.set_xlabel('Trajectory probability (%)', fontsize=fontsize['xlab'])

  for ax in axes[1::2]:
      ax.set_yticklabels([])

  handles, labels = axes[0].get_legend_handles_labels()
  fig.legend(
    handles, labels,
    loc='lower center',
    ncol=2,
    fontsize=fontsize['legend']
  )

  #plt.tight_layout(rect=[0, 0.05, 1, 1])
  plt.savefig(output_path)
  plt.close()

if __name__ == '__main__':

  hemi_data = []
  for sub in subs:
    for side in ['left', 'right']:
      for target in target_regions:
        results = os.path.join(native_cortical_intersections, sub, side, target, 'tris.npy')
        if os.path.exists(results):
          hemi_path = os.path.join(fsdir, sub, 'surf', side[0] + 'h.pial')
          annot_path = os.path.join(fsdir, sub, 'label', side[0] + atlas)
          hemi_data.append({'sub': sub, 'side': side, 'surf': hemi_path, 'annot': annot_path})
          break

  with Pool() as p:
    result = p.map(process_hemi, hemi_data)

  df = pd.DataFrame([x for y in result for x in y])

  inputs = []

  for target in target_regions:
    dk_means = {'left': load_atlas_names(), 'right': load_atlas_names(hemi='rh')} # this just helps maintain the order
    dk_sems = {'left': load_atlas_names(), 'right': load_atlas_names(hemi='rh')} # this just helps maintain the order
    for side in ['left', 'right']:
      valid_subs = [s for s in subs if df.loc[(df['sub'] == s) & (df['side'] == side) & (df['target'] == target), 'probability'].sum()]
      print(f'Valid subs for {side} {target}: {len(valid_subs)} / {len(subs)}')
      for name in dk_means[side]:
        select = (df['sub'].isin(valid_subs)) & (df['side'] == side) & (df['target'] == target) & (df['region'] == name)
        dk_means[side][name] = df.loc[select, 'probability'].mean()
        dk_sems[side][name] = df.loc[select, 'probability'].std() / np.sqrt(np.sum(select))
    title = f'{target.replace("_", " ")}'.capitalize()
    inputs.append({
      'mean_lh': dk_means['left'],
      'sem_lh': dk_sems['left'],
      'mean_rh': dk_means['right'],
      'sem_rh': dk_sems['right'],
      'title': title
    })

  plot_quad(inputs, output_path = f'{reg_type}_combined.svg')
