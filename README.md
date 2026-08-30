# Modelling gene therapy delivery to the striatum in Huntington’s disease: optimising target coverage

Collection of select code, meshes, volumes used for the manuscript

## Common

`common.py`:

- collection of code repeatedly referenced by multiple figure/analysis scripts
- also acts as a path setter to all relevant source data
- paths have been left empty as source data cannot be openly shared

## Preparation

- `_preparation_transform_VDmax_trajectories.py`:
  - transform cortical intersection points native → fsaverage space
  - intersection point data are stored as triangle ID and corresponding barycentric coordinates
  - used e.g. for surface mesh generation (Fig. 4)

- `_preparation_generate_native_space_masks.py`:
  - generate 3D meshes of deposits, rasterise target nucleus meshes and deposits, clean out-of-bounds spillovers
  - deposit rasterisation assumes deposit merging
  - used for volume mask generation (Fig. 5A) and related analyses

## Figure and analysis scripts

- `figure_2_pca_plots.py`:
  - produce PCA biplots (Fig. 2B)

- `figure_4_create_VDmax_maps.py`:
  - generate 3D scenes with valid trajectory probability (Fig. 4A) or best trajectories (Fig. 4E) in fsaverage space
  - requires `_preparation_transform_VDmax_trajectories.py` outputs
  - map type is manually switched in script

- `figure_4_create_access_maps.py`:
  - generate 3D scenes with vertex-wise change in access surface (Fig. 4C) and valid trajectory probability (Supp 3) with change in atrophy markers
  - requires `_preparation_transform_VDmax_trajectories.py` outputs
  - map type is manually switched in script

- `figure_4_regional_accessibility_and_best.py`:
  - accessibility and probability of best trajectories per region (Fig. 4B,D)
  - requires `_preparation_transform_VDmax_trajectories.py` inputs (native cortical intersection barycentric cds)

- `figure_4_global_change_in_accessibility_ratio.py`:
  - run LME modelling change in accessibility ration with change in atrophy markers (Supp. 2)
  - generate box plots illustrating the same (Fig. 4D)

- `figure_4_permanova_permdist.py`:
  - run PERMANOVA and PERMDIST statistics on standardised trajectory vectors per region

- `figure_4_plot_angles.py`:
  - project best trajectories on AP / LAT planes and generate distribution plots (Fig. 4G)

- `figure_4_volumes_testing.py`:
  - multivariate regression with blocked permutations investigating relationship between atrophy markers and trajectories
  - looks at atrophy effects on best trajectory (Fig. 4H)
  
- `figure_5_put_traj_LME.py`:
  - comparisons between the number of trajectories to the putamen (1 vs 2) including t-test for vdmax, LME for vdmax difference
  - generate Figs. 5B-E
  
- `figure_5_VDmax_corr_plots.py`: 
  - coverage correlation plot Fig. 5F

- `figure_5_LOSO.py`:
  - LOSO RMSE comparison (stats) and LOSO MAE comparison (violin plots) Fig. 5G

## Surface meshes for Fig.4A,C,E + Supp 3

- Valid trajectory probabilty (Fig. 4A)
- Vertex-wise change in access surface with change in atrophy markers (Fig. 4C)
- Vertex-wise change in probability of valid trajectories with change in atrophy markers (Supp 3)
- Cortical entry points of best trajectories (Fib. 4E)

## Volume masks for Fig.5A

- Striatal region in MNI space
- Probability of caudate in cohort in MNI space
- Probability of putamen in cohort in MNI space
- Probability of best distribution in caudate in MNI space
- Probability of best distribution in putamen in MNI space