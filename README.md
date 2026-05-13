# proyect-brian-pipeline

Per-tumour pipeline for predicting glioblastoma growth on healthy HCP brains
using diffusion-MRI tractography simulation.

Input is a real UPENN-GBM patient (T1 + tumour segmentation labels 1 = enhancing
and 4 = necrotic). For each patient the pipeline registers the tumour into MNI
and into 20 healthy HCP training brains, runs an SPSA + Adam + curriculum
optimiser to search a seed location / angle / length / cutoff in tractography
space whose simulated tumour matches the real one, aggregates the 20 best
simulated masks in MNI, and validates the resulting parameters on 65 unseen
HCP brains.

The pipeline was developed for the BRIAN project at the University of Malta
and is designed to run as a SLURM array on the `radagast` cluster (one array
task = one tumour).

## Pipeline stages

| stage | script | role |
|---|---|---|
| 0 | `run_full_tumor_pipeline.py` | orchestrator: filters segmentation labels 1\|4, drives all steps below |
| 1a | `prep_upenn_for_mni.sh` | RAS reorient, UPENN brain mask excluding tumour. Runs **once per tumour** |
| 1b | `register_upenn_to_mni.sh` | ANTs Rigid + Affine + SyN UPENN T1 → MNI, tumour excluded from metric. **Once per tumour** |
| 1c | `register_one_hcp_subject.sh` + `cache_hcp_to_mni.sh` | For each of 20 train + 65 val HCP subjects: fills (flock-guarded, atomic) HCP → MNI cache the first time it is needed, then symlinks cache files + per-tumour UPENN files into the per-subject work dir and composes the `(HCP→MNI)⁻¹ ∘ (UPENN→MNI)` transform with `antsApplyTransforms` to warp the tumour mask into the HCP grid |
| 2 | inside orchestrator | `mrconvert T1_hcp_RAS.nii.gz → .mif` (no-op when cache already pre-built it) |
| 3 | `rand_seed_v2.py` | per-train-subject optimiser: SPSA + Adam, curriculum A → B → C, 5 percentile × 3 PCA-axis seed init grid; uses `--optim-select 10000` streamlines per evaluation and a final re-run at `--select 50000` to produce the saved best NIfTI |
| 4 | `collect_best_to_mni_3.py` | per train subject: binarise best simulated NIfTI at the optimised `threshold`, warp to MNI, write `best_bin_MNI.nii.gz`. Produces `best_params_all.csv` plus `.median.csv` / `.mean.csv` summaries |
| 5 | `warp_mni_to_subject_coords.py` + `validate_single_run_currC.py` | for each of 65 validation subjects: warp the MNI median / mean seed to native space, run **one** tractography simulation, write `metrics.json` / `metrics.csv` |
| 5b | `collect_best_to_mni_3.py --mode validation` | binarise and warp each validation NIfTI to MNI |
| 6 | `warp_upenn_tumor_to_mni.py` + `average_mni_across_participants.py` + `compare_average_tumors_to_reference.py` | warp the real UPENN tumour to MNI (shared once), voxel-average the 20 train and 65 val masks in MNI, write `training_avg.nii.gz` + `validation_avg.nii.gz` + `training_majority.nii.gz` (>=0.5) + `validation_majority.nii.gz` + `average_vs_tumor_metrics.csv` |

See `Scripts/PIPELINE_FLOW_CLUSTER.md` for the per-stage mermaid diagram and
`Scripts/DRY_RUN_FULL_PIPELINE_EXAMPLE.txt` for a worked invocation.

## Repository layout

```
proyect-brian-pipeline/
├── rand_seed_v2.py                          # SPSA+Adam+curriculum optimiser
└── Scripts/
    ├── run_full_tumor_pipeline.py           # orchestrator
    ├── run_tumor_pipeline_cluster.sh        # bash wrapper (1 tumour = 1 invocation)
    ├── prep_upenn_for_mni.sh                # step 1a
    ├── register_upenn_to_mni.sh             # step 1b
    ├── register_one_hcp_subject.sh          # step 1c
    ├── cache_hcp_to_mni.sh                  # shared HCP→MNI cache, flock+atomic
    ├── prewarm_hcp_to_mni.sh                # optional batch cache warmup
    ├── collect_best_to_mni_3.py             # step 4 / 5b
    ├── warp_mni_to_subject_coords.py        # step 5
    ├── validate_single_run_currC.py         # step 5
    ├── warp_upenn_tumor_to_mni.py           # step 6
    ├── average_mni_across_participants.py   # step 6
    ├── compare_average_tumors_to_reference.py # step 6
    ├── pipeline_original_T1_MNI.sh          # legacy monolithic registration (fallback)
    ├── submit_tumor_array_15.slurm          # SLURM array launcher, batch 1
    ├── submit_tumor_array_batch2.slurm      # SLURM array launcher, batch 2
    ├── submit_validation_array.slurm        # validation-only resubmit (batch 1 phase 1)
    ├── submit_validation_remaining.slurm    # validation-only resubmit (batch 1 phase 2)
    ├── submit_prewarm_hcp_to_mni.slurm      # optional 85-HCP cache prewarm
    ├── monitor_tumor_jobs.sh                # live status snapshot
    ├── check_cluster_inputs.sh              # sanity-check input paths
    ├── build_hcp_ids_list.sh                # emit 20+65 HCP IDs as flat file
    ├── PIPELINE_FLOW_CLUSTER.md             # mermaid pipeline diagram
    └── DRY_RUN_FULL_PIPELINE_EXAMPLE.txt    # worked dry-run example
```

## Dependencies

External binaries (cluster-side):

- **ANTs 2.5.3+** — `antsRegistration`, `antsApplyTransforms`, `antsApplyTransformsToPoints`
- **MRtrix3 3.0.4+** — `mrconvert`, `mrcalc`, `maskfilter`, `mrtransform`, `tckgen`, `tckmap`
- **flock** (util-linux), **python3 ≥ 3.10**

Python (conda env `simulation` on the cluster):

- numpy, scipy, pandas, nibabel
- (the optimiser uses only the standard scientific stack — no torch / TF)

SLURM environment:

- partition `research_cpu` (or `research_long` for >14 d), account `brian`,
  node `ori`, 5 cpus / 15 GB / 14 d per array task by default
- env vars set by every job: `ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS`,
  `MRTRIX_NTHREADS` ← `SLURM_CPUS_PER_TASK`; OMP / OPENBLAS / MKL / NUMEXPR
  pinned to 1

## Inputs expected on cluster NFS

| dataset | path |
|---|---|
| HCP 1200 T1 | `/opt/nfs/neuroinformatics/HCP/HCP_1200_qc_data/<ID>/T1w/T1w_acpc_dc_restore_brain.nii.gz` |
| HCP diffusion ODFs (train 20) | `/opt/nfs/neuroinformatics/BRIAN/dODFs/odf_generation_results_20/<ID>/odf_sh_order12_mrtrix.mif` |
| HCP diffusion ODFs (val 65)   | `/opt/nfs/neuroinformatics/BRIAN/dODFs/odf_generation_results_66/<ID>/odf_sh_order12_mrtrix.mif` |
| UPENN-GBM tumour              | `<UPENN_ROOT>/<TUMOR_ID>/{<TUMOR_ID>_T1.nii.gz, <TUMOR_ID>_automated_approx_segm.nii.gz}` |
| MNI template                  | `/opt/nfs/neuroinformatics/BRIAN/tpl-MNI152NLin2009cAsym_res-01_desc-brain_T1w.nii.gz` |

`UPENN_ROOT` defaults to `/opt/nfs/neuroinformatics/BRIAN/UPENN-15` and is
overridden via env var by the batch-2 launcher to point at a user-writable
input dir.

## Running

Submit one array job per batch (1 array task = 1 tumour):

```bash
# Batch 1: end-to-end (training + collect + validation + averages)
sbatch Scripts/submit_tumor_array_15.slurm /path/to/tumor_ids_15.txt

# Batch 2: same, with UPENN_ROOT override baked in
sbatch Scripts/submit_tumor_array_batch2.slurm /path/to/tumor_ids_batch2.txt
```

The `tumor_ids_*.txt` file is one UPENN tumour ID per line. The SLURM
`%15` throttle controls how many array tasks run concurrently. Per-tumour
output lands at `runs/<TUMOR_ID>/` under the configured `BASE_ROOT`
(default `/opt/users/aalbe03/Parallel_runs`).

For a deferred validation pass (training already done, validation
not):

```bash
sbatch Scripts/submit_validation_array.slurm /path/to/tumor_ids_done.txt
```

This sets `--skip-registration --skip-mrconvert --skip-optimizer --skip-collect`
on the orchestrator so it jumps straight to validation + averages.

## Monitoring

```bash
bash Scripts/monitor_tumor_jobs.sh
```

reports SLURM queue state, per-tumour `done / live / todo` train-subject
counts, throughput (evals/min) on each active subject, and cluster-wide
aggregate.

## Outputs

Per tumour, after a complete end-to-end run:

```
runs/<TUMOR_ID>/
├── tumor_inputs/                              # filtered tumour seg, labels 1|4
├── registration/
│   ├── upenn_workdir/                         # per-tumour UPENN→MNI artefacts
│   ├── registration_20/<HCP>/                 # per-train symlink + tumour compose
│   ├── registration_65/<HCP>/                 # per-val   symlink + tumour compose
│   └── tumor_upenn_MNI.nii.gz                 # real UPENN tumour warped to MNI
├── training_results/<HCP>/
│   ├── analysis_participant_<HCP>_best.nii.gz # 50K-streamline final NIfTI
│   ├── analysis_participant_<HCP>_best.txt
│   ├── analysis_participant_<HCP>_best_params.json
│   └── simulation.csv                         # one row per optimisation eval
├── validation_results/<HCP>/
│   ├── analysis_participant_<HCP>.nii.gz
│   ├── metrics.json
│   └── metrics.csv
├── best_mni_outputs/
│   ├── <HCP>/best_bin_MNI.nii.gz              # per-train binary → MNI
│   ├── best_params_all.csv
│   ├── best_params_all.median.csv
│   └── best_params_all.mean.csv
├── validation_mni_outputs/
│   ├── <HCP>/analysis_bin_MNI.nii.gz
│   └── validation_all.csv
└── avg_mni_outputs/
    ├── training_avg.nii.gz                    # probabilistic 0–1
    ├── training_sum.nii.gz
    ├── training_majority.nii.gz               # avg ≥ 0.5
    ├── validation_avg.nii.gz
    ├── validation_sum.nii.gz
    ├── validation_majority.nii.gz
    └── average_vs_tumor_metrics.csv           # final Dice / HD / vol-err vs GT
```

## Metric definitions (validation step)

All produced by `validate_single_run_currC.py`:

- **dice** = `1 − dice_loss` (Sørensen–Dice on binary masks, intrinsic [0,1])
- **hd_norm** = `min(HD_mm / 20, 1)` — directed Hausdorff distance capped at `HD_DMAX_MM = 20 mm`
- **vol_err** = `min(|pred_vox − gt_vox| / gt_vox, 1)` — clipped relative volume error
- **surface_dice** = fraction of surface points within `SURF_TOL_MM = 2 mm` of the GT surface
- **proj_dice_worst** = `min(Dice_xy, Dice_xz, Dice_yz)` over the three axis-aligned projections

The composite **objective** weights stage-C losses: `w_hd · hd_norm + w_dice · dice_loss + w_proj · proj_loss_worst` with `(w_hd, w_dice, w_proj) = (0.3, 0.6, 0.2)` and `dice_loss = 1 − dice`.

## Notes

- The HCP → MNI ANTs registration depends only on the HCP subject + MNI
  template (not on the tumour), so it is computed **once per HCP subject**
  and reused across every tumour. The shared cache at
  `cache/hcp_to_mni/<HCP>/` is flock-guarded and published atomically
  (build in `<TARGET>.tmp.$$`, `mv` rename, then `touch .cache_done`).
- The UPENN → MNI ANTs registration depends only on the tumour + MNI
  template, so it is computed **once per tumour** (`upenn_workdir/`) and
  symlinked into all 85 per-subject directories.
- Together these two refactors are what made the 15-tumour pipeline fit
  inside the 14-day SLURM cap, vs the previous monolithic version that
  re-registered 85× per tumour.

## License

MIT — see `LICENSE`.
