# proyect-brian-pipeline

Per-tumour pipeline that simulates the spread of the MRI-visible glioblastoma
core on healthy HCP brains by streamline propagation through the diffusion ODF.

The parameters are fitted to each tumour against its own reference mask, and the
training/validation split is over host brains, not over tumours. The results
therefore describe the reproduction of known lesions and the transfer of a fitted
configuration across host anatomy, not the prediction of unseen tumours.

The results reported in the M.Sc. dissertation (University of Malta, 2026) come
from the base pipeline below **plus the October 2026 re-runs and analysis** in
`rerun_2026-10/`, `analysis/` and `results/` (see the sections of the same names).

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
| 3 | `rand_seed_v2.py` | per-train-subject optimiser: SPSA + Adam, curriculum A → B → C, 5 percentile × 3 PCA-axis seed init grid; uses `--optim-select 10000` streamlines per evaluation and is meant to re-run the selected configuration at `--select 50000` (in the June 2026 campaign this re-run did not replace the 10 000-streamline image; the 50 000-streamline training masks were produced afterwards by `rerun_2026-10/rerun_driver.py shore50k`) |
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
├── rand_seed_v2_resume.py                   # optimiser variant: resume from a saved run
├── rerun_2026-10/                           # October 2026 re-runs reported in the dissertation
├── analysis/                                # final metrics, contrasts, tables and figures
├── results/                                 # per-host and aggregate metrics, contrasts (CSV)
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
    ├── DRY_RUN_FULL_PIPELINE_EXAMPLE.txt    # worked dry-run example
    └── arms/                                # multi-arm tractography experiments (see below)
        ├── run_stage_c_2.py                 # SHORE dODF + iFOD2 — training optimiser driver
        ├── run_stage_c_2_w2.py              # SHORE arm, Workstation-2 variant
        ├── run_postfit_c2.py                # SHORE arm — post-fit collect / validate / average
        ├── run_stage_null.py                # NULL (NullDist2) — direction-blind baseline driver
        ├── run_postfit_null.py              # NULL arm — post-fit
        ├── run_postfit_null50k.py           # NULL arm, 50k-streamline variant — post-fit
        ├── run_stage_dti.py                 # DTI (Tensor_Prob) — training driver
        ├── run_postfit_dti.py               # DTI arm — post-fit
        ├── run_stage_tensor.py              # tensor dODF + iFOD2 — training driver
        ├── run_postfit_tensor.py            # tensor arm — post-fit
        ├── compute_odfs_basis_fix.py        # build SHORE dODFs (all shells b=0/1000/2000/3000)
        ├── compute_odfs_tensor.py           # build single-tensor dODFs (b≤1000)
        ├── build_tensor_odfs_batch.sh       # batch wrapper for compute_odfs_tensor
        ├── extract_b1000_dwi.py             # extract b=1000 shell for the DTI arm
        ├── launch_null_w1.sh                # NULL launcher, Workstation-1
        ├── launch_null_w2.sh                # NULL launcher, Workstation-2
        ├── launch_null50k_w1.sh             # NULL-50k launcher, Workstation-1
        ├── chain_dti_postfit.sh             # DTI stage → post-fit chain
        └── tensor_9tum_chain.sh             # tensor arm, 9-tumour chain
```

## Tracking arms (July 2026 drafts, superseded)

`Scripts/arms/` holds the July 2026 drivers for the diffusion-model / tracker
comparison. They are kept for provenance. The exception is `run_stage_c_2.py`, which
produced the stage C of the SHORE configurations used everywhere in the dissertation.
The comparisons reported in the
dissertation were re-run in October 2026 with one protocol for all conditions
(`rerun_2026-10/`), and the DTI (`Tensor_Prob`) arm is not reported. The July drivers
were run on top of the base pipeline. All arms share the same optimiser
(`rand_seed_v2.py`) and post-fit machinery; they differ only in the diffusion
model and the `tckgen -algorithm`, selected via the `--algorithm` flag
(`iFOD2` | `NullDist2` | `Tensor_Det` | `Tensor_Prob`):

| arm | diffusion model | tracker | driver |
|---|---|---|---|
| **SHORE** | SHORE dODF (all shells) | `iFOD2` | `run_stage_c_2.py` / `run_postfit_c2.py` |
| **TENSOR** | single-tensor dODF (b≤1000) | `iFOD2` | `run_stage_tensor.py` / `run_postfit_tensor.py` |
| **DTI** | tensor | `Tensor_Prob` | `run_stage_dti.py` / `run_postfit_dti.py` |
| **NULL** | SHORE dODF (same file as SHORE) | `NullDist2` | `run_stage_null.py` / `run_postfit_null.py` |

NULL and SHORE read the same ODF file but use different trackers. `NullDist2`
never evaluates the ODF amplitude (so `cutoff` is inert in the null) and it also
stops streamlines by a different rule from `iFOD2`. The NULL–SHORE contrast is
therefore one between a diffusion-informed and a random-tracking pipeline, not a
test of orientation alone. SHORE and TENSOR share the tracker and differ in the
diffusion model and in the shells used to fit it.

## October 2026 re-runs (reported in the dissertation)

`rerun_2026-10/` holds the scripts exactly as they ran on the Malta workstation
(`/mnt/bsc-das-01/BRIAN/rerun_2026-10-15/`) and on the `ori` node. The absolute
paths at the top of each script are site configuration.

All three conditions of the ten-tumour comparison (UPENN-GBM-00001, 00019, 00020,
00023, 00024, 00025, 00031, 00032, 00034, 00042) follow one protocol:

1. stages A and B of each condition's own search, with the campaign code
   (`rand_seed_v2.py`, stopped on entering stage C; `patch_nullAB.py` adds `--stop_at_C`);
2. stage C restarted from that condition's own end-of-B state with
   `rand_seed_v2_resume_algo.py` (the campaign `rand_seed_v2_resume.py` plus an
   `--algorithm` flag and a final image written to `<best>_50k`): stage-C weights,
   selection from stage-C rows only, patience 5;
3. 10 000 streamlines per evaluation in the search, 50 000 for the final training
   image and for every validation run;
4. validation on the 65 hosts with the median configuration, MNI collection,
   host-frequency averages and comparison with the reference.

| condition | dODF / tracker | stages A–B and C | post-fit |
|---|---|---|---|
| SHORE | SHORE dODF / `iFOD2` | stages A–B of the June campaign; stage C by `Scripts/arms/run_stage_c_2.py` (the `stage_c_2` run, also used for the 30-tumour main campaign); 50k training masks by `rerun_driver.py shore50k` | `postfit_driver.py shore` |
| null | SHORE dODF / `NullDist2` | `rerun_driver.py nullProto` | `postfit_driver.py null` |
| tensor | single-tensor dODF (b ≤ 1000) / `iFOD2` | `cluster_tensor.py train` (ori); `malta_tensor.py` is the same logic with Malta paths, used for 00020, 00032 and 00042 | `cluster_tensor.py` / `malta_tensor.py post` |

The `nullC` and `nullFull` jobs of `rerun_driver.py` (and `patch_nullfull.py`) are
earlier variants that are not reported. `tensor_full.sbatch` is the Slurm launcher
used on `ori`.

**Known implementation facts.** These are described in the dissertation:
- In `rand_seed_v2.py` the objective takes its weights as default arguments, so the
  stage-A weights `(0.55, 0.45, 0.0)` stayed in force during stages A and B. Stage C
  of every reported configuration was re-run with the correct weights `(0.20, 0.60, 0.20)`.
- `tckgen` received neither `-minlength` nor `-step`, so the defaults applied
  (6.25 mm and 0.625 mm).
- The per-host reference removes a sphere of radius 1.5 host voxels around the seed
  (`create_roi_mask(..., radius=3)` halves its argument) with `np.subtract`. The
  reported per-host scores use a logical subtraction instead (`analysis/metrics_final.py`).

## Analysis and results

- `analysis/metrics_final.py` re-scores the saved masks: Dice, HD95 and Hausdorff
  agreement, the boundary coverage used by the optimiser, the **standard surface
  Dice** at 2 mm (Nikolov et al., 2021, computed with DeepMind `surface-distance`),
  volume match and worst-projection Dice. The per-host reference uses the logical
  seed-sphere subtraction; the aggregate masks keep voxels with host frequency > 0.02
  and are compared with the complete MNI reference.
- `analysis/contrasts_final.py results_out/` reproduces the tumour-level contrasts
  (n = 10): one-sample t-test on the per-tumour differences, Holm over the five
  metrics of each contrast, uncorrected Wilcoxon, Cohen's d_z and 95 % CI, with the
  aggregate-mask and the per-host estimators. Run it from the repository root; its
  output equals `results/contrasts_final_asrun.csv`.
- `analysis/tfm_numbers.py` writes every number, table row and Chapter 5 figure of
  the dissertation from `results/`.
- `results/`: `shore_c2_*` (main campaign, 30 tumours × 65 hosts, and the averages),
  `rerun_*` (null condition and the 50k SHORE averages), `tensornew_*` (tensor
  condition), `threshold_sweep.csv` and `tumour_volumes.csv`. Read them with
  `keep_default_na=False`, or pandas turns the label `null` into NaN.

## Dependencies

External binaries (cluster-side):

- **ANTs 2.5.3+** — `antsRegistration`, `antsApplyTransforms`, `antsApplyTransformsToPoints`
- **MRtrix3 3.0.4+** — `mrconvert`, `mrcalc`, `maskfilter`, `mrtransform`, `tckgen`, `tckmap`
- **flock** (util-linux), **python3 ≥ 3.10**

Python (conda env `simulation` on the cluster):

- numpy, scipy, pandas, nibabel
- (the optimiser uses only the standard scientific stack — no torch / TF)
- for `analysis/`: matplotlib and DeepMind `surface-distance`
  (`pip install git+https://github.com/google-deepmind/surface-distance.git`)

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
- **hd_norm** = `min(HD95_mm / 20, 1)` — 95th-percentile symmetric Hausdorff distance capped at `HD_DMAX_MM = 20 mm`; the dissertation reports `1 − hd_norm` (Hausdorff agreement)
- **vol_err** = `min(|pred_vox − gt_vox| / gt_vox, 1)` — clipped relative volume error
- **surface_dice** = mean of the two directional fractions of surface voxels within `SURF_TOL_MM = 2 mm` of the other surface. This is the *boundary coverage* used by the optimiser (`sd_loss = 1 − surface_dice`), not the standard surface Dice; every surface Dice reported in the dissertation is the standard one from `analysis/metrics_final.py`
- **proj_dice_worst** = `min(Dice_xy, Dice_xz, Dice_yz)` over the three axis-aligned projections

The composite **objective** weights the losses: `w_hd · hd_norm + w_dice · dice_loss + w_proj · proj_loss_worst` with `dice_loss = 1 − dice`. The curriculum is designed to set the weights per stage — A = `(0.55, 0.45, 0.0)`, B = `(0.35, 0.55, 0.10)`, C = `(0.20, 0.60, 0.20)`. In `rand_seed_v2.py` the stage-A values stay in force during stages A and B (default-argument binding); the stage-C runs of `rand_seed_v2_resume.py` use `(0.20, 0.60, 0.20)`, and every reported configuration was selected from stage-C rows only.

## Notes

- **Integration step.** `tckgen` is called without `-step`, so iFOD2 uses its
  default step of half the voxel size (0.625 mm on the 1.25 mm dODF grid).
  This is the setting of the published runs; the `--step` / `-st` arguments
  are parsed for compatibility only and are not passed to `tckgen`.
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
