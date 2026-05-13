# Flujo del pipeline en el cluster

```mermaid
flowchart TD
    A[run_tumor_pipeline_cluster.sh\n1 job = 1 tumor UPENN] --> B[run_full_tumor_pipeline.py]
    B --> C[0. Filtrar labels del tumor UPENN\nlabels 1 y 4]
    C --> D[1. Registration TRAIN 20\npipeline_original_T1_MNI.sh]
    C --> E[1. Registration VALID 65\npipeline_original_T1_MNI.sh]
    D --> F[2. mrconvert\nT1_hcp_RAS.nii.gz -> T1_hcp_RAS.mif]
    E --> G[2. mrconvert\nT1_hcp_RAS.nii.gz -> T1_hcp_RAS.mif]
    F --> H[3. Optimizer TRAIN\nrand_seed_v2.py]
    H --> I[4. Collect best -> MNI\ncollect_best_to_mni_3.py]
    I --> J[4b. Cleanup opcional\nborra NIfTIs intermedios]
    I --> K[best_params_all.mean/median.csv]
    K --> L[5. Warp MNI coords -> native\nwarp_mni_to_subject_coords.py]
    L --> M[5. Validation single run\nvalidate_single_run_currC.py]
    M --> N[5b. Validation outputs -> MNI\ncollect_best_to_mni_3.py]
    N --> O[6. Tumor UPENN -> MNI compartido]
    O --> P[6. Averages TRAIN/VALID\naverage_mni_across_participants.py]
    P --> Q[6. Métricas finales\ncompare_average_tumors_to_reference.py]
```

## Qué sí está cubierto ya

- **Conversión `nii.gz -> mif`**: no falta un script adicional si usas `run_full_tumor_pipeline.py`; el paso 2 llama a `mrconvert` internamente.
- **Tumor original en MNI**: tampoco falta un script aparte en el flujo principal; el paso 6 lo hace dentro de `run_full_tumor_pipeline.py` y lo guarda en `registration/tumor_upenn_MNI.nii.gz`.
- **CSV de métricas**:
  - training: `simulation.csv` por sujeto con métricas completas
  - validation: `metrics.csv` / `metrics.json` por sujeto
  - collect: `best_params_all.csv` y `validation_all.csv` arrastran las métricas del best/single run
  - average final: `avg_mni_outputs/average_vs_tumor_metrics.csv`

## Qué faltaba en el paquete original y queda añadido aquí

- Wrapper principal de cluster: `run_tumor_pipeline_cluster.sh`
- Plantilla Slurm para 15 tumores: `submit_tumor_array_15.slurm`
- Chequeo de inputs compartidos: `check_cluster_inputs.sh`
- Cleanup standalone del training: `cleanup_training_keep_best.py`

## Observación importante

El único paso repetitivo que **podría** optimizarse en el futuro es que el registro `UPENN -> MNI` se recalcula dentro de cada carpeta de sujeto HCP, aunque conceptualmente depende solo del tumor UPENN y del template MNI. No falta para ejecutar el pipeline, pero sí es una posible optimización futura.
