#!/usr/bin/env python3
"""
Pipeline unificado para calcular dODFs (ODFs discretizadas en una esfera) con SHORE
y, opcionalmente, convertirlas a coeficientes SH listos para MRtrix.

Incluye:
1) Reorientación del volumen DWI a RAS+ y rotación/permuta coherente de bvecs.
2) Ajuste del modelo SHORE y cálculo de ODFs sobre una esfera (SF: sampled function).
3) (Opcional) Conversión de ODF SF a coeficientes SH **en convención MRtrix** usando amp2sh,
   reorientando coeficientes con `mrtransform -reorient_fod yes`.

Requisitos: nibabel, dipy, numpy y MRtrix3 en PATH (para salida SH con MRtrix).

Notas de rendimiento:
- Por defecto usa 16 procesos para el ajuste SHORE+ODF.
- Pasa también `-nthreads` a MRtrix.
- Se fijan los hilos BLAS a 1 por proceso para evitar oversubscription.
"""
import os

# Evitar oversubscription cuando usamos multiprocessing: 1 hilo BLAS por proceso.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import sys
import shutil
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

import numpy as np
import nibabel as nib
from nibabel.orientations import (
    io_orientation, axcodes2ornt, ornt_transform,
    apply_orientation, inv_ornt_aff
)

from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from dipy.reconst.shore import ShoreModel
from dipy.io.image import save_nifti
from dipy.reconst.shm import sf_to_sh


def log(msg):
    print(msg, flush=True)


def cmd_exists(name):
    return shutil.which(name) is not None


def run(cmd):
    log(f"$ {' '.join(cmd)}")
    subprocess.check_call(cmd)


def ornt_to_perm_mat(ornt):
    """Matriz 3x3 P (permuta+signos) que aplica la misma
    transformación de ejes que 'ornt' sobre vectores (bvecs).
    Si data_new = apply_orientation(data_old, ornt),
    entonces v_new = P @ v_old.
    """
    P = np.zeros((3, 3), dtype=float)
    for new_axis, (old_axis, sgn) in enumerate(ornt):
        P[new_axis, int(old_axis)] = int(sgn)
    return P


def reorient_to_ras_and_rotate_bvecs(in_nii, in_bvals, in_bvecs, out_nii, out_bvals, out_bvecs):
    """Reorienta el volumen a RAS+ y rota bvecs en coherencia."""
    img = nib.load(in_nii)
    data = img.get_fdata()
    aff  = img.affine

    axcodes = nib.aff2axcodes(aff)
    log(f"AxCodes imagen (antes): {axcodes}")

    ornt_in  = io_orientation(aff)
    ornt_ras = axcodes2ornt(('R','A','S'))
    xform    = ornt_transform(ornt_in, ornt_ras)

    if np.allclose(xform, np.array([[0,1],[1,1],[2,1]])):
        log("La imagen ya está en RAS+. No se reorienta el volumen.")
        data_ras = data
        aff_ras  = aff
    else:
        data_ras = apply_orientation(data, xform)
        aff_ras  = aff @ inv_ornt_aff(xform, data.shape[:3])
        log(f"Aplicada reorientación a RAS+. xform=\n{xform}")

    # --- robust bvals/bvecs load (y forzar (3,N)) ---
    bvals = np.asarray(np.loadtxt(in_bvals), dtype=float).ravel()
    bvecs = np.asarray(np.loadtxt(in_bvecs), dtype=float)
    if bvecs.ndim == 1:
        n = bvals.size
        assert bvecs.size == 3*n, "bvecs size inconsistent with bvals"
        bvecs = bvecs.reshape(3, n)
    elif bvecs.shape[0] != 3 and bvecs.shape[1] == 3:
        bvecs = bvecs.T
    assert bvecs.shape[0] == 3, "Se esperaban bvecs con forma (3, N)"

    # Rotar/permuta bvecs
    P = ornt_to_perm_mat(xform)
    bvecs_rot = P @ bvecs

    # Renormalizar solo b>0
    dwi_mask = bvals > 50
    if dwi_mask.any():
        norms = np.linalg.norm(bvecs_rot[:, dwi_mask], axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        bvecs_rot[:, dwi_mask] = bvecs_rot[:, dwi_mask] / norms

    # --- Guardar con CABECERA FRESCA y sform/qform coherentes ---
    img_ras = nib.Nifti1Image(data_ras.astype(np.float32), aff_ras)
    img_ras.set_sform(aff_ras, code=1)
    img_ras.set_qform(aff_ras, code=1)
    nib.save(img_ras, out_nii)

    np.savetxt(out_bvals, bvals, fmt="%.6f")
    np.savetxt(out_bvecs, bvecs_rot, fmt="%.10f")

    new_axcodes = nib.aff2axcodes(aff_ras)
    log(f"Nueva orientación: {new_axcodes}")
    log(f"Guardado: {out_nii}")
    log(f"Guardado: {out_bvals} y {out_bvecs}")

    return out_nii, out_bvals, out_bvecs


# ---------- SHORE parallel workers ----------

_SHORE_MODEL = None
_SHORE_SPHERE = None


def _init_shore_worker(bvals, bvecs, sphere_name, radial_order, lambdaN, lambdaL):
    """Inicializa una vez por proceso el modelo SHORE y la esfera."""
    global _SHORE_MODEL, _SHORE_SPHERE
    gtab = gradient_table(np.asarray(bvals, dtype=float), bvecs=np.asarray(bvecs, dtype=float))
    _SHORE_MODEL = ShoreModel(
        gtab,
        radial_order=radial_order,
        lambdaN=lambdaN,
        lambdaL=lambdaL,
    )
    _SHORE_SPHERE = get_sphere(sphere_name)


def _fit_odf_chunk(chunk_data):
    """Ajusta SHORE y calcula ODF para un chunk de señales (n_vox, n_grad)."""
    global _SHORE_MODEL, _SHORE_SPHERE
    if _SHORE_MODEL is None or _SHORE_SPHERE is None:
        raise RuntimeError("Worker SHORE no inicializado.")
    n_vox = int(chunk_data.shape[0])
    n_dirs = int(_SHORE_SPHERE.vertices.shape[0])
    out = np.empty((n_vox, n_dirs), dtype=np.float32)
    for i in range(n_vox):
        fit = _SHORE_MODEL.fit(chunk_data[i])
        out[i, :] = fit.odf(_SHORE_SPHERE).astype(np.float32, copy=False)
    return out


def _auto_vox_per_chunk(n_vox, nthreads):
    if n_vox <= 0:
        return 1
    # ~8 chunks por proceso, acotado para no disparar la RAM.
    est = int(np.ceil(n_vox / max(1, nthreads * 8)))
    return int(max(256, min(4096, est)))


def _prepare_mask_in_ras(mask_path, data_shape):
    """Carga la máscara y la reorienta a RAS+ si es necesario."""
    if mask_path is None:
        return None

    log(f"Aplicando máscara (auto-reorient si necesario): {mask_path}")
    mimg = nib.load(mask_path)
    mdata = mimg.get_fdata()
    ornt_in  = io_orientation(mimg.affine)
    ornt_ras = axcodes2ornt(('R', 'A', 'S'))
    xform_m  = ornt_transform(ornt_in, ornt_ras)
    if not np.allclose(xform_m, np.array([[0, 1], [1, 1], [2, 1]])):
        mdata = apply_orientation(mdata, xform_m)
    if mdata.shape[:3] != data_shape[:3]:
        raise RuntimeError(f"La máscara ({mdata.shape}) no coincide con los datos ({data_shape}).")
    return (mdata > 0).astype(bool)


def fit_shore_and_compute_odf(
    data_nii,
    bvals_path,
    bvecs_path,
    out_odf_sf,
    sphere_name="repulsion724",
    radial_order=6,
    lambdaN=1e-8,
    lambdaL=1e-8,
    mask_path=None,
    nthreads=16,
    vox_per_chunk=0,
):
    """Ajusta SHORE y calcula ODFs discretizadas en la esfera especificada.

    Estrategia:
    - Reorienta la máscara a RAS+ si hace falta.
    - Procesa los voxels en chunks.
    - Usa multiprocessing para hacer fit+ODF en paralelo.
    """
    log("Cargando datos y gradientes…")
    img = nib.load(data_nii)
    data = img.get_fdata()
    bvals, bvecs = read_bvals_bvecs(bvals_path, bvecs_path)  # (N,), (N,3)
    gtab = gradient_table(bvals, bvecs=bvecs)
    sphere = get_sphere(sphere_name)
    n_dirs = int(sphere.vertices.shape[0])

    # Máscara opcional (reorientada a RAS si hace falta)
    mask = _prepare_mask_in_ras(mask_path, data.shape)

    flat_data = data.reshape(-1, data.shape[-1])
    if mask is None:
        mask_idx = np.arange(flat_data.shape[0], dtype=np.int64)
        log("Sin máscara: se procesarán todos los voxels del volumen.")
    else:
        mask_idx = np.flatnonzero(mask.ravel(order="C"))
        log(f"Máscara activa: {int(mask.sum())} voxels a procesar.")

    n_vox = int(mask_idx.size)
    if n_vox == 0:
        raise RuntimeError("La máscara no contiene voxels > 0; no hay nada que procesar.")

    nthreads = int(max(1, nthreads))
    if vox_per_chunk is None or int(vox_per_chunk) <= 0:
        vox_per_chunk = _auto_vox_per_chunk(n_vox, nthreads)
    else:
        vox_per_chunk = int(vox_per_chunk)

    total_chunks = int(np.ceil(n_vox / vox_per_chunk))
    log(
        f"Creando y ajustando el modelo SHORE… "
        f"(voxels={n_vox}, dirs={n_dirs}, nthreads={nthreads}, vox_per_chunk={vox_per_chunk}, chunks={total_chunks})"
    )

    # Reservamos directamente la salida final y escribimos por chunks.
    odf = np.zeros(list(data.shape[:3]) + [n_dirs], dtype=np.float32)
    odf_flat = odf.reshape(-1, n_dirs)

    # Inicialización serial o paralela
    if nthreads == 1:
        _init_shore_worker(bvals, bvecs, sphere_name, radial_order, lambdaN, lambdaL)
        log("Modo serial para SHORE+ODF.")
        log_every = max(1, total_chunks // 20)
        for chunk_no, start in enumerate(range(0, n_vox, vox_per_chunk), start=1):
            end = min(start + vox_per_chunk, n_vox)
            chunk_data = flat_data[mask_idx[start:end], :]
            odf_chunk = _fit_odf_chunk(chunk_data)
            odf_flat[mask_idx[start:end], :] = odf_chunk
            if chunk_no == total_chunks or chunk_no % log_every == 0:
                log(f"  Progreso SHORE+ODF: {chunk_no}/{total_chunks} chunks completados")
    else:
        max_inflight = max(2 * nthreads, 1)
        log(f"Modo paralelo para SHORE+ODF con {nthreads} procesos (máx. {max_inflight} chunks en vuelo).")
        log_every = max(1, total_chunks // 20)

        with ProcessPoolExecutor(
            max_workers=nthreads,
            initializer=_init_shore_worker,
            initargs=(bvals, bvecs, sphere_name, radial_order, lambdaN, lambdaL),
        ) as ex:
            futures = {}
            next_start = 0
            chunks_done = 0

            def _submit_one(start_idx):
                end_idx = min(start_idx + vox_per_chunk, n_vox)
                chunk_data = flat_data[mask_idx[start_idx:end_idx], :]
                fut = ex.submit(_fit_odf_chunk, chunk_data)
                futures[fut] = (start_idx, end_idx)

            # Rellenar la cola inicial
            while next_start < n_vox and len(futures) < max_inflight:
                _submit_one(next_start)
                next_start += vox_per_chunk

            while futures:
                done, _ = wait(list(futures.keys()), return_when=FIRST_COMPLETED)
                for fut in done:
                    start, end = futures.pop(fut)
                    odf_chunk = fut.result()
                    odf_flat[mask_idx[start:end], :] = odf_chunk
                    chunks_done += 1
                    if chunks_done == total_chunks or chunks_done % log_every == 0:
                        log(f"  Progreso SHORE+ODF: {chunks_done}/{total_chunks} chunks completados")

                    if next_start < n_vox:
                        _submit_one(next_start)
                        next_start += vox_per_chunk

    # Guardar salida
    log("Guardando ODFs (SF)…")
    # img.affine corresponde ya al DWI RAS porque data_nii es data_ras.nii.gz
    save_nifti(out_odf_sf, odf.astype(np.float32), img.affine)
    log(f"Guardado: {out_odf_sf} (shape {odf.shape})")

    return out_odf_sf, sphere_name


# ---------- SH export paths ----------

def write_sphere_txt(sphere_name, out_txt):
    """Guarda la esfera de DIPY en texto Nx3 en el MISMO orden de vértices."""
    verts = get_sphere(sphere_name).vertices
    np.savetxt(out_txt, verts, fmt="%.8f")
    log(f"Guardado: {out_txt} (dirs={verts.shape[0]})")
    return out_txt


def sf_to_sh_coeffs_dipy(odf_sf_path, out_sh_path, sh_order=8, sphere_name="repulsion724"):
    """DIPY fallback. Útil si no hay MRtrix, pero MRtrix suele ser preferible."""
    log("Convirtiendo ODF SF a SH con DIPY…")
    img = nib.load(odf_sf_path)
    odf_sf = img.get_fdata()  # (X,Y,Z,K)
    sphere = get_sphere(sphere_name)
    K = sphere.vertices.shape[0]
    if odf_sf.shape[-1] != K:
        raise RuntimeError(
            f"Última dimensión ({odf_sf.shape[-1]}) != #dirs esfera {sphere_name} ({K})."
        )
    odf_sh = sf_to_sh(odf_sf, sphere, sh_order=sh_order, smooth=0.0, basis_type="tournier07")
    img_sh = nib.Nifti1Image(odf_sh.astype(np.float32), img.affine)
    img_sh.set_sform(img.affine, code=1)
    img_sh.set_qform(img.affine, code=1)
    nib.save(img_sh, out_sh_path)
    log(f"Guardado (DIPY): {out_sh_path} -> shape {odf_sh.shape}")
    return out_sh_path


def sf_to_sh_coeffs_mrtrix(
    odf_sf_path,
    dwi_nii,
    bvals,
    bvecs,
    out_sh_path,
    sh_order=12,
    sphere_name="repulsion724",
    nthreads=16,
):
    """
    Convierte ODF SF a SH usando MRtrix (amp2sh) y reorienta coeficientes con -reorient_fod yes.
    Alinea la salida al grid y ejes de la DWI RAS para DEC correcto.
    """
    for tool in ("mrconvert", "amp2sh", "mrtransform"):
        if not cmd_exists(tool):
            raise RuntimeError(f"Se requiere MRtrix: '{tool}' no encontrado en PATH.")

    work_dir = os.path.dirname(out_sh_path)
    sphere_txt = os.path.join(work_dir, f"{sphere_name}.txt")
    sf_mif     = os.path.join(work_dir, "sf.mif")
    dwi_mif    = os.path.join(work_dir, "dwi.mif")
    sh_tmp_mif = os.path.join(work_dir, "sh_tmp.mif")
    sh_in_dwi  = os.path.join(work_dir, "sh_in_dwi.mif")
    nthreads = int(max(1, nthreads))

    # 0) Esfera exacta usada en SF
    write_sphere_txt(sphere_name, sphere_txt)

    # 1) Convertir SF a MIF (strides canónicos)
    run([
        "mrconvert", odf_sf_path, sf_mif,
        "-strides", "1,2,3,4",
        "-nthreads", str(nthreads),
        "-force",
    ])

    # 2) Template DWI en MIF (coordenadas/grilla de referencia)
    run([
        "mrconvert", dwi_nii, dwi_mif,
        "-fslgrad", bvecs, bvals,
        "-nthreads", str(nthreads),
        "-force",
    ])

    # 3) SF -> SH con MRtrix (misma esfera)
    run([
        "amp2sh", sf_mif, sh_tmp_mif,
        "-lmax", str(sh_order),
        "-directions", sphere_txt,
        "-nthreads", str(nthreads),
        "-force",
    ])

    # 4) Reorientar coeficientes a ejes del DWI y poner en el mismo grid
    run([
        "mrtransform", sh_tmp_mif,
        "-template", dwi_mif,
        "-reorient_fod", "yes",
        sh_in_dwi,
        "-nthreads", str(nthreads),
        "-force",
    ])

    # 5) Salida final
    run([
        "mrconvert", sh_in_dwi, out_sh_path,
        "-nthreads", str(nthreads),
        "-force",
    ])

    log(f"Guardado (MRtrix): {out_sh_path}")
    return out_sh_path


def main():
    p = argparse.ArgumentParser(
        description="Pipeline para calcular dODFs con SHORE y opcionalmente SH listo para MRtrix."
    )
    p.add_argument('--in_nii', required=True, help='DWI NIfTI de entrada (4D)')
    p.add_argument('--in_bvals', required=True, help='bvals (FSL)')
    p.add_argument('--in_bvecs', required=True, help='bvecs (FSL)')
    p.add_argument('--out_dir', required=True, help='Directorio de salida')

    p.add_argument('--mask', default=None, help='Máscara binaria opcional (NIfTI)')

    # Parámetros SHORE
    p.add_argument('--radial_order', type=int, default=6)
    p.add_argument('--lambdaN', type=float, default=1e-8)
    p.add_argument('--lambdaL', type=float, default=1e-8)

    # Paralelización
    p.add_argument(
        '--nthreads',
        type=int,
        default=16,
        help='Nº de procesos/hilos a usar para SHORE y MRtrix (default: 16)',
    )
    p.add_argument(
        '--vox_per_chunk',
        type=int,
        default=0,
        help='Voxels por chunk en SHORE. 0 = auto (recomendado).',
    )

    # Esfera / ODF
    p.add_argument('--sphere', default='repulsion724', help='Nombre de esfera dipy (p.ej. repulsion724)')

    # SH opcional
    p.add_argument('--save_sh', action='store_true', help='Si se pasa, convierte la ODF SF a coeficientes SH')
    p.add_argument('--sh_order', type=int, default=12, help='Orden SH par (p.ej. 8, 10, 12)')
    p.add_argument(
        '--sh_backend',
        choices=['mrtrix', 'dipy'],
        default='mrtrix',
        help='Generar SH con MRtrix (recomendado) o DIPY.',
    )

    args = p.parse_args()
    args.nthreads = int(max(1, args.nthreads))

    # Subprocesos MRtrix heredarán esto; el flag -nthreads lo sobreescribe igualmente.
    os.environ["MRTRIX_NTHREADS"] = str(args.nthreads)

    os.makedirs(args.out_dir, exist_ok=True)

    out_nii   = os.path.join(args.out_dir, 'data_ras.nii.gz')
    out_bvals = os.path.join(args.out_dir, 'bvals_ras')
    out_bvecs = os.path.join(args.out_dir, 'bvecs_ras')

    odf_sf_path = os.path.join(args.out_dir, 'odf_sf_ras_shore.nii.gz')
    # Nombres de salida SH
    if args.sh_backend == 'mrtrix':
        odf_sh_path = os.path.join(args.out_dir, f'odf_sh_order{args.sh_order}_mrtrix.nii.gz')
    else:
        odf_sh_path = os.path.join(args.out_dir, f'odf_sh_order{args.sh_order}.nii.gz')

    log(f"Configuración de paralelización: nthreads={args.nthreads}, vox_per_chunk={args.vox_per_chunk or 'auto'}")

    # 1) Reorientar imagen y bvecs a RAS+
    log("[1/3] Reorientando a RAS+ y rotando bvecs…")
    ras_nii, ras_bvals, ras_bvecs = reorient_to_ras_and_rotate_bvecs(
        args.in_nii, args.in_bvals, args.in_bvecs, out_nii, out_bvals, out_bvecs
    )

    # 2) Ajustar SHORE y calcular ODFs en la esfera
    log("[2/3] Ajustando SHORE y calculando dODFs (SF)…")
    odf_sf_path, sphere_used = fit_shore_and_compute_odf(
        ras_nii,
        ras_bvals,
        ras_bvecs,
        odf_sf_path,
        sphere_name=args.sphere,
        radial_order=args.radial_order,
        lambdaN=args.lambdaN,
        lambdaL=args.lambdaL,
        mask_path=args.mask,
        nthreads=args.nthreads,
        vox_per_chunk=args.vox_per_chunk,
    )

    # 3) (Opcional) Convertir a coeficientes SH
    if args.save_sh:
        log("[3/3] Convirtiendo ODF SF a coeficientes SH…")
        if args.sh_backend == 'mrtrix':
            sf_to_sh_coeffs_mrtrix(
                odf_sf_path,
                ras_nii,
                ras_bvals,
                ras_bvecs,
                odf_sh_path,
                sh_order=args.sh_order,
                sphere_name=sphere_used,
                nthreads=args.nthreads,
            )
        else:
            sf_to_sh_coeffs_dipy(
                odf_sf_path,
                odf_sh_path,
                sh_order=args.sh_order,
                sphere_name=sphere_used,
            )
    else:
        log("[3/3] Conversión a SH omitida (use --save_sh para activarla).")

    log("\nListo. Salidas en: " + os.path.abspath(args.out_dir))
    log("- data_ras.nii.gz, bvals_ras, bvecs_ras")
    log("- odf_sf_ras_shore.nii.gz")
    if args.save_sh:
        log(f"- {os.path.basename(odf_sh_path)}")


if __name__ == '__main__':
    sys.exit(main())
