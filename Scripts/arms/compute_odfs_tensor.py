#!/usr/bin/env python3
"""Single-fibre control for the dODF arm.

Mirror of compute_odfs_basis_fix.py but swaps the SHORE model for a single
diffusion TENSOR (b<=bmax). Everything else is identical: same sphere
(repulsion724), same amp2sh lmax, same `mrtransform -reorient_fod yes` path,
so the output odf_sh_order12_tensor_mrtrix.mif is a drop-in replacement for the
SHORE dODF and tracks with iFOD2 under identical cutoff semantics.

The ONLY difference vs the dODF arm is the angular model:
  SHORE  -> can show multiple lobes (crossings)
  tensor -> exactly one ellipsoidal lobe (single fibre)

Spatial support is taken from the existing SHORE SF (nonzero voxels) so the two
arms cover the identical voxel set.
"""
import os, sys, argparse, subprocess, shutil
import numpy as np
import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.core.sphere import Sphere          # avoid dipy.data (pulls h5py)
from dipy.reconst.dti import TensorModel


def log(m): print(m, flush=True)

def run(cmd):
    log("$ " + " ".join(cmd)); subprocess.check_call(cmd)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_nii", required=True, help="data_ras.nii.gz (already RAS+)")
    p.add_argument("--in_bvals", required=True)
    p.add_argument("--in_bvecs", required=True)
    p.add_argument("--mask_nii", required=True, help="precomputed brain mask (nonzero support of SHORE SF)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--sphere_txt", required=True, help="repulsion724.txt vertices used by the SHORE dODF (Nx3)")
    p.add_argument("--template_mif", required=True, help="existing SHORE odf_sh_order12_mrtrix.mif (grid template for reorient)")
    p.add_argument("--bmax", type=float, default=1100.0, help="keep shells with b<=bmax (b0 + b1000)")
    p.add_argument("--sphere", default="repulsion724")
    p.add_argument("--sh_order", type=int, default=12)
    p.add_argument("--nthreads", type=int, default=16)
    p.add_argument("--cleanup", action="store_true", help="delete SF/mask intermediates after building the FOD")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    log("[1/4] loading DWI + gradients")
    img = nib.load(a.in_nii); aff = img.affine
    data = img.get_fdata(dtype=np.float32)
    bvals = np.loadtxt(a.in_bvals).ravel().astype(float)
    bvecs = np.loadtxt(a.in_bvecs).astype(float)
    if bvecs.shape[0] != 3 and bvecs.shape[1] == 3:
        bvecs = bvecs.T                      # want (3,N)
    assert bvecs.shape[0] == 3, bvecs.shape

    sel = bvals <= a.bmax
    log(f"      keeping {int(sel.sum())}/{bvals.size} volumes (b<= {a.bmax})  "
        f"shells~{sorted(set(np.round(bvals[sel]/1000)*1000))}")
    data = data[..., sel]
    gtab = gradient_table(bvals[sel], bvecs=bvecs[:, sel].T)

    log("[2/4] loading precomputed brain mask")
    mask = np.asarray(nib.load(a.mask_nii).get_fdata()) > 0
    log(f"      mask voxels: {int(mask.sum())}")

    log("[3/4] fitting single tensor + computing diffusion ODF on sphere")
    verts = np.loadtxt(a.sphere_txt)
    sphere = Sphere(xyz=verts)
    tm = TensorModel(gtab)
    fit = tm.fit(data, mask=mask)
    odf = fit.odf(sphere).astype(np.float32)     # (X,Y,Z,Kdir)
    odf[~mask] = 0.0                              # identical support to SHORE
    odf = np.nan_to_num(odf, nan=0.0, posinf=0.0, neginf=0.0)
    sf_out = os.path.join(a.out_dir, "odf_sf_ras_tensor.nii.gz")
    out_img = nib.Nifti1Image(odf, aff)
    out_img.set_sform(aff, code=1); out_img.set_qform(aff, code=1)
    nib.save(out_img, sf_out)
    log(f"      saved {sf_out}  shape {odf.shape}")
    # free the big in-RAM arrays before spawning amp2sh (else python's ~12GB resident
    # collides with amp2sh's ~10GB preload -> ~23GB/job -> OOM under any parallelism)
    del odf, data, out_img, fit
    import gc; gc.collect()

    log("[4/4] SF -> SH (amp2sh) + reorient_fod  (identical to dODF path)")
    wd = a.out_dir
    sphere_txt = a.sphere_txt                    # exact same directions as SHORE dODF
    sh_tmp = os.path.join(wd, "sh_tmp_tensor.mif")
    out_mif = os.path.join(wd, f"odf_sh_order{a.sh_order}_tensor_mrtrix.mif")
    nt = str(a.nthreads)
    # amp2sh reads the SF nii.gz directly (one decompress; no 5GB intermediate mif)
    run(["amp2sh", sf_out, sh_tmp, "-lmax", str(a.sh_order),
         "-directions", sphere_txt, "-nthreads", nt, "-force"])
    # reorient onto the EXISTING SHORE FOD grid -> byte-identical grid as the dODF arm
    run(["mrtransform", sh_tmp, "-template", a.template_mif, "-reorient_fod", "yes",
         out_mif, "-nthreads", nt, "-force"])
    log(f"DONE -> {out_mif}")

    if a.cleanup:
        for f in (sf_out, sh_tmp, a.mask_nii):
            try: os.remove(f)
            except OSError: pass
        log("      cleaned intermediates (kept only the tensor FOD mif)")


if __name__ == "__main__":
    sys.exit(main())
