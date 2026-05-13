#!/usr/bin/env python3
"""warp_mni_to_subject_coords.py

Convierte una coordenada (x,y,z) en espacio MNI a coordenadas en el espacio
nativo de cada sujeto (p.ej. HCP) usando los warps ya calculados en:
  <reg_root>/<SUBJECT_ID>/

Por defecto usa los transforms "hcpT1_2mni_*" (moving=HCP, fixed=MNI), y aplica
la regla correcta para POINTSETS según la wiki oficial de ANTs:
- fixed -> moving: usar *forward* transforms (movingToFixed_1Warp + movingToFixed_0GenericAffine)
- moving -> fixed: usar inversos ([Affine,1] + movingToFixed_1InverseWarp)

Soporta RAS<->LPS porque antsApplyTransformsToPoints trabaja en ITK physical (LPS).

Outputs:
- Un CSV con una fila por sujeto y las coordenadas transformadas.
- (Opcional) check round-trip MNI->HCP->MNI.

"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# -------------------------
# Coords helpers
# -------------------------

def ras_to_lps(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """NIfTI RAS -> ITK LPS (flip X,Y)."""
    return (-float(x), -float(y), float(z))


def lps_to_ras(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """ITK LPS -> NIfTI RAS (flip X,Y)."""
    return (-float(x), -float(y), float(z))


@dataclass
class TransformSpec:
    warp: Optional[Path]
    affine: Optional[Path]
    inv_warp: Optional[Path]


def _safe_run(cmd: List[str], quiet: bool = True) -> None:
    """Run external command with basic error reporting."""
    try:
        if quiet:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.run(cmd, check=True)
    except FileNotFoundError:
        raise RuntimeError(f"No encuentro el ejecutable '{cmd[0]}'. ¿Está ANTs en el PATH?")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Fallo ejecutando: {' '.join(cmd)}\nReturn code: {e.returncode}")


def _write_single_point_csv(path: Path, x: float, y: float, z: float) -> None:
    # antsApplyTransformsToPoints exige cabecera x,y,z,t
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['x', 'y', 'z', 't'])
        w.writerow([f"{x:.10f}", f"{y:.10f}", f"{z:.10f}", "0"])


def _read_single_point_csv(path: Path) -> Tuple[float, float, float]:
    df = pd.read_csv(path)
    if df.shape[0] < 1:
        raise RuntimeError(f"CSV vacío: {path}")
    return float(df.loc[0, 'x']), float(df.loc[0, 'y']), float(df.loc[0, 'z'])


def _discover_subject_ids(reg_root: Path) -> List[str]:
    ids = []
    for p in reg_root.iterdir():
        if p.is_dir() and re.fullmatch(r"\d+", p.name):
            ids.append(p.name)
    return sorted(ids)


def _get_transforms(subj_dir: Path, prefix: str) -> TransformSpec:
    affine = subj_dir / f"{prefix}_0GenericAffine.mat"
    warp = subj_dir / f"{prefix}_1Warp.nii.gz"
    inv_warp = subj_dir / f"{prefix}_1InverseWarp.nii.gz"

    return TransformSpec(
        warp=warp if warp.exists() else None,
        affine=affine if affine.exists() else None,
        inv_warp=inv_warp if inv_warp.exists() else None,
    )


def _apply_points_fixed_to_moving(
    in_csv: Path,
    out_csv: Path,
    spec: TransformSpec,
    quiet: bool = True,
) -> None:
    """fixed -> moving: usar forward warp + forward affine."""
    cmd = [
        "antsApplyTransformsToPoints",
        "-d",
        "3",
        "-i",
        str(in_csv),
        "-o",
        str(out_csv),
    ]

    if spec.warp is not None:
        cmd += ["-t", str(spec.warp)]
    if spec.affine is not None:
        cmd += ["-t", str(spec.affine)]

    if (spec.warp is None) and (spec.affine is None):
        raise RuntimeError("No hay transforms disponibles (ni warp ni affine).")

    _safe_run(cmd, quiet=quiet)


def _apply_points_moving_to_fixed(
    in_csv: Path,
    out_csv: Path,
    spec: TransformSpec,
    quiet: bool = True,
) -> None:
    """moving -> fixed: usar inverse affine + inverse warp."""
    cmd = [
        "antsApplyTransformsToPoints",
        "-d",
        "3",
        "-i",
        str(in_csv),
        "-o",
        str(out_csv),
    ]

    # Importante: para points, el orden (según la wiki) es:
    #  -t [Affine,1]  -t InverseWarp
    if spec.affine is not None:
        cmd += ["-t", f"[{spec.affine},1]"]
    if spec.inv_warp is not None:
        cmd += ["-t", str(spec.inv_warp)]

    if (spec.inv_warp is None) and (spec.affine is None):
        raise RuntimeError("No hay transforms disponibles (ni inverse warp ni affine).")

    _safe_run(cmd, quiet=quiet)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Warp de coordenadas desde MNI -> espacio nativo por sujeto usando ANTs (points)."
    )
    ap.add_argument("--reg-root", required=True, type=str, help="Ruta a registration_results_65")

    ap.add_argument("--x", required=True, type=float, help="X en MNI (por defecto en RAS mm)")
    ap.add_argument("--y", required=True, type=float, help="Y en MNI (por defecto en RAS mm)")
    ap.add_argument("--z", required=True, type=float, help="Z en MNI (por defecto en RAS mm)")

    ap.add_argument(
        "--space",
        choices=["hcp", "upenn", "both"],
        default="hcp",
        help="Qué transforms usar: hcpT1_2mni, upT1_2mni, o ambos.",
    )

    ap.add_argument(
        "--out-csv",
        required=True,
        type=str,
        help="CSV de salida con coords por sujeto",
    )

    ap.add_argument(
        "--assume-input-ras",
        action="store_true",
        default=True,
        help="Asume que las coords de entrada están en RAS y convierte a LPS para ANTs.",
    )

    ap.add_argument(
        "--no-assume-input-ras",
        action="store_false",
        dest="assume_input_ras",
        help="Si tus coords ya están en LPS (ITK), desactiva la conversión.",
    )

    ap.add_argument(
        "--emit-output-ras",
        action="store_true",
        default=True,
        help="Convierte salida LPS->RAS (recomendado si vas a pasar coords a MRtrix).",
    )

    ap.add_argument(
        "--emit-output-lps",
        action="store_false",
        dest="emit_output_ras",
        help="Deja la salida en LPS (ITK).",
    )

    ap.add_argument(
        "--roundtrip-check",
        action="store_true",
        default=False,
        help="Hace MNI->native->MNI y reporta error (mm).",
    )

    ap.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Silencia stdout/stderr de ANTs.",
    )

    args = ap.parse_args()

    reg_root = Path(args.reg_root)
    if not reg_root.exists():
        print(f"[ERROR] reg-root no existe: {reg_root}", file=sys.stderr)
        return 2

    subj_ids = _discover_subject_ids(reg_root)
    if not subj_ids:
        print(f"[ERROR] No encontré subcarpetas numéricas en: {reg_root}", file=sys.stderr)
        return 2

    # Input point
    x_in, y_in, z_in = float(args.x), float(args.y), float(args.z)
    if args.assume_input_ras:
        x_itk, y_itk, z_itk = ras_to_lps(x_in, y_in, z_in)
    else:
        x_itk, y_itk, z_itk = x_in, y_in, z_in

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Temp point files (reutilizables)
    tmp_dir = out_csv.parent / (out_csv.stem + "_tmp_points")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    in_csv = tmp_dir / "point_in.csv"
    _write_single_point_csv(in_csv, x_itk, y_itk, z_itk)

    rows: List[Dict[str, float]] = []

    def _process_one(prefix: str, sid: str) -> Dict[str, float]:
        subj_dir = reg_root / sid
        spec = _get_transforms(subj_dir, prefix)

        out_pt_csv = tmp_dir / f"{sid}_{prefix}_out.csv"
        _apply_points_fixed_to_moving(in_csv, out_pt_csv, spec, quiet=args.quiet)

        xo, yo, zo = _read_single_point_csv(out_pt_csv)
        # salida de ANTs está en ITK physical (LPS)
        if args.emit_output_ras:
            xo, yo, zo = lps_to_ras(xo, yo, zo)

        out: Dict[str, float] = {
            "subject_id": sid,
            f"x_{prefix}": float(xo),
            f"y_{prefix}": float(yo),
            f"z_{prefix}": float(zo),
        }

        if args.roundtrip_check:
            # native -> MNI (moving -> fixed) para points usa inverse affine + inverse warp
            # Necesitamos dar a ANTs la coord en LPS.
            # Si emitimos output RAS, revertimos a LPS para el check.
            if args.emit_output_ras:
                xr_lps, yr_lps, zr_lps = ras_to_lps(xo, yo, zo)
            else:
                xr_lps, yr_lps, zr_lps = xo, yo, zo

            rt_in = tmp_dir / f"{sid}_{prefix}_rt_in.csv"
            rt_out = tmp_dir / f"{sid}_{prefix}_rt_out.csv"
            _write_single_point_csv(rt_in, xr_lps, yr_lps, zr_lps)
            _apply_points_moving_to_fixed(rt_in, rt_out, spec, quiet=args.quiet)
            xrt, yrt, zrt = _read_single_point_csv(rt_out)

            # roundtrip error en ITK (LPS)
            dx = xrt - x_itk
            dy = yrt - y_itk
            dz = zrt - z_itk
            out[f"roundtrip_err_{prefix}_mm"] = float((dx * dx + dy * dy + dz * dz) ** 0.5)

        return out

    for sid in subj_ids:
        base_row: Dict[str, float] = {
            "subject_id": sid,
            "x_mni_in": float(x_in),
            "y_mni_in": float(y_in),
            "z_mni_in": float(z_in),
        }

        try:
            if args.space in ("hcp", "both"):
                d = _process_one("hcpT1_2mni", sid)
                base_row.update({
                    "x_native": d["x_hcpT1_2mni"],
                    "y_native": d["y_hcpT1_2mni"],
                    "z_native": d["z_hcpT1_2mni"],
                })
                # también guardo explícito por si quieres ambos
                base_row.update(d)

            if args.space in ("upenn", "both"):
                d2 = _process_one("upT1_2mni", sid)
                base_row.update(d2)

            rows.append(base_row)

        except Exception as e:
            # Seguimos con los demás sujetos
            rows.append({
                "subject_id": sid,
                "x_mni_in": float(x_in),
                "y_mni_in": float(y_in),
                "z_mni_in": float(z_in),
                "error": str(e),
            })

    df = pd.DataFrame(rows)

    # Orden columnas "bonito"
    preferred = [
        "subject_id",
        "x_mni_in",
        "y_mni_in",
        "z_mni_in",
        "x_native",
        "y_native",
        "z_native",
        "roundtrip_err_hcpT1_2mni_mm",
        "roundtrip_err_upT1_2mni_mm",
        "error",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]

    df.to_csv(out_csv, index=False, float_format="%.10f")
    print(f"[OK] Guardado: {out_csv}  (n={len(df)})")
    if args.roundtrip_check:
        for c in ["roundtrip_err_hcpT1_2mni_mm", "roundtrip_err_upT1_2mni_mm"]:
            if c in df.columns:
                vals = pd.to_numeric(df[c], errors='coerce').dropna()
                if len(vals):
                    print(f"[RoundTrip] {c}: mean={vals.mean():.4f} mm | median={vals.median():.4f} mm | max={vals.max():.4f} mm")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
