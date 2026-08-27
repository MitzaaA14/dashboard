#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
solidify_map.py — transformă o hartă PCD (nor de puncte SLAM) într-o hartă cu
structuri SOLIDE și continue, eliminând zgomotul și punctele izolate (ex. cele
văzute prin geam / reflexii).

Pipeline:
  1) Statistical Outlier Removal (SOR) — scoate fulgii izolați.
  2) Voxelizare + componente conexe 3D — păstrează structurile mari conectate
     și aruncă insulele mici, îndepărtate (geam, reflexii).
  3) Închidere morfologică — umple golurile mici -> pereți/suprafețe continue.
  4) Export ca .pcd (doar x y z), lăsând originalul neatins.

Dependințe:  pip install numpy scipy
Format acceptat la intrare: PCD ascii sau binary (necompresat). Câmpurile în
plus (ex. rgb) sunt ignorate; se păstrează doar x y z.

Exemple:
  python3 solidify_map.py harta.pcd                     # salvează harta_solid.pcd în ACELAȘI folder
  python3 solidify_map.py harta.pcd out.pcd             # sau nume de ieșire explicit
  python3 solidify_map.py harta.pcd --voxel 0.08 --close-iter 2
  python3 solidify_map.py harta.pcd --keep-frac 0.01    # curățare mai agresivă

Dacă nu dai fișierul de ieșire, rezultatul se salvează lângă cel de intrare, ca
<nume>_solid.pcd (deci dacă harta e în folderul map, și rezultatul ajunge acolo).
"""

import argparse
import os
import struct
import sys

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


# ----------------------------- citire / scriere PCD -----------------------------

def load_pcd_xyz(path):
    """Citește un PCD (ascii sau binary necompresat) și întoarce un array Nx3 (x,y,z)."""
    fields, sizes, types, counts = [], [], [], []
    npoints, data_fmt = None, None
    header_len = 0
    with open(path, "rb") as f:
        while True:
            raw = f.readline()
            if not raw:
                raise ValueError("Header PCD incomplet.")
            header_len += len(raw)
            line = raw.decode("ascii", "replace").strip()
            if line.startswith("#") or line == "":
                continue
            key, _, val = line.partition(" ")
            key = key.upper()
            if key == "FIELDS":
                fields = val.split()
            elif key == "SIZE":
                sizes = [int(x) for x in val.split()]
            elif key == "TYPE":
                types = val.split()
            elif key == "COUNT":
                counts = [int(x) for x in val.split()]
            elif key == "POINTS":
                npoints = int(val)
            elif key == "WIDTH" and npoints is None:
                npoints = int(val)
            elif key == "DATA":
                data_fmt = val.split()[0].lower()
                break
        if not counts:
            counts = [1] * len(fields)
        ix, iy, iz = fields.index("x"), fields.index("y"), fields.index("z")

        if data_fmt == "ascii":
            arr = np.loadtxt(f, usecols=(ix, iy, iz), dtype=np.float64)
            return arr.reshape(-1, 3)

        if data_fmt == "binary":
            # offset (bytes) și tipul struct pentru fiecare câmp
            np_type = {("F", 4): "f", ("F", 8): "d", ("U", 1): "B", ("U", 2): "H",
                       ("U", 4): "I", ("I", 1): "b", ("I", 2): "h", ("I", 4): "i"}
            offsets, fmts, stride = [], [], 0
            for t, s, c in zip(types, sizes, counts):
                offsets.append(stride)
                fmts.append(np_type.get((t, s), "f"))
                stride += s * c
            buf = f.read(npoints * stride)
            out = np.empty((npoints, 3), dtype=np.float64)
            for row in range(npoints):
                base = row * stride
                out[row, 0] = struct.unpack_from("<" + fmts[ix], buf, base + offsets[ix])[0]
                out[row, 1] = struct.unpack_from("<" + fmts[iy], buf, base + offsets[iy])[0]
                out[row, 2] = struct.unpack_from("<" + fmts[iz], buf, base + offsets[iz])[0]
            return out

        raise ValueError(f"DATA '{data_fmt}' neacceptat (folosește ascii sau binary, "
                         f"nu binary_compressed).")


def save_pcd_ascii(path, P):
    """Scrie un array Nx3 ca PCD ascii (doar x y z)."""
    P = np.asarray(P, dtype=np.float32)
    n = len(P)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    header = ("# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\n"
              "COUNT 1 1 1\nWIDTH %d\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
              "POINTS %d\nDATA ascii\n" % (n, n))
    with open(path, "w") as f:
        f.write(header)
        np.savetxt(f, P, fmt="%.4f")


# --------------------------------- pipeline ------------------------------------

def statistical_outlier_removal(P, k=16, std_ratio=2.0):
    """Elimină punctele a căror distanță medie la K vecini e mult peste medie."""
    tree = cKDTree(P)
    d, _ = tree.query(P, k=k + 1)          # +1 = punctul însuși
    mean_d = d[:, 1:].mean(axis=1)
    thr = mean_d.mean() + std_ratio * mean_d.std()
    return P[mean_d <= thr], int((mean_d > thr).sum())


def solidify(P, voxel=0.05, keep_frac=0.003, close_iter=1, dilate_iter=0, fill_holes=False):
    """Voxelizare -> păstrează componentele mari -> închidere morfologică
    (opțional dilatare + umplere de cavități pentru suprafețe mai solide).
    Întoarce noul nor de puncte (centre de voxel ocupat)."""
    origin = P.min(axis=0)
    idx = np.floor((P - origin) / voxel).astype(np.int64)
    dims = tuple(idx.max(axis=0) + 1)
    grid = np.zeros(dims, dtype=bool)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    occ0 = int(grid.sum())

    # componente conexe 3D (26-conectivitate): păstrează structurile mari
    lbl, ncomp = ndimage.label(grid, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    thr = max(keep_frac * sizes.max(), 8)
    keep = np.where(sizes >= thr)[0]
    grid &= np.isin(lbl, keep)
    kept, dropped = len(keep), ncomp - len(keep)

    # închidere morfologică: pereți/suprafețe continue, fără găuri mici
    if close_iter > 0:
        grid = ndimage.binary_closing(grid, structure=np.ones((3, 3, 3)),
                                      iterations=close_iter)
    # dilatare: îngroașă și unește suprafețele (mai "solid"). Atenție: prea mult
    # închide uși și mănâncă din spațiul liber -> rău pentru navigație.
    if dilate_iter > 0:
        grid = ndimage.binary_dilation(grid, structure=np.ones((3, 3, 3)),
                                       iterations=dilate_iter)
    # umple cavitățile complet închise (volume solide)
    if fill_holes:
        grid = ndimage.binary_fill_holes(grid)

    pts = origin + (np.argwhere(grid) + 0.5) * voxel
    stats = dict(occ0=occ0, ncomp=ncomp, kept=kept, dropped=dropped,
                 thr=int(thr), occ_final=int(grid.sum()))
    return pts, stats


def main():
    ap = argparse.ArgumentParser(description="Solidifică o hartă PCD (denoise + voxel + închidere morfologică).")
    ap.add_argument("input", help="fișier PCD de intrare")
    ap.add_argument("output", nargs="?", default=None,
                    help="fișier PCD de ieșire (opțional; implicit <nume>_solid.pcd lângă intrare)")
    ap.add_argument("--voxel", type=float, default=0.05, help="mărimea voxelului în metri (implicit 0.05 = 5 cm)")
    ap.add_argument("--sor-k", type=int, default=16, help="nr. de vecini pentru SOR (implicit 16)")
    ap.add_argument("--sor-std", type=float, default=2.0, help="prag SOR = medie + std*sigma (implicit 2.0; mai mic = mai agresiv)")
    ap.add_argument("--keep-frac", type=float, default=0.003, help="păstrează componentele >= frac*cea_mai_mare (implicit 0.003; mai mare = mai agresiv)")
    ap.add_argument("--close-iter", type=int, default=1, help="iterații închidere morfologică (implicit 1; 2 = pereți mai plini; 0 = deloc)")
    ap.add_argument("--dilate", type=int, default=0, help="iterații de dilatare pentru suprafețe mai solide/unite (implicit 0; 1 = clar mai solid; atenție: îngustează ușile)")
    ap.add_argument("--fill-holes", action="store_true", help="umple cavitățile complet închise (volume solide)")
    ap.add_argument("--no-sor", action="store_true", help="sari peste eliminarea de zgomot SOR")
    args = ap.parse_args()

    # ieșire implicită: lângă fișierul de intrare, ca <nume>_solid.pcd
    if args.output is None:
        base, _ = os.path.splitext(args.input)
        args.output = base + "_solid.pcd"

    if os.path.abspath(args.input) == os.path.abspath(args.output):
        sys.exit("EROARE: intrarea și ieșirea sunt același fișier. Alege alt nume de ieșire.")

    P = load_pcd_xyz(args.input)
    n0 = len(P)
    print(f"[1] încărcat: {n0} puncte din {args.input}")

    if not args.no_sor:
        P, removed = statistical_outlier_removal(P, args.sor_k, args.sor_std)
        print(f"[2] SOR: scos {removed} fulgi izolați -> {len(P)} puncte")

    pts, s = solidify(P, args.voxel, args.keep_frac, args.close_iter,
                      args.dilate, args.fill_holes)
    print(f"[3] voxel {args.voxel*100:.0f} cm: {s['occ0']} voxeli; componente {s['ncomp']} "
          f"-> păstrate {s['kept']} (prag {s['thr']}), aruncate {s['dropped']} insule")
    print(f"[4] închidere x{args.close_iter}: {s['occ_final']} voxeli finali")

    save_pcd_ascii(args.output, pts)
    print(f"[5] scris {len(pts)} puncte -> {args.output}")
    print(f"    rezumat: {n0} -> {len(pts)} puncte")


if __name__ == "__main__":
    main()
