#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-lesion and per-patient SUV metric extraction for the lesion graph.

Fills LesionNode SUV attributes from registered SUV images.
"""

import os
import numpy as np
import SimpleITK as sitk
from typing import Dict, List, Tuple

from lesion_graph import PatientGraph, LesionNode


# ---------------------------------------------------------------------------
# Per-lesion SUV extraction
# ---------------------------------------------------------------------------

def extract_lesion_suv_metrics(suv_arr: np.ndarray,
                               cc_arr: np.ndarray,
                               cc_label: int,
                               spacing: Tuple[float, float, float]
                               ) -> Dict[str, float]:
    """Extract SUV metrics for a single connected component.

    Parameters
    ----------
    suv_arr : np.ndarray
        SUV image array (z, y, x).
    cc_arr : np.ndarray
        Connected component label array (z, y, x), same grid as suv_arr.
    cc_label : int
        Label of the CC to extract metrics for.
    spacing : tuple
        Voxel spacing (x, y, z) in mm.

    Returns
    -------
    dict with keys: suv_mean, suv_max, suv_peak, suv_std, suv_median,
                    volume_ml, tlg
    """
    mask = cc_arr == cc_label
    if not mask.any():
        return dict(suv_mean=0, suv_max=0, suv_peak=0, suv_std=0,
                    suv_median=0, volume_ml=0, tlg=0)

    voxel_values = suv_arr[mask]
    voxel_vol_ml = (spacing[0] * spacing[1] * spacing[2]) / 1000.0  # mm³ → mL

    suv_mean = float(np.mean(voxel_values))
    suv_max = float(np.max(voxel_values))
    suv_std = float(np.std(voxel_values))
    suv_median = float(np.median(voxel_values))
    volume_ml = float(mask.sum()) * voxel_vol_ml
    tlg = suv_mean * volume_ml

    # SUV_peak: mean SUV in 1 cm³ sphere centered on voxel of max uptake (PERCIST)
    suv_peak = _compute_suv_peak(suv_arr, mask, spacing)

    return dict(
        suv_mean=suv_mean,
        suv_max=suv_max,
        suv_peak=suv_peak,
        suv_std=suv_std,
        suv_median=suv_median,
        volume_ml=volume_ml,
        tlg=tlg,
    )


def _compute_suv_peak(suv_arr, mask, spacing):
    """PERCIST SUV_peak: mean within 1 cm³ sphere at hottest voxel.

    The sphere radius for 1 cm³ = (3/(4π))^(1/3) ≈ 0.6204 cm = 6.204 mm.
    """
    radius_mm = 6.204

    # Find hottest voxel within the lesion mask
    masked_suv = np.where(mask, suv_arr, -np.inf)
    max_idx = np.unravel_index(np.argmax(masked_suv), suv_arr.shape)

    # Build sphere mask around hottest voxel
    # spacing is (x, y, z), array is (z, y, x)
    sz, sy, sx = spacing[2], spacing[1], spacing[0]

    # Determine search range in voxels
    rz = int(np.ceil(radius_mm / sz)) + 1
    ry = int(np.ceil(radius_mm / sy)) + 1
    rx = int(np.ceil(radius_mm / sx)) + 1

    zc, yc, xc = max_idx
    z_lo = max(0, zc - rz)
    z_hi = min(suv_arr.shape[0], zc + rz + 1)
    y_lo = max(0, yc - ry)
    y_hi = min(suv_arr.shape[1], yc + ry + 1)
    x_lo = max(0, xc - rx)
    x_hi = min(suv_arr.shape[2], xc + rx + 1)

    # Create local coordinate arrays
    zz, yy, xx = np.mgrid[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi]
    dist_sq = ((zz - zc) * sz) ** 2 + ((yy - yc) * sy) ** 2 + ((xx - xc) * sx) ** 2
    sphere_mask = dist_sq <= radius_mm ** 2

    sphere_values = suv_arr[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi][sphere_mask]
    # Exclude background fill values (e.g. -1024 outside body from registration)
    sphere_values = sphere_values[sphere_values >= 0]

    if len(sphere_values) == 0:
        return float(suv_arr[max_idx])

    return float(np.mean(sphere_values))


# ---------------------------------------------------------------------------
# Patient-level aggregate metrics
# ---------------------------------------------------------------------------

def compute_patient_level_metrics(nodes: List[LesionNode]) -> Dict[str, float]:
    """Compute patient-aggregate metrics from nodes at a single timepoint.

    Returns volume-weighted suv_mean/peak/std, suv_max (global), tmtv,
    tlg_total, area_total, lesion_count.
    """
    if not nodes:
        return dict(suv_mean_wt=0, suv_max=0, suv_peak_max=0, suv_std_wt=0,
                    tmtv=0, tlg_total=0, area_total=0, lesion_count=0)

    volumes = np.array([n.volume for n in nodes])  # mm³
    volumes_ml = volumes / 1000.0
    suv_means = np.array([n.suv_mean for n in nodes])
    suv_maxs = np.array([n.suv_max for n in nodes])
    suv_peaks = np.array([n.suv_peak for n in nodes])
    suv_stds = np.array([n.suv_std for n in nodes])
    areas = np.array([getattr(n, 'area', 0) for n in nodes])
    tlgs = np.array([n.tlg for n in nodes])

    total_vol_ml = float(volumes_ml.sum())
    if total_vol_ml > 0:
        suv_mean_wt = float(np.average(suv_means, weights=volumes_ml))
        suv_std_wt = float(np.average(suv_stds, weights=volumes_ml))
    else:
        suv_mean_wt = 0.0
        suv_std_wt = 0.0

    return dict(
        suv_mean_wt=suv_mean_wt,
        suv_max=float(suv_maxs.max()) if len(suv_maxs) else 0.0,
        suv_peak_max=float(suv_peaks.max()) if len(suv_peaks) else 0.0,
        suv_std_wt=suv_std_wt,
        tmtv=total_vol_ml,
        tlg_total=float(tlgs.sum()),
        area_total=float(areas.sum()),
        lesion_count=len(nodes),
    )


# ---------------------------------------------------------------------------
# Populate graph with SUV metrics
# ---------------------------------------------------------------------------

def populate_graph_suv_metrics(graph: PatientGraph,
                               registered_root: str,
                               ref_date: str):
    """Fill SUV attributes on all graph nodes from registered SUV images.

    Each timepoint's PETseg CC labels and SUV image are on the same grid,
    so no cross-timepoint resampling is needed.
    """
    for tp in graph.timepoints:
        tp_dir = os.path.join(registered_root, tp)
        nodes = graph.get_nodes_at_tp(tp)
        if not nodes:
            continue

        # Load SUV image
        if tp == ref_date:
            suv_path = os.path.join(tp_dir, "PET_ref_SUV.nii.gz")
            seg_path = os.path.join(tp_dir, "PETseg_ref.nii.gz")
        else:
            suv_path = os.path.join(tp_dir, f"SUV_to_{ref_date}.nii.gz")
            seg_path = os.path.join(tp_dir, f"PETseg_to_{ref_date}.nii.gz")

        if not os.path.exists(suv_path) or not os.path.exists(seg_path):
            print(f"  [{tp}] SUV/seg not found, skipping SUV extraction")
            continue

        suv_img = sitk.ReadImage(suv_path)
        seg_img = sitk.ReadImage(seg_path)

        # For reference timepoint, PETseg is on PET grid and SUV is also on PET grid
        # For follow-ups, both are on CT grid. So they always match within a timepoint.
        suv_arr = sitk.GetArrayFromImage(suv_img)
        spacing = suv_img.GetSpacing()  # (x, y, z)

        # Run CC on the segmentation (same as graph building)
        cc_filter = sitk.ConnectedComponentImageFilter()
        cc_img = cc_filter.Execute(seg_img)
        cc_arr = sitk.GetArrayFromImage(cc_img)

        for node in nodes:
            metrics = extract_lesion_suv_metrics(
                suv_arr, cc_arr, node.cc_label, spacing)

            node.suv_mean = metrics['suv_mean']
            node.suv_max = metrics['suv_max']
            node.suv_peak = metrics['suv_peak']
            node.suv_std = metrics['suv_std']
            node.suv_median = metrics['suv_median']
            node.tlg = metrics['tlg']
            # Update volume to mL-based value for consistency
            # (node.volume is already in mm³ from graph building)

        print(f"  [{tp}] SUV extracted for {len(nodes)} lesion(s)")
