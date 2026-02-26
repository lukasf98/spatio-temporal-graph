"""
Bidirectional centroid matching for edge type classification.

Given registered PET segmentations across timepoints, classifies temporal
edges between lesion connected components as continuation, split, or merge
based on the cardinality of forward/backward nearest-neighbor sets.

See paper Section 3.2 (Centroid Matching and Edge Classification).
"""

import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
from collections import defaultdict


def load_cc_data(reg_root, ref_date, date_str):
    """Load registered PET segmentation and extract connected component centroids.

    Parameters
    ----------
    reg_root : str
        Path to registered_to_<ref_date> directory.
    ref_date : str
        Reference date (YYYYMMDD).
    date_str : str
        Timepoint date (YYYYMMDD).

    Returns
    -------
    list of (int, np.ndarray)
        List of (cc_label, centroid_mm) for each connected component.
    """
    tp_dir = os.path.join(reg_root, date_str)
    if date_str == ref_date:
        seg_path = os.path.join(tp_dir, "PETseg_ref.nii.gz")
    else:
        seg_path = os.path.join(tp_dir, f"PETseg_to_{ref_date}.nii.gz")
    if not os.path.exists(seg_path):
        return []
    seg_img = sitk.ReadImage(seg_path)
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_img = cc_filter.Execute(seg_img)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc_img)
    return [(label, np.array(stats.GetCentroid(label))) for label in stats.GetLabels()]


def bidirectional_matching(reg_root, ref_date, timepoints, max_dist=30.0):
    """Bidirectional centroid matching between consecutive timepoints.

    For each pair of consecutive timepoints, computes forward (prev → curr)
    and backward (curr → prev) nearest-neighbor matches. The union of both
    directions forms the edge set. Edge types are classified by the
    cardinality of the children/parent sets.

    Parameters
    ----------
    reg_root : str
        Path to registered_to_<ref_date> directory.
    ref_date : str
        Reference date (YYYYMMDD).
    timepoints : list of str
        Sorted list of timepoint dates (YYYYMMDD).
    max_dist : float
        Maximum matching distance in mm (default: 30.0).

    Returns
    -------
    total_splits : int
        Number of split events (one parent → multiple children).
    total_merges : int
        Number of merge events (multiple parents → one child).
    total_edges : int
        Total number of edges across all consecutive pairs.
    """
    tp_ccs = {tp: load_cc_data(reg_root, ref_date, tp) for tp in timepoints}
    total_splits, total_merges, total_edges = 0, 0, 0

    for i in range(len(timepoints) - 1):
        prev_ccs = tp_ccs[timepoints[i]]
        curr_ccs = tp_ccs[timepoints[i + 1]]
        if not prev_ccs or not curr_ccs:
            continue

        edge_set = set()

        # Forward: each prev → nearest curr
        for p_label, p_cent in prev_ccs:
            best_d, best_c = float("inf"), None
            for c_label, c_cent in curr_ccs:
                d = float(np.linalg.norm(p_cent - c_cent))
                if d < best_d:
                    best_d, best_c = d, c_label
            if best_d <= max_dist and best_c is not None:
                edge_set.add((p_label, best_c))

        # Backward: each curr → nearest prev
        for c_label, c_cent in curr_ccs:
            best_d, best_p = float("inf"), None
            for p_label, p_cent in prev_ccs:
                d = float(np.linalg.norm(c_cent - p_cent))
                if d < best_d:
                    best_d, best_p = d, p_label
            if best_d <= max_dist and best_p is not None:
                edge_set.add((best_p, c_label))

        prev_children = defaultdict(set)
        curr_parents = defaultdict(set)
        for p, c in edge_set:
            prev_children[p].add(c)
            curr_parents[c].add(p)

        total_splits += sum(1 for p in prev_children if len(prev_children[p]) > 1)
        total_merges += sum(1 for c in curr_parents if len(curr_parents[c]) > 1)
        total_edges += len(edge_set)

    return total_splits, total_merges, total_edges


def forward_only_matching(reg_root, ref_date, timepoints, max_dist=30.0):
    """Forward-only nearest-neighbor matching (baseline).

    Each node at time t is matched to its nearest neighbor at t+1.
    By construction, splits cannot be detected (each parent gets at most
    one child).

    Parameters
    ----------
    reg_root : str
        Path to registered_to_<ref_date> directory.
    ref_date : str
        Reference date (YYYYMMDD).
    timepoints : list of str
        Sorted list of timepoint dates (YYYYMMDD).
    max_dist : float
        Maximum matching distance in mm (default: 30.0).

    Returns
    -------
    total_splits : int
        Always 0 by construction.
    total_merges : int
        Number of merge events detected.
    """
    tp_ccs = {tp: load_cc_data(reg_root, ref_date, tp) for tp in timepoints}
    total_splits, total_merges = 0, 0

    for i in range(len(timepoints) - 1):
        prev_ccs = tp_ccs[timepoints[i]]
        curr_ccs = tp_ccs[timepoints[i + 1]]
        if not prev_ccs or not curr_ccs:
            continue

        edge_set = set()
        for p_label, p_cent in prev_ccs:
            best_d, best_c = float("inf"), None
            for c_label, c_cent in curr_ccs:
                d = float(np.linalg.norm(p_cent - c_cent))
                if d < best_d:
                    best_d, best_c = d, c_label
            if best_d <= max_dist and best_c is not None:
                edge_set.add((p_label, best_c))

        prev_children = defaultdict(set)
        curr_parents = defaultdict(set)
        for p, c in edge_set:
            prev_children[p].add(c)
            curr_parents[c].add(p)

        total_splits += sum(1 for p in prev_children if len(prev_children[p]) > 1)
        total_merges += sum(1 for c in curr_parents if len(curr_parents[c]) > 1)

    return total_splits, total_merges


def classify_edges(reg_root, ref_date, timepoints, max_dist=30.0):
    """Classify all edges between consecutive timepoints.

    Returns a detailed list of edges with type labels.

    Parameters
    ----------
    reg_root : str
        Path to registered_to_<ref_date> directory.
    ref_date : str
        Reference date (YYYYMMDD).
    timepoints : list of str
        Sorted list of timepoint dates (YYYYMMDD).
    max_dist : float
        Maximum matching distance in mm (default: 30.0).

    Returns
    -------
    pd.DataFrame
        Columns: tp_prev, tp_curr, label_prev, label_curr, distance_mm, edge_type
        where edge_type is one of 'continuation', 'split', 'merge', 'split+merge'.
    """
    tp_ccs = {tp: load_cc_data(reg_root, ref_date, tp) for tp in timepoints}
    rows = []

    for i in range(len(timepoints) - 1):
        tp_prev, tp_curr = timepoints[i], timepoints[i + 1]
        prev_ccs = tp_ccs[tp_prev]
        curr_ccs = tp_ccs[tp_curr]
        if not prev_ccs or not curr_ccs:
            continue

        # Build centroid lookup
        prev_cents = {label: cent for label, cent in prev_ccs}
        curr_cents = {label: cent for label, cent in curr_ccs}

        edge_set = set()

        # Forward
        for p_label, p_cent in prev_ccs:
            best_d, best_c = float("inf"), None
            for c_label, c_cent in curr_ccs:
                d = float(np.linalg.norm(p_cent - c_cent))
                if d < best_d:
                    best_d, best_c = d, c_label
            if best_d <= max_dist and best_c is not None:
                edge_set.add((p_label, best_c))

        # Backward
        for c_label, c_cent in curr_ccs:
            best_d, best_p = float("inf"), None
            for p_label, p_cent in prev_ccs:
                d = float(np.linalg.norm(c_cent - p_cent))
                if d < best_d:
                    best_d, best_p = d, p_label
            if best_d <= max_dist and best_p is not None:
                edge_set.add((best_p, c_label))

        # Classify
        prev_children = defaultdict(set)
        curr_parents = defaultdict(set)
        for p, c in edge_set:
            prev_children[p].add(c)
            curr_parents[c].add(p)

        for p, c in edge_set:
            n_children = len(prev_children[p])
            n_parents = len(curr_parents[c])
            if n_children == 1 and n_parents == 1:
                edge_type = "continuation"
            elif n_children > 1 and n_parents == 1:
                edge_type = "split"
            elif n_children == 1 and n_parents > 1:
                edge_type = "merge"
            else:
                edge_type = "split+merge"

            dist = float(np.linalg.norm(prev_cents[p] - curr_cents[c]))
            rows.append({
                "tp_prev": tp_prev,
                "tp_curr": tp_curr,
                "label_prev": p,
                "label_curr": c,
                "distance_mm": round(dist, 2),
                "edge_type": edge_type,
            })

    return pd.DataFrame(rows)
