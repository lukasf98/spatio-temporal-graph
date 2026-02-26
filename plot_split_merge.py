#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split/merge diagram for a single lesion lineage.

Uses bidirectional nearest-centroid matching between consecutive timepoints
to detect splits (one prev CC -> multiple curr CCs) and merges
(multiple prev CCs -> one curr CC). Produces a two-panel figure:
  (a) Split/merge topology with nodes colored by SUV_peak and sized by volume.
  (b) Volume-weighted SUV_peak trajectory for the target lineage vs. all others.

Usage:
    python plot_split_merge.py \\
        --registered-root /path/to/registered_to_YYYYMMDD \\
        --lineage 1 \\
        --timepoints 20220401 20221110 20230509 20230901 20240117 20240410 20250122 \\
        --output split_merge.png

See paper Section 4.2 (Topology Changes) and Figure 3.
"""

import os
import argparse
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from collections import defaultdict
from datetime import datetime


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def compute_suv_peak(suv_arr, mask, spacing):
    """PERCIST SUV_peak: mean within 1 cm^3 sphere at hottest voxel."""
    radius_mm = 6.204
    masked_suv = np.where(mask, suv_arr, -np.inf)
    max_idx = np.unravel_index(np.argmax(masked_suv), suv_arr.shape)

    sz, sy, sx = spacing[2], spacing[1], spacing[0]
    rz = int(np.ceil(radius_mm / sz)) + 1
    ry = int(np.ceil(radius_mm / sy)) + 1
    rx = int(np.ceil(radius_mm / sx)) + 1

    zc, yc, xc = max_idx
    z_lo, z_hi = max(0, zc - rz), min(suv_arr.shape[0], zc + rz + 1)
    y_lo, y_hi = max(0, yc - ry), min(suv_arr.shape[1], yc + ry + 1)
    x_lo, x_hi = max(0, xc - rx), min(suv_arr.shape[2], xc + rx + 1)

    zz, yy, xx = np.mgrid[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi]
    dist_sq = ((zz - zc) * sz)**2 + ((yy - yc) * sy)**2 + ((xx - xc) * sx)**2
    sphere_mask = dist_sq <= radius_mm**2

    sphere_values = suv_arr[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi][sphere_mask]
    sphere_values = sphere_values[sphere_values >= 0]

    if len(sphere_values) == 0:
        return float(suv_arr[max_idx])
    return float(np.mean(sphere_values))


def load_cc_data(reg_root, ref_date, date_str):
    """Load PETseg + SUV, run CC, return list of (label, centroid, volume_ml, suv_peak)."""
    tp_dir = os.path.join(reg_root, date_str)
    if date_str == ref_date:
        seg_path = os.path.join(tp_dir, "PETseg_ref.nii.gz")
        suv_path = os.path.join(tp_dir, "PET_ref_SUV.nii.gz")
    else:
        seg_path = os.path.join(tp_dir, f"PETseg_to_{ref_date}.nii.gz")
        suv_path = os.path.join(tp_dir, f"SUV_to_{ref_date}.nii.gz")

    if not os.path.exists(seg_path):
        return []

    seg_img = sitk.ReadImage(seg_path)
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_img = cc_filter.Execute(seg_img)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc_img)

    suv_arr = None
    spacing = seg_img.GetSpacing()  # (x, y, z)
    if os.path.exists(suv_path):
        suv_img = sitk.ReadImage(suv_path)
        suv_arr = sitk.GetArrayFromImage(suv_img)
    cc_arr = sitk.GetArrayFromImage(cc_img)

    voxel_vol_ml = (spacing[0] * spacing[1] * spacing[2]) / 1000.0

    results = []
    for label in stats.GetLabels():
        centroid = stats.GetCentroid(label)
        volume_ml = stats.GetNumberOfPixels(label) * voxel_vol_ml

        suv_peak = 0.0
        if suv_arr is not None:
            mask = cc_arr == label
            suv_peak = compute_suv_peak(suv_arr, mask, spacing)

        results.append((label, np.array(centroid), volume_ml, suv_peak))
    return results


# ---------------------------------------------------------------------------
# Lineage filtering via bidirectional matching
# ---------------------------------------------------------------------------

def filter_lineage_ccs(tp_ccs, timepoints, max_dist=30.0):
    """Filter CCs to those belonging to the seed lineage using bidirectional matching.

    Seeds from all CCs at the first timepoint, then propagates forward using
    bidirectional nearest-neighbor matching between consecutive timepoints.

    Parameters
    ----------
    tp_ccs : dict
        tp -> list of (label, centroid, volume_ml, suv_peak)
    timepoints : list of str
        Sorted timepoints to track.
    max_dist : float
        Maximum matching distance in mm.

    Returns
    -------
    dict : tp -> list of (label, centroid, volume_ml, suv_peak) for matched CCs.
    """
    lineage_ccs = {}
    lineage_ccs[timepoints[0]] = tp_ccs[timepoints[0]][:]

    for i in range(len(timepoints) - 1):
        tp_prev = timepoints[i]
        tp_curr = timepoints[i + 1]

        prev_ccs = lineage_ccs[tp_prev]
        curr_ccs = tp_ccs[tp_curr]

        if not prev_ccs or not curr_ccs:
            lineage_ccs[tp_curr] = []
            continue

        matched_curr_labels = set()

        # Forward: for each prev CC, find nearest curr CC
        for _, p_cent, _, _ in prev_ccs:
            best_dist = float('inf')
            best_label = None
            for c_label, c_cent, _, _ in curr_ccs:
                d = float(np.linalg.norm(p_cent - c_cent))
                if d < best_dist:
                    best_dist = d
                    best_label = c_label
            if best_dist <= max_dist and best_label is not None:
                matched_curr_labels.add(best_label)

        # Backward: for each curr CC, find nearest prev CC
        for c_label, c_cent, _, _ in curr_ccs:
            best_dist = float('inf')
            for _, p_cent, _, _ in prev_ccs:
                d = float(np.linalg.norm(c_cent - p_cent))
                if d < best_dist:
                    best_dist = d
            if best_dist <= max_dist:
                matched_curr_labels.add(c_label)

        lineage_ccs[tp_curr] = [(l, c, v, s) for l, c, v, s in curr_ccs
                                 if l in matched_curr_labels]

    return lineage_ccs


# ---------------------------------------------------------------------------
# Edge classification
# ---------------------------------------------------------------------------

def classify_lineage_edges(lineage_ccs, timepoints, max_dist=30.0):
    """Bidirectional edge matching + classification for the filtered lineage.

    Returns list of (tp_prev, prev_label, tp_curr, curr_label, distance, edge_type).
    """
    all_edges = []

    for i in range(len(timepoints) - 1):
        tp_prev = timepoints[i]
        tp_curr = timepoints[i + 1]

        prev_ccs = lineage_ccs[tp_prev]
        curr_ccs = lineage_ccs[tp_curr]

        if not prev_ccs or not curr_ccs:
            continue

        edge_set = set()
        edge_dists = {}

        # Forward
        for p_label, p_cent, _, _ in prev_ccs:
            best_dist = float('inf')
            best_c = None
            for c_label, c_cent, _, _ in curr_ccs:
                d = float(np.linalg.norm(p_cent - c_cent))
                if d < best_dist:
                    best_dist = d
                    best_c = c_label
            if best_dist <= max_dist and best_c is not None:
                edge_set.add((p_label, best_c))
                edge_dists[(p_label, best_c)] = best_dist

        # Backward
        for c_label, c_cent, _, _ in curr_ccs:
            best_dist = float('inf')
            best_p = None
            for p_label, p_cent, _, _ in prev_ccs:
                d = float(np.linalg.norm(c_cent - p_cent))
                if d < best_dist:
                    best_dist = d
                    best_p = p_label
            if best_dist <= max_dist and best_p is not None:
                edge_set.add((best_p, c_label))
                edge_dists[(best_p, c_label)] = best_dist

        # Classify
        prev_children = defaultdict(set)
        curr_parents = defaultdict(set)
        for p_label, c_label in edge_set:
            prev_children[p_label].add(c_label)
            curr_parents[c_label].add(p_label)

        for p_label, c_label in edge_set:
            n_children = len(prev_children[p_label])
            n_parents = len(curr_parents[c_label])
            dist = edge_dists.get((p_label, c_label), 0)

            if n_children > 1 and n_parents > 1:
                etype = "split+merge"
            elif n_children > 1:
                etype = "split"
            elif n_parents > 1:
                etype = "merge"
            else:
                etype = "continuation"

            all_edges.append((tp_prev, p_label, tp_curr, c_label, dist, etype))

    return all_edges


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_split_merge(lineage_ccs, all_edges, timepoints,
                     nodes_csv=None, target_lineage=None,
                     tracer_filter=None, output_path="split_merge.png"):
    """Generate the two-panel split/merge figure.

    Parameters
    ----------
    lineage_ccs : dict
        tp -> list of (label, centroid, volume_ml, suv_peak).
    all_edges : list
        Output of classify_lineage_edges().
    timepoints : list of str
        Sorted timepoints.
    nodes_csv : str, optional
        Path to lesion_nodes.csv for the SUV trajectory panel.
    target_lineage : int, optional
        Lineage ID for trajectory highlighting.
    tracer_filter : str, optional
        Tracer to filter in trajectory panel (e.g. 'PSMA').
    output_path : str
        Output path for the figure.
    """
    # Collect SUV_peak and volume for all lineage CCs
    all_suv_peaks = []
    all_volumes = []
    node_data = {}

    for tp in timepoints:
        for label, centroid, vol, spk in lineage_ccs[tp]:
            node_data[(tp, label)] = (vol, spk)
            all_suv_peaks.append(spk)
            all_volumes.append(vol)

    if not all_suv_peaks:
        print("No nodes to plot!")
        return

    # Setup figure
    has_trajectory = nodes_csv is not None and target_lineage is not None
    if has_trajectory:
        fig = plt.figure(figsize=(18, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1.5, 1], wspace=0.15)
        ax = fig.add_subplot(gs[0, 0])
    else:
        fig, ax = plt.subplots(figsize=(12, 6))

    # Colormap for SUV_peak
    vmin = max(0, min(all_suv_peaks) - 0.5)
    vmax = max(all_suv_peaks) + 0.5
    cmap = cm.hot_r
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Size scaling for volume
    vol_min = min(all_volumes) if all_volumes else 0.01
    vol_max = max(all_volumes) if all_volumes else 1.0
    SIZE_MIN, SIZE_MAX = 150, 1200

    def vol_to_size(vol):
        if vol_max == vol_min:
            return (SIZE_MIN + SIZE_MAX) / 2
        frac = (vol - vol_min) / (vol_max - vol_min)
        return SIZE_MIN + frac * (SIZE_MAX - SIZE_MIN)

    # X positions: calendar-based spacing (months from first timepoint)
    tp_dates = [datetime.strptime(tp, '%Y%m%d') for tp in timepoints]
    date_min = tp_dates[0]
    x_positions = {}
    for tp, dt in zip(timepoints, tp_dates):
        x_positions[tp] = (dt - date_min).days / 30.0

    # Y positions: stack CCs within each timepoint, centered
    y_positions = {}
    node_spacing = 1.5

    for tp in timepoints:
        ccs = lineage_ccs[tp]
        n = len(ccs)
        if n == 0:
            continue
        for j, (label, _, _, _) in enumerate(sorted(ccs, key=lambda x: x[0])):
            y_positions[(tp, label)] = (j - (n - 1) / 2.0) * node_spacing

    # Draw edges
    for tp_prev, p_label, tp_curr, c_label, dist, etype in all_edges:
        x0 = x_positions[tp_prev]
        y0 = y_positions.get((tp_prev, p_label), 0)
        x1 = x_positions[tp_curr]
        y1 = y_positions.get((tp_curr, c_label), 0)

        if etype == "continuation":
            ax.plot([x0, x1], [y0, y1], '-', color='gray', lw=2, alpha=0.7)
        elif etype == "split":
            ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle='->', color='blue',
                                        lw=2, ls='--', alpha=0.8))
        elif etype == "merge":
            ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle='->', color='red',
                                        lw=2, ls='--', alpha=0.8))
        elif etype == "split+merge":
            ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle='->', color='purple',
                                        lw=2, ls=':', alpha=0.8))

    # Draw nodes
    for tp in timepoints:
        ccs = lineage_ccs[tp]
        x = x_positions[tp]
        for label, centroid, vol, spk in ccs:
            y = y_positions.get((tp, label), 0)
            sz = vol_to_size(vol)
            color = cmap(norm(spk))
            ax.scatter(x, y, s=sz, c=[color], edgecolors='black',
                       linewidths=0.8, zorder=5)
            ax.text(x, y, str(label), ha='center', va='center',
                    fontsize=7, fontweight='bold', color='black', zorder=6)

    # Event labels between timepoints
    for i in range(len(timepoints) - 1):
        tp_prev = timepoints[i]
        tp_curr = timepoints[i + 1]
        n_prev = len(lineage_ccs[tp_prev])
        n_curr = len(lineage_ccs[tp_curr])

        tp_edges = [(e[4], e[5]) for e in all_edges
                    if e[0] == tp_prev and e[2] == tp_curr]
        etypes = set(e[1] for e in tp_edges)

        if "split" in etypes and "merge" in etypes:
            event_label, event_color = "SPLIT+MERGE", "purple"
        elif "split" in etypes:
            event_label, event_color = f"SPLIT ({n_prev}\u2192{n_curr})", "blue"
        elif "merge" in etypes:
            event_label, event_color = f"MERGE ({n_prev}\u2192{n_curr})", "red"
        else:
            event_label, event_color = f"CONT ({n_prev}\u2192{n_curr})", "gray"

        mid_x = (x_positions[tp_prev] + x_positions[tp_curr]) / 2
        y_top = max(y_positions.get((tp, l), 0)
                    for tp in [tp_prev, tp_curr]
                    for l, _, _, _ in lineage_ccs[tp]) + 1.0
        ax.text(mid_x, y_top, event_label, ha='center', va='bottom',
                fontsize=9, fontweight='bold', color=event_color)

    # Timepoint labels
    x_tick_positions = [x_positions[tp] for tp in timepoints]
    baseline_dt = tp_dates[0]
    rel_labels = []
    for tp, dt in zip(timepoints, tp_dates):
        delta_days = (dt - baseline_dt).days
        if delta_days == 0:
            rel_labels.append('Baseline')
        elif delta_days < 60:
            rel_labels.append(f'+{delta_days}d')
        else:
            months = round(delta_days / 30)
            rel_labels.append(f'+{months}mo')
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(rel_labels, rotation=0, ha='center')

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, shrink=0.7, aspect=20)
    cbar.set_label("SUV$_{peak}$", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Legend
    edge_handles = [
        plt.Line2D([0], [0], color='gray', lw=2, label='Continuation'),
        plt.Line2D([0], [0], color='blue', lw=2, ls='--', label='Split'),
        plt.Line2D([0], [0], color='red', lw=2, ls='--', label='Merge'),
    ]
    ax.legend(handles=edge_handles, loc='lower left', fontsize=8, framealpha=0.9)

    lineage_label = f"Lineage {target_lineage}" if target_lineage else "Lineage"
    ax.set_title(f"{lineage_label} Split/Merge Diagram",
                 fontsize=11, fontweight='bold')
    ax.set_ylabel("CC index (centered)", fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_ylim(ax.get_ylim()[0] - 0.5, ax.get_ylim()[1] + 2.0)

    if has_trajectory:
        ax.text(0.01, 0.98, '(a)', transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top')

    # ── Right panel: SUV trajectory ──────────────────────────────────
    if has_trajectory:
        ax2 = fig.add_subplot(gs[0, 1])

        nodes_df = pd.read_csv(nodes_csv)
        if tracer_filter:
            nodes_df = nodes_df[nodes_df['tracer'] == tracer_filter]
        nodes_df = nodes_df[nodes_df['suv_peak'] > 0].copy()

        baseline_tp = int(timepoints[0])

        # Target lineage: volume-weighted SUV peak per timepoint
        lin_nodes = nodes_df[nodes_df['lineage_id'] == target_lineage]
        lin_suv = []
        for tp_val, grp in lin_nodes.groupby('timepoint'):
            if tp_val < baseline_tp:
                continue
            weights = grp['volume_mm3'].values
            if weights.sum() > 0:
                wt_suv = float(np.average(grp['suv_peak'].values, weights=weights))
            else:
                wt_suv = float(grp['suv_peak'].mean())
            lin_suv.append((tp_val, wt_suv))
        lin_suv.sort()

        # All other lineages combined
        other_nodes = nodes_df[nodes_df['lineage_id'] != target_lineage]
        other_suv = []
        for tp_val, grp in other_nodes.groupby('timepoint'):
            if tp_val < baseline_tp:
                continue
            weights = grp['volume_mm3'].values
            if weights.sum() > 0:
                wt_suv = float(np.average(grp['suv_peak'].values, weights=weights))
            else:
                wt_suv = float(grp['suv_peak'].mean())
            other_suv.append((tp_val, wt_suv))
        other_suv.sort()

        # Convert to x positions (months from baseline)
        tp_x_map = {int(t): (datetime.strptime(t, '%Y%m%d') - date_min).days / 30.0
                     for t in timepoints}
        tp_ints = [int(t) for t in timepoints]

        def tp_to_x(tp_val):
            if tp_val in tp_x_map:
                return tp_x_map[tp_val]
            nearest = min(tp_ints, key=lambda t: abs(t - tp_val))
            return tp_x_map[nearest]

        other_x = [tp_to_x(tp) for tp, _ in other_suv]
        other_y = [s for _, s in other_suv]
        lin_x = [tp_to_x(tp) for tp, _ in lin_suv]
        lin_y = [s for _, s in lin_suv]

        ax2.plot(other_x, other_y, '-s', color='gray', lw=2, markersize=6,
                 markerfacecolor='gray', markeredgecolor='black',
                 markeredgewidth=0.5, label='All other lineages', alpha=0.8, zorder=3)
        ax2.plot(lin_x, lin_y, '-o', color='red', lw=2.5, markersize=7,
                 markerfacecolor='red', markeredgecolor='black',
                 markeredgewidth=0.5, label=f'Lineage {target_lineage}', zorder=4)

        ax2.set_xticks(x_tick_positions)
        ax2.set_xticklabels(rel_labels, rotation=45, ha='right', fontsize=7)
        ax2.set_ylabel("SUV$_{peak}$ (volume-weighted)", fontsize=9)
        ax2.set_title("SUV$_{peak}$ Trajectory", fontsize=11, fontweight='bold')
        ax2.legend(fontsize=8, framealpha=0.9, loc='lower right')
        ax2.grid(True, alpha=0.2)
        ax2.text(0.01, 0.98, '(b)', transform=ax2.transAxes,
                 fontsize=12, fontweight='bold', va='top')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    if output_path.endswith('.png'):
        pdf_path = output_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        print(f"Saved: {pdf_path}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Split/merge diagram for a single lesion lineage.")
    parser.add_argument("--registered-root", required=True,
                        help="Path to registered_to_YYYYMMDD directory")
    parser.add_argument("--timepoints", required=True, nargs='+',
                        help="Sorted timepoints (YYYYMMDD) to include")
    parser.add_argument("--lineage", type=int, default=None,
                        help="Lineage ID for SUV trajectory panel (requires --nodes-csv)")
    parser.add_argument("--tracer", default=None,
                        help="Filter trajectory panel to this tracer (e.g. PSMA)")
    parser.add_argument("--nodes-csv", default=None,
                        help="Path to lesion_nodes.csv for SUV trajectory panel")
    parser.add_argument("--max-distance", type=float, default=30.0,
                        help="Max matching distance in mm (default: 30.0)")
    parser.add_argument("--output", default="split_merge.png",
                        help="Output path (default: split_merge.png)")
    args = parser.parse_args()

    reg_root = args.registered_root
    ref_date = os.path.basename(reg_root).replace("registered_to_", "")
    timepoints = sorted(args.timepoints)

    # Load CC data for each timepoint
    print("Loading CC data...")
    tp_ccs = {}
    for tp in timepoints:
        tp_ccs[tp] = load_cc_data(reg_root, ref_date, tp)
        print(f"  {tp}: {len(tp_ccs[tp])} CCs")

    # Filter to lineage CCs using bidirectional matching
    print("\nFiltering lineage CCs...")
    lineage_ccs = filter_lineage_ccs(tp_ccs, timepoints, args.max_distance)
    for tp in timepoints:
        labels = [l for l, _, _, _ in lineage_ccs[tp]]
        print(f"  {tp}: {len(labels)} CCs, labels={labels}")

    # Classify edges
    print("\nClassifying edges...")
    all_edges = classify_lineage_edges(lineage_ccs, timepoints, args.max_distance)
    for e in all_edges:
        print(f"  {e[0]}_{e[1]} -> {e[2]}_{e[3]}: {e[5]} ({e[4]:.1f}mm)")

    # Auto-detect nodes CSV if not provided
    nodes_csv = args.nodes_csv
    if nodes_csv is None and args.lineage is not None:
        candidate = os.path.join(reg_root, "lesion_graph", "lesion_nodes.csv")
        if os.path.exists(candidate):
            nodes_csv = candidate

    # Plot
    print(f"\nGenerating figure...")
    plot_split_merge(
        lineage_ccs, all_edges, timepoints,
        nodes_csv=nodes_csv,
        target_lineage=args.lineage,
        tracer_filter=args.tracer,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
