#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizations for the spatio-temporal lesion graph.

A. Forest diagram — lineage lanes over calendar time
B. Metric trajectories — SUVmean, SUVpeak, SUVstd, Volume, Area
C. Heterogeneity analysis — CV, deviation, waterfall
D. Treatment event loading from ProstateTreatments.xlsx
"""

import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

from lesion_graph import PatientGraph, EdgeType, LesionNode
from suv_extraction import compute_patient_level_metrics

# Color palette reused from create_mip_qc.py
INSTANCE_COLORS = [
    (1.0, 0.0, 0.0),    # red
    (0.0, 1.0, 0.0),    # green
    (0.0, 0.5, 1.0),    # blue
    (1.0, 1.0, 0.0),    # yellow
    (1.0, 0.0, 1.0),    # magenta
    (0.0, 1.0, 1.0),    # cyan
    (1.0, 0.5, 0.0),    # orange
    (0.5, 1.0, 0.0),    # lime
    (0.5, 0.0, 1.0),    # purple
    (1.0, 0.5, 0.5),    # salmon
    (0.0, 1.0, 0.5),    # spring green
    (1.0, 1.0, 0.5),    # light yellow
]
FALLBACK_COLOR = (0.5, 0.5, 0.5)


def _tp_to_date(tp: str) -> datetime:
    """Convert YYYYMMDD string to datetime."""
    return datetime.strptime(tp, "%Y%m%d")


def _lineage_color(lineage_id: int) -> tuple:
    if lineage_id is None:
        return FALLBACK_COLOR
    return INSTANCE_COLORS[(lineage_id - 1) % len(INSTANCE_COLORS)]


def _collect_lineage_data(graph):
    """Return dict {lineage_id: [nodes sorted by timepoint]}."""
    lineages = sorted(set(
        n.lineage_id for n in graph.nodes.values()
        if n.lineage_id is not None))
    lineage_data = {}
    for lid in lineages:
        nodes = sorted(
            [n for n in graph.nodes.values() if n.lineage_id == lid],
            key=lambda n: n.timepoint)
        if nodes:
            lineage_data[lid] = nodes
    return lineage_data


def _collect_aggregates(graph):
    """Return dict {tp: aggregate_metrics}."""
    aggregate = {}
    for tp in graph.timepoints:
        nodes = graph.get_nodes_at_tp(tp)
        if nodes:
            aggregate[tp] = compute_patient_level_metrics(nodes)
    return aggregate


# ---------------------------------------------------------------------------
# D. Treatment event loading
# ---------------------------------------------------------------------------

def load_treatment_events(excel_path: str, patient_id: str) -> List[Dict]:
    """Parse treatment events from ProstateTreatments.xlsx Sheet1."""
    if not os.path.exists(excel_path):
        return []

    try:
        df = pd.read_excel(excel_path, sheet_name='Sheet1')
    except Exception as e:
        print(f"  Warning: Could not read Excel: {e}")
        return []

    row = df[df['ID'] == patient_id]
    if row.empty:
        return []
    row = row.iloc[0]

    events = []

    for i in range(1, 9):
        col = f'Lu-177-PSMA {i}'
        if col in row.index and pd.notna(row[col]):
            events.append(dict(
                date=pd.to_datetime(row[col]),
                label=f'Lu-177 #{i}', color='green'))

    for i in range(1, 5):
        col = f'EBRT {i}'
        if col in row.index and pd.notna(row[col]):
            events.append(dict(
                date=pd.to_datetime(row[col]),
                label=f'EBRT #{i}', color='orange'))

    col = 'ARPI start'
    if col in row.index and pd.notna(row[col]):
        events.append(dict(
            date=pd.to_datetime(row[col]),
            label='ARPI start', color='royalblue'))

    for col, lbl in [('CTx start', 'CTx start'), ('CTx end', 'CTx end')]:
        if col in row.index and pd.notna(row[col]):
            events.append(dict(
                date=pd.to_datetime(row[col]),
                label=lbl, color='hotpink'))

    col = 'Ra-223-RaCl'
    if col in row.index and pd.notna(row[col]):
        events.append(dict(
            date=pd.to_datetime(row[col]),
            label='Ra-223', color='darkviolet'))

    for i in range(1, 3):
        col = f'SIRT {i}'
        if col in row.index and pd.notna(row[col]):
            events.append(dict(
                date=pd.to_datetime(row[col]),
                label=f'SIRT #{i}', color='brown'))

    col = 'death'
    if col in row.index and pd.notna(row[col]):
        events.append(dict(
            date=pd.to_datetime(row[col]),
            label='Death', color='black'))

    return sorted(events, key=lambda e: e['date'])


def _draw_treatment_lines(ax, events, ymin=None, ymax=None):
    """Draw vertical treatment event lines on an axes."""
    for ev in events:
        ax.axvline(ev['date'], color=ev['color'], linestyle='--',
                   alpha=0.6, linewidth=1.0, zorder=0)
        y_pos = ymax if ymax is not None else ax.get_ylim()[1]
        ax.text(ev['date'], y_pos, ev['label'],
                rotation=90, va='top', ha='right',
                fontsize=6, color=ev['color'], alpha=0.7)


# ---------------------------------------------------------------------------
# A. Forest diagram
# ---------------------------------------------------------------------------

def plot_lesion_forest(graph: PatientGraph,
                       treatment_events: List[Dict],
                       output_path: str):
    """X=calendar dates, Y=lineage lanes. Nodes sized by volume, colored by suv_mean."""

    if not graph.nodes:
        return

    lineages = sorted(set(
        n.lineage_id for n in graph.nodes.values()
        if n.lineage_id is not None))
    lineage_lane = {lid: i for i, lid in enumerate(lineages)}

    fig, ax = plt.subplots(figsize=(14, max(4, len(lineages) * 0.8 + 2)))

    suv_vals = [n.suv_mean for n in graph.nodes.values() if n.suv_mean > 0]
    vmin = min(suv_vals) if suv_vals else 0
    vmax = max(suv_vals) if suv_vals else 1
    cmap = plt.cm.hot_r

    # Draw edges
    for edge in graph.edges:
        if edge.source_id not in graph.nodes or edge.target_id not in graph.nodes:
            continue
        src = graph.nodes[edge.source_id]
        tgt = graph.nodes[edge.target_id]
        x_src = _tp_to_date(src.timepoint)
        x_tgt = _tp_to_date(tgt.timepoint)
        y_src = lineage_lane.get(src.lineage_id, 0)
        y_tgt = lineage_lane.get(tgt.lineage_id, 0)

        ax.plot([x_src, x_tgt], [y_src, y_tgt], '-', color='gray',
                linewidth=1.5, alpha=0.6, zorder=1)

    # Draw nodes
    for node in graph.nodes.values():
        x = _tp_to_date(node.timepoint)
        y = lineage_lane.get(node.lineage_id, 0)
        vol_ml = node.volume / 1000.0
        size = np.clip(vol_ml * 50, 20, 300)
        if vmax > vmin:
            norm_val = (node.suv_mean - vmin) / (vmax - vmin)
        else:
            norm_val = 0.5
        color = cmap(norm_val)
        ax.scatter(x, y, s=size, c=[color], edgecolors='black',
                   linewidths=0.5, zorder=3)

    _draw_treatment_lines(ax, treatment_events, ymax=len(lineages) - 0.5)

    ax.set_yticks(range(len(lineages)))
    ax.set_yticklabels([f'Lesion {lid}' for lid in lineages], fontsize=8)
    ax.set_ylim(-0.5, len(lineages) - 0.5)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    ax.set_xlabel('Date')
    ax.set_ylabel('Tracked lesion')
    ax.set_title(f'Patient {graph.patient_id} — Lesion Forest Diagram')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, aspect=30)
    cbar.set_label('SUV mean')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved forest diagram: {output_path}")


# ---------------------------------------------------------------------------
# B. Metric trajectories (SUVmean, SUVpeak, SUVstd, Volume, Area)
# ---------------------------------------------------------------------------

def plot_suv_trajectories(graph: PatientGraph,
                          treatment_events: List[Dict],
                          output_path: str):
    """Per-lesion colored lines + thick black patient-aggregate.
    5 subplots: SUV mean, SUV peak, SUV std, Volume (mL), Area (mm²).
    6th panel holds the legend."""

    if not graph.nodes:
        return

    lineage_data = _collect_lineage_data(graph)
    aggregate = _collect_aggregates(graph)

    # (node_attribute, y-label, aggregate_key)
    metrics = [
        ('suv_mean',  'SUV Mean',      'suv_mean_wt'),
        ('suv_peak',  'SUV Peak',      'suv_peak_max'),
        ('suv_std',   'SUV Std',       'suv_std_wt'),
        ('volume',    'Volume (mL)',   'tmtv'),
        ('area',      'Area (mm\u00b2)', 'area_total'),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    axes_flat = axes.flatten()

    for idx, (attr, ylabel, agg_key) in enumerate(metrics):
        ax = axes_flat[idx]

        # Per-lineage lines
        for lid, nodes in lineage_data.items():
            dates = [_tp_to_date(n.timepoint) for n in nodes]
            if attr == 'volume':
                values = [n.volume / 1000.0 for n in nodes]
            elif attr == 'area':
                values = [n.area for n in nodes]
            else:
                values = [getattr(n, attr) for n in nodes]
            color = _lineage_color(lid)
            ax.plot(dates, values, '-o', color=color, linewidth=1.0,
                    markersize=3, alpha=0.6)

        # Patient aggregate (thick black)
        if aggregate:
            agg_dates = [_tp_to_date(tp) for tp in sorted(aggregate)]
            agg_vals = [aggregate[tp].get(agg_key, 0)
                        for tp in sorted(aggregate)]
            ax.plot(agg_dates, agg_vals, '-s', color='black', linewidth=2.5,
                    markersize=6, zorder=10)

        _draw_treatment_lines(ax, treatment_events)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 6th panel: legend
    ax_legend = axes_flat[5]
    ax_legend.axis('off')
    handles = []
    labels_list = []
    for lid in sorted(lineage_data.keys())[:20]:
        handles.append(Line2D([0], [0], color=_lineage_color(lid),
                              linewidth=1, marker='o', markersize=3))
        labels_list.append(f'Lesion {lid}')
    handles.append(Line2D([0], [0], color='black', linewidth=2.5,
                          marker='s', markersize=6))
    labels_list.append('Aggregate')
    ax_legend.legend(handles, labels_list, loc='center', fontsize=7,
                     ncol=max(1, len(handles) // 10 + 1))
    ax_legend.set_title('Legend', fontsize=10)

    fig.suptitle(f'Patient {graph.patient_id} — Lesion Metrics Over Time',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved metric trajectories: {output_path}")


# ---------------------------------------------------------------------------
# C. Heterogeneity analysis
# ---------------------------------------------------------------------------

def plot_heterogeneity_analysis(graph: PatientGraph,
                                treatment_events: List[Dict],
                                output_path: str):
    """4 panels:
    (A) SUV mean trajectories + aggregate
    (B) CV across lesions per timepoint
    (C) Per-lesion deviation from aggregate
    (D) Waterfall % change baseline->latest
    """
    if not graph.nodes:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    lineage_data = _collect_lineage_data(graph)
    aggregate = _collect_aggregates(graph)

    # --- Panel A: trajectories + aggregate ---
    ax = axes[0, 0]
    for lid, nodes in lineage_data.items():
        dates = [_tp_to_date(n.timepoint) for n in nodes]
        vals = [n.suv_mean for n in nodes]
        ax.plot(dates, vals, '-o', color=_lineage_color(lid),
                linewidth=1.0, markersize=4, alpha=0.7)

    if aggregate:
        agg_dates = [_tp_to_date(tp) for tp in sorted(aggregate)]
        agg_vals = [aggregate[tp]['suv_mean_wt'] for tp in sorted(aggregate)]
        ax.plot(agg_dates, agg_vals, '-s', color='black', linewidth=2.5,
                markersize=6, label='Aggregate', zorder=10)

    _draw_treatment_lines(ax, treatment_events)
    ax.set_ylabel('SUV mean')
    ax.set_title('(A) SUV mean trajectories')
    ax.grid(True, alpha=0.3)

    # --- Panel B: CV across lesions per timepoint ---
    ax = axes[0, 1]
    cv_dates = []
    cv_vals = []
    for tp in sorted(graph.timepoints):
        nodes = graph.get_nodes_at_tp(tp)
        if len(nodes) > 1:
            vals = [n.suv_mean for n in nodes]
            mean_val = np.mean(vals)
            if mean_val > 0:
                cv = np.std(vals) / mean_val
            else:
                cv = 0
            cv_dates.append(_tp_to_date(tp))
            cv_vals.append(cv)

    if cv_dates:
        ax.plot(cv_dates, cv_vals, '-o', color='darkred', linewidth=1.5,
                markersize=5)
        ax.fill_between(cv_dates, 0, cv_vals, alpha=0.2, color='darkred')

    _draw_treatment_lines(ax, treatment_events)
    ax.set_ylabel('CV (SUV mean)')
    ax.set_title('(B) Inter-lesion heterogeneity (CV)')
    ax.grid(True, alpha=0.3)

    # --- Panel C: Per-lesion deviation from aggregate ---
    ax = axes[1, 0]
    for lid, nodes in lineage_data.items():
        dates = []
        devs = []
        for n in nodes:
            agg = aggregate.get(n.timepoint)
            if agg and agg['suv_mean_wt'] > 0:
                dev = (n.suv_mean - agg['suv_mean_wt']) / agg['suv_mean_wt'] * 100
                dates.append(_tp_to_date(n.timepoint))
                devs.append(dev)
        if dates:
            ax.plot(dates, devs, '-o', color=_lineage_color(lid),
                    linewidth=1.0, markersize=4, alpha=0.7)

    ax.axhline(0, color='black', linewidth=1, linestyle='-', alpha=0.5)
    _draw_treatment_lines(ax, treatment_events)
    ax.set_ylabel('Deviation from aggregate (%)')
    ax.set_title('(C) Per-lesion deviation from aggregate')
    ax.grid(True, alpha=0.3)

    # --- Panel D: Waterfall % change baseline->latest ---
    ax = axes[1, 1]
    waterfall = []
    for lid, nodes in lineage_data.items():
        if len(nodes) >= 2:
            baseline = nodes[0].suv_mean
            latest = nodes[-1].suv_mean
            if baseline > 0:
                pct = (latest - baseline) / baseline * 100
            else:
                pct = 0
            waterfall.append((lid, pct))

    if waterfall:
        waterfall.sort(key=lambda x: x[1])
        lids_sorted = [w[0] for w in waterfall]
        pcts = [w[1] for w in waterfall]
        colors = [_lineage_color(lid) for lid in lids_sorted]
        ax.bar(range(len(pcts)), pcts, color=colors, edgecolor='black',
               linewidth=0.5)
        ax.set_xticks(range(len(pcts)))
        ax.set_xticklabels([f'L{lid}' for lid in lids_sorted], fontsize=8,
                           rotation=45)
        ax.axhline(0, color='black', linewidth=1)

    ax.set_ylabel('% change SUV mean')
    ax.set_title('(D) Waterfall: baseline \u2192 latest')
    ax.grid(True, alpha=0.3, axis='y')

    # Format x-axes on time plots
    for ax_time in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        ax_time.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax_time.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax_time.xaxis.get_majorticklabels(), rotation=45, ha='right')

    fig.suptitle(f'Patient {graph.patient_id} \u2014 Heterogeneity Analysis',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved heterogeneity analysis: {output_path}")
