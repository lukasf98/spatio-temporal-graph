#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core data structures and graph builder for spatio-temporal lesion tracking.

Tracks individual lesion identities across registered PET/CT timepoints using
centroid-based matching across all previous timepoints (same method as
track_lesions.py). No same-tracer restriction for spatial matching.
"""

import os
import glob
import json
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import SimpleITK as sitk


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class EdgeType(Enum):
    CONTINUATION = "continuation"
    APPEARANCE = "appearance"
    DISAPPEARANCE = "disappearance"


@dataclass
class LesionNode:
    node_id: str          # "{tp}_{label}"
    patient_id: str
    timepoint: str        # YYYYMMDD
    cc_label: int
    centroid: Tuple[float, float, float]  # physical mm
    volume: float         # mm³
    area: float           # surface area mm²
    organ: str
    tracer: str           # PSMA / FDG
    # SUV metrics — filled later by suv_extraction
    suv_mean: float = 0.0
    suv_max: float = 0.0
    suv_peak: float = 0.0
    suv_std: float = 0.0
    suv_median: float = 0.0
    tlg: float = 0.0
    # Set during centroid matching
    lineage_id: Optional[int] = None


@dataclass
class LesionEdge:
    source_id: str
    target_id: str
    edge_type: EdgeType
    distance_mm: float = 0.0


@dataclass
class PatientGraph:
    patient_id: str
    nodes: Dict[str, LesionNode] = field(default_factory=dict)
    edges: List[LesionEdge] = field(default_factory=list)
    timepoints: List[str] = field(default_factory=list)

    def get_nodes_at_tp(self, tp: str) -> List[LesionNode]:
        return [n for n in self.nodes.values() if n.timepoint == tp]

    def get_parents(self, node_id: str) -> List[LesionEdge]:
        return [e for e in self.edges if e.target_id == node_id]

    def get_children(self, node_id: str) -> List[LesionEdge]:
        return [e for e in self.edges if e.source_id == node_id]

    def to_dataframe(self):
        import pandas as pd
        rows = []
        for n in self.nodes.values():
            rows.append({
                'node_id': n.node_id,
                'patient_id': n.patient_id,
                'timepoint': n.timepoint,
                'cc_label': n.cc_label,
                'centroid_x': n.centroid[0],
                'centroid_y': n.centroid[1],
                'centroid_z': n.centroid[2],
                'volume_mm3': n.volume,
                'area_mm2': n.area,
                'organ': n.organ,
                'tracer': n.tracer,
                'suv_mean': n.suv_mean,
                'suv_max': n.suv_max,
                'suv_peak': n.suv_peak,
                'suv_std': n.suv_std,
                'suv_median': n.suv_median,
                'tlg': n.tlg,
                'lineage_id': n.lineage_id,
            })
        return pd.DataFrame(rows)

    def to_edge_dataframe(self):
        import pandas as pd
        rows = []
        for e in self.edges:
            rows.append({
                'source_id': e.source_id,
                'target_id': e.target_id,
                'edge_type': e.edge_type.value,
                'distance_mm': e.distance_mm,
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _euclidean_distance(c1, c2):
    return float(np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 + (c1[2]-c2[2])**2))


def _load_organ_names(json_path: str) -> Dict[int, str]:
    organ_map = {0: "Background"}
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
            labels = data.get('labels', {})
            for label_str, name in labels.items():
                organ_map[int(label_str)] = name
    return organ_map


def _get_organ_at_centroid(centroid_physical, organ_seg, organ_names):
    try:
        idx = organ_seg.TransformPhysicalPointToIndex(
            [float(centroid_physical[0]),
             float(centroid_physical[1]),
             float(centroid_physical[2])]
        )
        label = organ_seg.GetPixel(idx)
        return organ_names.get(label, f"Unknown_{label}")
    except Exception:
        return "Outside"


def _determine_tracer(raw_root, patient_id, date_str):
    """Read PET.json to determine tracer type."""
    pet_json = os.path.join(raw_root, patient_id, date_str, "PET.json")
    if os.path.exists(pet_json):
        with open(pet_json, 'r') as f:
            data = json.load(f)
        rp = data.get("Radiopharmaceutical", "").lower()
        if "psma" in rp:
            return "PSMA"
        elif "fdg" in rp or "fluorodeoxyglucose" in rp:
            return "FDG"
    return "Unknown"


def _connected_components(seg_img: sitk.Image):
    """Run CC labelling with perimeter computation; returns (cc_img, stats)."""
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_img = cc_filter.Execute(seg_img)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.ComputePerimeterOn()
    stats.Execute(cc_img)
    return cc_img, stats


# ---------------------------------------------------------------------------
# Main graph builder
# ---------------------------------------------------------------------------

def build_patient_graph(patient_id: str,
                        registered_root: str,
                        raw_root: str,
                        organ_seg_path: str,
                        organ_json_path: str,
                        max_distance_mm: float = 30.0) -> PatientGraph:
    """Build a spatio-temporal lesion graph for one patient.

    Uses centroid-based matching across ALL previous timepoints (same
    algorithm as track_lesions.py). No same-tracer restriction — lesions
    at the same location are tracked across PSMA/FDG scans.
    """
    ref_date = os.path.basename(registered_root).replace("registered_to_", "")
    graph = PatientGraph(patient_id=patient_id)

    # Load organ segmentation
    organ_seg = None
    organ_names = {}
    if os.path.exists(organ_seg_path) and os.path.exists(organ_json_path):
        organ_seg = sitk.ReadImage(organ_seg_path)
        organ_names = _load_organ_names(organ_json_path)

    # Discover timepoints from run_log.json
    run_log_path = os.path.join(registered_root, "run_log.json")
    if os.path.exists(run_log_path):
        with open(run_log_path, 'r') as f:
            run_log = json.load(f)
        valid_dates = []
        for entry in run_log:
            date_str = entry['date']
            if entry['status'] in ('ok', 'reference'):
                valid_dates.append(date_str)
            elif entry['status'] == 'error':
                tp_dir = os.path.join(registered_root, date_str)
                if date_str == ref_date:
                    seg_check = os.path.join(tp_dir, "PETseg_ref.nii.gz")
                else:
                    seg_check = os.path.join(tp_dir, f"PETseg_to_{ref_date}.nii.gz")
                if os.path.exists(seg_check):
                    valid_dates.append(date_str)
        valid_dates = sorted(valid_dates)
    else:
        tp_dirs = sorted(glob.glob(os.path.join(registered_root, "????????")))
        valid_dates = [os.path.basename(d) for d in tp_dirs]

    graph.timepoints = valid_dates
    print(f"  Found {len(valid_dates)} valid timepoints: {valid_dates}")

    if not valid_dates:
        return graph

    # Per-timepoint: load PETseg, run CC, create LesionNodes
    tp_nodes = {}

    for date_str in valid_dates:
        tp_dir = os.path.join(registered_root, date_str)
        if date_str == ref_date:
            seg_path = os.path.join(tp_dir, "PETseg_ref.nii.gz")
        else:
            seg_path = os.path.join(tp_dir, f"PETseg_to_{ref_date}.nii.gz")

        if not os.path.exists(seg_path):
            print(f"  [{date_str}] PETseg not found, skipping")
            tp_nodes[date_str] = []
            continue

        seg_img = sitk.ReadImage(seg_path)
        cc_img, stats = _connected_components(seg_img)

        tracer = _determine_tracer(raw_root, patient_id, date_str)
        nodes = []

        for label in stats.GetLabels():
            centroid = stats.GetCentroid(label)
            volume = stats.GetPhysicalSize(label)
            area = stats.GetPerimeter(label)

            organ = "Unknown"
            if organ_seg is not None:
                organ = _get_organ_at_centroid(centroid, organ_seg, organ_names)

            node_id = f"{date_str}_{label}"
            node = LesionNode(
                node_id=node_id,
                patient_id=patient_id,
                timepoint=date_str,
                cc_label=label,
                centroid=tuple(centroid),
                volume=volume,
                area=area,
                organ=organ,
                tracer=tracer,
            )
            graph.nodes[node_id] = node
            nodes.append(node)

        tp_nodes[date_str] = nodes
        print(f"  [{date_str}] {len(nodes)} lesion(s), tracer={tracer}")

    # ------------------------------------------------------------------
    # Centroid-based matching across ALL previous timepoints
    # (same algorithm as track_lesions.py — match_lesions_across_time)
    # ------------------------------------------------------------------
    next_lineage = 1
    all_matched = []  # list of (centroid, lineage_id) accumulated over time

    for tp in valid_dates:
        nodes = tp_nodes.get(tp, [])
        for node in nodes:
            best_lineage = None
            min_dist = float('inf')

            for prev_centroid, prev_lid in all_matched:
                dist = _euclidean_distance(node.centroid, prev_centroid)
                if dist < min_dist:
                    min_dist = dist
                    best_lineage = prev_lid

            if min_dist <= max_distance_mm and best_lineage is not None:
                node.lineage_id = best_lineage
            else:
                node.lineage_id = next_lineage
                next_lineage += 1

            all_matched.append((node.centroid, node.lineage_id))

    # ------------------------------------------------------------------
    # Build edges between consecutive appearances of same lineage
    # ------------------------------------------------------------------
    lineage_nodes_map = defaultdict(list)
    for node in graph.nodes.values():
        lineage_nodes_map[node.lineage_id].append(node)

    for lid, lnodes in lineage_nodes_map.items():
        lnodes_sorted = sorted(lnodes, key=lambda n: n.timepoint)
        for i in range(len(lnodes_sorted) - 1):
            src = lnodes_sorted[i]
            tgt = lnodes_sorted[i + 1]
            dist = _euclidean_distance(src.centroid, tgt.centroid)
            graph.edges.append(LesionEdge(
                source_id=src.node_id,
                target_id=tgt.node_id,
                edge_type=EdgeType.CONTINUATION,
                distance_mm=dist))

    n_lineages = len(lineage_nodes_map)
    n_edges = len(graph.edges)
    print(f"  Graph: {len(graph.nodes)} nodes, {n_edges} edges, "
          f"{n_lineages} lineages")

    return graph
