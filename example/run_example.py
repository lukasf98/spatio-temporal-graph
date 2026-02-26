#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal usage example for the spatio-temporal lesion graph pipeline.

Requires registered PET/CT data for one patient.
Adjust paths below to match your data layout.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lesion_graph import build_patient_graph
from suv_extraction import populate_graph_suv_metrics
from edge_classification import bidirectional_matching, classify_edges


def main():
    # --- Configure paths (adjust to your data) ---
    patient_id = "EXAMPLE_PATIENT"
    registered_root = "/path/to/registered_to_YYYYMMDD"
    raw_root = "/path/to/raw_data"
    organ_seg_path = "/path/to/CTseg_cads.nii.gz"
    organ_json_path = "/path/to/CTseg.json"

    ref_date = os.path.basename(registered_root).replace("registered_to_", "")

    # --- Step 1: Build the lesion graph ---
    print("Building patient graph...")
    graph = build_patient_graph(
        patient_id=patient_id,
        registered_root=registered_root,
        raw_root=raw_root,
        organ_seg_path=organ_seg_path,
        organ_json_path=organ_json_path,
        max_distance_mm=30.0,
    )

    print(f"  Nodes: {len(graph.nodes)}")
    print(f"  Edges: {len(graph.edges)}")
    print(f"  Lineages: {len(set(n.lineage_id for n in graph.nodes.values()))}")

    # --- Step 2: Extract SUV metrics ---
    print("\nExtracting SUV metrics...")
    populate_graph_suv_metrics(graph, registered_root, ref_date)

    # --- Step 3: Classify edges (bidirectional matching) ---
    print("\nClassifying edges...")
    splits, merges, total = bidirectional_matching(
        registered_root, ref_date, graph.timepoints, max_dist=30.0)
    print(f"  Bidirectional edges: {total}")
    print(f"  Splits: {splits}, Merges: {merges}")

    edge_df = classify_edges(
        registered_root, ref_date, graph.timepoints, max_dist=30.0)
    print(f"\nEdge type breakdown:")
    print(edge_df["edge_type"].value_counts().to_string())

    # --- Step 4: Export to CSV ---
    nodes_df = graph.to_dataframe()
    edges_df = graph.to_edge_dataframe()
    nodes_df.to_csv("lesion_nodes.csv", index=False)
    edges_df.to_csv("lesion_edges.csv", index=False)
    edge_df.to_csv("edge_classification.csv", index=False)
    print("\nSaved: lesion_nodes.csv, lesion_edges.csv, edge_classification.csv")


if __name__ == "__main__":
    main()
