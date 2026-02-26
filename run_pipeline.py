#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Driver script for the spatio-temporal lesion graph pipeline.

Usage:
    python run_pipeline.py --patient PID \\
        --registered-root /path/to/registered_to_YYYYMMDD \\
        --raw-root /path/to/raw_data \\
        --organ-seg /path/to/CTseg_cads.nii.gz \\
        --organ-json /path/to/CTseg.json

    python run_pipeline.py --patient PID \\
        --registered-root /path/to/registered_to_YYYYMMDD \\
        --raw-root /path/to/raw_data \\
        --classify-edges
"""

import os
import sys
import argparse

from lesion_graph import build_patient_graph
from suv_extraction import populate_graph_suv_metrics
from edge_classification import bidirectional_matching, forward_only_matching, classify_edges
from plot_lesion_graph import (
    plot_lesion_forest,
    plot_suv_trajectories,
    plot_heterogeneity_analysis,
    load_treatment_events,
)


def main():
    parser = argparse.ArgumentParser(
        description="Spatio-temporal lesion graph: build, extract SUV, visualize."
    )
    parser.add_argument("--patient", required=True,
                        help="Patient ID")
    parser.add_argument("--registered-root", required=True,
                        help="Path to registered_to_YYYYMMDD directory")
    parser.add_argument("--raw-root", required=True,
                        help="Raw data root containing patient directories")
    parser.add_argument("--organ-seg", default=None,
                        help="Path to organ segmentation (CTseg_cads.nii.gz)")
    parser.add_argument("--organ-json", default=None,
                        help="Path to organ label JSON (CTseg.json)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: <registered-root>/lesion_graph/)")
    parser.add_argument("--excel", default=None,
                        help="Path to treatment events Excel file (optional)")
    parser.add_argument("--max-distance", type=float, default=30.0,
                        help="Max centroid distance for matching in mm (default: 30.0)")
    parser.add_argument("--classify-edges", action="store_true",
                        help="Run bidirectional matching for edge type classification")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--no-suv", action="store_true",
                        help="Skip SUV extraction")

    args = parser.parse_args()

    patient_id = args.patient
    registered_root = args.registered_root
    raw_root = args.raw_root
    ref_date = os.path.basename(registered_root).replace("registered_to_", "")

    if not os.path.isdir(registered_root):
        print(f"ERROR: Registered root not found: {registered_root}")
        return 1

    # Organ segmentation (optional)
    organ_seg_path = args.organ_seg or ""
    organ_json_path = args.organ_json or ""

    # Output directory
    output_dir = args.output_dir or os.path.join(registered_root, "lesion_graph")
    os.makedirs(output_dir, exist_ok=True)

    print(f"{'='*70}")
    print(f"LESION GRAPH PIPELINE — Patient {patient_id}")
    print(f"{'='*70}")
    print(f"  Raw root:        {raw_root}")
    print(f"  Registered root: {registered_root}")
    print(f"  Reference date:  {ref_date}")
    print(f"  Output:          {output_dir}")
    print(f"  Max distance:    {args.max_distance} mm")
    print()

    # Step 1: Build graph
    print("[1] Building patient graph...")
    graph = build_patient_graph(
        patient_id=patient_id,
        registered_root=registered_root,
        raw_root=raw_root,
        organ_seg_path=organ_seg_path,
        organ_json_path=organ_json_path,
        max_distance_mm=args.max_distance,
    )

    if not graph.nodes:
        print("  No lesions found. Exiting.")
        return 0

    # Step 2: Extract SUV metrics
    if not args.no_suv:
        print("\n[2] Extracting SUV metrics...")
        populate_graph_suv_metrics(graph, registered_root, ref_date)
    else:
        print("\n[2] Skipping SUV extraction (--no-suv)")

    # Step 3: Export CSVs
    print("\n[3] Exporting CSVs...")
    nodes_df = graph.to_dataframe()
    edges_df = graph.to_edge_dataframe()
    nodes_csv = os.path.join(output_dir, "lesion_nodes.csv")
    edges_csv = os.path.join(output_dir, "lesion_edges.csv")
    nodes_df.to_csv(nodes_csv, index=False)
    edges_df.to_csv(edges_csv, index=False)
    print(f"  Saved {len(nodes_df)} nodes -> {nodes_csv}")
    print(f"  Saved {len(edges_df)} edges -> {edges_csv}")

    # Step 4: Edge classification (optional)
    if args.classify_edges:
        print("\n[4] Running bidirectional edge classification...")
        timepoints = graph.timepoints

        splits, merges, total = bidirectional_matching(
            registered_root, ref_date, timepoints, args.max_distance)
        print(f"  Bidirectional: {total} edges, {splits} splits, {merges} merges")

        fwd_splits, fwd_merges = forward_only_matching(
            registered_root, ref_date, timepoints, args.max_distance)
        print(f"  Forward-only:  {fwd_splits} splits, {fwd_merges} merges")

        edge_df = classify_edges(
            registered_root, ref_date, timepoints, args.max_distance)
        edge_class_csv = os.path.join(output_dir, "edge_classification.csv")
        edge_df.to_csv(edge_class_csv, index=False)
        print(f"  Saved {len(edge_df)} classified edges -> {edge_class_csv}")
    else:
        print("\n[4] Skipping edge classification (use --classify-edges to enable)")

    # Step 5: Plots
    if not args.no_plots:
        print("\n[5] Generating plots...")
        treatment_events = []
        if args.excel and os.path.exists(args.excel):
            treatment_events = load_treatment_events(args.excel, patient_id)
            print(f"  Found {len(treatment_events)} treatment event(s)")

        plot_lesion_forest(
            graph, treatment_events,
            os.path.join(output_dir, "forest_diagram.png"))
        plot_suv_trajectories(
            graph, treatment_events,
            os.path.join(output_dir, "suv_trajectories.png"))
        plot_heterogeneity_analysis(
            graph, treatment_events,
            os.path.join(output_dir, "heterogeneity_analysis.png"))
    else:
        print("\n[5] Skipping plots (--no-plots)")

    # Summary
    n_lineages = len(set(n.lineage_id for n in graph.nodes.values()))
    print(f"\n{'='*70}")
    print(f"DONE — Patient {patient_id}")
    print(f"  Nodes:      {len(graph.nodes)}")
    print(f"  Edges:      {len(graph.edges)}")
    print(f"  Lineages:   {n_lineages}")
    print(f"  Timepoints: {len(graph.timepoints)}")
    print(f"  Output:     {output_dir}")
    print(f"{'='*70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
