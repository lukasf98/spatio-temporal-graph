# Spatio-Temporal Lesion Graphs for Longitudinal PET/CT Tracking

Code for the paper: *Spatio-Temporal Lesion Graphs for Longitudinal PET/CT Lesion Tracking in Metastatic Prostate Cancer* .

## Method Overview

The pipeline builds a spatio-temporal graph tracking individual lesion identities across longitudinally registered PET/CT scans:

1. **Graph Construction** (`lesion_graph.py`): Connected component analysis on registered PET segmentations extracts lesion nodes (centroid, volume, organ). Nodes are matched to the nearest centroid across all previous timepoints within a distance threshold (default 30 mm) to assign lineage identities.

2. **Edge Type Classification** (`edge_classification.py`): Bidirectional nearest-neighbor matching between consecutive timepoints classifies edges as continuation (1:1), split (1:N), or merge (N:1) based on forward/backward match cardinality.

3. **SUV Extraction** (`suv_extraction.py`): Per-lesion SUV metrics (mean, max, peak, TLG) are extracted from registered SUV images. SUV_peak follows the PERCIST 1 cm³ sphere definition.

4. **Visualization** (`plot_lesion_graph.py`): Forest diagrams, SUV trajectories, and intra-tumoral heterogeneity analysis.

5. **Split/Merge Diagram** (`plot_split_merge.py`): Per-lineage topology visualization showing split and merge events between consecutive timepoints, with nodes colored by SUV_peak and sized by volume.

## Installation

```bash
pip install -r requirements.txt
```

## Expected Input Data

The pipeline expects registered PET/CT data organized as:

```
registered_to_YYYYMMDD/
├── run_log.json                    # Registration log with status per timepoint
├── YYYYMMDD/                       # Reference timepoint
│   ├── PETseg_ref.nii.gz          # Binary PET segmentation
│   └── PET_ref_SUV.nii.gz         # SUV image (optional, for SUV extraction)
├── YYYYMMDD/                       # Follow-up timepoint
│   ├── PETseg_to_YYYYMMDD.nii.gz  # Registered PET segmentation
│   └── SUV_to_YYYYMMDD.nii.gz     # Registered SUV image (optional)
└── ...
```

Each patient also needs a raw data directory with `PET.json` files containing the `Radiopharmaceutical` field for tracer identification.

## Usage

```bash
python run_pipeline.py \
    --patient PATIENT_ID \
    --registered-root /path/to/registered_to_YYYYMMDD \
    --raw-root /path/to/raw_data \
    --classify-edges
```

### Options

| Flag | Description |
|------|-------------|
| `--patient` | Patient ID (required) |
| `--registered-root` | Path to `registered_to_YYYYMMDD` directory (required) |
| `--raw-root` | Raw data root with patient directories (required) |
| `--organ-seg` | Organ segmentation NIfTI (optional) |
| `--organ-json` | Organ label JSON (optional) |
| `--output-dir` | Output directory (default: `<registered-root>/lesion_graph/`) |
| `--max-distance` | Matching threshold in mm (default: 30.0) |
| `--classify-edges` | Run bidirectional edge type classification |
| `--no-suv` | Skip SUV extraction |
| `--no-plots` | Skip plot generation |

## Output

| File | Description |
|------|-------------|
| `lesion_nodes.csv` | Node attributes: centroid, volume, organ, tracer, SUV metrics, lineage ID |
| `lesion_edges.csv` | Lineage-based edges (continuation) |
| `edge_classification.csv` | Bidirectional edge classification (continuation/split/merge) |
| `forest_diagram.png` | Lesion forest visualization |
| `suv_trajectories.png` | Per-lineage SUV time courses |
| `heterogeneity_analysis.png` | Intra-tumoral heterogeneity over time |

### Split/Merge Diagram

```bash
python plot_split_merge.py \
    --registered-root /path/to/registered_to_YYYYMMDD \
    --timepoints 20220401 20221110 20230509 20230901 20240117 20240410 20250122 \
    --lineage 1 --tracer PSMA \
    --output split_merge.png
```
