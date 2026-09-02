"""Canonical repository paths for data inputs and pipeline outputs."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"

ESSENTIAL_WORKERS_DATA = DATA_DIR / "essential_workers"
SCALE_UP_DATA = DATA_DIR / "scale_up"

ESSENTIAL_WORKERS_RESULTS = RESULTS_DIR / "essential_workers"
SCALE_UP_RESULTS = RESULTS_DIR / "scale_up"
PACS_PRIORITIZED_RESULTS = SCALE_UP_RESULTS / "PACs_prioritized"
CR_BOXES_PRIORITIZED_RESULTS = SCALE_UP_RESULTS / "CR_boxes_prioritized"
VISUALIZATIONS_RESULTS = RESULTS_DIR / "visualizations"
