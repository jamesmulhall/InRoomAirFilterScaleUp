"""Pytest configuration for the InRoomAirFilterScaleUp test suite.

A ``--full-data`` CLI flag toggles between the committed mini fixtures in
``tests/fixtures/`` (default; fast) and the real ``data/`` directory
(slow; runs the global / per-country sense checks against the real ILO
figures).
"""

from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_DATA = REPO_ROOT / "data"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def pytest_addoption(parser):
    parser.addoption(
        "--full-data",
        action="store_true",
        default=False,
        help=(
            "Run tests against the real data/ folder rather than the mini "
            "fixtures in tests/fixtures/. Enables global magnitude and "
            "strict per-country ILO sense checks."
        ),
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "full_data: test requires the real data/ folder (only runs with --full-data).",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--full-data"):
        return
    skip_full = pytest.mark.skip(reason="needs --full-data")
    for item in items:
        if "full_data" in item.keywords:
            item.add_marker(skip_full)


@pytest.fixture(scope="session")
def use_full_data(request) -> bool:
    return bool(request.config.getoption("--full-data"))


@pytest.fixture(scope="session")
def data_dir(use_full_data) -> Path:
    return REAL_DATA if use_full_data else FIXTURES


@pytest.fixture(scope="session")
def results_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("results")


@pytest.fixture(scope="session")
def ew_outputs(data_dir, results_dir):
    """Run the essential-worker pipeline once and share the result."""
    from essential_workers import run_pipeline

    return run_pipeline(data_dir=data_dir, results_dir=results_dir, write=True)


@pytest.fixture(scope="session")
def countries_outputs(data_dir, results_dir, ew_outputs):  # noqa: ARG001
    """Run the countries pipeline once (depends on EW outputs being on disk)."""
    from countries import run_pipeline

    return run_pipeline(data_dir=data_dir, results_dir=results_dir, write=False)
