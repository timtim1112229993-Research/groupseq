"""Test configuration: make the source package importable and build synthetic records."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR / "src"))

from groupseq import tables  # noqa: E402

BASE = pd.Timestamp("2030-01-01 09:00:00")


def make_records(specs: list[tuple]) -> pd.DataFrame:
    """Build a synthetic record table.

    Each spec is (seconds, group, record_type, trigger, action, source, off_topic).
    Rows are numbered in the order given, which stands for file order.
    """
    rows = []
    for n, (seconds, group, record_type, trigger, action, source, off_topic) in enumerate(specs, start=1):
        moment = BASE + pd.Timedelta(seconds=seconds)
        rows.append(
            {
                "record_id": f"Z-R{n:04d}",
                "class": "Z",
                "source_row": n,
                "time_local": moment.strftime(tables.ISO_SECONDS),
                "t": moment,
                "group": group,
                "stage": 1,
                "record_type": record_type,
                "trigger": trigger,
                "action": action,
                "source": source,
                "off_topic": off_topic,
                "help_count": "0" if record_type == "DECISION" else "",
                "items_done": "0" if record_type == "DECISION" else "",
                "items_total": "0" if record_type == "DECISION" else "",
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def pipeline_config() -> dict:
    return tables.read_json(PIPELINE_DIR / "config" / "pipeline.json")
