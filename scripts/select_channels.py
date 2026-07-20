#!/usr/bin/env python3
"""Select an OpenBCI-like 16-electrode montage from a named EEG array."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TARGET_CHANNELS = (
    "Fp1", "Fp2", "F7", "F3", "F4", "F8", "T7", "C3",
    "C4", "T8", "P7", "P3", "P4", "P8", "O1", "O2",
)
ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


def normalize(name: str) -> str:
    compact = "".join(name.split())
    return ALIASES.get(compact.upper(), compact).upper()


def channel_indices(channel_names: list[str]) -> list[int]:
    lookup = {normalize(name): index for index, name in enumerate(channel_names)}
    missing = [name for name in TARGET_CHANNELS if normalize(name) not in lookup]
    if missing:
        raise ValueError(f"Missing required channels: {', '.join(missing)}")
    return [lookup[normalize(name)] for name in TARGET_CHANNELS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="NumPy array [N, C, T]")
    parser.add_argument("--channel-names", type=Path, required=True, help="JSON list of C names")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    eeg = np.load(args.input, mmap_mode="r")
    names = json.loads(args.channel_names.read_text(encoding="utf-8"))
    if eeg.ndim != 3 or eeg.shape[1] != len(names):
        raise ValueError("Expected input [N, C, T] and one name for each channel.")
    indices = channel_indices(names)
    np.save(args.output, np.asarray(eeg[:, indices, :]))
    print(json.dumps({"channels": TARGET_CHANNELS, "indices": indices}, indent=2))


if __name__ == "__main__":
    main()
