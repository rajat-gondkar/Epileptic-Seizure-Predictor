#!/usr/bin/env python3
"""
Step 1: Download CHB-MIT EDF files + Preprocess
=================================================
Downloads only seizure + limited interictal files from PhysioNet,
generates genetic profiles, and runs full EEG preprocessing.

This reuses the project's existing scripts.

Usage:
    python 01_download_and_preprocess.py
    python 01_download_and_preprocess.py --patients chb01 chb03  # specific patients
"""

import argparse
import sys
from pathlib import Path

# Resolve project root (cloud_training/ is inside the project)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.selective_download_and_preprocess import main as download_main


if __name__ == "__main__":
    # Forward all arguments to the existing download script
    download_main()
