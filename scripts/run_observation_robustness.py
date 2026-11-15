#!/usr/bin/env python
"""Entry point for the local DATA8002 observation-robustness investigation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sgdiff.engine.observation_robustness import main

if __name__ == "__main__":
    main()
