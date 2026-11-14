"""Pytest configuration and global environment fixtures."""

import os
from pathlib import Path

# Configure environment for offline deterministic test execution
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

mpl_dir = Path(__file__).resolve().parent.parent / ".mpl_cache"
mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_dir)
