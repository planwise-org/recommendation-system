"""
Configuration file for pytest.
Sets up the Python path for all tests to access modules correctly.
"""
import os
import sys
from pathlib import Path

# Add the parent directory (reco) to the Python path
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, parent_dir)

# Add the planwise directory to the path
planwise_dir = str(Path(__file__).resolve().parent.parent.joinpath('planwise'))
sys.path.insert(0, planwise_dir) 