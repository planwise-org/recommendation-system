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

# Add the streamlit directory to the path
streamlit_dir = str(Path(__file__).resolve().parent.parent.joinpath('streamlit'))
sys.path.insert(0, streamlit_dir)

# Make pathway.py findable by tests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.joinpath('streamlit'))) 