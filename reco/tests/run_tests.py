#!/usr/bin/env python
"""
Test runner for the recommendation system.
This script discovers and runs all tests in the tests directory.
"""

import unittest
import sys
import os
from pathlib import Path

# Add the parent directory to the Python path to be able to import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Add the streamlit directory to the path
streamlit_dir = str(Path(__file__).resolve().parent.parent.joinpath('planwise'))
sys.path.insert(0, streamlit_dir)

def run_tests():
    """Discover and run all tests in the tests directory."""
    # Import conftest to set up path properly
    try:
        import conftest
    except ImportError:
        pass
    
    test_loader = unittest.TestLoader()
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    test_suite = test_loader.discover(tests_dir, pattern='test_*.py')
    
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)  # Return non-zero exit code if tests fail 