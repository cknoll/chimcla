"""
This module contains general utility functions
"""


import os

pjoin = os.path.join
# assuming that the package is installed with `pip install -e .`
CHIMCLA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CHIMCLA_DATA = pjoin(CHIMCLA_ROOT, "data")
