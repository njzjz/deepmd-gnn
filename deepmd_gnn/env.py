"""Configurations read from environement variables."""
import os

DP_GNN_USE_MAPPING = os.environ.get("DP_GNN_USE_MAPPING", "0") == "1"
