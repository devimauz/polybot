"""Scalp variant of the model strategy: enter on p_model - ask >= EDGE_IN, exit when
p_model' - bid <= EDGE_OUT (sell at bid). Multiple round trips per window, one position at a time."""
from model import run as _unused  # noqa  (loads module-level ws etc.)  -- careful: model.py main guarded