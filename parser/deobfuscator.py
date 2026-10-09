"""
deobfuscator.py — standalone deobfuscation helpers extracted from GenosEngine.
Re-exports from the genos/deobfuscator.py module so parser/ scripts can run
standalone (with only parser/ on sys.path) without duplicating the logic.
"""

import sys
import os
import importlib.util

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_root_deob_path = os.path.join(_ROOT_DIR, "genos", "deobfuscator.py")

_spec = importlib.util.spec_from_file_location("_root_deobfuscator", _root_deob_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

for _attr in dir(_mod):
    if not _attr.startswith("__"):
        globals()[_attr] = getattr(_mod, _attr)

MAX_LAYERS = getattr(_mod, "DEFAULT_MAX_LAYERS", 5)
