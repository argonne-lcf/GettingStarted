"""
vllm_build_all_modelinfo_caches.py

Iterate over vllm.model_executor.models.* modules and pre-generate
vLLM model-info caches using internal registry APIs.

This avoids the model-inspection subprocess codepath.

Safe behavior:
- Skips modules that fail to import
- Skips classes that fail inspection
- Continues on errors
- Uses vLLM's own cache naming + serialization

Usage:
  export VLLM_CACHE_ROOT=$PWD/.vllm_cache
  python vllm_build_all_modelinfo_caches.py

Optional:
  python vllm_build_all_modelinfo_caches.py --verbose
"""

from __future__ import annotations

import argparse
import pkgutil
import importlib
import inspect
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable, Type


def is_candidate_model_class(cls: Type) -> bool:
    """
    Heuristic: vLLM model classes usually
    - are classes
    - defined in this module
    - have 'ForCausalLM', 'Model', 'LM', or 'Vision' in name
    """
    if not inspect.isclass(cls):
        return False
    name = cls.__name__
    return any(
        key in name
        for key in (
            "ForCausalLM",
            "Model",
            "LM",
            "Vision",
            "VL",
        )
    )


def iter_model_classes(module: ModuleType) -> Iterable[Type]:
    for _, obj in vars(module).items():
        if is_candidate_model_class(obj):
            yield obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # Import vLLM registry internals
    try:
        import vllm.model_executor.models.registry as r
        import vllm.model_executor.models as models_pkg
    except Exception as e:
        print(f"ERROR: failed to import vLLM internals: {e!r}", file=sys.stderr)
        return 1

    Lazy = getattr(r, "_LazyRegisteredModel", None)
    ModelInfo = getattr(r, "_ModelInfo", None)
    if Lazy is None or ModelInfo is None:
        print("ERROR: vLLM registry internals not available", file=sys.stderr)
        return 2

    cache_root = os.environ.get("VLLM_CACHE_ROOT")
    if not cache_root:
        print("ERROR: VLLM_CACHE_ROOT must be set", file=sys.stderr)
        return 3

    print(f"Using VLLM_CACHE_ROOT={cache_root}")

    success = 0
    skipped = 0
    failed = 0

    for modinfo in pkgutil.iter_modules(models_pkg.__path__):
        modname = modinfo.name
        fqmod = f"{models_pkg.__name__}.{modname}"

        try:
            module = importlib.import_module(fqmod)
        except Exception as e:
            skipped += 1
            if args.verbose:
                print(f"[SKIP] {fqmod}: import failed ({e.__class__.__name__})")
            continue

        module_file = getattr(module, "__file__", None)
        if not module_file:
            skipped += 1
            continue

        try:
            module_bytes = Path(module_file).read_bytes()
            module_hash = r.safe_hash(
                module_bytes, usedforsecurity=False
            ).hexdigest()
        except Exception:
            skipped += 1
            continue

        for cls in iter_model_classes(module):
            cls_name = cls.__name__

            try:
                lazy = Lazy(module_name=fqmod, class_name=cls_name)
                model_cls = lazy.load_model_cls()
                mi = ModelInfo.from_model_cls(model_cls)
                lazy._save_modelinfo_to_cache(mi, module_hash)

                cache_path = lazy._get_cache_dir() / lazy._get_cache_filename()
                success += 1
                print(f"[OK] {fqmod}:{cls_name}")
                if args.verbose:
                    print(f"     -> {cache_path}")

            except Exception as e:
                failed += 1
                if args.verbose:
                    print(f"[FAIL] {fqmod}:{cls_name} ({e.__class__.__name__})")
                continue

    print("\nSummary:")
    print(f"  success: {success}")
    print(f"  failed : {failed}")
    print(f"  skipped: {skipped}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
