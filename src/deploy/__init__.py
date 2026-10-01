"""
`deploy` package.

Keep this module import-light.

Some submodules depend on optional third-party packages (and heavyweight scientific
stacks). Importing them here makes *any* `import deploy.*` fail if an optional
dependency is missing. Instead, we expose the historical symbols via lazy imports.
"""

from __future__ import annotations

from typing import Any

_LAZY_EXPORTS = {
    "estimator_test": ("deploy.deploy_cls", "estimator_test"),
    "estimator": ("deploy.apply", "estimator"),
    "load_data": ("deploy.apply", "load_data"),
    "sampler": ("deploy.apply", "sampler"),
    "infer_func": ("deploy.infer_func", "infer_func"),
    "pyscf_func": ("deploy.pyscf_func", "pyscf_func"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(name)
    mod_name, attr = _LAZY_EXPORTS[name]
    import importlib

    mod = importlib.import_module(mod_name)
    v = getattr(mod, attr)
    globals()[name] = v
    return v


__all__ = list(_LAZY_EXPORTS.keys())
