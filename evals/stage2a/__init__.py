"""Stage 2A case/worker/transport preflight contracts."""

__all__ = [
    "CONTRACT_VERSION",
    "PROOF_VERSION",
    "ContractError",
    "freeze_case",
    "preflight_case",
    "require_preflight",
]


def __getattr__(name):
    # Keep ``python -m evals.stage2a.contracts`` free of runpy's duplicate
    # module warning while preserving the convenient package-level API.
    if name in __all__:
        from . import contracts

        return getattr(contracts, name)
    raise AttributeError(name)
