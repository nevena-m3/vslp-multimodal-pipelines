"""Alignment APIs are loaded only when needed.

Feature-facing names (AlignmentStore, load_final_alignment) come from the
MFA-free ``contract`` module. Producer names (running or freezing an aligner)
come from ``stage`` and are used only by the separate Alignment app.
"""

_CONTRACT = {"AlignmentStore", "load_final_alignment"}
_PRODUCER = {"AlignmentConfig", "MFAProvider", "PromptManifest", "freeze_alignment",
             "list_alignment_runs", "run_acoustic_alignment"}
__all__ = sorted(_CONTRACT | _PRODUCER)


def __getattr__(name: str):
    if name in _CONTRACT:
        from . import contract

        return getattr(contract, name)
    if name in _PRODUCER:
        from . import stage

        return getattr(stage, name)
    raise AttributeError(name)
