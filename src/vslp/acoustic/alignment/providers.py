"""Provider boundary for the unchanged feature-facing AlignmentStore."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from .stage import AlignmentStore


class FinalAlignmentProvider(Protocol):
    def load(self, root: Path, *, require_current_trials: bool,
             require_feature_contract: bool) -> tuple[AlignmentStore | None, str]: ...


class EmbeddedAlignmentProvider:
    def load(self, root: Path, *, require_current_trials: bool,
             require_feature_contract: bool) -> tuple[AlignmentStore | None, str]:
        from .stage import _load_embedded_final_alignment

        return _load_embedded_final_alignment(
            root, require_current_trials=require_current_trials,
            require_feature_contract=require_feature_contract)


class ImportedPackageAlignmentProvider:
    def load(self, root: Path, *, require_current_trials: bool,
             require_feature_contract: bool) -> tuple[AlignmentStore | None, str]:
        from .package_v1 import load_imported_alignment

        return load_imported_alignment(root)


def selected_alignment_provider(root: Path) -> FinalAlignmentProvider:
    # Only an explicit user-selected import changes precedence.
    if (root / "alignment_imports" / "active.json").is_file():
        return ImportedPackageAlignmentProvider()
    return EmbeddedAlignmentProvider()
