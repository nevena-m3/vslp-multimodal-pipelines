"""Feature-facing, MFA-free Alignment contract.

This module only reads frozen or imported alignment products. It never runs,
imports or configures an aligner, so the Acoustic GUI can depend on it without
any Montreal Forced Aligner code being loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd

from .audio_identity import audio_content_sha256, legacy_peak_timestamp_only_match


PHONE_SET = "ARPABET_CMU_39"
ARPABET_VOWELS = frozenset({
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY",
    "OW", "OY", "UH", "UW",
})
ARPABET_CONSONANTS = frozenset({
    "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N",
    "NG", "P", "R", "S", "SH", "T", "TH", "V", "W", "Y", "Z", "ZH",
})
WORD_COLUMNS = (
    "alignment_run_id", "recording_id", "file_name", "task_id", "prompt_version",
    "word_index", "word", "start_sec", "end_sec", "duration_sec", "start_sample",
    "end_sample", "sequence_id", "alignment_status", "alignment_source",
)
PHONE_COLUMNS = (
    "alignment_run_id", "recording_id", "file_name", "task_id", "prompt_version",
    "word_index", "phone_index", "phone", "phone_raw", "phone_normalized",
    "stress", "is_vowel", "start_sec", "end_sec", "duration_sec", "start_sample",
    "end_sample", "sequence_id", "alignment_status", "alignment_source",
)
DIAGNOSTIC_COLUMNS = (
    "alignment_run_id", "recording_id", "file_name", "status", "reason",
    "expected_word_count", "aligned_expected_word_count", "word_coverage",
    "missing_words", "extra_tokens", "n_words", "n_phones", "invalid_intervals",
    "overlapping_intervals", "nonmonotonic_intervals", "outside_domain_tokens",
    "canonical_sample_rate_hz", "working_sample_rate_hz",
)


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_transcript(text: str) -> str:
    """vslp_prompt_nfkc_v1: uppercase, preserve apostrophes, split hyphens.

    Digits and abbreviations require a manually reviewed spoken form; no
    implicit number expansion or ASR transcription is performed.
    """
    value = unicodedata.normalize("NFKC", text).replace("’", "'").replace("-", " ")
    value = re.sub(r"[^\w'\s]", " ", value, flags=re.UNICODE)
    value = " ".join(value.upper().split())
    if any(char.isdigit() for char in value):
        raise ValueError("TRANSCRIPT_NUMBER_REQUIRES_REVIEW")
    return value


def normalize_phone(label: str, phone_set: str = PHONE_SET) -> tuple[str, str, bool]:
    if phone_set != PHONE_SET:
        raise ValueError("unsupported_phone_set")
    raw = str(label).strip().upper()
    match = re.fullmatch(r"([A-Z]+)([012])?", raw)
    if not match:
        raise ValueError("invalid_phone_label")
    normalized, stress = match.group(1), match.group(2) or ""
    if normalized not in ARPABET_VOWELS | ARPABET_CONSONANTS:
        raise ValueError("PHONESET_MAPPING_REQUIRED")
    return normalized, stress, normalized in ARPABET_VOWELS


def _paths(root: Path) -> tuple[Path, Path]:
    base = root / "acoustic" / "004_alignment"
    return base, base / "final"


@dataclass(frozen=True)
class AlignmentStore:
    words: pd.DataFrame
    phones: pd.DataFrame
    diagnostics: pd.DataFrame
    manifest: dict
    trials: pd.DataFrame | None = None
    canonical_words: pd.DataFrame | None = None
    canonical_phones: pd.DataFrame | None = None

    def words_for_trial(self, recording_id: str, trial_id: str) -> pd.DataFrame:
        if self.canonical_words is None:
            raise ValueError("final_alignment_contract_required")
        return self.canonical_words.loc[
            self.canonical_words.recording_id.astype(str).eq(str(recording_id))
            & self.canonical_words.trial_id.astype(str).eq(str(trial_id))].copy()

    def phones_for_trial(self, recording_id: str, trial_id: str) -> pd.DataFrame:
        if self.canonical_phones is None:
            raise ValueError("final_alignment_contract_required")
        return self.canonical_phones.loc[
            self.canonical_phones.recording_id.astype(str).eq(str(recording_id))
            & self.canonical_phones.trial_id.astype(str).eq(str(trial_id))].copy()

    def vowels_for_trial(self, recording_id: str, trial_id: str) -> pd.DataFrame:
        phones = self.phones_for_trial(recording_id, trial_id)
        return phones.loc[phones.is_vowel.astype(str).str.lower().isin(["true", "1"])].copy()

    def phones_for_word(self, word_id: str) -> pd.DataFrame:
        if self.canonical_phones is None:
            raise ValueError("final_alignment_contract_required")
        return self.canonical_phones.loc[
            self.canonical_phones.parent_word_id.astype(str).eq(str(word_id))].copy()

    def canonical_word(self, recording_id: str, trial_id: str,
                       canonical_word_index: int) -> pd.Series:
        matches = self.words_for_trial(recording_id, trial_id)
        matches = matches.loc[matches.canonical_word_index.astype(int).eq(
            int(canonical_word_index))]
        if len(matches) != 1:
            raise ValueError("canonical_word_not_unique")
        return matches.iloc[0].copy()

    def target_phone(self, recording_id: str, trial_id: str,
                     phone_index_in_trial: int) -> pd.Series:
        matches = self.phones_for_trial(recording_id, trial_id)
        matches = matches.loc[matches.phone_index_in_trial.astype(int).eq(
            int(phone_index_in_trial))]
        if len(matches) != 1:
            raise ValueError("target_phone_not_unique")
        return matches.iloc[0].copy()

    def get_word_tokens(self, recording_id: str) -> pd.DataFrame:
        source = self.canonical_words if self.canonical_words is not None else self.words
        return source.loc[source.recording_id.astype(str).eq(str(recording_id))].copy()

    def get_phone_tokens(self, recording_id: str) -> pd.DataFrame:
        source = self.canonical_phones if self.canonical_phones is not None else self.phones
        return source.loc[source.recording_id.astype(str).eq(str(recording_id))].copy()

    def get_vowel_tokens(self, recording_id: str) -> pd.DataFrame:
        phones = self.get_phone_tokens(recording_id)
        return phones.loc[phones.is_vowel.astype(str).str.lower().isin(["true", "1"])].copy()

    def get_tokens_by_label(self, recording_id: str, label: str) -> pd.DataFrame:
        phones = self.get_phone_tokens(recording_id)
        return phones.loc[phones.phone_normalized.astype(str).eq(label.upper())].copy()

    def get_trial_tokens(self, recording_id: str, trial_id: str,
                         token_type: str = "word") -> pd.DataFrame:
        """Get final feature-facing tokens, never raw provider tokens."""
        if self.canonical_words is not None:
            return (self.words_for_trial(recording_id, trial_id) if token_type == "word"
                    else self.phones_for_trial(recording_id, trial_id))
        path = self.manifest.get("final_trial_tokens_path", "")
        if not path:
            return pd.DataFrame()
        mapping = pd.read_csv(path, keep_default_na=False)
        indices = mapping.loc[mapping.recording_id.astype(str).eq(str(recording_id))
                              & mapping.trial_id.astype(str).eq(str(trial_id))
                              & mapping.token_type.eq(token_type), "token_index"]
        source = self.words if token_type == "word" else self.phones
        index_name = "word_index" if token_type == "word" else "phone_index"
        return source.loc[source.recording_id.astype(str).eq(str(recording_id))
                          & source[index_name].isin(indices)].copy()

    def family07_rows(self) -> pd.DataFrame:
        if self.canonical_words is None or self.canonical_phones is None:
            raise ValueError("final_alignment_contract_required")
        words = self.canonical_words.rename(columns={"word": "label"}).copy()
        words["token_type"] = "word"
        vowels = self.canonical_phones.loc[
            self.canonical_phones.is_vowel.astype(str).str.lower().isin(["true", "1"])
        ].rename(columns={"phone": "label"}).copy()
        vowels["token_type"] = "vowel"
        combined = pd.concat([words, vowels], ignore_index=True)
        expected = self.canonical_words.groupby("recording_id").size().to_dict()
        combined["expected_count"] = combined.recording_id.map(expected)
        combined["valid"] = True
        return combined


def _load_embedded_final_alignment(output_root: str | Path, *,
                                   require_current_trials: bool = True,
                                   require_feature_contract: bool = True) -> tuple[AlignmentStore | None, str]:
    root = Path(output_root)
    _, final = _paths(root)
    manifest_path = final / "final_alignment_manifest.json"
    if not manifest_path.is_file():
        return None, "missing_alignment"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        from .final_contract import FINAL_ALIGNMENT_CONTRACT_VERSION
        version = manifest.get("final_alignment_contract_version", "")
        if require_feature_contract and version != FINAL_ALIGNMENT_CONTRACT_VERSION:
            return None, "unsupported_final_alignment_contract"
        if manifest.get("alignment_status") != "FROZEN":
            return None, "alignment_not_frozen"
        profile_path = str(manifest.get("mfa_profile_source_path", ""))
        if profile_path and (not Path(profile_path).is_file() or
                             _hash(Path(profile_path)) != manifest.get("mfa_profile_sha256")):
            return None, "stale_alignment_profile"
        if manifest.get("alignment_review_policy_version"):
            from .review_policy import ALIGNMENT_REVIEW_POLICY_VERSION

            if manifest["alignment_review_policy_version"] != ALIGNMENT_REVIEW_POLICY_VERSION:
                return None, "stale_alignment_review_policy"
        review = root / "acoustic" / "003_segmentation_review" / "final"
        if (manifest["source_final_decisions_sha256"] != _hash(review / "final_segmentation_decisions.csv")
                or manifest["source_final_intervals_sha256"] != _hash(review / "final_segmentation_intervals.csv")):
            return None, "alignment_review_source_mismatch"
        source_hashes = manifest.get("source_audio_sha256_by_recording", {})
        content_hashes = manifest.get("source_audio_content_sha256_by_recording")
        if source_hashes or content_hashes:
            decisions = pd.read_csv(review / "final_segmentation_decisions.csv",
                                    keep_default_na=False).set_index("recording_id")
            if content_hashes is not None and source_hashes and set(content_hashes) != set(source_hashes):
                return None, "alignment_source_audio_changed"
            identities = content_hashes if content_hashes is not None else source_hashes
            for identity in identities:
                if identity not in decisions.index:
                    return None, "alignment_source_audio_changed"
                audio = Path(str(decisions.loc[identity].analysis_wav_path))
                if content_hashes is not None:
                    expected_content = content_hashes.get(identity)
                    if (not expected_content or
                            audio_content_sha256(audio) != expected_content):
                        return None, "alignment_source_audio_changed"
                elif _hash(audio) != source_hashes[identity]:
                    project_manifest = json.loads((root / "project_manifest.json").read_text(
                        encoding="utf-8")) if (root / "project_manifest.json").is_file() else {}
                    run_manifest_path = (root / "acoustic" / "004_alignment" / "runs" /
                                         manifest["alignment_run_id"] / "logs" /
                                         "stage_manifest.json")
                    run_manifest = json.loads(run_manifest_path.read_text(
                        encoding="utf-8")) if run_manifest_path.is_file() else {}
                    if not legacy_peak_timestamp_only_match(
                            audio, source_hashes[identity], project_manifest.get("created_at_utc", ""),
                            run_manifest.get("created_utc", "")):
                        return None, "alignment_source_audio_changed"
        choices_path = root / "configs" / "alignment_choices.json"
        if (require_current_trials and manifest.get("trial_contract_version") == "reviewed_trials_v1"
                and choices_path.is_file()):
            choices = json.loads(choices_path.read_text(encoding="utf-8"))
            if choices.get("pending_trial_revision"):
                return None, "stale_alignment_trial_revision"
            plan_path = (root / "acoustic" / "004_alignment" / "runs" /
                         manifest["alignment_run_id"] / "configs" / "alignment_trials.json")
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            frozen_trials = {(str(item["recording_id"]), str(item["trial_id"])): item
                             for item in plan["trials"]}
            current_trials = {(str(identity), str(item["trial_id"])): item
                              for identity, items in choices.get("trials", {}).items()
                              for item in items}
            if current_trials and (set(current_trials) != set(frozen_trials) or any(
                    current_trials[key].get("prompt_id", "") != frozen_trials[key].get("prompt_id", "")
                    or abs(float(current_trials[key]["start_sec"]) -
                           float(frozen_trials[key]["start_sec"])) > 1e-9
                    or abs(float(current_trials[key]["end_sec"]) -
                           float(frozen_trials[key]["end_sec"])) > 1e-9
                    for key in current_trials)):
                return None, "stale_alignment_trial_revision"
        words_path = final / "final_alignment_words.csv"
        phones_path = final / "final_alignment_phones.csv"
        diagnostics_path = final / "final_alignment_diagnostics.csv"
        if (manifest["final_words_sha256"] != _hash(words_path)
                or manifest["final_phones_sha256"] != _hash(phones_path)
                or manifest["final_diagnostics_sha256"] != _hash(diagnostics_path)):
            return None, "alignment_hash_mismatch"
        for name, expected in manifest.get("final_trial_artifact_sha256", {}).items():
            if _hash(final / name) != expected:
                return None, "alignment_trial_hash_mismatch"
        for name, expected in manifest.get("final_alignment_contract_artifact_sha256", {}).items():
            if _hash(final / name) != expected:
                return None, "alignment_contract_hash_mismatch"
        for name, expected in manifest.get("final_alignment_evidence_sha256", {}).items():
            if _hash(final / name) != expected:
                return None, "alignment_evidence_hash_mismatch"
        for artifact in manifest.get("final_alignment_qc_artifacts", {}).values():
            if _hash(Path(artifact["path"])) != artifact["sha256"]:
                return None, "alignment_qc_hash_mismatch"
        words = pd.read_csv(words_path, keep_default_na=False)
        phones = pd.read_csv(phones_path, keep_default_na=False)
        diagnostics = pd.read_csv(diagnostics_path, keep_default_na=False)
        if not set(WORD_COLUMNS) <= set(words) or not set(PHONE_COLUMNS) <= set(phones):
            return None, "alignment_schema_missing"
        if version == FINAL_ALIGNMENT_CONTRACT_VERSION:
            trials = pd.read_csv(final / "final_alignment_contract_trials.csv",
                                 keep_default_na=False)
            canonical_words = pd.read_csv(final / "final_alignment_canonical_words.csv",
                                          keep_default_na=False)
            canonical_phones = pd.read_csv(final / "final_alignment_canonical_phones.csv",
                                           keep_default_na=False)
            valid = {"AUTO_ACCEPTED_STRUCTURAL", "HUMAN_ACCEPTED"}
            if (trials.empty or not trials.alignment_review_status.isin(valid).all()
                    or not canonical_words.alignment_review_status.isin(valid).all()
                    or not canonical_phones.alignment_review_status.isin(valid).all()):
                return None, "unresolved_final_alignment"
            if (trials.duplicated(["recording_id", "trial_id"]).any()
                    or canonical_words.duplicated(["recording_id", "trial_id", "word_id"]).any()
                    or canonical_phones.duplicated(["recording_id", "trial_id", "phone_id"]).any()):
                return None, "invalid_final_alignment_identity"
            trial_bounds = trials.set_index(["recording_id", "trial_id"])
            word_bounds = canonical_words.set_index("word_id")
            tolerance = 0.5 / max(1, int(manifest.get("working_sample_rate_hz", 16000)))
            for table in (canonical_words, canonical_phones):
                for token in table.itertuples():
                    key = str(token.recording_id), str(token.trial_id)
                    if key not in trial_bounds.index:
                        return None, "invalid_final_alignment_trial_reference"
                    trial = trial_bounds.loc[key]
                    start, end = float(token.start_sec), float(token.end_sec)
                    if (not np.isfinite([start, end]).all() or not
                            float(trial.start_sec) - tolerance <= start < end <=
                            float(trial.end_sec) + tolerance or
                            str(token.sequence_id) != str(trial.sequence_id)):
                        return None, "invalid_final_alignment_original_time"
            for phone in canonical_phones.itertuples():
                if str(phone.parent_word_id) not in word_bounds.index:
                    return None, "invalid_final_alignment_parent_word"
                word = word_bounds.loc[str(phone.parent_word_id)]
                if (str(word.trial_id) != str(phone.trial_id) or
                        not float(word.start_sec) - tolerance <= float(phone.start_sec)
                        < float(phone.end_sec) <= float(word.end_sec) + tolerance):
                    return None, "invalid_final_alignment_parent_word"
            from .prompts import task_registry

            registered = {str(prompt["prompt_id"]): prompt
                          for task in task_registry()["tasks"]
                          for prompt in task.get("prompts", [])}
            for trial in trials.itertuples():
                prompt = registered.get(str(trial.prompt_id))
                if prompt is not None and (str(prompt["prompt_version"]) != str(
                        trial.prompt_version) or sha256(str(prompt["exact_expected_text"]).encode(
                            "utf-8")).hexdigest() != str(trial.prompt_sha256)):
                    return None, "stale_alignment_prompt"
            return AlignmentStore(words, phones, diagnostics, manifest, trials,
                                  canonical_words, canonical_phones), ""
        return AlignmentStore(words, phones, diagnostics, manifest), ""
    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        return None, "invalid_final_alignment"


def load_final_alignment(output_root: str | Path, *,
                         require_current_trials: bool = True,
                         require_feature_contract: bool = True) -> tuple[AlignmentStore | None, str]:
    """Return the explicit imported provider, or the original embedded provider."""
    from .providers import selected_alignment_provider

    provider = selected_alignment_provider(Path(output_root))
    return provider.load(Path(output_root), require_current_trials=require_current_trials,
                         require_feature_contract=require_feature_contract)
