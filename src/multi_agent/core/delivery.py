"""Validated, transactionally persisted terminal report deliveries.

Each target file replacement is atomic. A process crash between replacements cannot
be made atomic across the filesystem, so an unfinished transaction marker retains
the prior generation and is recovered to that generation before the next delivery
operation in the same run directory.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from multi_agent.core.artifact_paths import RunArtifactPaths
from multi_agent.core.report_document import (
    REQUIRED_SECTION_KEYS,
    ReportDocument,
    render_markdown,
    render_recommendation,
    render_structured_report,
)
from multi_agent.core.review_contracts import FinalDecisionRecord


EXPECTED_DELIVERY_STATE = {
    "passed": "formal_report",
    "evidence_limited": "evidence_limited_report",
    "blocked": "blocked_notice",
}
_ALLOWED_STANCES = {
    "formal_report": {"buy", "hold", "sell"},
    "evidence_limited_report": {"watch"},
    "blocked_notice": {"blocked"},
}
ReplaceFile = Callable[[str | os.PathLike[str], str | os.PathLike[str]], None]


class DeliveryValidationResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    valid: bool
    errors: list[str] = Field(default_factory=list)


class DeliveryPackage:
    """Immutable byte snapshots for every terminal delivery projection.

    Public model/dict accessors deserialize fresh copies. Mutating a returned
    object therefore cannot alter the bytes that were validated or will be written.
    """

    __slots__ = (
        "_decision_json",
        "_document_json",
        "_markdown",
        "_recommendation_json",
        "_structured_report_json",
    )

    def __init__(
        self,
        *,
        decision_json: str,
        document_json: str,
        markdown: str,
        recommendation_json: str,
        structured_report_json: str,
    ) -> None:
        object.__setattr__(self, "_decision_json", decision_json)
        object.__setattr__(self, "_document_json", document_json)
        object.__setattr__(self, "_markdown", markdown)
        object.__setattr__(self, "_recommendation_json", recommendation_json)
        object.__setattr__(self, "_structured_report_json", structured_report_json)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("DeliveryPackage is immutable")

    @classmethod
    def from_document(
        cls, *, decision: FinalDecisionRecord, document: ReportDocument
    ) -> "DeliveryPackage":
        decision_json = _json_text(decision.model_dump(mode="json"))
        document_json = _json_text(document.model_dump(mode="json"))
        decision_snapshot = FinalDecisionRecord.model_validate_json(decision_json)
        document_snapshot = ReportDocument.model_validate_json(document_json)
        return cls(
            decision_json=decision_json,
            document_json=document_json,
            markdown=render_markdown(document_snapshot),
            recommendation_json=_json_text(render_recommendation(document_snapshot)),
            structured_report_json=_json_text(render_structured_report(document_snapshot)),
        )

    @property
    def decision(self) -> FinalDecisionRecord:
        return FinalDecisionRecord.model_validate_json(self._decision_json)

    @property
    def document(self) -> ReportDocument:
        return ReportDocument.model_validate_json(self._document_json)

    @property
    def markdown(self) -> str:
        return self._markdown

    @property
    def recommendation(self) -> dict[str, object]:
        return _json_object(self._recommendation_json)

    @property
    def structured_report(self) -> dict[str, object]:
        return _json_object(self._structured_report_json)

    def payloads(self, paths: RunArtifactPaths) -> tuple[tuple[Path, str], ...]:
        return (
            (paths.report_document_path, self._document_json),
            (paths.final_report_path, self._markdown),
            (paths.structured_recommendation_path, self._recommendation_json),
            (paths.structured_report_path, self._structured_report_json),
            (paths.final_decision_path, self._decision_json),
        )


class DeliveryValidator:
    """Reject delivery drift before any terminal artifact is written."""

    def validate(
        self, *, decision: FinalDecisionRecord, document: ReportDocument
    ) -> DeliveryValidationResult:
        errors: list[str] = []
        expected_mode = EXPECTED_DELIVERY_STATE.get(decision.final_decision)
        if expected_mode is None:
            errors.append("final_decision_not_terminal")
        elif (
            decision.final_delivery_state != expected_mode
            or document.report_mode != expected_mode
        ):
            errors.append("report_mode_mismatch")
        if document.trust_score != decision.trust_score:
            errors.append("trust_score_mismatch")
        if expected_mode and document.stance not in _ALLOWED_STANCES[expected_mode]:
            errors.append("stance_mismatch")

        for key in REQUIRED_SECTION_KEYS:
            section = document.sections.get(key)
            if section is None or not section.content.strip():
                errors.append(f"section_empty:{key}")

        source_ids = {source.source_id for source in document.sources}
        section_claim_ids = {
            claim_id
            for section in document.sections.values()
            for claim_id in section.claim_ids
        }
        for claim in document.claims:
            if claim.critical and not set(claim.source_ids).intersection(source_ids):
                errors.append(f"source_unbound:{claim.claim_id}")
            if claim.critical and claim.claim_id not in section_claim_ids:
                errors.append(f"claim_unplaced:{claim.claim_id}")

        return DeliveryValidationResult(valid=not errors, errors=sorted(set(errors)))

    def validate_package(self, package: DeliveryPackage) -> DeliveryValidationResult:
        decision = package.decision
        document = package.document
        result = self.validate(decision=decision, document=document)
        errors = list(result.errors)
        if package.markdown != render_markdown(document):
            errors.append("markdown_projection_mismatch")
        if package.recommendation != render_recommendation(document):
            errors.append("recommendation_projection_mismatch")
        if package.structured_report != render_structured_report(document):
            errors.append("structured_report_projection_mismatch")

        sections = package.structured_report.get("sections")
        if not isinstance(sections, dict):
            errors.append("sections_missing")
        else:
            for key in REQUIRED_SECTION_KEYS:
                if not str(sections.get(key, "")).strip():
                    errors.append(f"section_empty:{key}")
        if package.recommendation.get("status") != decision.final_decision:
            errors.append("recommendation_decision_mismatch")
        if package.recommendation.get("final_delivery_state") != decision.final_delivery_state:
            errors.append("recommendation_delivery_state_mismatch")
        if package.recommendation.get("stance") != document.stance:
            errors.append("recommendation_stance_mismatch")
        if package.structured_report.get("final_decision") != decision.final_decision:
            errors.append("structured_report_decision_mismatch")
        if package.structured_report.get("final_delivery_state") != decision.final_delivery_state:
            errors.append("structured_report_delivery_state_mismatch")
        if package.structured_report.get("stance") != document.stance:
            errors.append("structured_report_stance_mismatch")
        return DeliveryValidationResult(valid=not errors, errors=sorted(set(errors)))


def write_delivery_package(
    *,
    paths: RunArtifactPaths,
    package: DeliveryPackage,
    replace_file: ReplaceFile = os.replace,
) -> None:
    """Commit the five terminal artifacts or restore their entire prior generation."""
    recover_incomplete_delivery_transactions(paths.run_dir)
    result = DeliveryValidator().validate_package(package)
    if not result.valid:
        raise ValueError(f"delivery package validation failed: {', '.join(result.errors)}")
    _commit_payloads(paths.run_dir, package.payloads(paths), replace_file=replace_file)


def invalidate_delivery_package(paths: RunArtifactPaths) -> None:
    """Remove all terminal truth artifacts as one recoverable invalidation operation."""
    recover_incomplete_delivery_transactions(paths.run_dir)
    payload_paths = tuple(path for path, _ in _terminal_payload_paths(paths))
    _invalidate_paths(paths.run_dir, payload_paths)


def recover_incomplete_delivery_transactions(run_dir: Path) -> None:
    root = _transaction_root(run_dir)
    if not root.exists():
        return
    for transaction_dir in root.iterdir():
        manifest_path = transaction_dir / "manifest.json"
        if not manifest_path.exists():
            shutil.rmtree(transaction_dir, ignore_errors=True)
            continue
        manifest = _json_object(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("state") != "committed":
            _rollback_entries(manifest.get("entries", []))
        _cleanup_transaction(manifest.get("entries", []), transaction_dir)
    root.rmdir() if root.exists() and not any(root.iterdir()) else None


def _commit_payloads(
    run_dir: Path,
    payloads: tuple[tuple[Path, str], ...],
    *,
    replace_file: ReplaceFile,
) -> None:
    transaction_dir, entries, manifest_path = _prepare_transaction(run_dir, payloads, "delivery")
    try:
        # Every payload is staged and fsynced before the current generation moves.
        for entry, (_, content) in zip(entries, payloads, strict=True):
            entry["stage"] = str(_stage_text(Path(entry["target"]), content))
        _write_manifest(manifest_path, {"state": "prepared", "entries": entries})

        # Preserve all originals before replacing any target.
        for entry in entries:
            if entry["original_exists"]:
                replace_file(entry["target"], entry["backup"])

        for entry in entries:
            replace_file(entry["stage"], entry["target"])

        _write_manifest(manifest_path, {"state": "committed", "entries": entries})
    except Exception:
        _rollback_entries(entries)
        raise
    finally:
        _cleanup_transaction(entries, transaction_dir)


def _invalidate_paths(run_dir: Path, paths: tuple[Path, ...]) -> None:
    transaction_dir, entries, manifest_path = _prepare_transaction(
        run_dir, tuple((path, "") for path in paths), "invalidation"
    )
    try:
        _write_manifest(manifest_path, {"state": "prepared", "entries": entries})
        for entry in entries:
            if entry["original_exists"]:
                os.replace(entry["target"], entry["backup"])
        _write_manifest(manifest_path, {"state": "committed", "entries": entries})
    except Exception:
        _rollback_entries(entries)
        raise
    finally:
        _cleanup_transaction(entries, transaction_dir)


def _prepare_transaction(
    run_dir: Path,
    payloads: tuple[tuple[Path, str], ...],
    kind: str,
) -> tuple[Path, list[dict[str, object]], Path]:
    transaction_dir = _transaction_root(run_dir) / f"{kind}-{uuid4().hex}"
    transaction_dir.mkdir(parents=True, exist_ok=False)
    entries: list[dict[str, object]] = []
    for target, _ in payloads:
        target.parent.mkdir(parents=True, exist_ok=True)
        token = uuid4().hex
        entries.append(
            {
                "target": str(target),
                "backup": str(target.parent / f".{target.name}.delivery-backup-{token}"),
                "stage": "",
                "original_exists": target.exists(),
            }
        )
    return transaction_dir, entries, transaction_dir / "manifest.json"


def _rollback_entries(entries: object) -> None:
    if not isinstance(entries, list):
        return
    for raw_entry in reversed(entries):
        if not isinstance(raw_entry, dict):
            continue
        target = Path(str(raw_entry.get("target", "")))
        backup = Path(str(raw_entry.get("backup", "")))
        if backup.exists():
            if target.exists():
                target.unlink()
            os.replace(backup, target)
        elif raw_entry.get("original_exists") is False and target.exists():
            target.unlink()


def _cleanup_transaction(entries: object, transaction_dir: Path) -> None:
    if isinstance(entries, list):
        for raw_entry in entries:
            if not isinstance(raw_entry, dict):
                continue
            for key in ("stage", "backup"):
                value = str(raw_entry.get(key, "") or "")
                candidate = Path(value)
                if value and candidate.exists():
                    candidate.unlink()
    shutil.rmtree(transaction_dir, ignore_errors=True)
    try:
        transaction_dir.parent.rmdir()
    except OSError:
        pass


def _transaction_root(run_dir: Path) -> Path:
    root = run_dir / ".delivery-transactions"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    _write_text(path, _json_text(payload))


def _stage_text(target: Path, content: str) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.delivery-stage-", suffix=".tmp", dir=target.parent, text=True
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    return Path(temporary_name)


def _write_text(path: Path, content: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_name, path)


def _terminal_payload_paths(paths: RunArtifactPaths) -> tuple[tuple[Path, str], ...]:
    return (
        (paths.report_document_path, ""),
        (paths.final_report_path, ""),
        (paths.structured_recommendation_path, ""),
        (paths.structured_report_path, ""),
        (paths.final_decision_path, ""),
    )


def _json_text(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _json_object(payload: str) -> dict[str, object]:
    decoded = json.loads(payload)
    if not isinstance(decoded, dict):
        raise ValueError("delivery JSON snapshot must be an object")
    return decoded
