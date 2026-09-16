"""Conservative, backed-up review suggestions; never edits dataset files.

Run without arguments to apply; use --dry-run to preview. Existing decisions
and even notes on undecided rows are preserved. Suggestions require spot-checks.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import math
import os
from pathlib import Path
import re
import tempfile


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "dataset_audit"
DECISIONS = REPORTS / "manual_review_decisions.csv"
EVIDENCE = REPORTS / "similarity_review.csv"
BACKUP = REPORTS / "manual_review_decisions_before_autofill.csv"
THRESHOLDS = {
    "phash_distance_max": 2,
    "dhash_distance_max": 2,
    "normalized_mae_max": 0.015,
    "color_histogram_intersection_min": 0.98,
    "filename_number_gap_max": 1,
}
NOTE = (
    "Auto-prefilled from very strong multi-metric similarity; same class/capture "
    "sequence. Requires spot-check."
)


def read_csv_bytes(raw):
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    fields = reader.fieldnames
    if not fields or len(set(fields)) != len(fields):
        raise ValueError("Missing or duplicate CSV column names.")
    rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("Malformed CSV row; fix manually before autofill.")
    return fields, rows


def pair_key(row):
    return tuple(sorted((row["filepath_a"], row["filepath_b"])))


def capture_tokens(path):
    # Require the actual project's full date + IMG naming pattern, not any digits.
    match = re.fullmatch(
        r"(\d{4}_\d{2}_\d{2})_IMG\s*\((\d+)\)", Path(path).stem,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    try:
        date = datetime.strptime(match[1], "%Y_%m_%d").date()
    except ValueError:
        return None
    return date, int(match[2])


def qualifies(row):
    """All conditions must pass. Missing/non-finite metrics never qualify."""
    if not row["class_a"] or row["class_a"] != row["class_b"]:
        return False
    if row["split_a"] == row["split_b"]:
        return False
    if not {row["split_a"], row["split_b"]} <= {"train", "val", "test"}:
        return False
    first, second = capture_tokens(row["filepath_a"]), capture_tokens(row["filepath_b"])
    if not first or not second or first[0] != second[0]:
        return False
    if abs(first[1] - second[1]) > THRESHOLDS["filename_number_gap_max"]:
        return False
    try:
        phash, dhash, mae, color = (
            float(row[key]) for key in (
                "phash_distance", "dhash_distance", "normalized_pixel_mae",
                "color_histogram_intersection",
            )
        )
    except (KeyError, ValueError, TypeError):
        return False
    return (
        all(math.isfinite(value) for value in (phash, dhash, mae, color))
        and phash.is_integer() and dhash.is_integer()
        and 0 <= phash <= THRESHOLDS["phash_distance_max"]
        and 0 <= dhash <= THRESHOLDS["dhash_distance_max"]
        and 0 <= mae <= THRESHOLDS["normalized_mae_max"]
        and THRESHOLDS["color_histogram_intersection_min"] <= color <= 1
    )


def verify_source(row, checked):
    """Reject stale suggestions using the previously recorded image hashes."""
    for side in ("a", "b"):
        relative = row[f"filepath_{side}"]
        path = (ROOT / relative).resolve()
        if not path.is_relative_to((ROOT / "dataset").resolve()) or not path.is_file():
            raise ValueError(f"Missing/unsafe image reference: {relative}")
        if relative not in checked:
            checked[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        if checked[relative] != row[f"sha256_{side}"]:
            raise ValueError(f"Image changed since similarity audit: {relative}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Preview only; no writes.")
    args = parser.parse_args()
    original = DECISIONS.read_bytes()
    evidence_bytes = EVIDENCE.read_bytes()
    fields, decisions = read_csv_bytes(original)
    _, evidence = read_csv_bytes(evidence_bytes)
    required = {"pair_id", "filepath_a", "filepath_b", "class_a", "class_b",
                "split_a", "split_b", "human_decision", "review_notes"}
    if not required <= set(fields):
        raise ValueError("Decision CSV is missing required columns.")
    if len({row["pair_id"] for row in decisions}) != len(decisions):
        raise ValueError("Duplicate decision pair IDs.")
    if len({pair_key(row) for row in decisions}) != len(decisions):
        raise ValueError("Duplicate decision image pairs.")
    lookup = {}
    for row in evidence:
        key = pair_key(row)
        if key in lookup:
            raise ValueError(f"Duplicate similarity evidence: {key}")
        lookup[key] = row
    suggestions, checked = [], {}
    for decision in decisions:
        # Preserve all existing text, including notes on an undecided case.
        if decision["human_decision"].strip() or decision["review_notes"].strip():
            continue
        row = lookup.get(pair_key(decision))
        if row is None:
            raise ValueError(f"No evidence for {decision['pair_id']}")
        for side in ("a", "b"):
            source_side = "a" if decision[f"filepath_{side}"] == row["filepath_a"] else "b"
            for field in ("class", "split"):
                if decision[f"{field}_{side}"] != row[f"{field}_{source_side}"]:
                    raise ValueError(f"Conflicting evidence for {decision['pair_id']}")
        if qualifies(row):
            verify_source(row, checked)
            suggestions.append((decision, row))

    print("All conditions required: same class, cross-split, same valid date token,")
    print("number gap <= 1, and every metric threshold below. Filename is not proof.")
    for key, value in THRESHOLDS.items():
        print(f"  {key}: {value}")
    for decision, _ in suggestions:
        decision["human_decision"] = "SAME_SOURCE"
        decision["review_notes"] = NOTE
    if suggestions and not args.dry_run:
        if DECISIONS.read_bytes() != original or EVIDENCE.read_bytes() != evidence_bytes:
            raise RuntimeError("Inputs changed during analysis; no update applied.")
        # Preserve the first requested backup; later runs get a separate snapshot.
        backup = BACKUP
        if backup.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            backup = BACKUP.with_name(f"{BACKUP.stem}_{stamp}.csv")
        with backup.open("xb") as handle:
            handle.write(original)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                             dir=REPORTS, prefix=".autofill_", suffix=".tmp",
                                             delete=False) as handle:
                temp_path = Path(handle.name)
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(decisions)
                handle.flush()
                os.fsync(handle.fileno())
            if DECISIONS.read_bytes() != original:
                raise RuntimeError("Decision file changed before saving; update aborted.")
            try:
                temp_path.replace(DECISIONS)
            except PermissionError:
                # Windows can permit file writes while denying atomic replacement.
                # Preserve the existing backup and recheck before writing in place.
                if DECISIONS.read_bytes() != original:
                    raise RuntimeError("Decision file changed; update aborted.")
                replacement = temp_path.read_bytes()
                with DECISIONS.open("wb") as destination:
                    destination.write(replacement)
                    destination.flush()
                    os.fsync(destination.fileno())
                print("Used backed-up in-place write: Windows denied file replacement.")
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()
        print(f"Backup: {backup}")
    label = "Would automatically mark" if args.dry_run else "Automatically marked"
    print(f"{label} SAME_SOURCE: {len(suggestions)}")
    print("Left for review (blank or UNSURE):", sum(
        row["human_decision"].strip() in {"", "UNSURE"} for row in decisions))
    print("Auto-prefilled cases still require spot-check; no DIFFERENT assigned.")
    print(f"Sample auto-filled pairs (up to 10; available {len(suggestions)}):")
    for decision, row in suggestions[:10]:
        print(f"  {decision['pair_id']}: {decision['filepath_a']} <-> {decision['filepath_b']}")
        print(f"    pHash={row['phash_distance']}, dHash={row['dhash_distance']}, "
              f"MAE={row['normalized_pixel_mae']}, color={row['color_histogram_intersection']}")


if __name__ == "__main__":
    main()
