"""Read local KILT/HotpotQA rows without rewriting or normalizing their data."""
import json
import os
from pathlib import Path


def resolve_dataset_path(name, path=None):
    value = path or os.getenv(f"{name.upper()}_DATASET_PATH") or os.getenv("DATASET_PATH")
    if not value:
        raise ValueError(f"Set --dataset-path, {name.upper()}_DATASET_PATH or DATASET_PATH; no dataset is downloaded")
    resolved = Path(os.path.expandvars(str(value))).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Dataset file not found: {resolved}")
    return resolved


def iter_dataset(name, path=None):
    source = resolve_dataset_path(name, path)
    if source.suffix.lower() == ".arrow":
        # Read the Arrow IPC itself: newer HF feature metadata is incompatible
        # with the pinned datasets package, but row data needs no conversion.
        import pyarrow as pa
        with pa.memory_map(str(source), "r") as handle:
            try:
                reader = pa.ipc.open_stream(handle)
            except pa.ArrowInvalid:
                handle.seek(0)
                reader = pa.ipc.open_file(handle)
                for index in range(reader.num_record_batches):
                    yield from reader.get_batch(index).to_pylist()
                return
            for batch in reader:
                yield from batch.to_pylist()
        return
    with source.open(encoding="utf-8-sig") as handle:
        if source.suffix.lower() == ".json":
            rows = json.load(handle)
            if not isinstance(rows, list):
                raise ValueError("Dataset JSON must be an array of original records")
            for index, row in enumerate(rows, start=1):
                if not isinstance(row, dict):
                    raise ValueError(f"Dataset record {index} must be an object")
                yield row
        else:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid dataset JSON at {source}:{line_number}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Dataset record at line {line_number} must be an object")
                yield row


def load_dataset(name, path=None):
    return list(iter_dataset(name, path))


def question_fields(row, dataset_format="kilt"):
    if dataset_format == "kilt":
        identifier, question = row.get("id"), row.get("input")
    elif dataset_format == "hotpotqa":
        identifier, question = row.get("_id", row.get("id")), row.get("question")
    else:
        raise ValueError(f"Unknown dataset format: {dataset_format}")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("Question ID must be a nonempty original string")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Missing question text for {identifier!r} using format {dataset_format}")
    return identifier, question
