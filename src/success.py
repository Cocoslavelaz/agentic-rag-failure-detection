"""Explicit serialization equivalents; never infer success from an explanation."""
from dataclasses import dataclass

MISSING = object()


class SuccessValidationError(ValueError):
    pass


@dataclass(frozen=True)
class SuccessNormalization:
    raw: object
    normalized: str | None
    status: str

    def require(self):
        if self.normalized is None:
            raise SuccessValidationError(f"Invalid success serialization ({self.status}): {self.raw!r}")
        return self.normalized

    def fields(self):
        return {"success_raw": self.raw if self.raw is not MISSING else None,
                "success_normalized": self.normalized,
                "success_normalization_status": self.status}


def normalize_success(value=MISSING):
    if value is MISSING:
        return SuccessNormalization(value, None, "missing")
    if type(value) is bool:
        return SuccessNormalization(value, "yes" if value else "no", "mapped")
    if type(value) is int and value in (0, 1):
        return SuccessNormalization(value, "yes" if value == 1 else "no", "mapped")
    if isinstance(value, str):
        key = value.strip().lower()
        if key in {"yes", "true", "1", "no", "false", "0"}:
            normalized = "yes" if key in {"yes", "true", "1"} else "no"
            return SuccessNormalization(value, normalized, "canonical" if value == normalized else "mapped")
    return SuccessNormalization(value, None, "invalid")


def should_summarize(plan, outputs):
    # Validate even the last step: malformed success must never bypass validation.
    last = normalize_success(outputs[-1].get("success", MISSING)).require() if outputs else None
    return len(outputs) == len(plan) or last == "no"


def validate_qa_output(payload, schema):
    if not isinstance(payload, dict):
        raise SuccessValidationError("QA/Aggregate output must be an object")
    norm = normalize_success(payload.get("success", MISSING))
    # The existing Pydantic/wire schema remains unchanged. Only its temporary
    # validation input is canonical; the stored success retains its original type.
    canonical = dict(payload, success=norm.require())
    validated = schema.model_validate(canonical).model_dump()
    validated["success"] = payload["success"]
    return {**validated, **norm.fields()}
