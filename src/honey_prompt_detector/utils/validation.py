import re
from dataclasses import dataclass
from typing import List

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]

class InputValidator:
    MAX_TEXT_LENGTH = 1_000_000

    @classmethod
    def validate_text_input(cls, text: str) -> ValidationResult:
        errors, warnings = [], []

        if not text:
            errors.append("Input text cannot be empty")
            return ValidationResult(False, errors, warnings)

        if len(text) > cls.MAX_TEXT_LENGTH:
            errors.append(f"Input exceeds maximum length ({cls.MAX_TEXT_LENGTH} chars)")

        if '\x00' in text:
            errors.append("Input contains null characters")

        if re.search(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', text):
            warnings.append("Input contains control characters")

        return ValidationResult(not errors, errors, warnings)
