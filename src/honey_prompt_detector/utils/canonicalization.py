"""
Robust Text Canonicalization Module

Addresses reviewer feedback on obfuscation vulnerabilities:
- Unicode NFKC normalization
- Homoglyph normalization
- Zero-width character removal
- Bidi control stripping
- Multiple encoding detection and decoding

This module hardens the detector against:
- Unicode smuggling attacks
- Homoglyph substitution attacks
- Invisible character injection
- Bidirectional text attacks
- Multi-encoding obfuscation (Base64, hex, URL, ROT13, etc.)
"""

import base64
import codecs
import html
import re
import unicodedata
from typing import Dict, List, Set
from urllib.parse import unquote

# Homoglyph mapping: visually similar characters to ASCII
# Note: We do NOT include leet speak here to avoid breaking normal text
HOMOGLYPH_MAP: Dict[str, str] = {
    # Cyrillic lookalikes (lowercase)
    "а": "a",
    "е": "e",
    "і": "i",
    "о": "o",
    "р": "p",
    "с": "c",
    "у": "y",
    "х": "x",
    # Cyrillic lookalikes (uppercase)
    "А": "A",
    "В": "B",
    "Е": "E",
    "І": "I",
    "К": "K",
    "М": "M",
    "Н": "H",
    "О": "O",
    "Р": "P",
    "С": "C",
    "Т": "T",
    "Х": "X",
    # Greek lookalikes (uppercase)
    "Α": "A",
    "Β": "B",
    "Ε": "E",
    "Ζ": "Z",
    "Η": "H",
    "Ι": "I",
    "Κ": "K",
    "Μ": "M",
    "Ν": "N",
    "Ο": "O",
    "Ρ": "P",
    "Τ": "T",
    "Υ": "Y",
    "Χ": "X",
    # Greek lookalikes (lowercase)
    "α": "a",
    "ο": "o",
    "ν": "v",
    "ρ": "p",
    "τ": "t",
    "υ": "u",
    "χ": "x",
    # Mathematical/special lookalikes
    "ℓ": "l",
    "ℐ": "I",
    "ℑ": "I",
    "ℒ": "L",
    "ℕ": "N",
    "ℙ": "P",
    "ℚ": "Q",
    "ℛ": "R",
    "ℜ": "R",
    "ℤ": "Z",
    "ℨ": "Z",
    "ℬ": "B",
    "ℭ": "C",
    "ℯ": "e",
    "ℰ": "E",
    "ℱ": "F",
    "ℳ": "M",
    "ℴ": "o",
    "ℹ": "i",
    # Common substitutions
    "ı": "i",
    "ȷ": "j",
    "ɑ": "a",
    "ɡ": "g",
    "ɪ": "I",
    # Fullwidth digits -> ASCII digits
    "０": "0",
    "１": "1",
    "２": "2",
    "３": "3",
    "４": "4",
    "５": "5",
    "６": "6",
    "７": "7",
    "８": "8",
    "９": "9",
}

# Zero-width and invisible characters to remove
INVISIBLE_CHARS: Set[str] = {
    "\u200b",  # Zero-width space
    "\u200c",  # Zero-width non-joiner
    "\u200d",  # Zero-width joiner
    "\u200e",  # Left-to-right mark
    "\u200f",  # Right-to-left mark
    "\u2060",  # Word joiner
    "\u2061",  # Function application
    "\u2062",  # Invisible times
    "\u2063",  # Invisible separator
    "\u2064",  # Invisible plus
    "\ufeff",  # Zero-width no-break space (BOM)
    "\u00ad",  # Soft hyphen
    "\u034f",  # Combining grapheme joiner
    "\u061c",  # Arabic letter mark
    "\u115f",  # Hangul choseong filler
    "\u1160",  # Hangul jungseong filler
    "\u17b4",  # Khmer vowel inherent AQ
    "\u17b5",  # Khmer vowel inherent AA
    "\u180e",  # Mongolian vowel separator
    "\u3164",  # Hangul filler
    "\uffa0",  # Halfwidth hangul filler
}

# Bidi control characters
BIDI_CONTROLS: Set[str] = {
    "\u200e",
    "\u200f",
    "\u202a",
    "\u202b",
    "\u202c",
    "\u202d",
    "\u202e",
    "\u2066",
    "\u2067",
    "\u2068",
    "\u2069",
}

# Various whitespace characters to normalize
WHITESPACE_CHARS: Set[str] = {
    "\u2000",  # En quad
    "\u2001",  # Em quad
    "\u2002",  # En space
    "\u2003",  # Em space
    "\u2004",  # Three-per-em space
    "\u2005",  # Four-per-em space
    "\u2006",  # Six-per-em space
    "\u2007",  # Figure space
    "\u2008",  # Punctuation space
    "\u2009",  # Thin space
    "\u200a",  # Hair space
    "\u3000",  # Ideographic space
    "\u00a0",  # Non-breaking space
}

# Morse code mapping
MORSE_TO_ALPHA: Dict[str, str] = {
    ".-": "a",
    "-...": "b",
    "-.-.": "c",
    "-..": "d",
    ".": "e",
    "..-.": "f",
    "--.": "g",
    "....": "h",
    "..": "i",
    ".---": "j",
    "-.-": "k",
    ".-..": "l",
    "--": "m",
    "-.": "n",
    "---": "o",
    ".--.": "p",
    "--.-": "q",
    ".-.": "r",
    "...": "s",
    "-": "t",
    "..-": "u",
    "...-": "v",
    ".--": "w",
    "-..-": "x",
    "-.--": "y",
    "--..": "z",
    "-----": "0",
    ".----": "1",
    "..---": "2",
    "...--": "3",
    "....-": "4",
    ".....": "5",
    "-....": "6",
    "--...": "7",
    "---..": "8",
    "----.": "9",
    "..--..": "?",
    ".-.-.-": ".",
    "--..--": ",",
}


class TextCanonicalizer:
    """
    Robust text canonicalization for detecting obfuscated tokens.

    Implements defense against:
    - Unicode normalization attacks
    - Homoglyph substitution
    - Invisible character injection
    - Bidirectional text manipulation
    - Multiple encoding layers
    """

    def __init__(self, max_decode_iterations: int = 3):
        """
        Initialize canonicalizer.

        Args:
            max_decode_iterations: Maximum iterations for recursive decoding
        """
        self.max_decode_iterations = max_decode_iterations
        self.homoglyph_map = HOMOGLYPH_MAP
        self.invisible_chars = INVISIBLE_CHARS
        self.bidi_controls = BIDI_CONTROLS
        self.whitespace_chars = WHITESPACE_CHARS

    def canonicalize(self, text: str, aggressive: bool = True) -> str:
        """
        Fully canonicalize text for robust token matching.

        Args:
            text: Input text to canonicalize
            aggressive: If True, apply all transformations

        Returns:
            Canonicalized text
        """
        if not text:
            return text

        # Step 1: Remove invisible characters first
        text = self.remove_invisible_chars(text)

        # Step 2: Remove bidi controls
        text = self.remove_bidi_controls(text)

        # Step 3: Normalize whitespace characters to regular space
        text = self.normalize_whitespace_chars(text)

        # Step 4: Unicode NFKC normalization (handles fullwidth, ligatures, etc.)
        text = self.unicode_normalize(text)

        # Step 5: Homoglyph normalization
        text = self.normalize_homoglyphs(text)

        # Step 6: Recursive decoding of encodings
        text = self.decode_all_encodings(text)

        if aggressive:
            # Step 7: Normalize case
            text = text.lower()

            # Step 8: Normalize whitespace (collapse multiple spaces)
            text = self.normalize_whitespace(text)

        return text

    def remove_invisible_chars(self, text: str) -> str:
        """Remove zero-width and invisible characters."""
        return "".join(c for c in text if c not in self.invisible_chars)

    def remove_bidi_controls(self, text: str) -> str:
        """Remove bidirectional control characters."""
        return "".join(c for c in text if c not in self.bidi_controls)

    def normalize_whitespace_chars(self, text: str) -> str:
        """Convert various whitespace characters to regular space."""
        return "".join(" " if c in self.whitespace_chars else c for c in text)

    def unicode_normalize(self, text: str, form: str = "NFKC") -> str:
        """
        Apply Unicode normalization.

        NFKC is used as it:
        - Decomposes characters to canonical form
        - Replaces compatibility characters with canonical equivalents
        - Recomposes to canonical form
        """
        try:
            return unicodedata.normalize(form, text)
        except Exception:
            return text

    def normalize_homoglyphs(self, text: str) -> str:
        """Replace visually similar characters with ASCII equivalents."""
        return "".join(self.homoglyph_map.get(c, c) for c in text)

    def normalize_whitespace(self, text: str) -> str:
        """Normalize all whitespace to single spaces."""
        return " ".join(text.split())

    def decode_all_encodings(self, text: str) -> str:
        """
        Recursively decode all detected encodings.

        Handles: Base64, hex, URL encoding, HTML entities, ROT13, Unicode escapes
        """
        prev_text = None
        iterations = 0

        while prev_text != text and iterations < self.max_decode_iterations:
            prev_text = text
            iterations += 1

            # Try each decoder - order matters!
            text = self.decode_html_entities(text)
            text = self.decode_url(text)
            text = self.decode_unicode_escapes(text)
            text = self.decode_hex(text)
            text = self.decode_base64(text)
            text = self.decode_rot13(text)
            text = self.decode_morse(text)

        return text

    def decode_base64(self, text: str) -> str:
        """Decode Base64 encoded segments."""
        # Pattern for potential Base64 strings (at least 8 chars)
        b64_pattern = r"[A-Za-z0-9+/]{8,}={0,2}"

        def try_decode(match):
            encoded = match.group(0)
            try:
                # Ensure proper padding
                padding_needed = 4 - (len(encoded) % 4)
                if padding_needed != 4:
                    encoded_padded = encoded + "=" * padding_needed
                else:
                    encoded_padded = encoded

                decoded = base64.b64decode(encoded_padded).decode("utf-8", errors="strict")

                # Only return if it looks like readable text (mostly printable)
                if decoded and len(decoded) > 2:
                    printable_ratio = sum(1 for c in decoded if c.isprintable() or c.isspace()) / len(decoded)
                    if printable_ratio > 0.8:
                        return decoded
            except Exception:
                pass
            return match.group(0)

        return re.sub(b64_pattern, try_decode, text)

    def decode_hex(self, text: str) -> str:
        """Decode hexadecimal encoded segments."""

        # Pattern for \xNN sequences
        def decode_escaped_hex(text):
            pattern = r"(?:\\x[0-9a-fA-F]{2})+"

            def replace(m):
                try:
                    hex_str = m.group(0).replace("\\x", "")
                    return bytes.fromhex(hex_str).decode("utf-8", errors="strict")
                except Exception:
                    return m.group(0)

            return re.sub(pattern, replace, text)

        # Pattern for 0xNNNN sequences
        def decode_0x_hex(text):
            pattern = r"0x([0-9a-fA-F]{2,})"

            def replace(m):
                try:
                    decoded = bytes.fromhex(m.group(1)).decode("utf-8", errors="strict")
                    if decoded and all(c.isprintable() or c.isspace() for c in decoded):
                        return decoded
                except Exception:
                    pass
                return m.group(0)

            return re.sub(pattern, replace, text)

        text = decode_escaped_hex(text)
        text = decode_0x_hex(text)
        return text

    def decode_url(self, text: str) -> str:
        """Decode URL-encoded segments."""
        try:
            return unquote(text)
        except Exception:
            return text

    def decode_html_entities(self, text: str) -> str:
        """Decode HTML entities."""
        try:
            return html.unescape(text)
        except Exception:
            return text

    def decode_unicode_escapes(self, text: str) -> str:
        """Decode Unicode escape sequences."""

        # \uXXXX format
        def decode_u_escapes(text):
            pattern = r"\\u([0-9a-fA-F]{4})"

            def replace(m):
                try:
                    return chr(int(m.group(1), 16))
                except Exception:
                    return m.group(0)

            return re.sub(pattern, replace, text)

        # U+XXXX format
        def decode_uplus_escapes(text):
            pattern = r"U\+([0-9a-fA-F]{4,6})"

            def replace(m):
                try:
                    return chr(int(m.group(1), 16))
                except Exception:
                    return m.group(0)

            return re.sub(pattern, replace, text)

        text = decode_u_escapes(text)
        text = decode_uplus_escapes(text)
        return text

    def decode_rot13(self, text: str) -> str:
        """
        Decode ROT13 if detected.

        Only decodes if the result contains more common English words.
        """
        # Don't decode if text already has common attack words
        common_attack_words = {"ignore", "previous", "instructions", "tell", "system", "prompt", "forget", "disregard"}
        text_lower = text.lower()
        if any(word in text_lower for word in common_attack_words):
            return text

        try:
            decoded = codecs.decode(text, "rot_13")
            decoded_lower = decoded.lower()

            # Check if decoded has more attack words
            decoded_matches = sum(1 for word in common_attack_words if word in decoded_lower)
            original_matches = sum(1 for word in common_attack_words if word in text_lower)

            if decoded_matches > original_matches and decoded_matches > 0:
                return decoded
        except Exception:
            pass

        return text

    def decode_morse(self, text: str) -> str:
        """Decode Morse code if detected."""
        # Only try if text looks like Morse code (dots, dashes, spaces, slashes)
        if not re.match(r"^[\.\-\s/]+$", text.strip()):
            return text

        try:
            # Split by word separator (/ or multiple spaces)
            words = re.split(r"\s*/\s*|\s{3,}", text.strip())
            decoded_words = []

            for word in words:
                if not word.strip():
                    continue
                # Split by letter separator (single or double space)
                letters = re.split(r"\s+", word.strip())
                decoded_letters = []

                for letter in letters:
                    letter = letter.strip()
                    if letter in MORSE_TO_ALPHA:
                        decoded_letters.append(MORSE_TO_ALPHA[letter])
                    elif letter:
                        return text  # Unknown Morse, return original

                if decoded_letters:
                    decoded_words.append("".join(decoded_letters))

            if decoded_words:
                return " ".join(decoded_words)
        except Exception:
            pass

        return text

    def get_all_variants(self, text: str) -> List[str]:
        """
        Generate all canonical variants of text for matching.

        Returns multiple variants to catch partial obfuscation.
        """
        variants = set()

        # Original
        variants.add(text)

        # Full canonicalization
        variants.add(self.canonicalize(text, aggressive=True))

        # Partial canonicalizations
        variants.add(self.canonicalize(text, aggressive=False))
        variants.add(self.remove_invisible_chars(text))
        variants.add(self.unicode_normalize(text))
        variants.add(self.normalize_homoglyphs(text))
        variants.add(text.lower())
        variants.add(text.upper())

        # Decode-only variants
        variants.add(self.decode_all_encodings(text))

        return [v for v in variants if v]  # Remove empty strings

    def detect_obfuscation(self, text: str) -> dict:
        """
        Detect what types of obfuscation are present in text.

        Returns dict with detected obfuscation types and severity.
        """
        detections = {
            "has_invisible_chars": False,
            "has_bidi_controls": False,
            "has_homoglyphs": False,
            "has_encoding": False,
            "encoding_types": [],
            "severity": "none",
        }

        # Check for invisible characters
        if any(c in self.invisible_chars for c in text):
            detections["has_invisible_chars"] = True

        # Check for bidi controls
        if any(c in self.bidi_controls for c in text):
            detections["has_bidi_controls"] = True

        # Check for homoglyphs
        if any(c in self.homoglyph_map for c in text):
            detections["has_homoglyphs"] = True

        # Check for encodings
        # Base64: at least 12 chars of base64 alphabet with optional padding
        if re.search(r"[A-Za-z0-9+/]{12,}={0,2}", text):
            detections["has_encoding"] = True
            detections["encoding_types"].append("base64")

        if re.search(r"0x[0-9a-fA-F]+|\\x[0-9a-fA-F]{2}", text):
            detections["has_encoding"] = True
            detections["encoding_types"].append("hex")

        if re.search(r"%[0-9a-fA-F]{2}", text):
            detections["has_encoding"] = True
            detections["encoding_types"].append("url")

        if re.search(r"&[a-zA-Z]+;|&#\d+;|&#x[0-9a-fA-F]+;", text):
            detections["has_encoding"] = True
            detections["encoding_types"].append("html")

        # Calculate severity
        obfuscation_count = sum(
            [
                detections["has_invisible_chars"],
                detections["has_bidi_controls"],
                detections["has_homoglyphs"],
                detections["has_encoding"],
            ]
        )

        if obfuscation_count == 0:
            detections["severity"] = "none"
        elif obfuscation_count == 1:
            detections["severity"] = "low"
        elif obfuscation_count == 2:
            detections["severity"] = "medium"
        else:
            detections["severity"] = "high"

        return detections


# Singleton instance for easy use
_canonicalizer = TextCanonicalizer()


def canonicalize(text: str, aggressive: bool = True) -> str:
    """Convenience function for text canonicalization."""
    return _canonicalizer.canonicalize(text, aggressive)


def get_variants(text: str) -> List[str]:
    """Convenience function to get all text variants."""
    return _canonicalizer.get_all_variants(text)


def detect_obfuscation(text: str) -> dict:
    """Convenience function for obfuscation detection."""
    return _canonicalizer.detect_obfuscation(text)
