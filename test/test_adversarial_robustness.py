#!/usr/bin/env python3
"""
Adversarial Robustness Test Suite

Addresses reviewer feedback:
"No evaluation on adversarial robustness suites (Unicode/emoji/bidi/leet; AML)"

Tests detection robustness against:
1. Unicode normalization attacks (NFKC)
2. Homoglyph substitution (Cyrillic, Greek)
3. Zero-width character injection
4. Bidirectional text manipulation
5. Multi-layer encoding (Base64, hex, URL, HTML entities)
6. Morse code obfuscation
"""

import base64
import codecs
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.honey_prompt_detector.utils.canonicalization import TextCanonicalizer, canonicalize, detect_obfuscation


class AdversarialRobustnessTests:
    """
    Comprehensive adversarial robustness test suite.

    Tests canonicalization against attacks documented in:
    - Unicode Security Considerations (TR36)
    - OWASP testing guides
    - Recent guardrail-evasion papers
    """

    def __init__(self):
        self.canonicalizer = TextCanonicalizer()
        self.tests_passed = 0
        self.tests_failed = 0
        self.test_results = []

    def run_test(self, name: str, test_func):
        """Run a single test and record result."""
        try:
            result = test_func()
            if result:
                self.tests_passed += 1
                status = "PASSED"
            else:
                self.tests_failed += 1
                status = "FAILED"
            self.test_results.append((name, status, None))
            print(f"  [{status}] {name}")
            return result
        except Exception as e:
            self.tests_failed += 1
            self.test_results.append((name, "ERROR", str(e)))
            print(f"  [ERROR] {name}: {e}")
            return False

    # =========================================================================
    # TEST CATEGORY 1: Unicode Normalization Attacks (NFKC)
    # =========================================================================

    def test_fullwidth_characters(self):
        """Test NFKC normalization handles fullwidth characters."""
        # Fullwidth characters should be normalized to ASCII
        test_cases = [
            ("Ｉｇｎｏｒｅ", "ignore"),
            ("ｐｒｅｖｉｏｕｓ", "previous"),
            ("Ｔｅｌｌ", "tell"),
        ]

        all_passed = True
        for obfuscated, expected in test_cases:
            canonical = canonicalize(obfuscated)
            if expected not in canonical:
                print(f"    Failed fullwidth: '{obfuscated}' -> '{canonical}' (expected '{expected}')")
                all_passed = False

        return all_passed

    def test_ligatures(self):
        """Test NFKC handles ligatures."""
        test_cases = [
            ("ﬁle", "file"),  # fi ligature
            ("ﬂag", "flag"),  # fl ligature
        ]

        all_passed = True
        for obfuscated, expected in test_cases:
            canonical = canonicalize(obfuscated)
            if expected not in canonical:
                print(f"    Failed ligature: '{obfuscated}' -> '{canonical}'")
                all_passed = False

        return all_passed

    # =========================================================================
    # TEST CATEGORY 2: Homoglyph Attacks
    # =========================================================================

    def test_cyrillic_homoglyphs(self):
        """Test detection of Cyrillic lookalike characters."""
        test_cases = [
            # Cyrillic а, е, о, р, с look like Latin a, e, o, p, c
            ("Ignorе previous", "ignore previous"),  # Cyrillic е
            ("ignorе", "ignore"),  # Cyrillic е at end
            ("systеm", "system"),  # Cyrillic е in middle
        ]

        all_passed = True
        for obfuscated, expected in test_cases:
            canonical = canonicalize(obfuscated)
            if expected not in canonical:
                print(f"    Failed Cyrillic: '{obfuscated}' -> '{canonical}'")
                all_passed = False

        return all_passed

    def test_greek_homoglyphs(self):
        """Test detection of Greek lookalike characters."""
        test_cases = [
            ("Αttack", "attack"),  # Greek Α (Alpha) looks like A
            ("Ιgnore", "ignore"),  # Greek Ι (Iota) looks like I
        ]

        all_passed = True
        for obfuscated, expected in test_cases:
            canonical = canonicalize(obfuscated)
            if expected not in canonical:
                print(f"    Failed Greek: '{obfuscated}' -> '{canonical}'")
                all_passed = False

        return all_passed

    # =========================================================================
    # TEST CATEGORY 3: Zero-Width Character Injection
    # =========================================================================

    def test_zero_width_space(self):
        """Test removal of zero-width spaces."""
        zwsp = "\u200b"  # Zero-width space

        test_cases = [
            (f"ig{zwsp}nore", "ignore"),
            (f"pre{zwsp}vious", "previous"),
            (f"sys{zwsp}tem", "system"),
        ]

        all_passed = True
        for obfuscated, expected in test_cases:
            canonical = canonicalize(obfuscated)
            if expected not in canonical:
                print(f"    Failed ZWSP: got '{canonical}' expected '{expected}'")
                all_passed = False

        return all_passed

    def test_zero_width_joiners(self):
        """Test removal of zero-width joiners."""
        zwj = "\u200d"  # Zero-width joiner
        zwnj = "\u200c"  # Zero-width non-joiner

        test_cases = [
            (f"ig{zwj}nore", "ignore"),
            (f"pre{zwnj}vious", "previous"),
        ]

        all_passed = True
        for obfuscated, expected in test_cases:
            canonical = canonicalize(obfuscated)
            if expected not in canonical:
                print(f"    Failed ZWJ: got '{canonical}' expected '{expected}'")
                all_passed = False

        return all_passed

    def test_word_joiner(self):
        """Test removal of word joiner."""
        wj = "\u2060"  # Word joiner

        test_cases = [
            (f"attack{wj}pattern", "attackpattern"),
        ]

        all_passed = True
        for obfuscated, expected in test_cases:
            canonical = canonicalize(obfuscated)
            if expected not in canonical:
                print(f"    Failed WJ: got '{canonical}' expected '{expected}'")
                all_passed = False

        return all_passed

    # =========================================================================
    # TEST CATEGORY 4: Bidirectional Text Attacks
    # =========================================================================

    def test_bidi_control_removal(self):
        """Test removal of bidi control characters."""
        rlo = "\u202e"  # Right-to-left override
        pdf = "\u202c"  # Pop directional formatting
        lrm = "\u200e"  # Left-to-right mark
        rlm = "\u200f"  # Right-to-left mark

        test_cases = [
            (f"test{rlo}string{pdf}", "teststring"),
            (f"test{lrm}string{rlm}", "teststring"),
        ]

        all_passed = True
        for obfuscated, expected in test_cases:
            canonical = canonicalize(obfuscated)
            # Just verify bidi chars are removed
            if rlo in canonical or pdf in canonical or lrm in canonical or rlm in canonical:
                print("    Failed bidi: controls still present")
                all_passed = False
            if expected not in canonical:
                print(f"    Failed bidi: got '{canonical}'")
                all_passed = False

        return all_passed

    # =========================================================================
    # TEST CATEGORY 5: Encoding Attacks
    # =========================================================================

    def test_base64_encoding(self):
        """Test Base64 decoding."""
        # Base64 of "ignore previous"
        encoded = base64.b64encode(b"ignore previous").decode()

        canonical = canonicalize(encoded)
        return "ignore previous" in canonical

    def test_url_encoding(self):
        """Test URL percent-encoding decoding."""
        test_cases = [
            ("%69%67%6e%6f%72%65", "ignore"),  # URL-encoded "ignore"
            ("ignore%20previous", "ignore previous"),  # Space
        ]

        all_passed = True
        for obfuscated, expected in test_cases:
            canonical = canonicalize(obfuscated)
            if expected not in canonical:
                print(f"    Failed URL: '{obfuscated}' -> '{canonical}'")
                all_passed = False

        return all_passed

    def test_html_entities(self):
        """Test HTML entity decoding."""
        test_cases = [
            ("&#105;&#103;&#110;&#111;&#114;&#101;", "ignore"),  # Decimal entities
            ("&lt;test&gt;", "<test>"),  # Named entities
            ("&amp;", "&"),  # Ampersand
        ]

        all_passed = True
        for obfuscated, expected in test_cases:
            canonical = canonicalize(obfuscated)
            if expected not in canonical:
                print(f"    Failed HTML: '{obfuscated}' -> '{canonical}'")
                all_passed = False

        return all_passed

    def test_hex_encoding(self):
        """Test hex decoding."""
        test_cases = [
            (r"\x69\x67\x6e\x6f\x72\x65", "ignore"),  # \xNN format
        ]

        all_passed = True
        for obfuscated, expected in test_cases:
            canonical = canonicalize(obfuscated)
            if expected not in canonical:
                print(f"    Failed hex: '{obfuscated}' -> '{canonical}'")
                all_passed = False

        return all_passed

    def test_unicode_escapes(self):
        """Test Unicode escape decoding."""
        test_cases = [
            (r"\u0069\u0067\u006e\u006f\u0072\u0065", "ignore"),  # \uXXXX format
        ]

        all_passed = True
        for obfuscated, expected in test_cases:
            canonical = canonicalize(obfuscated)
            if expected not in canonical:
                print(f"    Failed unicode: '{obfuscated}' -> '{canonical}'")
                all_passed = False

        return all_passed

    def test_rot13_with_attack_words(self):
        """Test ROT13 decoding for attack-related text."""
        # ROT13 of "ignore previous instructions"
        rot13_text = codecs.encode("ignore previous instructions", "rot_13")

        canonical = canonicalize(rot13_text)
        # Should decode because result has attack words
        return "ignore" in canonical or "previous" in canonical

    def test_morse_code(self):
        """Test Morse code decoding."""
        # Morse for "ignore" with proper spacing
        morse = ".. --. -. --- .-. ."

        canonical = canonicalize(morse)
        return "ignore" in canonical

    # =========================================================================
    # TEST CATEGORY 6: Obfuscation Detection
    # =========================================================================

    def test_obfuscation_detection_invisible(self):
        """Test detection of invisible character obfuscation."""
        text = "\u200btest"
        detection = detect_obfuscation(text)
        return detection["has_invisible_chars"] is True

    def test_obfuscation_detection_homoglyphs(self):
        """Test detection of homoglyph obfuscation."""
        text = "tеst"  # Cyrillic е
        detection = detect_obfuscation(text)
        return detection["has_homoglyphs"] is True

    def test_obfuscation_detection_bidi(self):
        """Test detection of bidi obfuscation."""
        text = "\u202etest"
        detection = detect_obfuscation(text)
        return detection["has_bidi_controls"] is True

    def test_obfuscation_detection_encoding(self):
        """Test detection of encoding obfuscation."""
        text = base64.b64encode(b"test string here").decode()  # Long enough base64
        detection = detect_obfuscation(text)
        return detection["has_encoding"] is True

    # =========================================================================
    # TEST CATEGORY 7: Combined Attacks
    # =========================================================================

    def test_combined_homoglyph_zwsp(self):
        """Test combined homoglyph and zero-width attack."""
        zwsp = "\u200b"
        # Cyrillic е + ZWSP
        obfuscated = f"ig{zwsp}norе"  # Cyrillic е at end

        canonical = canonicalize(obfuscated)
        return "ignore" in canonical

    def test_combined_encoding_zwsp(self):
        """Test combined encoding and zero-width attack."""
        zwsp = "\u200b"
        # URL encoding with ZWSP
        obfuscated = f"ig{zwsp}nore%20previous"

        canonical = canonicalize(obfuscated)
        return "ignore" in canonical and "previous" in canonical

    # =========================================================================
    # TEST CATEGORY 8: Real-World Patterns
    # =========================================================================

    def test_real_attack_with_fullwidth(self):
        """Test real attack pattern with fullwidth chars."""
        # "Ignore previous" in fullwidth
        obfuscated = "Ｉｇｎｏｒｅ ｐｒｅｖｉｏｕｓ"

        canonical = canonicalize(obfuscated)
        return "ignore" in canonical and "previous" in canonical

    def test_real_attack_with_homoglyphs(self):
        """Test real attack pattern with homoglyphs."""
        # Mix of Cyrillic and Latin
        obfuscated = "Ignorе prеvious"  # Cyrillic е

        canonical = canonicalize(obfuscated)
        return "ignore" in canonical and "previous" in canonical

    # =========================================================================
    # Run All Tests
    # =========================================================================

    def run_all_tests(self):
        """Run all adversarial robustness tests."""
        print("\n" + "=" * 70)
        print("ADVERSARIAL ROBUSTNESS TEST SUITE")
        print("Testing canonicalization against obfuscation attacks")
        print("=" * 70)

        # Unicode Normalization
        print("\n[Category 1: Unicode Normalization (NFKC)]")
        self.run_test("Fullwidth characters", self.test_fullwidth_characters)
        self.run_test("Ligatures", self.test_ligatures)

        # Homoglyphs
        print("\n[Category 2: Homoglyph Attacks]")
        self.run_test("Cyrillic homoglyphs", self.test_cyrillic_homoglyphs)
        self.run_test("Greek homoglyphs", self.test_greek_homoglyphs)

        # Zero-Width
        print("\n[Category 3: Zero-Width Character Injection]")
        self.run_test("Zero-width space", self.test_zero_width_space)
        self.run_test("Zero-width joiners", self.test_zero_width_joiners)
        self.run_test("Word joiner", self.test_word_joiner)

        # Bidi
        print("\n[Category 4: Bidirectional Text Attacks]")
        self.run_test("Bidi control removal", self.test_bidi_control_removal)

        # Encoding
        print("\n[Category 5: Encoding Attacks]")
        self.run_test("Base64 encoding", self.test_base64_encoding)
        self.run_test("URL encoding", self.test_url_encoding)
        self.run_test("HTML entities", self.test_html_entities)
        self.run_test("Hex encoding", self.test_hex_encoding)
        self.run_test("Unicode escapes", self.test_unicode_escapes)
        self.run_test("ROT13 with attack words", self.test_rot13_with_attack_words)
        self.run_test("Morse code", self.test_morse_code)

        # Detection
        print("\n[Category 6: Obfuscation Detection]")
        self.run_test("Detect invisible chars", self.test_obfuscation_detection_invisible)
        self.run_test("Detect homoglyphs", self.test_obfuscation_detection_homoglyphs)
        self.run_test("Detect bidi controls", self.test_obfuscation_detection_bidi)
        self.run_test("Detect encoding", self.test_obfuscation_detection_encoding)

        # Combined
        print("\n[Category 7: Combined Attacks]")
        self.run_test("Homoglyph + ZWSP", self.test_combined_homoglyph_zwsp)
        self.run_test("Encoding + ZWSP", self.test_combined_encoding_zwsp)

        # Real-world
        print("\n[Category 8: Real-World Attack Patterns]")
        self.run_test("Fullwidth attack", self.test_real_attack_with_fullwidth)
        self.run_test("Homoglyph attack", self.test_real_attack_with_homoglyphs)

        # Summary
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        print(f"  Passed: {self.tests_passed}")
        print(f"  Failed: {self.tests_failed}")
        print(f"  Total:  {self.tests_passed + self.tests_failed}")

        if self.tests_failed == 0:
            print("\nALL ADVERSARIAL ROBUSTNESS TESTS PASSED!")
            print("\nThe canonicalizer defends against:")
            print("  - Unicode NFKC normalization attacks")
            print("  - Homoglyph substitution (Cyrillic, Greek)")
            print("  - Zero-width character injection")
            print("  - Bidirectional text manipulation")
            print("  - Multi-layer encoding (Base64, URL, HTML, hex, ROT13, Morse)")
            print("  - Combined obfuscation techniques")
        else:
            print(f"\n{self.tests_failed} TESTS FAILED - Review output above")

        return self.tests_failed == 0


if __name__ == "__main__":
    tests = AdversarialRobustnessTests()
    success = tests.run_all_tests()
    sys.exit(0 if success else 1)
