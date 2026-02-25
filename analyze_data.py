#!/usr/bin/env python3
"""
Data analysis script for HIVE benchmark dataset.
Analyzes language distribution, script families, and data quality metrics.
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import unicodedata

def get_script_family(char):
    """
    Classify a character into a script family using Unicode blocks.
    Returns: 'Latin', 'CJK', 'Arabic', 'Cyrillic', 'Greek', 'Devanagari',
             'Hebrew', 'Emoji', 'Other', 'ASCII'
    """
    code_point = ord(char)

    # ASCII
    if code_point < 128:
        return 'ASCII'

    # Latin Extended (including accented characters)
    if 0x0100 <= code_point <= 0x017F:
        return 'Latin'
    if 0x0180 <= code_point <= 0x024F:
        return 'Latin'

    # Greek
    if 0x0370 <= code_point <= 0x03FF:
        return 'Greek'

    # Cyrillic
    if 0x0400 <= code_point <= 0x04FF:
        return 'Cyrillic'

    # Arabic
    if 0x0600 <= code_point <= 0x06FF:
        return 'Arabic'
    if 0x0750 <= code_point <= 0x077F:
        return 'Arabic'

    # Hebrew
    if 0x0590 <= code_point <= 0x05FF:
        return 'Hebrew'

    # Devanagari
    if 0x0900 <= code_point <= 0x097F:
        return 'Devanagari'

    # Thai
    if 0x0E00 <= code_point <= 0x0E7F:
        return 'Thai'

    # Hangul
    if 0xAC00 <= code_point <= 0xD7AF:
        return 'Hangul'

    # CJK Unified Ideographs
    if 0x4E00 <= code_point <= 0x9FFF:
        return 'CJK'
    if 0x3400 <= code_point <= 0x4DBF:
        return 'CJK'
    if 0x20000 <= code_point <= 0x2A6DF:
        return 'CJK'

    # CJK Symbols and Punctuation
    if 0x3000 <= code_point <= 0x303F:
        return 'CJK'

    # Hiragana and Katakana
    if 0x3040 <= code_point <= 0x309F:
        return 'Japanese'
    if 0x30A0 <= code_point <= 0x30FF:
        return 'Japanese'

    # Emoji ranges
    if 0x1F300 <= code_point <= 0x1F9FF:
        return 'Emoji'
    if 0x2600 <= code_point <= 0x27BF:
        return 'Emoji'

    return 'Other'

def analyze_text_scripts(text):
    """
    Analyze script distribution in a text.
    Returns a Counter of script families.
    """
    if not isinstance(text, str):
        return Counter()

    scripts = Counter()
    for char in text:
        if char.isspace() or char.isdigit() or ord(char) < 128 and not char.isalpha():
            continue
        script = get_script_family(char)
        scripts[script] += 1

    return scripts

def classify_text_language(scripts):
    """
    Determine primary language family based on script distribution.
    """
    if not scripts:
        return 'Unknown'

    total = sum(scripts.values())
    dominant_script, count = scripts.most_common(1)[0]
    ratio = count / total

    if ratio > 0.8:
        return dominant_script
    else:
        return 'Mixed'

def main():
    print("Loading dataset...")
    df = pd.read_csv('/sessions/hopeful-zen-feynman/mnt/honey-prompt-detector/data/unified_dataset.csv')

    print(f"\n{'='*80}")
    print("HIVE BENCHMARK DATA ANALYSIS")
    print(f"{'='*80}\n")

    # Basic statistics
    print(f"Total samples: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Label distribution
    print(f"\n--- Label Distribution ---")
    label_counts = df['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        pct = 100 * count / len(df)
        print(f"  Label {label}: {count:,} ({pct:.1f}%)")

    # Source distribution
    print(f"\n--- Source Distribution ---")
    source_counts = df['source'].value_counts()
    for source, count in source_counts.items():
        pct = 100 * count / len(df)
        print(f"  {source}: {count:,} ({pct:.1f}%)")

    # Text length statistics
    print(f"\n--- Text Length Statistics ---")
    df['text_length'] = df['text'].astype(str).str.len()
    print(f"  Mean: {df['text_length'].mean():.1f} characters")
    print(f"  Median: {df['text_length'].median():.1f} characters")
    print(f"  Min: {df['text_length'].min()} characters")
    print(f"  Max: {df['text_length'].max()} characters")
    print(f"  Std Dev: {df['text_length'].std():.1f} characters")

    # Word count (approximate)
    df['word_count'] = df['text'].astype(str).str.split().str.len()
    print(f"  Mean words: {df['word_count'].mean():.1f}")
    print(f"  Median words: {df['word_count'].median():.1f}")

    # Duplicate analysis
    print(f"\n--- Duplicate Analysis ---")
    total_texts = len(df)
    unique_texts = df['text'].nunique()
    duplicates = total_texts - unique_texts
    print(f"  Unique texts: {unique_texts:,}")
    print(f"  Duplicate texts: {duplicates:,} ({100*duplicates/total_texts:.2f}%)")

    # Find most common duplicates
    dup_texts = df['text'].value_counts()
    most_dup = dup_texts[dup_texts > 1].head(5)
    if len(most_dup) > 0:
        print(f"  Most duplicated texts (top 5):")
        for text, count in most_dup.items():
            preview = text[:60].replace('\n', ' ')
            print(f"    ({count}x) {preview}...")

    # Script/Language distribution
    print(f"\n--- Script Family Distribution ---")
    all_scripts = Counter()
    text_languages = []

    for idx, text in enumerate(df['text']):
        if idx % 5000 == 0:
            print(f"  Processing texts: {idx}/{len(df)}", end='\r')

        if not isinstance(text, str):
            text_languages.append('Unknown')
            continue

        scripts = analyze_text_scripts(text)
        all_scripts.update(scripts)
        language = classify_text_language(scripts)
        text_languages.append(language)

    print(f"  Processing texts: {len(df)}/{len(df)}        ")

    # Overall script statistics
    total_chars = sum(all_scripts.values())
    print(f"\n  Overall character distribution (alphabetic/symbolic only):")
    print(f"  Total characters analyzed: {total_chars:,}")
    for script, count in all_scripts.most_common():
        pct = 100 * count / total_chars
        print(f"    {script:15s}: {count:>8,} ({pct:>5.1f}%)")

    # Language family distribution (per-text)
    print(f"\n  Language family distribution (per text):")
    lang_counts = Counter(text_languages)
    for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(df)
        print(f"    {lang:15s}: {count:>6,} texts ({pct:>5.1f}%)")

    # Cross-tabulation: label vs language
    print(f"\n--- Label Distribution by Language Family ---")
    df['language'] = text_languages
    cross_tab = pd.crosstab(df['language'], df['label'], margins=True)
    print(cross_tab)

    # Per-source analysis
    print(f"\n--- Language Distribution by Source ---")
    for source in df['source'].unique():
        source_texts = df[df['source'] == source]['language'].value_counts()
        print(f"\n  {source}:")
        for lang, count in source_texts.items():
            pct = 100 * count / len(df[df['source'] == source])
            print(f"    {lang:15s}: {count:>6,} ({pct:>5.1f}%)")

    # Data quality summary
    print(f"\n--- Data Quality Metrics ---")
    print(f"  Missing values (text): {df['text'].isna().sum()}")
    print(f"  Missing values (label): {df['label'].isna().sum()}")
    print(f"  Missing values (source): {df['source'].isna().sum()}")

    # Empty texts
    empty_texts = (df['text'].astype(str).str.strip().str.len() == 0).sum()
    print(f"  Empty or whitespace-only texts: {empty_texts}")

    # Texts with only ASCII
    ascii_only = (df['language'] == 'ASCII').sum()
    print(f"  ASCII-only texts: {ascii_only} ({100*ascii_only/len(df):.1f}%)")

    # Summary statistics for data card
    print(f"\n{'='*80}")
    print("DATA CARD SUMMARY")
    print(f"{'='*80}")
    print(f"Total samples: {len(df):,}")
    print(f"Positive (injection) samples: {label_counts[1]:,} ({100*label_counts[1]/len(df):.1f}%)")
    print(f"Negative (benign) samples: {label_counts[0]:,} ({100*label_counts[0]/len(df):.1f}%)")
    print(f"Unique texts: {unique_texts:,}")
    print(f"Avg text length: {df['text_length'].mean():.0f} characters")
    print(f"\nPrimary language families:")
    for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1])[:5]:
        pct = 100 * count / len(df)
        print(f"  {lang}: {pct:.1f}%")

    # Get ASCII percentage
    ascii_pct = 100 * ascii_only / len(df)
    print(f"\nLanguage composition:")
    print(f"  Primarily Latin-based (ASCII/Latin): {ascii_pct:.1f}%")
    cjk_pct = 100 * lang_counts.get('CJK', 0) / len(df)
    print(f"  Primarily CJK: {cjk_pct:.1f}%")
    mixed_pct = 100 * lang_counts.get('Mixed', 0) / len(df)
    print(f"  Mixed scripts: {mixed_pct:.1f}%")
    other_pct = 100 * (len(df) - ascii_only - lang_counts.get('CJK', 0) - lang_counts.get('Mixed', 0)) / len(df)
    print(f"  Other: {other_pct:.1f}%")

if __name__ == '__main__':
    main()
