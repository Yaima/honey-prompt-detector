#!/usr/bin/env python3

import os
import pandas as pd
from datasets import load_dataset


def clean_text(text: str) -> str:
    """
    Removes any '####' substrings and strips extra whitespace from a given string.

    :param text: The text to be cleaned.
    :return: The cleaned text, free of '####' and trimmed of leading/trailing whitespace.
    """
    if not isinstance(text, str):
        return ""
    # Remove all occurrences of '####'
    text = text.replace("####", "")
    # Strip leading/trailing whitespace
    return text.strip()


def unify_datasets():
    """
    Unifies multiple datasets (both local and remote) into a single CSV file:
      1) Loads both train/test splits from deepset/prompt-injections.
      2) (Commented out) Optionally loads SPML_Chatbot_Prompt_Injection (skipped per your instructions).
      3) Loads xTRam1/safe-guard-prompt-injection (train/test).
      4) Loads local CSVs in the data directory, assigning labels based on filename rules.
      5) Loads local Parquet files in the data directory. (No default labels; must be in column.)
      6) Combines them, cleans text, removes duplicates, and saves to "unified_dataset.csv".

    NOTE: The labeling logic for CSV files is simple:
        - If filename includes "jailbreak" or "malicious_prompts" => label = 1
        - If filename includes "prompts_conversation" => label = 0
        - Otherwise, skip that file.
    """

    # -------------------------------------------------------------------------
    # 1) Load deepset/prompt-injections (train + test)
    # -------------------------------------------------------------------------
    print("[DEBUG] Loading 'deepset/prompt-injections' (train & test)...")

    ds_deepset_train = load_dataset("deepset/prompt-injections", split="train")
    ds_deepset_test = load_dataset("deepset/prompt-injections", split="test")

    df_deepset_train = ds_deepset_train.to_pandas()
    df_deepset_test = ds_deepset_test.to_pandas()

    df_deepset = pd.concat([df_deepset_train, df_deepset_test], ignore_index=True)

    if "text" not in df_deepset.columns:
        # Attempt to find a likely column containing text
        possible_text_cols = [col for col in df_deepset.columns if "text" in col.lower()]
        if possible_text_cols:
            df_deepset.rename(columns={possible_text_cols[0]: "text"}, inplace=True)
        else:
            print("[DEBUG] No 'text' column found in deepset. Creating empty 'text' column.")
            df_deepset["text"] = ""

    if "label" not in df_deepset.columns:
        # If no label column is present, set to None
        df_deepset["label"] = None

    df_deepset.dropna(subset=["text"], inplace=True)
    df_deepset["text"] = df_deepset["text"].apply(clean_text)
    df_deepset = df_deepset[df_deepset["text"] != ""]
    df_deepset["source"] = "hf_deepset_prompt_injections"

    # Keep only the columns of interest
    df_deepset = df_deepset[["text", "label", "source"]]

    # -------------------------------------------------------------------------
    # 2) (Optional/Commented out) Load SPML_Chatbot_Prompt_Injection
    #    - Skipped per your instructions
    # -------------------------------------------------------------------------
    # (Code intentionally omitted here)

    # -------------------------------------------------------------------------
    # 3) Load xTRam1/safe-guard-prompt-injection (train + test)
    # -------------------------------------------------------------------------
    print("[DEBUG] Loading 'xTRam1/safe-guard-prompt-injection' (train & test)...")

    ds_safeguard_train = load_dataset("xTRam1/safe-guard-prompt-injection", split="train")
    ds_safeguard_test = load_dataset("xTRam1/safe-guard-prompt-injection", split="test")

    df_safeguard_train = ds_safeguard_train.to_pandas()
    df_safeguard_test = ds_safeguard_test.to_pandas()

    df_safeguard = pd.concat([df_safeguard_train, df_safeguard_test], ignore_index=True)

    if "text" not in df_safeguard.columns:
        # Attempt to find a likely column containing text
        possible_text_cols = [col for col in df_safeguard.columns if "text" in col.lower()]
        if possible_text_cols:
            df_safeguard.rename(columns={possible_text_cols[0]: "text"}, inplace=True)
        else:
            print("[DEBUG] No 'text' column found in safeguard dataset. Creating empty 'text' column.")
            df_safeguard["text"] = ""

    if "label" not in df_safeguard.columns:
        df_safeguard["label"] = None

    df_safeguard.dropna(subset=["text"], inplace=True)
    df_safeguard["text"] = df_safeguard["text"].apply(clean_text)
    df_safeguard = df_safeguard[df_safeguard["text"] != ""]
    df_safeguard["source"] = "hf_safeguard_prompt_injection"

    df_safeguard = df_safeguard[["text", "label", "source"]]

    # -------------------------------------------------------------------------
    # 4) Local CSV files: label by filename
    # -------------------------------------------------------------------------
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.join(script_dir, "..", "data")

    local_csvs = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv"))
    dfs_local = []

    for csv_file in local_csvs:
        filename_lower = csv_file.lower()

        # Skip files containing 'act_as'
        if "act_as" in filename_lower:
            print(f"[DEBUG] Skipping '{csv_file}' (contains 'act_as').")
            continue

        # Decide label from filename
        if "jailbreak" in filename_lower or "malicious_prompts" in filename_lower:
            assigned_label = 1
        elif "prompts_conversation" in filename_lower:
            assigned_label = 0
        else:
            print(f"[DEBUG] Skipping '{csv_file}' (no matching rule).")
            continue

        path = os.path.join(data_dir, csv_file)
        print(f"[DEBUG] Reading {csv_file} => label={assigned_label}")

        try:
            df_local_temp = pd.read_csv(path, header=None, encoding="utf-8", keep_default_na=False)
        except Exception as e:
            print(f"[DEBUG] Error reading '{csv_file}': {e}")
            continue

        if df_local_temp.empty:
            print(f"[DEBUG] Warning: '{csv_file}' is empty. Skipping.")
            continue

        # Expecting single-column text
        df_local_temp = df_local_temp.iloc[:, :1]
        df_local_temp.columns = ["text"]

        df_local_temp["text"] = df_local_temp["text"].astype(str).apply(clean_text)
        df_local_temp = df_local_temp[df_local_temp["text"] != ""]

        df_local_temp["label"] = assigned_label
        df_local_temp["source"] = csv_file
        dfs_local.append(df_local_temp)

    if dfs_local:
        df_local_combined = pd.concat(dfs_local, ignore_index=True)
    else:
        df_local_combined = pd.DataFrame(columns=["text", "label", "source"])

    # -------------------------------------------------------------------------
    # 5) Local Parquet files
    # -------------------------------------------------------------------------
    parquet_files = [f for f in os.listdir(data_dir) if f.endswith(".parquet")]
    dfs_parquet = []

    for parquet_file in parquet_files:
        file_path = os.path.join(data_dir, parquet_file)
        try:
            df_parquet_temp = pd.read_parquet(file_path)
        except Exception as e:
            print(f"[DEBUG] Error reading parquet '{parquet_file}': {e}")
            continue

        # Optional: Ensure 'text' is present
        if "text" not in df_parquet_temp.columns:
            # Attempt to find a text col
            possible_text_cols = [col for col in df_parquet_temp.columns if "text" in col.lower()]
            if possible_text_cols:
                df_parquet_temp.rename(columns={possible_text_cols[0]: "text"}, inplace=True)
            else:
                print(f"[DEBUG] No 'text' column found in '{parquet_file}'. Creating empty 'text' column.")
                df_parquet_temp["text"] = ""

        df_parquet_temp["text"] = df_parquet_temp["text"].astype(str).apply(clean_text)
        df_parquet_temp = df_parquet_temp[df_parquet_temp["text"] != ""]

        # Append a source column
        df_parquet_temp["source"] = parquet_file
        dfs_parquet.append(df_parquet_temp)

    if dfs_parquet:
        df_parquet_combined = pd.concat(dfs_parquet, ignore_index=True)
    else:
        df_parquet_combined = pd.DataFrame(columns=["text", "label", "source"])

    # -------------------------------------------------------------------------
    # 6) Combine everything
    # -------------------------------------------------------------------------
    frames = [df_deepset, df_safeguard]

    # For clarity, add the merged local CSV data if not empty
    if not df_local_combined.empty:
        frames.append(df_local_combined)

    # For clarity, add merged parquet data if not empty
    if not df_parquet_combined.empty:
        frames.append(df_parquet_combined)

    # If none of these are available, bail out
    if not frames:
        print("[DEBUG] No data found. Exiting.")
        return

    df_all = pd.concat(frames, ignore_index=True)

    # Final cleanup
    df_all.dropna(subset=["text"], inplace=True)
    df_all = df_all[df_all["text"] != ""]
    # Remove duplicates based on the combination of text + label
    df_all.drop_duplicates(subset=["text", "label"], inplace=True)

    # -------------------------------------------------------------------------
    # Summaries + save
    # -------------------------------------------------------------------------
    total_rows = len(df_all)
    mal_count = (df_all["label"] == 1).sum()
    ben_count = (df_all["label"] == 0).sum()

    print(f"\n[DEBUG] Final unified dataset: {total_rows} total rows.")
    print(f"[DEBUG] Malicious (label=1): {mal_count}, Benign (label=0): {ben_count}")

    output_path = os.path.join(data_dir, "unified_dataset.csv")
    df_all.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[DEBUG] Saved unified dataset to '{output_path}'")


def main():
    unify_datasets()


if __name__ == "__main__":
    main()
