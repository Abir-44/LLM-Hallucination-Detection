"""
Sequence Builder and Exporter Module
Part of Member 3's pipeline: Construction and Export of Discrete Activation Sequences.
Prepares datasets for PrefixSpan Sequential Pattern Mining (Member 4's role).
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional


def build_ordered_sequences(
    tokens_df: pd.DataFrame,
    prompts_df: pd.DataFrame,
    cluster_labels: np.ndarray
) -> pd.DataFrame:
    """
    Maps each token activation to its discrete cluster state, groups them by Prompt_ID,
    and orders them temporally by generation step (Token_Step).

    Parameters:
    -----------
    tokens_df: DataFrame with Prompt_ID, Token_Step, Response_Label.
    prompts_df: DataFrame with prompt-level metadata (Question, Generated_Response, etc.).
    cluster_labels: 1D numpy array of discrete cluster IDs matching tokens_df rows.

    Returns:
    --------
    sequences_df: pd.DataFrame containing one row per prompt with its ordered discrete state sequence.
    """
    df = tokens_df.copy()
    df["State_ID"] = cluster_labels.astype(int)

    # Sort strictly by Prompt_ID and Token_Step to ensure temporal order
    df = df.sort_values(by=["Prompt_ID", "Token_Step"])

    # Aggregate discrete states into ordered list
    grouped_seqs = (
        df.groupby("Prompt_ID")["State_ID"]
        .apply(list)
        .reset_index()
        .rename(columns={"State_ID": "Sequence"})
    )

    # Merge with prompt-level metadata
    merged_df = pd.merge(prompts_df, grouped_seqs, on="Prompt_ID", how="inner")

    merged_df["Sequence_Length"] = merged_df["Sequence"].apply(len)
    # Format representation like <4, 12, 7, 9, 3>
    merged_df["Sequence_Str"] = merged_df["Sequence"].apply(
        lambda seq: "<" + ", ".join(str(s) for s in seq) + ">"
    )
    merged_df["Label_Name"] = merged_df["Response_Label"].map({0: "Truthful", 1: "Hallucinated"})

    # Sort by Prompt_ID
    merged_df = merged_df.sort_values("Prompt_ID").reset_index(drop=True)

    print(f"[Sequence Construction Summary]")
    print(f"  Total Sequences Built: {len(merged_df)}")
    print(f"  Truthful Sequences (0): {sum(merged_df['Response_Label'] == 0)}")
    print(f"  Hallucinated Sequences (1): {sum(merged_df['Response_Label'] == 1)}")
    print(f"  Min Length: {merged_df['Sequence_Length'].min()}")
    print(f"  Max Length: {merged_df['Sequence_Length'].max()}")
    print(f"  Mean Length: {merged_df['Sequence_Length'].mean():.2f}")

    return merged_df


def to_spmf_format(sequence: List[int]) -> str:
    """
    Converts a Python list of state IDs into SPMF sequence format.
    In SPMF format:
    - Each itemset ends with -1
    - Each sequence ends with -2
    Example: [4, 12, 7] -> "4 -1 12 -1 7 -1 -2"
    """
    items_part = " -1 ".join(str(s) for s in sequence)
    return f"{items_part} -1 -2"


def export_sequences(
    sequences_df: pd.DataFrame,
    output_dir: str = "data/processed/sequences",
    prefix: str = "sequences_k20"
) -> Dict[str, str]:
    """
    Exports the constructed sequences into multiple formats tailored for downstream tasks:
    1. CSV file with rich metadata (Question, Response, Label, Sequence)
    2. JSON file for structured Python consumption
    3. SPMF text files (all, truthful only, hallucinated only) for standard PrefixSpan / SPMF tools

    Returns:
    --------
    paths: Dictionary of created file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = {}

    # 1. Export Full CSV
    csv_path = os.path.join(output_dir, f"{prefix}.csv")
    export_df = sequences_df.copy()
    # Serialize sequence list to string for clean CSV representation
    export_df["Sequence_List"] = export_df["Sequence"].apply(lambda x: json.dumps(x))
    export_df.to_csv(csv_path, index=False)
    paths["csv"] = csv_path
    print(f"Exported tabular sequences to: '{csv_path}'")

    # 2. Export Structured JSON
    json_path = os.path.join(output_dir, f"{prefix}.json")
    json_records = []
    for _, row in sequences_df.iterrows():
        json_records.append({
            "prompt_id": int(row["Prompt_ID"]),
            "response_label": int(row["Response_Label"]),
            "label_name": row["Label_Name"],
            "sequence_length": int(row["Sequence_Length"]),
            "sequence": row["Sequence"],
            "sequence_str": row["Sequence_Str"],
            "question": row["Question"],
            "generated_response": row["Generated_Response"]
        })

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_records, f, indent=2, ensure_ascii=False)
    paths["json"] = json_path
    print(f"Exported JSON sequences to: '{json_path}'")

    # 3. Export SPMF / PrefixSpan format text files
    # All sequences
    all_spmf_path = os.path.join(output_dir, f"{prefix}_all_spmf.txt")
    with open(all_spmf_path, "w", encoding="utf-8") as f:
        for seq in sequences_df["Sequence"]:
            f.write(to_spmf_format(seq) + "\n")
    paths["spmf_all"] = all_spmf_path

    # Truthful sequences only (Label 0)
    truthful_df = sequences_df[sequences_df["Response_Label"] == 0]
    truthful_spmf_path = os.path.join(output_dir, "truthful_sequences_spmf.txt")
    with open(truthful_spmf_path, "w", encoding="utf-8") as f:
        for seq in truthful_df["Sequence"]:
            f.write(to_spmf_format(seq) + "\n")
    paths["spmf_truthful"] = truthful_spmf_path

    # Hallucinated sequences only (Label 1)
    hallucinated_df = sequences_df[sequences_df["Response_Label"] == 1]
    hallucinated_spmf_path = os.path.join(output_dir, "hallucinated_sequences_spmf.txt")
    with open(hallucinated_spmf_path, "w", encoding="utf-8") as f:
        for seq in hallucinated_df["Sequence"]:
            f.write(to_spmf_format(seq) + "\n")
    paths["spmf_hallucinated"] = hallucinated_spmf_path

    print(f"Exported SPMF format files:")
    print(f"  All: '{all_spmf_path}'")
    print(f"  Truthful (0): '{truthful_spmf_path}' ({len(truthful_df)} sequences)")
    print(f"  Hallucinated (1): '{hallucinated_spmf_path}' ({len(hallucinated_df)} sequences)")

    return paths
