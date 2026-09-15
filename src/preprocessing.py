"""
Preprocessing and Data Loading Module
Part of Member 3's pipeline: LLM Internal Activation Preprocessing & Discretization.
"""

import os
import zipfile
import io
import torch
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def extract_activations_archive(
    zip_path: str = "activations_50.zip",
    target_dir: str = "data/activations"
) -> str:
    """
    Extracts activations zip archive into target directory if not already present.
    """
    os.makedirs(target_dir, exist_ok=True)
    existing_pts = [f for f in os.listdir(target_dir) if f.endswith(".pt")]
    if len(existing_pts) >= 50:
        return target_dir

    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Activations zip archive not found at: {zip_path}")

    print(f"Extracting '{zip_path}' to '{target_dir}'...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(target_dir)
    print(f"Extraction complete. {len(os.listdir(target_dir))} files ready in '{target_dir}'.")
    return target_dir


def load_raw_activations(
    activations_dir: str = "data/activations",
    zip_fallback: Optional[str] = "activations_50.zip"
) -> List[Dict]:
    """
    Loads all activation .pt files from directory (or extracts from zip if needed).
    Returns a list of dictionaries with raw activation metadata and tensors.
    """
    if not os.path.exists(activations_dir) or len([f for f in os.listdir(activations_dir) if f.endswith(".pt")]) == 0:
        if zip_fallback and os.path.exists(zip_fallback):
            extract_activations_archive(zip_fallback, activations_dir)
        else:
            raise FileNotFoundError(f"Directory '{activations_dir}' does not exist and no fallback zip found.")

    pt_files = sorted([f for f in os.listdir(activations_dir) if f.endswith(".pt")])
    if not pt_files:
        raise ValueError(f"No .pt activation files found in '{activations_dir}'.")

    records = []
    for fname in pt_files:
        filepath = os.path.join(activations_dir, fname)
        try:
            # PyTorch 2.6+ defaults weights_only=True, set False for dictionary of tensors
            data = torch.load(filepath, map_location="cpu", weights_only=False)
            data["_filename"] = fname
            records.append(data)
        except Exception as e:
            print(f"[Warning] Failed to load '{fname}': {e}")

    print(f"Loaded {len(records)} raw activation files from '{activations_dir}'.")
    return records


def clean_and_prepare_dataset(
    raw_records: List[Dict],
    labels_csv_path: str = "data/processed/labeled_generated_responses_50.csv",
    selected_layer: Union[int, str] = 18
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Validates, filters invalid/incomplete records, and extracts token-level activation matrix
    along with prompt-level and token-level metadata DataFrames.

    Parameters:
    -----------
    raw_records: List of loaded activation dictionaries.
    labels_csv_path: Path to CSV with response-level labels (0=Truthful, 1=Hallucinated).
    selected_layer: Layer index (e.g. 10, 18, 26), 'concat' for all layers concatenated, or 'mean'.

    Returns:
    --------
    X_matrix: np.ndarray of shape (N_tokens, feature_dim)
    tokens_df: pd.DataFrame with metadata for each token row in X_matrix
    prompts_df: pd.DataFrame with prompt-level metadata and labels
    """
    if not os.path.exists(labels_csv_path):
        # Fallback to root if not found in data/processed/
        alt_path = os.path.basename(labels_csv_path)
        if os.path.exists(alt_path):
            labels_csv_path = alt_path
        else:
            raise FileNotFoundError(f"Labels CSV not found at '{labels_csv_path}'.")

    labels_df = pd.read_csv(labels_csv_path)
    # Ensure Prompt_ID is integer
    labels_df["Prompt_ID"] = labels_df["Prompt_ID"].astype(int)
    labels_map = labels_df.set_index("Prompt_ID").to_dict(orient="index")

    valid_prompts = []
    token_rows = []
    activation_vectors = []

    for rec in raw_records:
        prompt_id = int(rec.get("prompt_id", -1))
        if prompt_id not in labels_map:
            print(f"[Data Cleaning] Prompt ID {prompt_id} not found in labels CSV. Dropping.")
            continue

        activations_dict = rec.get("activations", {})
        if not activations_dict:
            print(f"[Data Cleaning] Prompt ID {prompt_id} contains empty activations. Dropping.")
            continue

        # Determine feature vectors for the requested layer
        available_layers = list(activations_dict.keys())
        if isinstance(selected_layer, int):
            if selected_layer not in activations_dict:
                print(f"[Data Cleaning] Selected layer {selected_layer} missing for Prompt {prompt_id}. Available: {available_layers}. Dropping.")
                continue
            act_tensor = activations_dict[selected_layer]
        elif selected_layer == "concat":
            sorted_layers = sorted(available_layers)
            act_tensor = torch.cat([activations_dict[l] for l in sorted_layers], dim=-1)
        elif selected_layer == "mean":
            stacked = torch.stack([activations_dict[l] for l in available_layers], dim=0)
            act_tensor = torch.mean(stacked, dim=0)
        else:
            raise ValueError(f"Invalid selected_layer specification: {selected_layer}")

        # Check for NaN, Inf, or empty tensor
        if torch.isnan(act_tensor).any() or torch.isinf(act_tensor).any():
            print(f"[Data Cleaning] NaN or Inf detected in activations for Prompt ID {prompt_id}. Dropping.")
            continue

        num_tokens = act_tensor.shape[0]
        if num_tokens == 0:
            print(f"[Data Cleaning] 0 tokens generated for Prompt ID {prompt_id}. Dropping.")
            continue

        # Convert to numpy float32
        act_np = act_tensor.detach().cpu().numpy().astype(np.float32)

        prompt_info = labels_map[prompt_id]
        response_label = int(prompt_info.get("Response_Label", 0))
        question = prompt_info.get("Question", rec.get("question", ""))
        gen_resp = prompt_info.get("Generated_Response", rec.get("generated_response", ""))

        valid_prompts.append({
            "Prompt_ID": prompt_id,
            "Response_Label": response_label,
            "Question": question,
            "Generated_Response": gen_resp,
            "Num_Tokens": num_tokens,
            "Selected_Layer": str(selected_layer),
            "Feature_Dim": act_np.shape[1],
            "Filename": rec.get("_filename", "")
        })

        # Append token level metadata and vectors
        for step in range(num_tokens):
            token_rows.append({
                "Prompt_ID": prompt_id,
                "Token_Step": step,
                "Response_Label": response_label
            })
            activation_vectors.append(act_np[step])

    if not activation_vectors:
        raise ValueError("No valid activation vectors remained after cleaning!")

    X_matrix = np.vstack(activation_vectors)
    tokens_df = pd.DataFrame(token_rows)
    prompts_df = pd.DataFrame(valid_prompts)

    print(f"[Data Cleaning Summary]")
    print(f"  Valid Prompts: {len(prompts_df)} / {len(raw_records)}")
    print(f"  Truthful Prompts (Label 0): {sum(prompts_df['Response_Label'] == 0)}")
    print(f"  Hallucinated Prompts (Label 1): {sum(prompts_df['Response_Label'] == 1)}")
    print(f"  Total Token Vectors Extracted: {X_matrix.shape[0]}")
    print(f"  Vector Dimensionality: {X_matrix.shape[1]}")

    return X_matrix, tokens_df, prompts_df


class ActivationPreprocessor:
    """
    Handles normalization / standardization and optional PCA dimensionality reduction.
    Saves and loads transformation state for inference and early warning pipelines.
    """

    def __init__(
        self,
        normalize: bool = True,
        use_pca: bool = False,
        pca_components: Union[int, float, None] = 0.95,
        random_state: int = 42
    ):
        self.normalize = normalize
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.random_state = random_state

        self.scaler: Optional[StandardScaler] = StandardScaler() if normalize else None
        self.pca: Optional[PCA] = None
        if self.use_pca:
            self.pca = PCA(n_components=self.pca_components, random_state=self.random_state)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fits scaler and optional PCA on X, and transforms X.
        """
        X_out = X.copy()
        if self.scaler is not None:
            X_out = self.scaler.fit_transform(X_out)

        if self.pca is not None:
            X_out = self.pca.fit_transform(X_out)
            print(f"[PCA] Reduced dimensions from {X.shape[1]} to {X_out.shape[1]} (Explained variance ratio sum: {np.sum(self.pca.explained_variance_ratio_):.4f})")

        return X_out

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Applies fitted transformations to unseen activation vectors.
        """
        X_out = X.copy()
        if self.scaler is not None:
            X_out = self.scaler.transform(X_out)
        if self.pca is not None:
            X_out = self.pca.transform(X_out)
        return X_out

    def save(self, output_dir: str = "models") -> Dict[str, str]:
        """
        Saves scaler and PCA models to disk using joblib.
        """
        os.makedirs(output_dir, exist_ok=True)
        paths = {}
        if self.scaler is not None:
            scaler_path = os.path.join(output_dir, "scaler.joblib")
            joblib.dump(self.scaler, scaler_path)
            paths["scaler"] = scaler_path
        if self.pca is not None:
            pca_path = os.path.join(output_dir, "pca.joblib")
            joblib.dump(self.pca, pca_path)
            paths["pca"] = pca_path
        return paths

    @classmethod
    def load(cls, models_dir: str = "models") -> "ActivationPreprocessor":
        """
        Loads fitted scaler and PCA from models_dir.
        """
        instance = cls(normalize=False, use_pca=False)
        scaler_path = os.path.join(models_dir, "scaler.joblib")
        pca_path = os.path.join(models_dir, "pca.joblib")

        if os.path.exists(scaler_path):
            instance.scaler = joblib.load(scaler_path)
            instance.normalize = True
        if os.path.exists(pca_path):
            instance.pca = joblib.load(pca_path)
            instance.use_pca = True

        return instance
