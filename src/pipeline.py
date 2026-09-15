"""
End-to-End Pipeline Orchestrator for Preprocessing & Discretization
Member 3 Role - LLM Hallucination Detection

Usage:
    python -m src.pipeline --k 20 --layer 18
    python -m src.pipeline --k 20 --layer 18 --use-pca --pca-components 0.95
"""

import os
import argparse
from typing import Dict, Any, Optional

from src.preprocessing import (
    load_raw_activations,
    clean_and_prepare_dataset,
    ActivationPreprocessor
)
from src.clustering import KMeansDiscretizer
from src.sequence_builder import (
    build_ordered_sequences,
    export_sequences
)


def run_preprocessing_pipeline(
    activations_dir: str = "data/activations",
    zip_path: str = "activations_50.zip",
    labels_csv_path: str = "data/processed/labeled_generated_responses_50.csv",
    selected_layer: int = 18,
    normalize: bool = True,
    use_pca: bool = False,
    pca_components: float = 0.95,
    k_clusters: int = 20,
    random_state: int = 42,
    output_sequences_dir: str = "data/processed/sequences",
    output_models_dir: str = "models"
) -> Dict[str, Any]:
    """
    Executes the entire Member 3 pipeline:
    1. Load & Validate raw activation vectors
    2. Normalize & (optionally) apply PCA
    3. Fit K-Means Discretizer (default K=20)
    4. Construct ordered temporal sequences
    5. Export sequences and save trained models
    """
    print("=" * 70)
    print("STAGE 1: DATA LOADING & HEALTH CHECKS")
    print("=" * 70)
    raw_records = load_raw_activations(activations_dir=activations_dir, zip_fallback=zip_path)
    X_raw, tokens_df, prompts_df = clean_and_prepare_dataset(
        raw_records=raw_records,
        labels_csv_path=labels_csv_path,
        selected_layer=selected_layer
    )

    print("\n" + "=" * 70)
    print("STAGE 2: NORMALIZATION & DIMENSIONALITY REDUCTION")
    print("=" * 70)
    preprocessor = ActivationPreprocessor(
        normalize=normalize,
        use_pca=use_pca,
        pca_components=pca_components,
        random_state=random_state
    )
    X_processed = preprocessor.fit_transform(X_raw)
    saved_preprocessors = preprocessor.save(output_dir=output_models_dir)
    print(f"Saved preprocessing models: {saved_preprocessors}")

    print("\n" + "=" * 70)
    print(f"STAGE 3: K-MEANS CLUSTERING & DISCRETIZATION (K={k_clusters})")
    print("=" * 70)
    kmeans_discretizer = KMeansDiscretizer(
        n_clusters=k_clusters,
        random_state=random_state
    )
    cluster_labels = kmeans_discretizer.fit_predict(X_processed)
    metrics = kmeans_discretizer.evaluate(X_processed, cluster_labels)
    print(f"Clustering Metrics:")
    print(f"  Inertia: {metrics['inertia']:.2f}")
    print(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
    print(f"  Davies-Bouldin Index: {metrics['davies_bouldin_index']:.4f}")
    print(f"  Calinski-Harabasz Index: {metrics['calinski_harabasz_index']:.2f}")
    print(f"  Min/Max Cluster Sizes: {metrics['min_cluster_size']} / {metrics['max_cluster_size']}")

    kmeans_model_path = os.path.join(output_models_dir, f"kmeans_k{k_clusters}.joblib")
    kmeans_discretizer.save(filepath=kmeans_model_path)

    print("\n" + "=" * 70)
    print("STAGE 4: TEMPORAL SEQUENCE CONSTRUCTION")
    print("=" * 70)
    sequences_df = build_ordered_sequences(
        tokens_df=tokens_df,
        prompts_df=prompts_df,
        cluster_labels=cluster_labels
    )

    print("\nSample Constructed Sequences:")
    for idx, row in sequences_df.head(3).iterrows():
        print(f"  Prompt {row['Prompt_ID']} ({row['Label_Name']}) [len={row['Sequence_Length']}]: {row['Sequence_Str'][:60]}...")

    print("\n" + "=" * 70)
    print("STAGE 5: EXPORT ARTIFACTS FOR PREFIXSPAN MINING")
    print("=" * 70)
    exported_paths = export_sequences(
        sequences_df=sequences_df,
        output_dir=output_sequences_dir,
        prefix=f"sequences_k{k_clusters}"
    )

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)

    return {
        "metrics": metrics,
        "sequences_df": sequences_df,
        "exported_paths": exported_paths,
        "kmeans_model_path": kmeans_model_path
    }


def main():
    parser = argparse.ArgumentParser(description="Member 3: Preprocessing & Discretization Pipeline")
    parser.add_argument("--k", type=int, default=20, help="Number of clusters K (default: 20)")
    parser.add_argument("--layer", type=int, default=18, help="Selected activation layer (default: 18)")
    parser.add_argument("--no-norm", action="store_true", help="Disable standardization")
    parser.add_argument("--use-pca", action="store_true", help="Enable PCA dimensionality reduction")
    parser.add_argument("--pca-components", type=float, default=0.95, help="PCA components or variance ratio (default: 0.95)")
    parser.add_argument("--activations-dir", type=str, default="data/activations", help="Path to raw activations directory")
    parser.add_argument("--zip-path", type=str, default="activations_50.zip", help="Path to fallback activations zip")
    parser.add_argument("--labels-csv", type=str, default="data/processed/labeled_generated_responses_50.csv", help="Path to labels CSV")
    parser.add_argument("--output-sequences", type=str, default="data/processed/sequences", help="Directory to save sequences")
    parser.add_argument("--output-models", type=str, default="models", help="Directory to save models")

    args = parser.parse_args()

    run_preprocessing_pipeline(
        activations_dir=args.activations_dir,
        zip_path=args.zip_path,
        labels_csv_path=args.labels_csv,
        selected_layer=args.layer,
        normalize=not args.no_norm,
        use_pca=args.use_pca,
        pca_components=args.pca_components,
        k_clusters=args.k,
        output_sequences_dir=args.output_sequences,
        output_models_dir=args.output_models
    )


if __name__ == "__main__":
    main()
