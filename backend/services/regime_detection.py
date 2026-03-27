import os
import json
import time
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


PHASE5_FEATURES = [
    "vix_level",
    "high_yield_spread",
    "us_cpi_yoy",
    "yield_spread",
    "spx_vol_3m",
]


def format_elapsed(seconds: float) -> str:
    total_seconds = int(seconds)
    hh = total_seconds // 3600
    mm = (total_seconds % 3600) // 60
    ss = total_seconds % 60
    ms = int((seconds - int(seconds)) * 1000)
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"


class ProgressTimer:
    def __init__(self, stage_name: str):
        self.stage_name = stage_name
        self.start_time = None

    def start(self):
        self.start_time = time.time()
        print(f"[{self.stage_name}] started | elapsed 00:00:00.000")

    def update(self, message: str):
        elapsed = time.time() - self.start_time
        print(f"[{self.stage_name}] {message} | elapsed {format_elapsed(elapsed)}")

    def done(self, message: str = "completed"):
        elapsed = time.time() - self.start_time
        print(f"[{self.stage_name}] {message} | elapsed {format_elapsed(elapsed)}")


@dataclass
class SplitData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


class RegimeDetectionService:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir

        self.features_path = os.path.join(
            base_dir, "data", "features", "features_monthly_full_history.csv"
        )
        self.regimes_dir = os.path.join(base_dir, "data", "regimes")
        self.models_dir = os.path.join(base_dir, "models", "phase5")

        os.makedirs(self.regimes_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

    def load_and_prepare_data(self) -> pd.DataFrame:
        timer = ProgressTimer("DATA_LOAD")
        timer.start()

        df = pd.read_csv(self.features_path)
        timer.update(f"loaded dataset with shape {df.shape}")

        if "date" not in df.columns:
            raise ValueError("Expected a 'date' column in features_monthly_full_history.csv")

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        missing_cols = [col for col in PHASE5_FEATURES if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required Phase 5 columns: {missing_cols}")

        df[PHASE5_FEATURES] = df[PHASE5_FEATURES].ffill()
        before_drop = len(df)
        df = df.dropna(subset=PHASE5_FEATURES).reset_index(drop=True)
        after_drop = len(df)

        timer.update(
            f"forward-filled required columns, dropped {before_drop - after_drop} remaining NaN rows"
        )
        timer.done(f"prepared modelling dataset with shape {df.shape}")
        return df

    def split_data_timewise(
        self, df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15
    ) -> SplitData:
        timer = ProgressTimer("DATA_SPLIT")
        timer.start()

        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

        timer.done(f"train={train_df.shape}, val={val_df.shape}, test={test_df.shape}")
        return SplitData(train=train_df, val=val_df, test=test_df)

    def fit_scaler(self, train_df: pd.DataFrame) -> StandardScaler:
        timer = ProgressTimer("SCALER_TRAIN")
        timer.start()

        scaler = StandardScaler()
        scaler.fit(train_df[PHASE5_FEATURES])

        timer.done("scaler fitted")
        return scaler

    def transform_splits(
        self, scaler: StandardScaler, split_data: SplitData
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        timer = ProgressTimer("SCALER_APPLY")
        timer.start()

        X_train = scaler.transform(split_data.train[PHASE5_FEATURES])
        timer.update(f"transformed train: {X_train.shape}")

        X_val = scaler.transform(split_data.val[PHASE5_FEATURES])
        timer.update(f"transformed validation: {X_val.shape}")

        X_test = scaler.transform(split_data.test[PHASE5_FEATURES])
        timer.update(f"transformed test: {X_test.shape}")

        timer.done("all splits transformed")
        return X_train, X_val, X_test

    def fit_models(self, X_train: np.ndarray) -> Tuple[GaussianMixture, KMeans]:
        timer = ProgressTimer("MODEL_TRAIN")
        timer.start()

        timer.update("fitting GaussianMixture")
        gmm = GaussianMixture(
            n_components=5,
            covariance_type="full",
            random_state=42,
            n_init=10,
            reg_covar=1e-6,
            max_iter=500,
            init_params="kmeans",
            warm_start=False,
        )
        gmm.fit(X_train)
        timer.update(
            f"GaussianMixture fitted | converged={gmm.converged_} | "
            f"n_iter={gmm.n_iter_} | lower_bound={gmm.lower_bound_:.6f}"
        )

        timer.update("fitting KMeans fallback")
        kmeans = KMeans(
            n_clusters=5,
            random_state=42,
            n_init=20,
            max_iter=500,
        )
        kmeans.fit(X_train)
        timer.update(
            f"KMeans fitted | n_iter={kmeans.n_iter_} | inertia={kmeans.inertia_:.6f}"
        )

        timer.done("all regime models trained")
        return gmm, kmeans

    def evaluate_model(self, model, X: np.ndarray, dataset_name: str) -> Dict:
        timer = ProgressTimer(f"{dataset_name.upper()}_EVAL")
        timer.start()

        labels = model.predict(X)
        unique_labels = np.unique(labels)

        if len(unique_labels) < 2:
            metrics = {
                "n_clusters_found": int(len(unique_labels)),
                "silhouette_score": None,
                "calinski_harabasz_score": None,
                "davies_bouldin_score": None,
            }
        else:
            metrics = {
                "n_clusters_found": int(len(unique_labels)),
                "silhouette_score": float(silhouette_score(X, labels)),
                "calinski_harabasz_score": float(calinski_harabasz_score(X, labels)),
                "davies_bouldin_score": float(davies_bouldin_score(X, labels)),
            }

        timer.done(
            f"{dataset_name} evaluation complete | "
            f"clusters={metrics['n_clusters_found']} | "
            f"silhouette={metrics['silhouette_score']} | "
            f"calinski_harabasz={metrics['calinski_harabasz_score']} | "
            f"davies_bouldin={metrics['davies_bouldin_score']}"
        )
        return metrics

    def print_model_metrics(
        self,
        train_metrics_gmm: Dict,
        val_metrics_gmm: Dict,
        test_metrics_gmm: Dict,
        train_metrics_km: Dict,
        val_metrics_km: Dict,
        test_metrics_km: Dict,
        gmm: GaussianMixture,
        X_train: np.ndarray,
    ) -> None:
        print("\n=== PHASE 5 VALIDATION METRICS ===")

        print("\n[GMM]")
        print(
            f"Train      | clusters={train_metrics_gmm['n_clusters_found']} | "
            f"silhouette={train_metrics_gmm['silhouette_score']} | "
            f"CH={train_metrics_gmm['calinski_harabasz_score']} | "
            f"DB={train_metrics_gmm['davies_bouldin_score']}"
        )
        print(
            f"Validation | clusters={val_metrics_gmm['n_clusters_found']} | "
            f"silhouette={val_metrics_gmm['silhouette_score']} | "
            f"CH={val_metrics_gmm['calinski_harabasz_score']} | "
            f"DB={val_metrics_gmm['davies_bouldin_score']}"
        )
        print(
            f"Test       | clusters={test_metrics_gmm['n_clusters_found']} | "
            f"silhouette={test_metrics_gmm['silhouette_score']} | "
            f"CH={test_metrics_gmm['calinski_harabasz_score']} | "
            f"DB={test_metrics_gmm['davies_bouldin_score']}"
        )
        print(f"BIC (train): {gmm.bic(X_train):.6f}")
        print(f"AIC (train): {gmm.aic(X_train):.6f}")

        print("\n[KMeans]")
        print(
            f"Train      | clusters={train_metrics_km['n_clusters_found']} | "
            f"silhouette={train_metrics_km['silhouette_score']} | "
            f"CH={train_metrics_km['calinski_harabasz_score']} | "
            f"DB={train_metrics_km['davies_bouldin_score']}"
        )
        print(
            f"Validation | clusters={val_metrics_km['n_clusters_found']} | "
            f"silhouette={val_metrics_km['silhouette_score']} | "
            f"CH={val_metrics_km['calinski_harabasz_score']} | "
            f"DB={val_metrics_km['davies_bouldin_score']}"
        )
        print(
            f"Test       | clusters={test_metrics_km['n_clusters_found']} | "
            f"silhouette={test_metrics_km['silhouette_score']} | "
            f"CH={test_metrics_km['calinski_harabasz_score']} | "
            f"DB={test_metrics_km['davies_bouldin_score']}"
        )

    def create_regime_mapping(
        self,
        gmm: GaussianMixture,
        scaler: StandardScaler,
    ) -> Tuple[Dict[int, str], pd.DataFrame]:
        timer = ProgressTimer("REGIME_INTERPRETATION")
        timer.start()

        centers_scaled = gmm.means_
        centers_original = scaler.inverse_transform(centers_scaled)

        centers_df = pd.DataFrame(centers_original, columns=PHASE5_FEATURES)
        centers_df["cluster"] = range(len(centers_df))

        mapping = {}
        used_labels = set()

        for _, row in centers_df.iterrows():
            cluster = int(row["cluster"])
            vix = row["vix_level"]
            credit = row["high_yield_spread"]
            inflation = row["us_cpi_yoy"]

            if (
                vix >= centers_df["vix_level"].quantile(0.75)
                and credit >= centers_df["high_yield_spread"].quantile(0.75)
            ):
                label = "crisis"
            elif inflation >= centers_df["us_cpi_yoy"].quantile(0.75):
                label = "inflation_stress"
            elif credit >= centers_df["high_yield_spread"].quantile(0.75):
                label = "credit_stress"
            else:
                label = "calm"

            if label in used_labels:
                alternatives = ["calm", "inflation_stress", "credit_stress", "crisis"]
                for alt in alternatives:
                    if alt not in used_labels:
                        label = alt
                        break

            mapping[cluster] = label
            used_labels.add(label)

        centers_df["regime_label"] = centers_df["cluster"].map(mapping)
        timer.done("regime labels assigned")
        return mapping, centers_df

    def build_regime_dataset(
        self,
        df: pd.DataFrame,
        scaler: StandardScaler,
        gmm: GaussianMixture,
        regime_mapping: Dict[int, str],
    ) -> pd.DataFrame:
        timer = ProgressTimer("REGIME_DATASET_BUILD")
        timer.start()

        X_all = scaler.transform(df[PHASE5_FEATURES])
        cluster_ids = gmm.predict(X_all)
        cluster_probs = gmm.predict_proba(X_all)
        max_probs = cluster_probs.max(axis=1)

        out = df.copy()
        out["regime_cluster"] = cluster_ids
        out["regime_label"] = pd.Series(cluster_ids).map(regime_mapping).values
        out["regime_confidence"] = max_probs

        timer.done(f"regime dataset built with shape {out.shape}")
        return out

    def create_summary(self, regime_df: pd.DataFrame, centers_df: pd.DataFrame) -> pd.DataFrame:
        timer = ProgressTimer("REGIME_SUMMARY")
        timer.start()

        counts = (
            regime_df.groupby(["regime_cluster", "regime_label"])
            .size()
            .reset_index(name="count_months")
        )

        summary = counts.merge(
            centers_df,
            left_on=["regime_cluster", "regime_label"],
            right_on=["cluster", "regime_label"],
            how="left",
        ).drop(columns=["cluster"])

        summary = summary.sort_values("count_months", ascending=False).reset_index(drop=True)

        timer.done("summary created")
        return summary

    def save_pickle(self, obj, path: str):
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    def save_json(self, obj, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)

    def save_artifacts(
        self,
        scaler: StandardScaler,
        gmm: GaussianMixture,
        kmeans: KMeans,
        regime_mapping: Dict[int, str],
        validation_metrics: Dict,
        regime_df: pd.DataFrame,
        summary_df: pd.DataFrame,
        split_data: SplitData,
    ):
        timer = ProgressTimer("SAVE_ARTIFACTS")
        timer.start()

        self.save_pickle(scaler, os.path.join(self.models_dir, "scaler.pkl"))
        timer.update("saved scaler")

        self.save_pickle(gmm, os.path.join(self.models_dir, "gmm_model.pkl"))
        timer.update("saved gmm model")

        self.save_pickle(kmeans, os.path.join(self.models_dir, "kmeans_model.pkl"))
        timer.update("saved kmeans model")

        self.save_json(regime_mapping, os.path.join(self.models_dir, "regime_label_mapping.json"))
        timer.update("saved regime label mapping")

        self.save_json(validation_metrics, os.path.join(self.models_dir, "validation_metrics.json"))
        timer.update("saved validation metrics")

        metadata = {
            "phase": "Phase 5 - Regime Detection",
            "primary_model": "GaussianMixture",
            "fallback_model": "KMeans",
            "n_clusters": 4,
            "feature_columns": PHASE5_FEATURES,
            "train_rows": len(split_data.train),
            "validation_rows": len(split_data.val),
            "test_rows": len(split_data.test),
            "train_date_range": [
                str(split_data.train["date"].min().date()),
                str(split_data.train["date"].max().date()),
            ],
            "validation_date_range": [
                str(split_data.val["date"].min().date()),
                str(split_data.val["date"].max().date()),
            ],
            "test_date_range": [
                str(split_data.test["date"].min().date()),
                str(split_data.test["date"].max().date()),
            ],
            "total_rows_used": int(len(regime_df)),
            "model_retrain_policy": (
                "Retrain full model when new data is appended. "
                "No true online learning for GMM/KMeans."
            ),
        }
        self.save_json(metadata, os.path.join(self.models_dir, "phase5_metadata.json"))
        timer.update("saved metadata")

        regime_df.to_csv(os.path.join(self.regimes_dir, "regime_dataset.csv"), index=False)
        timer.update("saved regime dataset")

        summary_df.to_csv(os.path.join(self.regimes_dir, "regime_summary.csv"), index=False)
        timer.update("saved regime summary")

        timer.done("all artifacts saved")

    def run(self):
        global_timer = ProgressTimer("PHASE5_PIPELINE")
        global_timer.start()

        df = self.load_and_prepare_data()
        split_data = self.split_data_timewise(df)

        scaler = self.fit_scaler(split_data.train)
        X_train, X_val, X_test = self.transform_splits(scaler, split_data)

        gmm, kmeans = self.fit_models(X_train)

        train_metrics_gmm = self.evaluate_model(gmm, X_train, "train_gmm")
        val_metrics_gmm = self.evaluate_model(gmm, X_val, "validation_gmm")
        test_metrics_gmm = self.evaluate_model(gmm, X_test, "test_gmm")

        train_metrics_km = self.evaluate_model(kmeans, X_train, "train_kmeans")
        val_metrics_km = self.evaluate_model(kmeans, X_val, "validation_kmeans")
        test_metrics_km = self.evaluate_model(kmeans, X_test, "test_kmeans")

        self.print_model_metrics(
            train_metrics_gmm=train_metrics_gmm,
            val_metrics_gmm=val_metrics_gmm,
            test_metrics_gmm=test_metrics_gmm,
            train_metrics_km=train_metrics_km,
            val_metrics_km=val_metrics_km,
            test_metrics_km=test_metrics_km,
            gmm=gmm,
            X_train=X_train,
        )

        regime_mapping, centers_df = self.create_regime_mapping(gmm, scaler)
        regime_df = self.build_regime_dataset(df, scaler, gmm, regime_mapping)
        summary_df = self.create_summary(regime_df, centers_df)

        validation_metrics = {
            "gmm": {
                "train": train_metrics_gmm,
                "validation": val_metrics_gmm,
                "test": test_metrics_gmm,
                "bic_train": float(gmm.bic(X_train)),
                "aic_train": float(gmm.aic(X_train)),
                "converged": bool(gmm.converged_),
                "n_iter": int(gmm.n_iter_),
                "lower_bound": float(gmm.lower_bound_),
            },
            "kmeans": {
                "train": train_metrics_km,
                "validation": val_metrics_km,
                "test": test_metrics_km,
                "n_iter": int(kmeans.n_iter_),
                "inertia": float(kmeans.inertia_),
            },
        }

        self.save_artifacts(
            scaler=scaler,
            gmm=gmm,
            kmeans=kmeans,
            regime_mapping=regime_mapping,
            validation_metrics=validation_metrics,
            regime_df=regime_df,
            summary_df=summary_df,
            split_data=split_data,
        )

        print("\n=== Phase 5 completed successfully ===")
        print("Primary model: GaussianMixture")
        print("Fallback model: KMeans")
        print(f"Rows labelled: {len(regime_df)}")
        print(f"Regime output file: {os.path.join(self.regimes_dir, 'regime_dataset.csv')}")
        print(f"Summary output file: {os.path.join(self.regimes_dir, 'regime_summary.csv')}")
        print(f"Model directory: {self.models_dir}")

        global_timer.done("full pipeline complete")