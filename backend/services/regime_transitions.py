import os
import pandas as pd
from collections import defaultdict


class RegimeTransitionService:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir

        self.input_path = os.path.join(
            base_dir, "data", "regimes", "regime_dataset.csv"
        )

        self.output_dir = os.path.join(
            base_dir, "data", "regimes"
        )

        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self):
        df = pd.read_csv(self.input_path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df

    def build_transition_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        df["next_regime"] = df["regime_label"].shift(-1)
        df["next_date"] = df["date"].shift(-1)

        transitions = df.dropna(subset=["next_regime"]).copy()

        transitions = transitions[
            ["date", "regime_label", "next_date", "next_regime"]
        ]

        transitions = transitions.rename(
            columns={
                "regime_label": "from_regime",
                "next_regime": "to_regime"
            }
        )

        return transitions

    def compute_transition_matrix(self, transitions: pd.DataFrame) -> pd.DataFrame:
        counts = pd.crosstab(
            transitions["from_regime"],
            transitions["to_regime"]
        )

        probs = counts.div(counts.sum(axis=1), axis=0)

        return probs.round(4)

    def compute_persistence(self, transitions: pd.DataFrame) -> pd.DataFrame:
        persistence = []

        regimes = transitions["from_regime"].unique()

        for regime in regimes:
            subset = transitions[transitions["from_regime"] == regime]

            stay_prob = (
                (subset["to_regime"] == regime).sum() / len(subset)
            )

            persistence.append({
                "regime": regime,
                "stay_probability": round(stay_prob, 4),
                "total_observations": len(subset)
            })

        return pd.DataFrame(persistence)

    def compute_regime_durations(self, df: pd.DataFrame) -> pd.DataFrame:
        durations = []

        current_regime = None
        length = 0

        for _, row in df.iterrows():
            regime = row["regime_label"]

            if regime == current_regime:
                length += 1
            else:
                if current_regime is not None:
                    durations.append({
                        "regime": current_regime,
                        "duration": length
                    })
                current_regime = regime
                length = 1

        if current_regime is not None:
            durations.append({
                "regime": current_regime,
                "duration": length
            })

        duration_df = pd.DataFrame(durations)

        summary = duration_df.groupby("regime")["duration"].agg([
            "mean", "max", "count"
        ]).reset_index()

        summary = summary.rename(columns={
            "mean": "avg_duration",
            "max": "max_duration",
            "count": "num_periods"
        })

        return summary.round(2)

    def run(self):
        print("\n=== Phase 5.5: Regime Transition Analysis ===")

        df = self.load_data()

        transitions = self.build_transition_dataset(df)
        transition_matrix = self.compute_transition_matrix(transitions)
        persistence = self.compute_persistence(transitions)
        durations = self.compute_regime_durations(df)

        transitions.to_csv(
            os.path.join(self.output_dir, "regime_transitions.csv"),
            index=False
        )

        transition_matrix.to_csv(
            os.path.join(self.output_dir, "transition_matrix.csv")
        )

        persistence.to_csv(
            os.path.join(self.output_dir, "regime_persistence.csv"),
            index=False
        )

        durations.to_csv(
            os.path.join(self.output_dir, "regime_durations.csv"),
            index=False
        )

        print("\nTransition Matrix:")
        print(transition_matrix)

        print("\nPersistence:")
        print(persistence)

        print("\nDurations:")
        print(durations)

        print("\nPhase 5.5 completed successfully.")