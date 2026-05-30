from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURE_COLUMNS = ["budget", "runtime", "rating", "genre"]
TARGET_COLUMN = "revenue"


def generate_sample_cinema_data(size: int = 120, random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic cinema data for quick local experimentation."""
    rng = pd.Series(range(size))
    budget = 10_000_000 + (rng * 250_000) % 140_000_000
    runtime = 80 + (rng * 7) % 70
    rating = 5.0 + (rng * 0.11) % 4.8
    genres = ["Action", "Drama", "Comedy", "Animation", "Thriller"]
    genre = pd.Series(genres[i % len(genres)] for i in rng)

    genre_boost = genre.map(
        {"Action": 55_000_000, "Drama": 25_000_000, "Comedy": 30_000_000, "Animation": 45_000_000, "Thriller": 35_000_000}
    )
    revenue = (budget * 1.45) + (rating * 12_000_000) + genre_boost - (runtime * 120_000)

    return pd.DataFrame(
        {
            "budget": budget,
            "runtime": runtime,
            "rating": rating.round(2),
            "genre": genre,
            "revenue": revenue.round(0),
        }
    )


def load_movie_data(csv_path: Path) -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    missing_columns = [column for column in (*FEATURE_COLUMNS, TARGET_COLUMN) if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    return data


def create_model() -> Pipeline:
    numeric_features = ["budget", "runtime", "rating"]
    categorical_features = ["genre"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(n_estimators=200, random_state=42)),
        ]
    )


def analyze_and_train(data: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame]:
    model_data = data.copy()
    model = create_model()

    x_train, x_test, y_train, y_test = train_test_split(
        model_data[FEATURE_COLUMNS],
        model_data[TARGET_COLUMN],
        test_size=0.2,
        random_state=42,
    )

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    metrics = pd.DataFrame(
        {
            "metric": ["MAE", "R2"],
            "value": [mean_absolute_error(y_test, predictions), r2_score(y_test, predictions)],
        }
    )
    return model, metrics


def create_visualizations(data: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    genre_revenue = data.groupby("genre", as_index=False)["revenue"].mean().sort_values("revenue", ascending=False)
    plt.figure(figsize=(9, 4.5))
    plt.bar(genre_revenue["genre"], genre_revenue["revenue"])
    plt.title("Average Revenue by Genre")
    plt.xlabel("Genre")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.savefig(output_dir / "genre_revenue.png")
    plt.close()

    plt.figure(figsize=(8, 4.5))
    scatter = plt.scatter(data["budget"], data["revenue"], c=data["rating"], cmap="viridis", alpha=0.7)
    plt.colorbar(scatter, label="Rating")
    plt.title("Budget vs Revenue")
    plt.xlabel("Budget")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.savefig(output_dir / "budget_vs_revenue.png")
    plt.close()


def run_pipeline(data: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    model, metrics = analyze_and_train(data)
    create_visualizations(data, output_dir)

    enriched = data.copy()
    enriched["predicted_revenue"] = model.predict(data[FEATURE_COLUMNS]).round(0)

    data.describe(include="all").to_csv(output_dir / "dataset_summary.csv")
    metrics.to_csv(output_dir / "model_metrics.csv", index=False)
    enriched.to_csv(output_dir / "movie_predictions.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cinema data analytics and predictive modeling pipeline using pandas and scikit-learn."
    )
    parser.add_argument("--input-csv", type=Path, help="Path to a CSV file with budget, runtime, rating, genre, revenue columns")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for analytics outputs")
    parser.add_argument("--sample-size", type=int, default=120, help="Rows of generated sample data when --input-csv is omitted")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.input_csv:
        data = load_movie_data(args.input_csv)
    else:
        data = generate_sample_cinema_data(size=args.sample_size)

    run_pipeline(data, args.output_dir)
    print(f"Pipeline completed. Outputs saved in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
