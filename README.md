# movie-data-analytics-prediction

Predictive analytics and data visualization project for cinema/movie datasets using **Python**, **Pandas**, and **Scikit-Learn**.

## What this project does

- Performs movie/cinema data analysis with Pandas
- Trains a predictive model for revenue estimation with Scikit-Learn
- Produces data visualizations for genre and budget/revenue trends

## Quick start

```bash
python -m pip install -r requirements.txt
python movie_analytics_pipeline.py --output-dir outputs
```

The command above runs the full machine-learning pipeline on generated sample data.

To run against your own CSV:

```bash
python movie_analytics_pipeline.py --input-csv /path/to/movies.csv --output-dir outputs
```

Required CSV columns:

- `budget`
- `runtime`
- `rating`
- `genre`
- `revenue`

## Generated artifacts

- `dataset_summary.csv` (data-analysis summary)
- `model_metrics.csv` (predictive-modeling metrics)
- `movie_predictions.csv` (actual vs predicted revenue)
- `genre_revenue.png` and `budget_vs_revenue.png` (data-visualization outputs)
