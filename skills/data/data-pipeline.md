---
description: Build data pipelines, ETL processes, and data transformations
triggers: data, pipeline, etl, transform, pandas, dataframe
---

# Data Pipeline Skill

## Stack
- **Polars** (fast) or **Pandas** (familiar) for transformations
- **DuckDB** for SQL analytics on files
- **Apache Airflow** or **Prefect** for orchestration
- **Pydantic** for data validation

## Template: ETL Pipeline
```python
import polars as pl
from pathlib import Path

def extract(source: str) -> pl.DataFrame:
    """Extract data from source."""
    if source.endswith(".csv"):
        return pl.read_csv(source)
    elif source.endswith(".parquet"):
        return pl.read_parquet(source)
    elif source.startswith("http"):
        import requests
        resp = requests.get(source)
        return pl.read_json(resp.text)

def transform(df: pl.DataFrame) -> pl.DataFrame:
    """Apply transformations."""
    return (
        df.lazy()
        .filter(pl.col("status") == "active")
        .with_columns([
            (pl.col("revenue") / pl.col("users")).alias("arpu"),
            pl.col("date").str.to_datetime().alias("parsed_date"),
        ])
        .group_by("region")
        .agg([
            pl.col("revenue").sum().alias("total_revenue"),
            pl.col("arpu").mean().alias("avg_arpu"),
        ])
        .collect()
    )

def load(df: pl.DataFrame, target: str):
    """Load data to target."""
    df.write_parquet(target)

# Run pipeline
df = extract("data/raw/sales.csv")
df = transform(df)
load(df, "data/processed/sales_summary.parquet")
```

## Performance Tips
- Use Polars over Pandas (10-100x faster)
- Process in chunks for large files: `pl.read_csv_batched()`
- Use Parquet over CSV (columnar, compressed)
- DuckDB for SQL queries on DataFrames: `duckdb.sql("SELECT * FROM df")`
