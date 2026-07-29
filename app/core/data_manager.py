"""
DataManager owns all loaded dataframes for a session: validation, profiling,
schema summaries used to ground the LLM, and basic data-quality checks.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetProfile:
    name: str
    n_rows: int
    n_cols: int
    columns: List[str]
    dtypes: Dict[str, str]
    null_counts: Dict[str, int]
    numeric_summary: Dict[str, dict]
    sample: List[dict]
    quality_warnings: List[str] = field(default_factory=list)


class DataValidationError(Exception):
    pass


class DataManager:
    """Holds one or more named dataframes for the current session."""

    def __init__(self):
        self.frames: Dict[str, pd.DataFrame] = {}
        self.profiles: Dict[str, DatasetProfile] = {}

    # ------------------------------------------------------------------ #
    # Loading & validation
    # ------------------------------------------------------------------ #
    def load_csv(self, name: str, file_bytes: bytes) -> DatasetProfile:
        """Load, validate, and profile a CSV file. Raises DataValidationError
        on unrecoverable problems (empty file, unparsable, etc.)."""
        if not file_bytes:
            raise DataValidationError(f"'{name}' is empty.")

        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
        except Exception as e:
            # Retry with a couple of common fallbacks before giving up
            for sep in [";", "\t"]:
                try:
                    df = pd.read_csv(io.BytesIO(file_bytes), sep=sep)
                    if df.shape[1] > 1:
                        break
                except Exception:
                    continue
            else:
                raise DataValidationError(
                    f"Could not parse '{name}' as CSV: {e}"
                ) from e

        if df.empty or df.shape[1] == 0:
            raise DataValidationError(f"'{name}' has no readable columns/rows.")

        df = self._clean_columns(df)
        df = self._infer_types(df)

        key = self._safe_key(name)
        self.frames[key] = df
        profile = self._profile(key, df)
        self.profiles[key] = profile
        logger.info(f"Loaded dataset '{key}' shape={df.shape}")
        return profile

    @staticmethod
    def _safe_key(name: str) -> str:
        base = name.rsplit(".", 1)[0]
        return "".join(c if c.isalnum() else "_" for c in base).strip("_").lower() or "dataset"

    @staticmethod
    def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [str(c).strip().replace(" ", "_").lower() for c in df.columns]
        # drop fully-empty unnamed columns often produced by trailing commas
        df = df.loc[:, ~df.columns.str.match(r"^unnamed.*$", na=False) | df.notna().any()]
        return df

    @staticmethod
    def _infer_types(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.columns:
            if df[col].dtype == object:
                # try datetime
                parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
                if parsed.notna().mean() > 0.8:
                    df[col] = parsed
                    continue
                # try numeric
                numeric = pd.to_numeric(
                    df[col].astype(str).str.replace(",", "", regex=False), errors="coerce"
                )
                if numeric.notna().mean() > 0.9:
                    df[col] = numeric
        return df

    # ------------------------------------------------------------------ #
    # Profiling / data quality
    # ------------------------------------------------------------------ #
    def _profile(self, key: str, df: pd.DataFrame) -> DatasetProfile:
        warnings: List[str] = []

        null_counts = df.isna().sum().to_dict()
        heavy_nulls = [c for c, n in null_counts.items() if n / len(df) > 0.3]
        if heavy_nulls:
            warnings.append(f"Columns with >30% missing values: {heavy_nulls}")

        dup_count = int(df.duplicated().sum())
        if dup_count:
            warnings.append(f"{dup_count} fully duplicated rows detected.")

        numeric_summary = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            desc = df[col].describe()
            numeric_summary[col] = {k: (None if pd.isna(v) else round(float(v), 4)) for k, v in desc.items()}
            if (df[col] < 0).any() and col not in ("profit",):
                warnings.append(f"Column '{col}' contains negative values — verify this is expected.")

        dtypes = {c: str(t) for c, t in df.dtypes.items()}
        sample = df.head(5).replace({np.nan: None}).to_dict(orient="records")

        return DatasetProfile(
            name=key,
            n_rows=len(df),
            n_cols=df.shape[1],
            columns=list(df.columns),
            dtypes=dtypes,
            null_counts={k: int(v) for k, v in null_counts.items()},
            numeric_summary=numeric_summary,
            sample=sample,
            quality_warnings=warnings,
        )

    # ------------------------------------------------------------------ #
    # Grounding context for the LLM
    # ------------------------------------------------------------------ #
    def schema_context(self) -> str:
        """A compact text description of all loaded datasets, used to ground
        the LLM so it knows column names without seeing raw data. Kept
        deliberately minimal — full detail (dtypes, sample rows, quality
        warnings) is available on demand via the get_dataset_info tool."""
        parts = []
        for key, profile in self.profiles.items():
            parts.append(
                f"Dataset '{key}' ({profile.n_rows} rows): {', '.join(profile.columns)}"
            )
        return "\n".join(parts) if parts else "No datasets loaded yet."

    def get(self, key: str) -> Optional[pd.DataFrame]:
        return self.frames.get(key)

    def all_frames(self) -> Dict[str, pd.DataFrame]:
        return self.frames

    def is_empty(self) -> bool:
        return len(self.frames) == 0
