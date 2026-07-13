"""Dataset integration for the AquaMind AI Prediction Agent (offline training).

Single responsibility: turn the six heterogeneous master datasets into ONE
balanced, enriched training DataFrame for the groundwater-level task -- without
letting the huge groundwater dataset dominate, and without fabricating joins.

Why not just concatenate or merge everything
---------------------------------------------
Only ``master_groundwater_level.csv`` carries the prediction target
(``groundwater_level_m``), and it holds millions of rows, while the other master
datasets hold thousands or fewer. Concatenating is impossible (no shared target)
and a naive merge would (a) bias the model toward the dominant dataset and
(b) risk inventing relationships across datasets that share no reliable key.

Strategy implemented here
-------------------------
1. **Balanced base.** The groundwater dataset is the base (it owns the target).
   It is reduced with GROUPED/STRATIFIED sampling across ``(district, year)`` so
   every district and year is represented and no heavily-monitored location
   dominates -- a statistically representative subset rather than an equal-size
   forced split.
2. **Validated enrichment joins.** Each other dataset is aggregated to a join
   granularity and LEFT-joined onto the base as extra FEATURES (never extra
   rows). Every join is validated: matched rows, unmatched rows and match
   percentage are recorded.
3. **Honest exclusion.** A source whose validated join matches fewer than the
   configured percentage of base rows (no reliable key) is EXCLUDED and the
   reason is documented -- no fabricated relationships.

Output: one unified DataFrame plus an "effective" task describing exactly which
enrichment features survived validation, so the existing FeatureEngineer,
ModelTrainer, ModelEvaluator and ModelRegistry consume it with no changes.

This module trains nothing and predicts nothing.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import pandas as pd

import config  # noqa: E402  (prediction_agent dir is on sys.path via the orchestrator)
from dataset_loader import DatasetLoader  # noqa: E402  (reused base loader)

logger = logging.getLogger("aquamind.prediction.dataset_integrator")

#: Temporary helper columns created during integration and dropped before output.
_DISTRICT_KEY = "district_key"
_YEAR_KEY = "_year_key"


# --------------------------------------------------------------------------- #
# Reports (also serialised into the model metadata)
# --------------------------------------------------------------------------- #

@dataclass
class SourceReport:
    """Row accounting and role for one master dataset."""

    name: str
    total_rows: int
    used_rows: int
    role: str  # 'base' | 'enrichment' | 'excluded'
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class JoinReport:
    """Validation outcome for one enrichment join against the base."""

    left: str
    right: str
    join_keys: list[str]
    join_type: str
    matched_rows: int
    unmatched_rows: int
    match_pct: float
    included: bool

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class IntegrationResult:
    """The unified dataset plus everything needed to train and report on it."""

    frame: pd.DataFrame
    effective_task: object
    source_reports: list[SourceReport] = field(default_factory=list)
    join_reports: list[JoinReport] = field(default_factory=list)
    strategy_description: str = ""
    feature_columns_added: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Enrichment plan
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class _Enrichment:
    """A prepared enrichment source: aggregated features keyed for joining."""

    name: str
    aggregated: pd.DataFrame
    join_keys: list[str]
    feature_columns: list[str]
    source_total_rows: int


class DatasetIntegrator:
    """Builds one balanced, enriched training DataFrame from all master datasets."""

    def __init__(self, master_dir: Path,
                 integration_config: config.IntegrationConfig = config.INTEGRATION_CONFIG,
                 random_state: int = 42) -> None:
        self._master_dir = Path(master_dir)
        self._config = integration_config
        self._random_state = random_state
        self._loader = DatasetLoader(self._master_dir, random_state=random_state)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def integrate(self, base_task, base_sample_target: int | None = None) -> IntegrationResult:
        """Produce the unified training DataFrame and its effective task."""
        target_rows = base_sample_target or self._config.base_sample_target

        source_reports: list[SourceReport] = []
        join_reports: list[JoinReport] = []

        # --- 1. Balanced base (groundwater; the only source with the target) ---
        base_full = self._loader.load(base_task, max_rows=None)
        base_total = self._count_csv_rows(self._master_dir / base_task.dataset_filename)
        base_full[_DISTRICT_KEY] = self._normalize_key(base_full[base_task.categorical_features[0]])
        base = self._stratified_sample(base_full, target_rows)
        base[_YEAR_KEY] = base["year"].round().astype("Int64")
        source_reports.append(SourceReport(
            name=base_task.dataset_filename,
            total_rows=base_total,
            used_rows=len(base),
            role="base",
            note=f"grouped/stratified sample across {list(self._config.stratify_columns)}",
        ))
        logger.info("Base '%s': %d total, %d after balanced sampling",
                    base_task.dataset_filename, base_total, len(base))

        # --- 2. Prepare enrichment sources ---
        enrichments = self._prepare_enrichments()

        # --- 3. Validate + apply each enrichment join ---
        added_features: list[str] = []
        for enrichment in enrichments:
            base, join_report, source_report = self._apply_enrichment(base, enrichment)
            join_reports.append(join_report)
            source_reports.append(source_report)
            if join_report.included:
                added_features.extend(enrichment.feature_columns)

        # --- 4. Clean helper columns ---
        base = base.drop(columns=[_DISTRICT_KEY, _YEAR_KEY], errors="ignore")

        # --- 5. Effective task (base features + surviving enrichment features) ---
        effective_numeric = tuple(base_task.numeric_features) + tuple(added_features)
        effective_task = replace(base_task, numeric_features=effective_numeric)

        strategy = self._describe_strategy(base_task, target_rows, added_features, join_reports)
        logger.info("Integration complete: %d rows, %d enrichment features added",
                    len(base), len(added_features))

        return IntegrationResult(
            frame=base.reset_index(drop=True),
            effective_task=effective_task,
            source_reports=source_reports,
            join_reports=join_reports,
            strategy_description=strategy,
            feature_columns_added=added_features,
        )

    # ------------------------------------------------------------------ #
    # Balancing
    # ------------------------------------------------------------------ #

    def _stratified_sample(self, frame: pd.DataFrame, target_rows: int) -> pd.DataFrame:
        """Grouped/stratified downsample capped per ``stratify_columns`` group.

        A deterministic shuffle followed by a per-group cap gives every group
        (district x year) balanced representation; groups smaller than the cap
        contribute all their rows. This keeps the base representative instead of
        letting a few heavily-monitored stations dominate.
        """
        if len(frame) <= target_rows:
            return frame.reset_index(drop=True)

        keys = list(self._config.stratify_columns)
        group_count = frame.groupby(keys, observed=True).ngroups
        per_group_cap = max(1, -(-target_rows // max(group_count, 1)))  # ceil division

        shuffled = frame.sample(frac=1.0, random_state=self._random_state)
        within_group_rank = shuffled.groupby(keys, observed=True).cumcount()
        capped = shuffled[within_group_rank < per_group_cap]

        if len(capped) > target_rows:  # many small groups overshoot -> trim deterministically
            capped = capped.sample(n=target_rows, random_state=self._random_state)
        logger.info("Stratified sample: %d groups, cap %d/group -> %d rows",
                    group_count, per_group_cap, len(capped))
        return capped.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Enrichment preparation (one aggregated frame per source)
    # ------------------------------------------------------------------ #

    def _prepare_enrichments(self) -> list[_Enrichment]:
        """Build every enrichment source; skip silently only if a file is absent."""
        builders = (
            self._district_features,
            self._firka_features,
            self._rainfall_features,
            self._river_level_features,
            self._river_discharge_features,
        )
        enrichments: list[_Enrichment] = []
        for builder in builders:
            enrichment = builder()
            if enrichment is not None:
                enrichments.append(enrichment)
        return enrichments

    def _district_features(self) -> "_Enrichment | None":
        """District-level assessment averages (stable per-district characteristics)."""
        columns = {
            "rainfall_mm_total": "district_rainfall_mm_total",
            "annual_ground_water_recharge_ham_total": "district_gw_recharge_total_ham",
            "stage_of_ground_water_extraction_percent_total": "district_extraction_stage_pct",
            "net_annual_ground_water_availability_for_future_use_ham_total": "district_net_gw_availability_ham",
        }
        frame = self._read_source("master_district.csv", ["district", *columns])
        if frame is None:
            return None
        total = len(frame)
        frame[_DISTRICT_KEY] = self._normalize_key(frame["district"])
        aggregated = (
            frame.groupby(_DISTRICT_KEY, as_index=False)[list(columns)].mean()
            .rename(columns=columns)
        )
        return _Enrichment("master_district.csv", aggregated, [_DISTRICT_KEY],
                           list(columns.values()), total)

    def _firka_features(self) -> "_Enrichment | None":
        """Per-district share of firkas categorised 'over_exploited' (firka-only signal)."""
        category_col = "categorization_of_assessment_unit_total"
        frame = self._read_source("master_firka.csv", ["district", category_col])
        if frame is None:
            return None
        total = len(frame)
        frame = frame.dropna(subset=[category_col])
        frame[_DISTRICT_KEY] = self._normalize_key(frame["district"])
        frame["_over_exploited"] = (
            frame[category_col].astype(str).str.strip().str.lower().eq("over_exploited")
        )
        aggregated = (
            frame.groupby(_DISTRICT_KEY, as_index=False)["_over_exploited"].mean()
            .rename(columns={"_over_exploited": "firka_over_exploited_ratio"})
        )
        return _Enrichment("master_firka.csv", aggregated, [_DISTRICT_KEY],
                           ["firka_over_exploited_ratio"], total)

    def _rainfall_features(self) -> "_Enrichment | None":
        """Mean recorded rainfall per district-year."""
        return self._district_year_mean(
            "master_rainfall.csv", "rainfall_mm", "rainfall_year_mm")

    def _river_level_features(self) -> "_Enrichment | None":
        """Mean river water level per district-year."""
        return self._district_year_mean(
            "master_river_water_level.csv", "river_water_level_m", "river_level_year_m")

    def _river_discharge_features(self) -> "_Enrichment | None":
        """Mean river discharge per district-year (very sparse; usually excluded)."""
        return self._district_year_mean(
            "master_river_discharge.csv", "river_discharge_m3s", "river_discharge_year_m3s")

    def _district_year_mean(self, filename: str, value_col: str,
                            output_col: str) -> "_Enrichment | None":
        """Aggregate ``value_col`` to a per-(district, year) mean for joining."""
        frame = self._read_source(filename, ["district", "year", value_col])
        if frame is None:
            return None
        total = len(frame)
        frame = frame.dropna(subset=["year", value_col])
        frame[_DISTRICT_KEY] = self._normalize_key(frame["district"])
        frame[_YEAR_KEY] = frame["year"].round().astype("Int64")
        aggregated = (
            frame.groupby([_DISTRICT_KEY, _YEAR_KEY], as_index=False)[value_col].mean()
            .rename(columns={value_col: output_col})
        )
        return _Enrichment(filename, aggregated, [_DISTRICT_KEY, _YEAR_KEY],
                           [output_col], total)

    # ------------------------------------------------------------------ #
    # Validated join
    # ------------------------------------------------------------------ #

    def _apply_enrichment(self, base: pd.DataFrame,
                          enrichment: _Enrichment) -> tuple[pd.DataFrame, JoinReport, SourceReport]:
        """Validate an enrichment join; apply it only if match% clears the threshold."""
        total = len(base)
        merged = base.merge(
            enrichment.aggregated, on=enrichment.join_keys, how="left", indicator="_match",
        )
        matched = int((merged["_match"] == "both").sum())
        unmatched = total - matched
        match_pct = (matched / total * 100.0) if total else 0.0
        included = match_pct >= self._config.min_join_match_pct

        join_report = JoinReport(
            left="base(groundwater)", right=enrichment.name,
            join_keys=list(enrichment.join_keys), join_type="left",
            matched_rows=matched, unmatched_rows=unmatched,
            match_pct=round(match_pct, 2), included=included,
        )

        if included:
            merged = merged.drop(columns="_match")
            note = f"joined on {enrichment.join_keys}; match {match_pct:.2f}%"
            role = "enrichment"
            result_frame = merged
        else:
            note = (f"excluded: join on {enrichment.join_keys} matched only "
                    f"{match_pct:.2f}% of base rows (< {self._config.min_join_match_pct}% "
                    f"threshold); no reliable key")
            role = "excluded"
            result_frame = base  # discard the merge; add no columns

        logger.info("Join base <- %-28s keys=%s match=%.2f%% -> %s",
                    enrichment.name, enrichment.join_keys, match_pct,
                    "INCLUDED" if included else "EXCLUDED")
        source_report = SourceReport(
            name=enrichment.name, total_rows=enrichment.source_total_rows,
            used_rows=len(enrichment.aggregated) if included else 0,
            role=role, note=note,
        )
        return result_frame, join_report, source_report

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _read_source(self, filename: str, columns: list[str]) -> "pd.DataFrame | None":
        """Read selected columns from a master dataset, or ``None`` if unavailable."""
        path = self._master_dir / filename
        if not path.exists():
            logger.warning("Enrichment source missing, skipping: %s", path)
            return None
        try:
            return pd.read_csv(path, usecols=columns)
        except ValueError as error:
            logger.warning("Cannot read %s (%s); skipping enrichment.", filename, error)
            return None

    @staticmethod
    def _normalize_key(series: pd.Series) -> pd.Series:
        """Normalise a district name to a robust join key (upper-case, trimmed)."""
        return series.astype(str).str.strip().str.upper()

    @staticmethod
    def _count_csv_rows(path: Path) -> int:
        """Count data rows (excluding the header) cheaply."""
        if not path.exists():
            return 0
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return max(sum(1 for _ in handle) - 1, 0)

    def _describe_strategy(self, base_task, target_rows: int,
                           added_features: list[str], join_reports: list[JoinReport]) -> str:
        """Human-readable summary of the balancing and join strategy."""
        included = [r.right for r in join_reports if r.included]
        excluded = [r.right for r in join_reports if not r.included]
        parts = [
            f"Base '{base_task.dataset_filename}' downsampled to ~{target_rows} rows via "
            f"grouped/stratified sampling across {list(self._config.stratify_columns)} "
            f"(per-group cap) so no district/year dominates.",
            "Other datasets are aggregated to their join granularity and LEFT-joined as "
            "features only (never adding base rows). Every join is validated by match %.",
        ]
        if included:
            parts.append(f"Included enrichment sources: {', '.join(included)}.")
        if excluded:
            parts.append(f"Excluded (unreliable key / low coverage): {', '.join(excluded)}.")
        return " ".join(parts)
