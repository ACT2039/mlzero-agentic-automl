"""
Controlled data noise robustness evaluation framework.
Creates temporary perturbed datasets to evaluate perception resilience,
routing accuracy, error recovery, and success under data quality anomalies.
"""
import csv
import shutil
from pathlib import Path
from typing import Any

from evaluation.cases import get_case
from evaluation.schemas import EvaluationRun


class RobustnessEvaluator:
    """Evaluates pipeline resilience against controlled data noise perturbations."""

    def __init__(self, temp_dir: Path | None = None, use_mock_llm: bool = True):
        self.temp_dir = temp_dir or Path("outputs/robustness_temp")
        self.use_mock_llm = use_mock_llm

    def run_robustness_suite(self, base_case_id: str = "case_01_tiny_cls") -> dict[str, Any]:
        """
        Run a suite of robustness perturbations on a base clean dataset:
        1. missing_values (inject blank cells into features)
        2. malformed_numerics (inject 'N/A' into numeric column)
        3. extra_columns (inject 5 irrelevant random noise columns)
        4. schema_mismatch (remove a feature column in test.csv)
        """
        base_case = get_case(base_case_id)
        if not base_case:
            raise ValueError(f"Base case {base_case_id} not found.")

        self.temp_dir.mkdir(parents=True, exist_ok=True)
        results: dict[str, Any] = {}
        raw_runs_list: list[dict[str, Any]] = []

        perturbations = {
            "missing_values": self._create_missing_values_variant,
            "malformed_numerics": self._create_malformed_numerics_variant,
            "extra_columns": self._create_extra_columns_variant,
            "schema_mismatch": self._create_schema_mismatch_variant,
        }

        from evaluation.runner import execute_case_run

        for pert_name, create_func in perturbations.items():
            variant_path = self.temp_dir / f"variant_{pert_name}"
            try:
                create_func(Path(base_case.dataset_path), variant_path)

                variant_case = base_case.model_copy(
                    update={
                        "case_id": f"{base_case.case_id}_{pert_name}",
                        "name": f"{base_case.name} ({pert_name})",
                        "dataset_path": str(variant_path),
                    }
                )

                run: EvaluationRun = execute_case_run(
                    case=variant_case,
                    run_index=1,
                    use_mock_llm=self.use_mock_llm,
                )

                results[pert_name] = run.model_dump()
                raw_runs_list.append({
                    "noise_type": pert_name,
                    "case_id": variant_case.case_id,
                    "run_id": run.evaluation_id,
                    "success": run.success,
                    "task_type_correct": run.task_type_correct,
                    "library_correct": run.library_correct,
                    "iterations": run.iterations,
                    "total_time": run.total_time,
                    "error_category": run.error_category,
                    "recovery": run.error_recovery_count,
                })
            finally:
                if variant_path.exists():
                    shutil.rmtree(variant_path, ignore_errors=True)

        return {
            "summaries": results,
            "raw_runs": raw_runs_list,
        }

    def _create_missing_values_variant(self, src: Path, dest: Path) -> None:
        shutil.copytree(src, dest, dirs_exist_ok=True)
        train_csv = dest / "train.csv"
        if train_csv.exists():
            rows, headers = self._read_csv(train_csv)
            if rows and len(headers) > 1:
                # Inject missing value in feature1 of 2nd row
                rows[1][headers[0]] = ""
                self._write_csv(train_csv, headers, rows)

    def _create_malformed_numerics_variant(self, src: Path, dest: Path) -> None:
        shutil.copytree(src, dest, dirs_exist_ok=True)
        train_csv = dest / "train.csv"
        if train_csv.exists():
            rows, headers = self._read_csv(train_csv)
            if rows and len(headers) > 1:
                # Inject "N/A" in numeric column
                rows[0][headers[0]] = "N/A"
                self._write_csv(train_csv, headers, rows)

    def _create_extra_columns_variant(self, src: Path, dest: Path) -> None:
        shutil.copytree(src, dest, dirs_exist_ok=True)
        train_csv = dest / "train.csv"
        if train_csv.exists():
            rows, headers = self._read_csv(train_csv)
            if headers:
                extra_headers = headers + ["noise_col1", "noise_col2", "irrelevant_id"]
                for r in rows:
                    r["noise_col1"] = "random_text"
                    r["noise_col2"] = "999"
                    r["irrelevant_id"] = "ABC"
                self._write_csv(train_csv, extra_headers, rows)

    def _create_schema_mismatch_variant(self, src: Path, dest: Path) -> None:
        shutil.copytree(src, dest, dirs_exist_ok=True)
        test_csv = dest / "test.csv"
        if test_csv.exists():
            rows, headers = self._read_csv(test_csv)
            if len(headers) > 2:
                # Drop first feature column in test.csv
                drop_col = headers[0]
                new_headers = [c for c in headers if c != drop_col]
                for r in rows:
                    r.pop(drop_col, None)
                self._write_csv(test_csv, new_headers, rows)

    def _read_csv(self, path: Path) -> tuple[list[dict[str, str]], list[str]]:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = list(reader.fieldnames or [])
            rows = [dict(r) for r in reader]
        return rows, headers

    def _write_csv(self, path: Path, headers: list[str], rows: list[dict[str, str]]) -> None:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
