"""
Ablation studies runner for MLZero Evaluation Framework.
Includes Semantic Memory, Episodic Memory, Execution Judge, and Retrieval Top-K ablations.
"""
from typing import Any

from evaluation.aggregator import aggregate_case_runs
from evaluation.cases import get_case
from evaluation.schemas import EvaluationRun


class AblationRunner:
    """Runs ablation experiments by toggling component switches."""

    def __init__(self, use_mock_llm: bool = True):
        self.use_mock_llm = use_mock_llm
        self.raw_runs: list[dict[str, Any]] = []

    def run_full_ablation_suite(self, case_ids: list[str] | None = None, runs_per_config: int = 1) -> dict[str, Any]:
        """Run all ablation studies (Semantic Memory, Episodic Memory, Execution Judge, Retrieval K)."""
        self.raw_runs = []
        
        sem_res = self.run_semantic_memory_ablation(case_ids=case_ids, runs_per_config=runs_per_config)
        epi_res = self.run_episodic_memory_ablation(case_ids=case_ids, runs_per_config=runs_per_config)
        judge_res = self.run_execution_judge_ablation(case_ids=case_ids, runs_per_config=runs_per_config)
        k_res = self.run_retrieval_k_ablation(case_ids=case_ids)

        return {
            "summaries": {
                "semantic_memory": sem_res,
                "episodic_memory": epi_res,
                "execution_judge": judge_res,
                "retrieval_k": k_res,
            },
            "raw_runs": self.raw_runs,
        }

    def run_semantic_memory_ablation(self, case_ids: list[str] | None = None, runs_per_config: int = 1) -> dict[str, Any]:
        targets = case_ids or ["case_01_tiny_cls", "case_03_house_price_faulty"]
        results: dict[str, Any] = {"semantic_memory_off": [], "semantic_memory_on": []}

        for cid in targets:
            case = get_case(cid)
            if not case:
                continue

            # Off runs
            off_runs = []
            for r_idx in range(1, runs_per_config + 1):
                run = self._run_single_config(case, r_idx, disable_semantic=True)
                off_runs.append(run)
                self._record_raw_run("semantic_memory", "OFF", case.case_id, r_idx, run)
            sum_off = aggregate_case_runs(off_runs)
            if sum_off:
                results["semantic_memory_off"].append(sum_off.model_dump())

            # On runs
            on_runs = []
            for r_idx in range(1, runs_per_config + 1):
                run = self._run_single_config(case, r_idx, disable_semantic=False)
                on_runs.append(run)
                self._record_raw_run("semantic_memory", "ON", case.case_id, r_idx, run)
            sum_on = aggregate_case_runs(on_runs)
            if sum_on:
                results["semantic_memory_on"].append(sum_on.model_dump())

        return results

    def run_episodic_memory_ablation(self, case_ids: list[str] | None = None, runs_per_config: int = 1) -> dict[str, Any]:
        targets = case_ids or ["case_03_house_price_faulty", "case_01_tiny_cls"]
        results: dict[str, Any] = {"episodic_memory_off": [], "episodic_memory_on": []}

        for cid in targets:
            case = get_case(cid)
            if not case:
                continue

            # Off runs
            off_runs = []
            for r_idx in range(1, runs_per_config + 1):
                run = self._run_single_config(case, r_idx, disable_episodic=True)
                off_runs.append(run)
                self._record_raw_run("episodic_memory", "OFF", case.case_id, r_idx, run)
            sum_off = aggregate_case_runs(off_runs)
            if sum_off:
                results["episodic_memory_off"].append(sum_off.model_dump())

            # On runs
            on_runs = []
            for r_idx in range(1, runs_per_config + 1):
                run = self._run_single_config(case, r_idx, disable_episodic=False)
                on_runs.append(run)
                self._record_raw_run("episodic_memory", "ON", case.case_id, r_idx, run)
            sum_on = aggregate_case_runs(on_runs)
            if sum_on:
                results["episodic_memory_on"].append(sum_on.model_dump())

        return results

    def run_execution_judge_ablation(self, case_ids: list[str] | None = None, runs_per_config: int = 1) -> dict[str, Any]:
        targets = case_ids or ["case_01_tiny_cls", "case_03_house_price_faulty"]
        results: dict[str, Any] = {"judge_off": [], "judge_on": []}

        for cid in targets:
            case = get_case(cid)
            if not case:
                continue

            off_runs = []
            for r in range(1, runs_per_config + 1):
                run = self._run_single_config(case, r, disable_judge=True)
                off_runs.append(run)
                self._record_raw_run("execution_judge", "OFF", case.case_id, r, run)
            sum_off = aggregate_case_runs(off_runs)
            if sum_off:
                results["judge_off"].append(sum_off.model_dump())

            on_runs = []
            for r in range(1, runs_per_config + 1):
                run = self._run_single_config(case, r, disable_judge=False)
                on_runs.append(run)
                self._record_raw_run("execution_judge", "ON", case.case_id, r, run)
            sum_on = aggregate_case_runs(on_runs)
            if sum_on:
                results["judge_on"].append(sum_on.model_dump())

        return results

    def run_retrieval_k_ablation(self, k_values: list[int] | None = None, case_ids: list[str] | None = None) -> dict[int, list[dict[str, Any]]]:
        ks = k_values or [0, 1, 3, 5, 10]
        targets = case_ids or ["case_01_tiny_cls", "case_03_house_price_faulty"]
        results: dict[int, list[dict[str, Any]]] = {}

        for k in ks:
            results[k] = []
            for cid in targets:
                case = get_case(cid)
                if not case:
                    continue
                run = self._run_single_config(case, run_index=1, retrieval_k=k)
                self._record_raw_run("retrieval_k", f"K={k}", case.case_id, 1, run)
                sum_k = aggregate_case_runs([run])
                if sum_k:
                    results[k].append(sum_k.model_dump())

        return results

    def _record_raw_run(self, component: str, variant: str, case_id: str, run_index: int, run: EvaluationRun) -> None:
        self.raw_runs.append({
            "experiment_id": f"abl_{component}_{variant}_{case_id}_r{run_index}",
            "component": component,
            "variant": variant,
            "case_id": case_id,
            "run_index": run_index,
            "success": run.success,
            "iterations": run.iterations,
            "recovery": run.error_recovery_count,
            "total_time": run.total_time,
            "selected_library": run.selected_library,
            "judge_decisions": run.judge_decisions,
            "metrics": run.metrics,
            "requested_k": run.requested_k,
            "actual_retrieved_k": run.actual_retrieved_k,
        })

    def _run_single_config(
        self,
        case: Any,
        run_index: int,
        disable_semantic: bool = False,
        disable_episodic: bool = False,
        disable_judge: bool = False,
        retrieval_k: int = 5,
    ) -> EvaluationRun:
        from evaluation.runner import execute_case_run
        return execute_case_run(
            case=case,
            run_index=run_index,
            use_mock_llm=self.use_mock_llm,
            disable_semantic=disable_semantic,
            disable_episodic=disable_episodic,
            disable_judge=disable_judge,
            retrieval_k=retrieval_k,
        )
