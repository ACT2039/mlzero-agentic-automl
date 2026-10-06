"""
Perception Agents for MLZero.
Includes implementations for File Grouping, File Perception, Task Perception, and ML Library Selection.
"""

import csv
import json
import os
import re
from pathlib import Path
from typing import Any, ClassVar

from mlzero.core.config import settings
from mlzero.core.llm import LLMClient
from mlzero.core.logger import setup_logger
from mlzero.schemas.perception import (
    AudioMetadataSummary,
    DataQualityReport,
    FileContext,
    FileGroup,
    FileMetadata,
    ImageMetadataSummary,
    InvalidNumericInfo,
    JSONStructureSummary,
    LibrarySelection,
    MissingValueInfo,
    PerceptualContext,
    TaskContext,
    TextSummary,
)

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TARGET_KEYWORDS = {
    "price", "target", "label", "class", "outcome", "result", "output",
    "churn", "fraud", "survived", "survival", "yield", "salary", "revenue",
    "score", "rating", "grade", "diagnosis", "risk", "default", "approved",
    "sale", "value", "cost", "income", "profit", "spend", "demand", "y",
}

_INSTRUCTION_TARGET_RE = re.compile(
    r"""
    (?:
        predict\s+(?:the\s+)?
        | use\s+(?:the\s+)?(?P<col1>[a-zA-Z_]\w*)\s+as\s+(?:the\s+)?target
        | (?:target|label|predict)\s+(?:column\s+)?(?:is\s+|=\s*)?(?P<col2>[a-zA-Z_]\w*)
    )
    (?P<col>[a-zA-Z_]\w*)?
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _try_float(val: str) -> bool:
    """Return True if val can be parsed as a float."""
    try:
        float(val.strip())
        return True
    except ValueError:
        return False


def _extract_instruction_target(instruction: str | None) -> str | None:
    """Extract an explicit target column name from a user instruction."""
    if not instruction:
        return None
    patterns = [
        r"(?:predict|forecast|estimate)\s+(?:the\s+)?(?:column\s+)?['\"]?([a-zA-Z_]\w*)['\"]?",
        r"(?:target|label)\s+(?:column\s+)?(?:is\s+|=\s*|:\s*)['\"]?([a-zA-Z_]\w*)['\"]?",
        r"use\s+['\"]?([a-zA-Z_]\w*)['\"]?\s+as\s+(?:the\s+)?(?:target|label)",
        r"['\"]([a-zA-Z_]\w*)['\"]?\s+(?:is\s+)?(?:as\s+)?(?:the\s+)?(?:target|label)",
    ]
    for pat in patterns:
        m = re.search(pat, instruction, re.IGNORECASE)
        if m:
            candidate = m.group(1).lower()
            if candidate not in {"the", "a", "an", "column", "field"}:
                return m.group(1)
    return None


# ---------------------------------------------------------------------------
# FileGroupingAgent
# ---------------------------------------------------------------------------

class FileGroupingAgent:
    """
    Agent responsible for analyzing dataset directory structure and grouping related files.
    Identifies relationships such as train/test pairs, image+metadata, documents+labels, etc.
    """

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir

    def process(self, file_contexts: list[FileContext]) -> list[FileGroup]:
        groups: list[FileGroup] = []
        if not file_contexts:
            return groups

        by_type: dict[str, list[FileContext]] = {}
        for fc in file_contexts:
            ftype = fc.metadata.file_type
            by_type.setdefault(ftype, []).append(fc)

        tabular_files = by_type.get("tabular", [])
        image_files = by_type.get("image", [])
        json_files = by_type.get("json", [])

        # 1. Tabular Train / Test pair
        train_tabs = [fc.metadata.path for fc in tabular_files if "train" in Path(fc.metadata.path).stem.lower()]
        test_tabs = [
            fc.metadata.path for fc in tabular_files
            if any(k in Path(fc.metadata.path).stem.lower() for k in ("test", "val", "validation"))
        ]

        if train_tabs:
            groups.append(
                FileGroup(
                    group_id="group_tabular_split",
                    files=train_tabs + test_tabs,
                    modality="tabular",
                    relationship="train_test_pair" if test_tabs else "standalone_table",
                    purpose="Primary tabular dataset",
                    confidence=0.95 if test_tabs else 0.85,
                )
            )

        # 2. Image + Labels/Metadata mapping
        if image_files:
            img_paths = [fc.metadata.path for fc in image_files[:50]]
            label_paths = [fc.metadata.path for fc in tabular_files + json_files]
            groups.append(
                FileGroup(
                    group_id="group_image_dataset",
                    files=img_paths + label_paths,
                    modality="image",
                    relationship="image_label_mapping" if label_paths else "image_collection",
                    purpose="Image dataset with annotations or metadata",
                    confidence=0.90,
                )
            )

        # 3. Retrieval Query + Corpus
        corpus_files = [
            fc.metadata.path for fc in json_files + tabular_files
            if any(k in Path(fc.metadata.path).stem.lower() for k in ("corpus", "doc", "passage"))
        ]
        query_files = [
            fc.metadata.path for fc in json_files + tabular_files
            if any(k in Path(fc.metadata.path).stem.lower() for k in ("query", "queries"))
        ]
        if corpus_files or query_files:
            groups.append(
                FileGroup(
                    group_id="group_retrieval_set",
                    files=corpus_files + query_files,
                    modality="text",
                    relationship="query_corpus",
                    purpose="Dense retrieval corpus and query set",
                    confidence=0.95,
                )
            )

        # 4. Multi-table relational group
        if len(tabular_files) > 1 and not (len(train_tabs) == 1 and len(test_tabs) == 1 and len(tabular_files) == 2):
            all_tabs = [fc.metadata.path for fc in tabular_files]
            if not any(g.group_id == "group_tabular_split" for g in groups):
                groups.append(
                    FileGroup(
                        group_id="group_multi_table",
                        files=all_tabs,
                        modality="tabular",
                        relationship="multi_table",
                        purpose="Multiple related data tables",
                        confidence=0.85,
                    )
                )

        if not groups:
            groups.append(
                FileGroup(
                    group_id="group_general",
                    files=[fc.metadata.path for fc in file_contexts[:20]],
                    modality="mixed",
                    relationship="unspecified",
                    purpose="General file context",
                    confidence=0.5,
                )
            )

        return groups


# ---------------------------------------------------------------------------
# DataProfiler
# ---------------------------------------------------------------------------

class DataProfiler:
    """
    Lightweight, deterministic profiler for tabular CSV datasets.
    Scans train and test CSV files to produce a DataQualityReport.
    """

    MAX_ROWS = 2_000
    MAX_INVALID_SAMPLES = 5

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir

    def profile(self, user_instruction: str | None = None) -> DataQualityReport:
        """Run profiling and return a DataQualityReport."""
        train_path = self._find_file("train")
        test_path = self._find_file("test")

        if train_path is None:
            csvs = list(self.dataset_dir.glob("*.csv"))
            train_path = csvs[0] if csvs else None

        if train_path is None:
            return DataQualityReport()

        train_cols, train_data = self._read_csv(train_path)
        test_cols = self._read_csv(test_path)[0] if test_path else []

        if not train_cols:
            return DataQualityReport()

        row_count = len(train_data)
        n = row_count if row_count > 0 else 1

        col_values: dict[str, list[str]] = {c: [] for c in train_cols}
        for row in train_data:
            for c in train_cols:
                col_values[c].append(row.get(c, ""))

        instruction_target = _extract_instruction_target(user_instruction)
        if instruction_target:
            col_lower = {c.lower(): c for c in train_cols}
            instruction_target = col_lower.get(instruction_target.lower(), instruction_target)

        train_only_cols = [c for c in train_cols if c not in test_cols] if test_cols else []
        keyword_matches = [c for c in train_cols if c.lower() in _TARGET_KEYWORDS]
        last_col = train_cols[-1]

        # Time series column detection heuristics
        ts_cols = [
            c for c in train_cols
            if c.lower() in {"timestamp", "date", "datetime", "time", "dt", "ts"} or
            (col_values.get(c) and any(re.match(r"^\d{4}[-/]\d{2}[-/]\d{2}", v.strip()) for v in col_values[c] if v.strip()))
        ]
        id_cols = [
            c for c in train_cols
            if c.lower() in {"item_id", "series_id", "entity_id", "id", "store", "product_id", "user_id", "symbol", "ticker"}
        ]

        ts_col = ts_cols[0] if ts_cols else None
        id_col = id_cols[0] if id_cols else None

        # Check for time-series forecasting structure (both timestamp AND id column, or timestamp + forecasting instruction)
        is_timeseries = False
        instr_lower = user_instruction.lower() if user_instruction else ""
        has_forecast_kw = any(k in instr_lower for k in ("forecast", "time_series", "timeseries", "horizon"))

        if ts_col and (id_col or has_forecast_kw):
            is_timeseries = True

        if is_timeseries:
            # For time-series, target column is numeric non-ts, non-id column
            num_targets = [
                c for c in train_cols
                if c not in (ts_col, id_col) and col_values.get(c) and sum(1 for v in col_values[c] if _try_float(v)) / max(len(col_values[c]), 1) >= 0.5
            ]
            if instruction_target and instruction_target in train_cols:
                target_col: str | None = instruction_target
            elif num_targets:
                target_col = num_targets[0]
            elif train_only_cols:
                target_col = train_only_cols[0]
            else:
                target_col = last_col

        elif instruction_target and instruction_target in train_cols:
            target_col = instruction_target
        elif len(train_only_cols) == 1:
            target_col = train_only_cols[0]
        elif keyword_matches:
            target_col = keyword_matches[0]
        else:
            target_col = last_col

        candidate_targets = []
        if instruction_target and instruction_target in train_cols:
            candidate_targets.append(instruction_target)
        if len(train_only_cols) == 1 and train_only_cols[0] not in candidate_targets:
            candidate_targets.append(train_only_cols[0])
        for k in keyword_matches:
            if k not in candidate_targets:
                candidate_targets.append(k)
        if last_col not in candidate_targets:
            candidate_targets.append(last_col)

        missing_info: list[MissingValueInfo] = []
        invalid_numeric: list[InvalidNumericInfo] = []
        constant_cols: list[str] = []

        for col in train_cols:
            vals = col_values[col]
            blanks = sum(1 for v in vals if v.strip() == "")
            if blanks:
                missing_info.append(
                    MissingValueInfo(
                        column=col,
                        missing_count=blanks,
                        missing_pct=round(blanks / n * 100, 2),
                    )
                )

            non_blank = [v for v in vals if v.strip() != ""]
            if not non_blank:
                continue

            numeric_count = sum(1 for v in non_blank if _try_float(v))
            is_mostly_numeric = numeric_count / len(non_blank) >= 0.6

            if is_mostly_numeric:
                invalid_vals = [v for v in non_blank if not _try_float(v)]
                if invalid_vals:
                    invalid_numeric.append(
                        InvalidNumericInfo(
                            column=col,
                            invalid_values=list(dict.fromkeys(invalid_vals))[: self.MAX_INVALID_SAMPLES],
                            invalid_count=len(invalid_vals),
                        )
                    )

            unique = {v for v in vals if v.strip() != ""}
            if len(unique) <= 1:
                constant_cols.append(col)

        malformed_target: list[str] = []
        if target_col and target_col in col_values:
            tvals = col_values[target_col]
            non_blank_t = [v for v in tvals if v.strip() != ""]
            if non_blank_t:
                numeric_t = sum(1 for v in non_blank_t if _try_float(v))
                if numeric_t / len(non_blank_t) >= 0.6:
                    malformed_target = list(dict.fromkeys(v for v in non_blank_t if not _try_float(v)))

        task_type: str | None = "time_series_forecasting" if is_timeseries else self._infer_task_type(target_col, col_values, user_instruction)

        schema_mismatches: list[str] = []
        if test_cols:
            for c in train_cols:
                if c not in test_cols and c != target_col:
                    schema_mismatches.append(f"Column '{c}' in train but not in test (non-target).")
            for c in test_cols:
                if c not in train_cols:
                    schema_mismatches.append(f"Column '{c}' in test but not in train.")

        col_lower_list = [c.lower() for c in train_cols]
        seen: dict[str, int] = {}
        dup_cols: list[str] = []
        for c in col_lower_list:
            seen[c] = seen.get(c, 0) + 1
        for c_lower, count in seen.items():
            if count > 1:
                dup_cols.extend(c for c in train_cols if c.lower() == c_lower)

        return DataQualityReport(
            target_column=target_col,
            timestamp_column=ts_col if is_timeseries else None,
            id_column=id_col if is_timeseries else None,
            task_type=task_type,
            row_count=row_count,
            column_names=train_cols,
            missing_values=missing_info,
            invalid_numeric_values=invalid_numeric,
            malformed_target_values=malformed_target,
            train_test_schema_mismatches=schema_mismatches,
            constant_columns=constant_cols,
            duplicate_columns=dup_cols,
            candidate_targets=candidate_targets,
        )

    def _infer_task_type(
        self,
        target_col: str | None,
        col_values: dict[str, list[str]],
        user_instruction: str | None,
    ) -> str | None:
        if user_instruction:
            instr_lower = user_instruction.lower()
            if any(k in instr_lower for k in ("regress", "price", "cost", "predict numeric", "continuous")):
                return "regression"
            if any(k in instr_lower for k in ("classif", "categor", "binary", "class")):
                return "classification"

        if not target_col or target_col not in col_values:
            return None

        tvals = [v.strip() for v in col_values[target_col] if v.strip() != ""]
        if not tvals:
            return None

        parseable = [v for v in tvals if _try_float(v)]
        if not parseable:
            return "classification"

        float_vals = [float(v) for v in parseable]
        all_int = all(v == int(v) for v in float_vals)
        unique_floats = set(float_vals)

        if all_int and len(unique_floats) <= 20:
            return "classification"

        return "regression"

    def _find_file(self, stem: str) -> Path | None:
        for ext in (".csv", ".tsv"):
            p = self.dataset_dir / f"{stem}{ext}"
            if p.exists():
                return p
        for p in self.dataset_dir.glob(f"{stem}*.csv"):
            return p
        return None

    def _read_csv(self, path: Path) -> tuple[list[str], list[dict[str, str]]]:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames or []
                rows = []
                for i, row in enumerate(reader):
                    if i >= self.MAX_ROWS:
                        break
                    rows.append(dict(row))
            return list(headers), rows
        except (OSError, csv.Error) as e:
            logger.warning(f"DataProfiler could not read {path}: {e}")
            return [], []


# ---------------------------------------------------------------------------
# FilePerceptionAgent
# ---------------------------------------------------------------------------

class FilePerceptionAgent:
    """
    Agent responsible for perceiving raw files across modalities
    (tabular, text, JSON, images, audio, binary).
    """

    IMAGE_EXTENSIONS: ClassVar[set[str]] = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
    AUDIO_EXTENSIONS: ClassVar[set[str]] = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

    def __init__(self, dataset_dir: str | Path):
        self.dataset_dir = Path(dataset_dir)

    def process(self) -> list[FileContext]:
        if not self.dataset_dir.exists() or not self.dataset_dir.is_dir():
            raise ValueError(f"Input directory does not exist or is not a directory: {self.dataset_dir}")

        file_contexts = []
        ignored_dirs = set(settings.perception.ignored_directories)
        max_size = settings.perception.max_file_size_bytes
        max_chars = settings.perception.max_text_characters
        max_rows = settings.perception.max_rows_sampled

        for root, dirs, files in os.walk(self.dataset_dir):
            dirs[:] = [d for d in dirs if d not in ignored_dirs and not d.startswith(".")]

            for file_name in files:
                file_path = Path(root) / file_name
                rel_path = file_path.relative_to(self.dataset_dir).as_posix()
                ext = file_path.suffix.lower()

                try:
                    size = file_path.stat().st_size
                except OSError as e:
                    logger.warning(f"Could not stat {file_path}: {e}")
                    continue

                if ext in [".csv", ".tsv", ".parquet"]:
                    file_type = "tabular"
                elif ext in [".json", ".jsonl"]:
                    file_type = "json"
                elif ext in [".txt", ".md"]:
                    file_type = "document"
                elif ext in self.IMAGE_EXTENSIONS:
                    file_type = "image"
                elif ext in self.AUDIO_EXTENSIONS:
                    file_type = "audio"
                else:
                    file_type = "binary_or_unsupported"

                metadata = FileMetadata(
                    path=rel_path,
                    absolute_path=str(file_path.absolute()),
                    extension=ext,
                    size_bytes=size,
                    file_type=file_type,
                )

                if size > max_size:
                    file_contexts.append(
                        FileContext(
                            metadata=metadata,
                            error=f"File exceeds max size limit of {max_size} bytes.",
                        )
                    )
                    continue

                try:
                    if file_type == "tabular":
                        ctx = self._inspect_tabular(file_path, metadata, max_rows)
                    elif file_type == "json":
                        ctx = self._inspect_json(file_path, metadata, max_chars)
                    elif file_type == "document":
                        ctx = self._inspect_document(file_path, metadata, max_chars)
                    elif file_type == "image":
                        ctx = self._inspect_image(file_path, metadata)
                    elif file_type == "audio":
                        ctx = self._inspect_audio(file_path, metadata)
                    else:
                        ctx = FileContext(metadata=metadata)
                except Exception as e:  # noqa: BLE001
                    ctx = FileContext(metadata=metadata, warnings=[f"Inspection warning: {e!s}"])

                file_contexts.append(ctx)

        return file_contexts

    def _inspect_tabular(self, file_path: Path, metadata: FileMetadata, max_rows: int) -> FileContext:
        delimiter = "\t" if metadata.extension == ".tsv" else ","
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f, delimiter=delimiter)
                headers = next(reader, None)
                if not headers:
                    return FileContext(metadata=metadata, error="Empty tabular file.")

                sample_rows: list[dict[str, Any]] = []
                row_count = 1
                for row in reader:
                    row_count += 1
                    if len(sample_rows) < max_rows:
                        row_dict = {
                            headers[i] if i < len(headers) else f"col_{i}": val
                            for i, val in enumerate(row)
                        }
                        sample_rows.append(row_dict)

                inferred: dict[str, str] = {}
                for col in headers:
                    vals = [str(r.get(col, "")) for r in sample_rows if r.get(col, "").strip() != ""]
                    if not vals:
                        inferred[col] = "unknown"
                    elif all(_try_float(v) for v in vals):
                        float_vals = [float(v) for v in vals]
                        if all(v == int(v) for v in float_vals) and len(set(float_vals)) <= 20:
                            inferred[col] = "integer_categorical"
                        elif all(v == int(v) for v in float_vals):
                            inferred[col] = "integer"
                        else:
                            inferred[col] = "float"
                    else:
                        inferred[col] = "string"

                return FileContext(
                    metadata=metadata,
                    columns=headers,
                    row_count=row_count,
                    sample_rows=sample_rows,
                    inferred_types=inferred,
                )
        except (OSError, ValueError, UnicodeDecodeError, csv.Error) as e:
            return FileContext(metadata=metadata, error=f"Error reading tabular file: {e!s}")

    def _inspect_json(self, file_path: Path, metadata: FileMetadata, max_chars: int) -> FileContext:
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(max_chars)

            data = json.loads(content)
            if isinstance(data, dict):
                keys = list(data.keys())[:20]
                summary = JSONStructureSummary(
                    keys=keys,
                    is_list=False,
                    nesting_depth=2,
                    sample_item={k: str(v)[:50] for k, v in list(data.items())[:3]},
                )
            elif isinstance(data, list):
                sample = data[0] if data else None
                keys = list(sample.keys())[:20] if isinstance(sample, dict) else []
                summary = JSONStructureSummary(
                    keys=keys,
                    is_list=True,
                    item_count=len(data),
                    sample_item=sample,
                )
            else:
                summary = JSONStructureSummary(keys=[], is_list=False, sample_item=str(data)[:100])

            return FileContext(metadata=metadata, json_summary=summary, extracted_text=content[:500])
        except Exception as e:  # noqa: BLE001
            return FileContext(metadata=metadata, warnings=[f"JSON parsing note: {e!s}"])

    def _inspect_document(self, file_path: Path, metadata: FileMetadata, max_chars: int) -> FileContext:
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                lines = [f.readline() for _ in range(50)]
                f.seek(0)
                content = f.read(max_chars)
                if len(content) == max_chars:
                    content += "\n...[TRUNCATED]"
            text_sum = TextSummary(
                line_count=len(lines),
                encoding="utf-8",
                first_lines=[l.strip() for l in lines[:10] if l.strip()],
            )
            return FileContext(metadata=metadata, extracted_text=content, text_summary=text_sum)
        except (OSError, ValueError, UnicodeDecodeError) as e:
            return FileContext(metadata=metadata, error=f"Error reading document file: {e!s}")

    def _inspect_image(self, file_path: Path, metadata: FileMetadata) -> FileContext:
        warnings = []
        try:
            from PIL import Image
            with Image.open(file_path) as img:
                w, h = img.size
                mode = img.mode
                fmt = img.format or metadata.extension.lstrip(".").upper()
                bands = len(img.getbands()) if hasattr(img, "getbands") else 3
                summary = ImageMetadataSummary(width=w, height=h, channels=bands, mode=mode, format=fmt)
        except Exception as e:  # noqa: BLE001
            summary = ImageMetadataSummary(format=metadata.extension.lstrip(".").upper())
            warnings.append(f"Lightweight PIL inspection fallback: {e!s}")

        return FileContext(metadata=metadata, image_summary=summary, warnings=warnings)

    def _inspect_audio(self, file_path: Path, metadata: FileMetadata) -> FileContext:
        fmt = metadata.extension.lstrip(".").upper()
        summary = AudioMetadataSummary(format=fmt)
        if metadata.extension == ".wav":
            try:
                import wave
                with wave.open(str(file_path), "rb") as wf:
                    summary.channels = wf.getnchannels()
                    summary.sample_rate = wf.getframerate()
                    frames = wf.getnframes()
                    if wf.getframerate() > 0:
                        summary.duration_sec = round(frames / float(wf.getframerate()), 2)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"WAV wave module inspection note: {e!s}")
        return FileContext(metadata=metadata, audio_summary=summary)


# ---------------------------------------------------------------------------
# TaskPerceptionAgent
# ---------------------------------------------------------------------------

class TaskPerceptionAgent:
    """Agent responsible for understanding the overall user task."""

    def __init__(self, llm_client: LLMClient, dataset_dir: Path | None = None):
        self.llm_client = llm_client
        self.dataset_dir = dataset_dir

    def process(
        self,
        file_contexts: list[FileContext],
        user_instruction: str | None = None,
        data_quality: DataQualityReport | None = None,
        file_groups: list[FileGroup] | None = None,
    ) -> TaskContext:
        readme_path, readme_content = self._find_readme(file_contexts)

        prompt = "Extract task information from perceptual context.\n"
        if user_instruction:
            prompt += f"Instruction: {user_instruction}\n"
        prompt += f"Files available: {len(file_contexts)}\n"
        data_file_list = [
            f"{fc.metadata.path} ({fc.metadata.file_type}): {fc.metadata.absolute_path}"
            for fc in file_contexts
            if not fc.error
        ]
        if data_file_list:
            prompt += "Discovered Files:\n" + "\n".join(f"- {f}" for f in data_file_list) + "\n"
        if file_groups:
            prompt += f"File Groups: {[g.relationship for g in file_groups]}\n"
        if readme_content:
            prompt += f"README Description:\n{readme_content[:1000]}\n"
        if data_quality:
            prompt += f"Profiled target: {data_quality.target_column}\n"
            prompt += f"Profiled task_type: {data_quality.task_type}\n"
            if data_quality.timestamp_column:
                prompt += f"Profiled timestamp_column: {data_quality.timestamp_column}\n"
            if data_quality.id_column:
                prompt += f"Profiled id_column: {data_quality.id_column}\n"

        llm_result: TaskContext = self.llm_client.generate_structured(prompt, TaskContext, max_tokens=1500)

        # Check for structural retrieval signals
        is_retrieval = False
        if file_groups and any(g.relationship == "query_corpus" for g in file_groups):
            is_retrieval = True
        for fc in file_contexts:
            if fc.json_summary and any(k in fc.json_summary.keys for k in ("query", "queries", "documents", "passages", "corpus")):
                is_retrieval = True

        instr_lower = user_instruction.lower() if user_instruction else ""
        readme_lower = readme_content.lower() if readme_content else ""
        if any(k in instr_lower or k in readme_lower for k in ("retrieval", "search query", "dense retrieval", "reranking", "corpus search")):
            is_retrieval = True

        # Resolve final task_type and metadata attributes
        final_task_type: str | None = None
        final_target_col: str | None = None
        final_ts_col: str | None = None
        final_id_col: str | None = None

        if is_retrieval:
            final_task_type = "retrieval"
            final_target_col = None
        elif data_quality and data_quality.task_type == "time_series_forecasting":
            final_task_type = "time_series_forecasting"
            final_target_col = data_quality.target_column or "target"
            final_ts_col = data_quality.timestamp_column
            final_id_col = data_quality.id_column
        elif llm_result.task_type in ("time_series_forecasting", "time_series"):
            final_task_type = "time_series_forecasting"
            final_target_col = llm_result.target_column or (data_quality.target_column if data_quality else "target")
            final_ts_col = llm_result.timestamp_column or (data_quality.timestamp_column if data_quality else None)
            final_id_col = llm_result.id_column or (data_quality.id_column if data_quality else None)
        elif llm_result.task_type in ("multimodal", "image_classification"):
            final_task_type = llm_result.task_type
            final_target_col = data_quality.target_column if (data_quality and data_quality.target_column) else llm_result.target_column
        elif data_quality and data_quality.task_type:
            final_task_type = data_quality.task_type
            final_target_col = data_quality.target_column
        else:
            final_task_type = llm_result.task_type
            final_target_col = llm_result.target_column

        llm_result = llm_result.model_copy(
            update={
                "task_type": final_task_type,
                "target_column": final_target_col,
                "timestamp_column": final_ts_col,
                "id_column": final_id_col,
                "data_quality": data_quality,
            }
        )

        if readme_path and not llm_result.relevant_description_files:
            llm_result = llm_result.model_copy(update={"relevant_description_files": [str(readme_path)]})

        # Always override input_data_files with actual discovered paths —
        # LLMs (mock or real) cannot know the real filesystem layout.
        data_files = [
            fc.metadata.absolute_path
            for fc in file_contexts
            if fc.metadata.file_type in ("tabular", "image", "json", "audio") and not fc.error
        ]
        if data_files:
            llm_result = llm_result.model_copy(update={"input_data_files": data_files})

        return llm_result

    def _find_readme(self, file_contexts: list[FileContext]) -> tuple[str | None, str | None]:
        for fc in file_contexts:
            if fc.metadata.extension in (".md", ".txt"):
                stem = Path(fc.metadata.path).stem.lower()
                if stem in ("readme", "task", "description"):
                    return fc.metadata.path, fc.extracted_text
        return None, None


# ---------------------------------------------------------------------------
# LibrarySelectorAgent
# ---------------------------------------------------------------------------

class LibrarySelectorAgent:
    """Agent responsible for selecting appropriate ML libraries."""

    REGISTRY: ClassVar[list[dict[str, Any]]] = [
        {
            "name": "autogluon.tabular",
            "version": "1.1.1",
            "description": "AutoML for tabular data classification and regression.",
            "modalities": ["tabular"],
            "task_types": ["classification", "regression"],
            "adapter_identifier": "TabularAdapter",
            "knowledge_identifier": "autogluon_tabular",
            "limitations": "Requires structured tabular data.",
        },
        {
            "name": "autogluon.multimodal",
            "version": "1.1.1",
            "description": "AutoML for images, text, and mixed multimodal data.",
            "modalities": ["image", "text", "multimodal"],
            "task_types": ["classification", "regression", "multimodal"],
            "adapter_identifier": "MultiModalAdapter",
            "knowledge_identifier": "autogluon_multimodal",
            "limitations": "Heavy memory usage, requires GPU for best performance.",
        },
        {
            "name": "autogluon.timeseries",
            "version": "1.1.1",
            "description": "AutoML for time series forecasting.",
            "modalities": ["timeseries"],
            "task_types": ["forecasting", "time_series", "time_series_forecasting"],
            "adapter_identifier": "TimeSeriesAdapter",
            "knowledge_identifier": "autogluon_timeseries",
            "limitations": "Requires timestamps and unique item IDs.",
        },
        {
            "name": "FlagEmbedding",
            "version": "1.2.0",
            "description": "Retrieval and reranking tasks.",
            "modalities": ["text"],
            "task_types": ["retrieval", "reranking", "embedding"],
            "adapter_identifier": "RetrievalAdapter",
            "knowledge_identifier": "flagembedding",
            "limitations": "Requires text documents to embed.",
        },
        {
            "name": "machine learning",
            "version": "1.0.0",
            "description": "General machine learning using standard libraries (scikit-learn, etc.).",
            "modalities": ["any"],
            "task_types": ["any", "general_ml"],
            "adapter_identifier": "GeneralMLAdapter",
            "knowledge_identifier": "general_ml",
            "limitations": "Requires manual pipeline building.",
        },
    ]

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def process(
        self,
        task_context: TaskContext,
        file_contexts: list[FileContext],
        perceptual_context: PerceptualContext | None = None,
    ) -> LibrarySelection:
        registry_str = json.dumps(self.REGISTRY, indent=2)
        prompt = f"Select library from registry:\n{registry_str}\n"
        prompt += f"Task type: {task_context.task_type}\n"
        if perceptual_context and perceptual_context.modalities:
            prompt += f"Data Modalities: {perceptual_context.modalities}\n"

        return self.llm_client.generate_structured(prompt, LibrarySelection, max_tokens=1500)
