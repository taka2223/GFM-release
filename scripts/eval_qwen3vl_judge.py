"""Blind, order-swapped Qwen3-VL judging for MorphBench sequences.

The inference path is deliberately offline and batch-size one.  Each request
contains the 17 original PNG frames from both anonymous sequences.  Raw model
answers are append-only so interrupted evaluations can be resumed safely;
derived CSV/JSON reports can be rebuilt from that JSONL without loading the
model by passing ``--summarize_only``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import json
import os
import platform
import re
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


PROMPT_VERSION = "qwen3vl-morphbench-blind-v1"
DEFAULT_MODEL_PATH = "Qwen/Qwen3-VL-32B-Instruct"
CRITERIA = (
    "visual_quality",
    "semantic_plausibility",
    "endpoint_content_preservation",
    "transition_smoothness",
    "progression_uniformity",
    "overall",
)
METHOD_NAME_PATTERN = re.compile(r"\b(?:gfm|c2f|slerp)\b", re.IGNORECASE)
RecordKey = Tuple[str, str, int, str]


@dataclass(frozen=True)
class PairSpec:
    subset: str
    name: str
    prompt_a: str
    prompt_b: str
    frames: Mapping[str, Tuple[str, ...]]


@dataclass(frozen=True)
class DiscoveryIssue:
    subset: str
    name: str
    error_type: str
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline Qwen3-VL blind A/B judge for MorphBench.",
    )
    parser.add_argument(
        "--out_root",
        default="./output/morphbench_c2f_sigma0p6_lambda20",
    )
    parser.add_argument("--methods", nargs=2, default=["gfm_c2f", "slerp"])
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["Metamorphosis", "Animation"],
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--expected_frames", type=int, default=17)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--bootstrap_repeats", type=int, default=10000)
    parser.add_argument("--pair_names", nargs="*", default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate only the first N discovered pairs (0 means all).",
    )
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--max_memory_per_gpu", default="76GiB")
    parser.add_argument("--raw_output", default=None)
    parser.add_argument("--per_pair_output", default=None)
    parser.add_argument("--summary_output", default=None)
    parser.add_argument(
        "--summarize_only",
        action="store_true",
        help="Rebuild reports from the existing JSONL without model inference.",
    )
    return parser.parse_args()


def _expected_frame_paths(
    out_root: str,
    method: str,
    subset: str,
    name: str,
    expected_frames: int,
) -> Tuple[str, ...]:
    frame_dir = Path(out_root) / method / subset / name / "frames"
    return tuple(
        str(frame_dir / f"frame_{index:03d}.png")
        for index in range(expected_frames)
    )


def _read_endpoint_prompts(meta_paths: Sequence[Path]) -> Tuple[str, str]:
    endpoints: List[Tuple[str, str]] = []
    for meta_path in meta_paths:
        with meta_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        prompt_a = metadata.get("promptA")
        prompt_b = metadata.get("promptB")
        if not isinstance(prompt_a, str) or not isinstance(prompt_b, str):
            raise ValueError(f"missing string promptA/promptB in {meta_path}")
        endpoints.append((prompt_a, prompt_b))
    if len(set(endpoints)) != 1:
        raise ValueError("endpoint prompts disagree between methods")
    return endpoints[0]


def discover_pairs(
    out_root: str,
    methods: Sequence[str],
    subsets: Sequence[str],
    expected_frames: int = 17,
    pair_names: Optional[Sequence[str]] = None,
) -> Tuple[List[PairSpec], List[DiscoveryIssue]]:
    """Discover the union of pair directories and retain only complete pairs."""
    if len(methods) != 2 or methods[0] == methods[1]:
        raise ValueError("exactly two distinct methods are required")
    requested = set(pair_names) if pair_names else None
    pairs: List[PairSpec] = []
    issues: List[DiscoveryIssue] = []

    for subset in subsets:
        candidates = set()
        for method in methods:
            subset_dir = Path(out_root) / method / subset
            if subset_dir.is_dir():
                candidates.update(path.name for path in subset_dir.iterdir())
        if requested is not None:
            candidates.intersection_update(requested)

        for name in sorted(candidates):
            frames_by_method: Dict[str, Tuple[str, ...]] = {}
            frame_problems = []
            for method in methods:
                expected = _expected_frame_paths(
                    out_root,
                    method,
                    subset,
                    name,
                    expected_frames,
                )
                missing = [path for path in expected if not Path(path).is_file()]
                frame_dir = Path(expected[0]).parent
                actual = (
                    sorted(frame_dir.glob("frame_*.png"))
                    if frame_dir.is_dir()
                    else []
                )
                if missing or len(actual) != expected_frames:
                    frame_problems.append(
                        f"{method}: expected {expected_frames}, "
                        f"found {len(actual)}, missing {len(missing)}"
                    )
                else:
                    frames_by_method[method] = expected
            if frame_problems:
                issues.append(
                    DiscoveryIssue(
                        subset,
                        name,
                        "missing_frames",
                        "; ".join(frame_problems),
                    )
                )
                continue

            meta_paths = [
                Path(out_root) / method / subset / name / "meta.json"
                for method in methods
            ]
            missing_meta = [str(path) for path in meta_paths if not path.is_file()]
            if missing_meta:
                issues.append(
                    DiscoveryIssue(
                        subset,
                        name,
                        "missing_metadata",
                        "missing " + ", ".join(missing_meta),
                    )
                )
                continue
            try:
                prompt_a, prompt_b = _read_endpoint_prompts(meta_paths)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                issues.append(
                    DiscoveryIssue(
                        subset,
                        name,
                        "invalid_metadata",
                        str(error),
                    )
                )
                continue
            pairs.append(
                PairSpec(
                    subset=subset,
                    name=name,
                    prompt_a=prompt_a,
                    prompt_b=prompt_b,
                    frames=frames_by_method,
                )
            )
    return sorted(pairs, key=lambda pair: (pair.subset, pair.name)), issues


def first_order_mapping(
    methods: Sequence[str],
    seed: int,
    subset: str,
    name: str,
) -> Dict[str, str]:
    payload = f"{seed}\0{subset}\0{name}".encode("utf-8")
    swap = hashlib.sha256(payload).digest()[0] & 1
    ordered = list(reversed(methods)) if swap else list(methods)
    return {"A": ordered[0], "B": ordered[1]}


def order_mapping(
    methods: Sequence[str],
    seed: int,
    subset: str,
    name: str,
    order_index: int,
) -> Dict[str, str]:
    if order_index not in (1, 2):
        raise ValueError("order_index must be 1 or 2")
    first = first_order_mapping(methods, seed, subset, name)
    if order_index == 1:
        return first
    return {"A": first["B"], "B": first["A"]}


def record_key(record: Mapping[str, Any]) -> Optional[RecordKey]:
    key = record.get("key")
    if not isinstance(key, Mapping):
        return None
    try:
        return (
            str(key["subset"]),
            str(key["name"]),
            int(key["order"]),
            str(key["prompt_version"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _criterion_schema_text() -> str:
    rows = []
    for criterion in CRITERIA:
        suffix = (
            ', "confidence": 0.85, "reason": "brief reason"'
            if criterion == "overall"
            else ""
        )
        rows.append(
            f'  "{criterion}": '
            f'{{"A": 1, "B": 1, "winner": "tie"{suffix}}}'
        )
    return "{\n" + ",\n".join(rows) + "\n}"


def evaluation_instructions(
    prompt_a: str,
    prompt_b: str,
    strict_json: bool = False,
) -> str:
    strict_prefix = (
        "Your previous response could not be parsed. Output a JSON object only: "
        "no Markdown fences, preamble, commentary, or trailing text.\n\n"
        if strict_json
        else ""
    )
    text = f"""\
{strict_prefix}Compare two anonymous 17-frame visual transition sequences.
Both are intended to progress from this starting endpoint description:
"{prompt_a}"
to this ending endpoint description:
"{prompt_b}"

Judge only what is visible in the supplied frames. Use these criteria:
- visual_quality: fidelity, clarity, artifacts, and aesthetic quality.
- semantic_plausibility: whether intermediate content is coherent and credible.
- endpoint_content_preservation: fidelity to both described endpoints and their
  important visible content.
- transition_smoothness: temporal continuity without abrupt structural jumps.
- progression_uniformity: whether perceptual change is distributed evenly over
  t=0/16 through t=16/16 rather than stalling or changing suddenly.
- overall: holistic preference, informed by but not mechanically averaging the
  five criteria above.

For every criterion, assign each sequence an integer score from 1 (poor) to 5
(excellent). The winner must match the scores: choose "A" when A is higher,
"B" when B is higher, and "tie" only when equal. For overall, confidence must
be a number from 0 to 1 and reason must be one short sentence.

Return exactly one JSON object with exactly this structure and no extra keys:
{_criterion_schema_text()}"""
    if METHOD_NAME_PATTERN.search(text):
        raise ValueError("method identifier leaked into the model prompt")
    return text


def build_messages(
    pair: PairSpec,
    mapping: Mapping[str, str],
    images_by_method: Mapping[str, Sequence[Any]],
    strict_json: bool = False,
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": evaluation_instructions(
                pair.prompt_a,
                pair.prompt_b,
                strict_json=strict_json,
            ),
        }
    ]
    denominator = len(images_by_method[mapping["A"]]) - 1
    for alias in ("A", "B"):
        images = images_by_method[mapping[alias]]
        if len(images) != denominator + 1:
            raise ValueError("both sequences must have the same frame count")
        content.append(
            {"type": "text", "text": f"Sequence {alias} follows."}
        )
        for index, image in enumerate(images):
            content.append(
                {
                    "type": "text",
                    "text": f"Sequence {alias}, t={index}/{denominator}",
                }
            )
            content.append({"type": "image", "image": image})
    all_text = "\n".join(
        item["text"] for item in content if item["type"] == "text"
    )
    if METHOD_NAME_PATTERN.search(all_text):
        raise ValueError("method identifier leaked into the model prompt")
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a meticulous evaluator of visual transition "
                        "sequences. Compare anonymous inputs impartially and "
                        "follow the requested output schema exactly."
                    ),
                }
            ],
        },
        {"role": "user", "content": content},
    ]


def prompt_text(messages: Sequence[Mapping[str, Any]]) -> str:
    blocks: List[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            blocks.append(content)
        elif isinstance(content, Sequence):
            blocks.extend(
                str(item["text"])
                for item in content
                if isinstance(item, Mapping) and item.get("type") == "text"
            )
    return "\n".join(blocks)


def extract_json_object(raw_response: str) -> Mapping[str, Any]:
    """Extract the largest decodable JSON object from a model response."""
    stripped = raw_response.strip()
    try:
        direct = json.loads(stripped)
        if isinstance(direct, Mapping):
            return direct
    except json.JSONDecodeError:
        pass

    first = stripped.find("{")
    last = stripped.rfind("}")
    if first >= 0 and last > first:
        try:
            outer = json.loads(stripped[first : last + 1])
            if isinstance(outer, Mapping):
                return outer
        except json.JSONDecodeError:
            pass

    decoder = json.JSONDecoder()
    candidates: List[Tuple[int, int, Mapping[str, Any]]] = []
    for match in re.finditer(r"\{", stripped):
        try:
            value, consumed = decoder.raw_decode(stripped[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            candidates.append((consumed, -match.start(), value))
    if not candidates:
        raise ValueError("no decodable JSON object found")
    return max(candidates, key=lambda candidate: (candidate[0], candidate[1]))[2]


def validate_judgment(value: Mapping[str, Any]) -> Dict[str, Any]:
    if set(value) != set(CRITERIA):
        missing = sorted(set(CRITERIA) - set(value))
        extra = sorted(set(value) - set(CRITERIA))
        raise ValueError(f"top-level schema mismatch; missing={missing}, extra={extra}")

    validated: Dict[str, Any] = {}
    for criterion in CRITERIA:
        item = value[criterion]
        if not isinstance(item, Mapping):
            raise ValueError(f"{criterion} must be an object")
        expected_keys = {"A", "B", "winner"}
        if criterion == "overall":
            expected_keys.update({"confidence", "reason"})
        if set(item) != expected_keys:
            raise ValueError(f"{criterion} has incorrect keys")
        scores = {}
        for alias in ("A", "B"):
            score = item[alias]
            if isinstance(score, bool) or not isinstance(score, int):
                raise ValueError(f"{criterion}.{alias} must be an integer")
            if not 1 <= score <= 5:
                raise ValueError(f"{criterion}.{alias} is outside [1, 5]")
            scores[alias] = score
        winner = item["winner"]
        if winner not in ("A", "B", "tie"):
            raise ValueError(f"{criterion}.winner is invalid")
        expected_winner = (
            "A"
            if scores["A"] > scores["B"]
            else "B"
            if scores["B"] > scores["A"]
            else "tie"
        )
        if winner != expected_winner:
            raise ValueError(
                f"{criterion}.winner conflicts with its numeric scores"
            )
        result: Dict[str, Any] = {**scores, "winner": winner}
        if criterion == "overall":
            confidence = item["confidence"]
            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not 0.0 <= float(confidence) <= 1.0
            ):
                raise ValueError("overall.confidence must be in [0, 1]")
            reason = item["reason"]
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("overall.reason must be a non-empty string")
            result["confidence"] = float(confidence)
            result["reason"] = reason.strip()
        validated[criterion] = result
    return validated


def parse_judgment(raw_response: str) -> Dict[str, Any]:
    return validate_judgment(extract_json_object(raw_response))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_environment(model_path: str) -> Dict[str, Any]:
    import torch
    import transformers

    config_path = Path(model_path) / "config.json"
    gpus = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpus.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "compute_capability": [
                        properties.major,
                        properties.minor,
                    ],
                }
            )
    return {
        "model_path": str(Path(model_path).resolve()),
        "model_config_sha256": (
            _sha256_file(config_path) if config_path.is_file() else None
        ),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "gpu": gpus,
    }


def load_model_and_processor(
    model_path: str,
    num_gpus: int,
    max_memory_per_gpu: str,
) -> Tuple[Any, Any]:
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if not Path(model_path).is_dir():
        raise FileNotFoundError(f"local model directory not found: {model_path}")
    if not torch.cuda.is_available() or torch.cuda.device_count() < num_gpus:
        raise RuntimeError(
            f"protocol requires {num_gpus} CUDA GPUs; "
            f"found {torch.cuda.device_count()}"
        )
    max_memory = {
        index: max_memory_per_gpu for index in range(num_gpus)
    }
    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="balanced",
        max_memory=max_memory,
        attn_implementation="sdpa",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    used_gpus = set()
    for device in getattr(model, "hf_device_map", {}).values():
        if isinstance(device, int):
            used_gpus.add(device)
        elif isinstance(device, str) and device.startswith("cuda:"):
            used_gpus.add(int(device.split(":", 1)[1]))
    expected_gpus = set(range(num_gpus))
    if not expected_gpus.issubset(used_gpus):
        raise RuntimeError(
            "balanced device map did not use every required GPU: "
            f"expected={sorted(expected_gpus)}, used={sorted(used_gpus)}"
        )
    return model, processor


def _load_pair_images(pair: PairSpec) -> Dict[str, List[Any]]:
    from PIL import Image

    result: Dict[str, List[Any]] = {}
    for method, paths in pair.frames.items():
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        result[method] = images
    return result


def _close_images(images_by_method: Mapping[str, Sequence[Any]]) -> None:
    for images in images_by_method.values():
        for image in images:
            close = getattr(image, "close", None)
            if close is not None:
                close()


def generate_response(
    model: Any,
    processor: Any,
    messages: Sequence[Mapping[str, Any]],
    max_new_tokens: int,
    num_gpus: int,
) -> Tuple[str, Dict[str, int]]:
    import torch

    for index in range(num_gpus):
        torch.cuda.reset_peak_memory_stats(index)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )
    generated_trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    peaks = {
        str(index): int(torch.cuda.max_memory_allocated(index))
        for index in range(num_gpus)
    }
    del generated_ids, generated_trimmed, inputs
    return response, peaks


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _base_record(
    pair: PairSpec,
    mapping: Mapping[str, str],
    order_index: int,
    environment: Mapping[str, Any],
    max_new_tokens: int,
) -> Dict[str, Any]:
    return {
        "record_type": "judgment",
        "key": {
            "subset": pair.subset,
            "name": pair.name,
            "order": order_index,
            "prompt_version": PROMPT_VERSION,
        },
        "created_at": _utc_now(),
        "mapping": {
            "sequence_A_method": mapping["A"],
            "sequence_B_method": mapping["B"],
        },
        "endpoint_descriptions": {
            "start": pair.prompt_a,
            "end": pair.prompt_b,
        },
        "generation": {
            "batch_size": 1,
            "dtype": "bfloat16",
            "device_map": "balanced",
            "attention_implementation": "sdpa",
            "do_sample": False,
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
            "n_images": sum(len(paths) for paths in pair.frames.values()),
        },
        "environment": dict(environment),
    }


def evaluate_order(
    pair: PairSpec,
    mapping: Mapping[str, str],
    order_index: int,
    model: Any,
    processor: Any,
    environment: Mapping[str, Any],
    max_new_tokens: int,
    num_gpus: int,
) -> Dict[str, Any]:
    import torch

    record = _base_record(
        pair,
        mapping,
        order_index,
        environment,
        max_new_tokens,
    )
    responses = []
    peak_by_gpu = {str(index): 0 for index in range(num_gpus)}
    images_by_method = _load_pair_images(pair)
    try:
        for attempt, strict_json in enumerate((False, True), start=1):
            messages = build_messages(
                pair,
                mapping,
                images_by_method,
                strict_json=strict_json,
            )
            if attempt == 1:
                text = prompt_text(messages)
                record["prompt_text"] = text
                record["prompt_sha256"] = hashlib.sha256(
                    text.encode("utf-8")
                ).hexdigest()
            try:
                raw_response, peaks = generate_response(
                    model,
                    processor,
                    messages,
                    max_new_tokens,
                    num_gpus,
                )
                for gpu, peak in peaks.items():
                    peak_by_gpu[gpu] = max(peak_by_gpu[gpu], peak)
            except Exception as error:
                for index in range(num_gpus):
                    peak_by_gpu[str(index)] = max(
                        peak_by_gpu[str(index)],
                        int(torch.cuda.max_memory_allocated(index)),
                    )
                error_type = (
                    "oom"
                    if isinstance(error, torch.cuda.OutOfMemoryError)
                    else "inference_error"
                )
                responses.append(
                    {
                        "attempt": attempt,
                        "prompt_mode": (
                            "strict_json_retry" if strict_json else "standard"
                        ),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                record.update(
                    {
                        "status": "error",
                        "error_type": error_type,
                        "error": str(error),
                        "responses": responses,
                        "peak_memory_allocated_bytes": peak_by_gpu,
                    }
                )
                return record
            try:
                parsed = parse_judgment(raw_response)
            except ValueError as error:
                responses.append(
                    {
                        "attempt": attempt,
                        "prompt_mode": (
                            "strict_json_retry" if strict_json else "standard"
                        ),
                        "raw_response": raw_response,
                        "parse_error": str(error),
                    }
                )
                if not strict_json:
                    continue
                record.update(
                    {
                        "status": "error",
                        "error_type": "parse_failure",
                        "error": str(error),
                        "responses": responses,
                        "peak_memory_allocated_bytes": peak_by_gpu,
                    }
                )
                return record
            responses.append(
                {
                    "attempt": attempt,
                    "prompt_mode": (
                        "strict_json_retry" if strict_json else "standard"
                    ),
                    "raw_response": raw_response,
                }
            )
            record.update(
                {
                    "status": "ok",
                    "responses": responses,
                    "raw_response": raw_response,
                    "parsed": parsed,
                    "peak_memory_allocated_bytes": peak_by_gpu,
                }
            )
            return record
    finally:
        _close_images(images_by_method)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    raise AssertionError("unreachable")


def append_jsonl(path: str, record: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not Path(path).is_file():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSONL at {path}:{line_number}: {error}"
                ) from error
            if isinstance(value, dict):
                records.append(value)
    return records


def is_valid_record(record: Mapping[str, Any]) -> bool:
    if record.get("status") != "ok" or record_key(record) is None:
        return False
    parsed = record.get("parsed")
    if not isinstance(parsed, Mapping):
        return False
    try:
        validate_judgment(parsed)
    except ValueError:
        return False
    mapping = record.get("mapping")
    return (
        isinstance(mapping, Mapping)
        and isinstance(mapping.get("sequence_A_method"), str)
        and isinstance(mapping.get("sequence_B_method"), str)
        and mapping["sequence_A_method"] != mapping["sequence_B_method"]
    )


def record_matches_mapping(
    record: Mapping[str, Any],
    mapping: Mapping[str, str],
) -> bool:
    stored = record.get("mapping")
    return (
        isinstance(stored, Mapping)
        and stored.get("sequence_A_method") == mapping["A"]
        and stored.get("sequence_B_method") == mapping["B"]
    )


def select_effective_records(
    records: Iterable[Mapping[str, Any]],
) -> Dict[RecordKey, Mapping[str, Any]]:
    """Select the latest valid record per key, otherwise the latest failure."""
    grouped: Dict[RecordKey, List[Mapping[str, Any]]] = {}
    for record in records:
        key = record_key(record)
        if key is not None:
            grouped.setdefault(key, []).append(record)
    selected = {}
    for key, candidates in grouped.items():
        valid = [record for record in candidates if is_valid_record(record)]
        selected[key] = valid[-1] if valid else candidates[-1]
    return selected


def _winner_in_method_space(record: Mapping[str, Any], criterion: str) -> str:
    winner = record["parsed"][criterion]["winner"]
    if winner == "tie":
        return "tie"
    return str(record["mapping"][f"sequence_{winner}_method"])


def _method_score(
    record: Mapping[str, Any],
    criterion: str,
    method: str,
) -> float:
    mapping = record["mapping"]
    alias = "A" if mapping["sequence_A_method"] == method else "B"
    return float(record["parsed"][criterion][alias])


def resolve_preference(
    winners: Sequence[str],
    methods: Sequence[str],
) -> Tuple[str, float]:
    """Return final preference and method-0 points averaged over both orders."""
    if len(winners) != 2:
        raise ValueError("two order-specific winners are required")
    directional = {winner for winner in winners if winner != "tie"}
    if set(methods).issubset(directional):
        preference = "tie"
    else:
        points = [
            0.5 if winner == "tie" else 1.0 if winner == methods[0] else 0.0
            for winner in winners
        ]
        mean_points = float(np.mean(points))
        preference = (
            methods[0]
            if mean_points > 0.5
            else methods[1]
            if mean_points < 0.5
            else "tie"
        )
        return preference, mean_points
    points = [
        0.5 if winner == "tie" else 1.0 if winner == methods[0] else 0.0
        for winner in winners
    ]
    return preference, float(np.mean(points))


def build_per_pair_rows(
    pairs: Sequence[PairSpec],
    records: Iterable[Mapping[str, Any]],
    methods: Sequence[str],
    prompt_version: str = PROMPT_VERSION,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    selected = select_effective_records(records)
    rows = []
    for pair in pairs:
        key1 = (pair.subset, pair.name, 1, prompt_version)
        key2 = (pair.subset, pair.name, 2, prompt_version)
        order_records = [selected.get(key1), selected.get(key2)]
        valid_records = [
            record
            for order_index, record in enumerate(order_records, start=1)
            if record is not None
            and is_valid_record(record)
            and (
                seed is None
                or record_matches_mapping(
                    record,
                    order_mapping(
                        methods,
                        seed,
                        pair.subset,
                        pair.name,
                        order_index,
                    ),
                )
            )
        ]
        row: Dict[str, Any] = {
            "subset": pair.subset,
            "name": pair.name,
            "status": "complete" if len(valid_records) == 2 else "incomplete",
            "valid_orders": len(valid_records),
        }
        if len(valid_records) != 2:
            errors = [
                str(record.get("error_type", "missing"))
                if record is not None
                else "missing"
                for record in order_records
            ]
            row["errors"] = ";".join(errors)
            rows.append(row)
            continue

        row["errors"] = ""
        for criterion in CRITERIA:
            winners = [
                _winner_in_method_space(record, criterion)
                for record in valid_records
            ]
            preference, method0_points = resolve_preference(winners, methods)
            scores = {
                method: float(
                    np.mean(
                        [
                            _method_score(record, criterion, method)
                            for record in valid_records
                        ]
                    )
                )
                for method in methods
            }
            prefix = criterion
            row[f"{prefix}_{methods[0]}_score"] = scores[methods[0]]
            row[f"{prefix}_{methods[1]}_score"] = scores[methods[1]]
            row[f"{prefix}_paired_difference"] = (
                scores[methods[0]] - scores[methods[1]]
            )
            row[f"{prefix}_order_1_winner"] = winners[0]
            row[f"{prefix}_order_2_winner"] = winners[1]
            row[f"{prefix}_preference"] = preference
            row[f"{prefix}_{methods[0]}_preference_points"] = method0_points
            row[f"{prefix}_exact_order_agreement"] = winners[0] == winners[1]
        rows.append(row)
    return rows


def bootstrap_mean_ci(
    values: Sequence[float],
    repeats: int,
    rng: np.random.Generator,
) -> List[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    if repeats <= 0:
        raise ValueError("bootstrap repeats must be positive")
    indices = rng.integers(0, array.size, size=(repeats, array.size))
    means = array[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return [float(low), float(high)]


def summarize_rows(
    pair_rows: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
    subsets: Sequence[str],
    bootstrap_repeats: int,
    seed: int,
) -> Dict[str, Any]:
    scopes: Dict[str, Any] = {}
    for scope_index, scope in enumerate([*subsets, "ALL"]):
        complete = [
            row
            for row in pair_rows
            if row["status"] == "complete"
            and (scope == "ALL" or row["subset"] == scope)
        ]
        scope_result: Dict[str, Any] = {
            "n_pairs": len(complete),
            "criteria": {},
        }
        if not complete:
            scopes[scope] = scope_result
            continue
        for criterion_index, criterion in enumerate(CRITERIA):
            rng = np.random.default_rng(
                seed + 1009 * scope_index + 9176 * criterion_index
            )
            method_scores = {
                method: [
                    float(row[f"{criterion}_{method}_score"])
                    for row in complete
                ]
                for method in methods
            }
            differences = [
                float(row[f"{criterion}_paired_difference"])
                for row in complete
            ]
            preferences = [
                str(row[f"{criterion}_preference"]) for row in complete
            ]
            points = [
                float(row[f"{criterion}_{methods[0]}_preference_points"])
                for row in complete
            ]
            agreements = [
                float(bool(row[f"{criterion}_exact_order_agreement"]))
                for row in complete
            ]
            score_result = {}
            for method in methods:
                score_result[method] = {
                    "mean": float(np.mean(method_scores[method])),
                    "bootstrap_ci95": bootstrap_mean_ci(
                        method_scores[method],
                        bootstrap_repeats,
                        rng,
                    ),
                }
            preference_result = {}
            for outcome in (*methods, "tie"):
                indicators = [
                    float(preference == outcome) for preference in preferences
                ]
                preference_result[outcome] = {
                    "proportion": float(np.mean(indicators)),
                    "bootstrap_ci95": bootstrap_mean_ci(
                        indicators,
                        bootstrap_repeats,
                        rng,
                    ),
                }
            scope_result["criteria"][criterion] = {
                "scores": score_result,
                "paired_difference": {
                    "definition": f"{methods[0]} - {methods[1]}",
                    "mean": float(np.mean(differences)),
                    "bootstrap_ci95": bootstrap_mean_ci(
                        differences,
                        bootstrap_repeats,
                        rng,
                    ),
                },
                "preference": preference_result,
                f"{methods[0]}_preference_points": {
                    "definition": "win=1, tie=0.5, loss=0; mean over orders",
                    "mean": float(np.mean(points)),
                    "bootstrap_ci95": bootstrap_mean_ci(
                        points,
                        bootstrap_repeats,
                        rng,
                    ),
                },
                "exact_order_agreement": {
                    "rate": float(np.mean(agreements)),
                    "bootstrap_ci95": bootstrap_mean_ci(
                        agreements,
                        bootstrap_repeats,
                        rng,
                    ),
                },
            }
        scopes[scope] = scope_result
    return scopes


def write_per_pair_csv(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    base_fields = ["subset", "name", "status", "valid_orders", "errors"]
    other_fields = sorted(
        {
            key
            for row in rows
            for key in row
            if key not in set(base_fields)
        }
    )
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[*base_fields, *other_fields],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    path: str,
    pair_rows: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    pairs: Sequence[PairSpec],
    issues: Sequence[DiscoveryIssue],
    methods: Sequence[str],
    subsets: Sequence[str],
    bootstrap_repeats: int,
    seed: int,
    environment: Mapping[str, Any],
) -> None:
    selected = select_effective_records(records)
    relevant = {
        key: record
        for key, record in selected.items()
        if key[3] == PROMPT_VERSION
        and any(
            pair.subset == key[0] and pair.name == key[1] for pair in pairs
        )
    }
    failures = []
    for key, record in sorted(relevant.items()):
        expected_mapping = order_mapping(
            methods,
            seed,
            key[0],
            key[1],
            key[2],
        )
        if not is_valid_record(record) or not record_matches_mapping(
            record,
            expected_mapping,
        ):
            failures.append(
                {
                    "subset": key[0],
                    "name": key[1],
                    "order": key[2],
                    "error_type": record.get(
                        "error_type",
                        (
                            "mapping_mismatch"
                            if is_valid_record(record)
                            else "invalid_record"
                        ),
                    ),
                    "error": record.get("error"),
                }
            )
    valid_judgments = sum(
        is_valid_record(record)
        and record_matches_mapping(
            record,
            order_mapping(methods, seed, key[0], key[1], key[2]),
        )
        for key, record in relevant.items()
    )
    summary = {
        "protocol": {
            "prompt_version": PROMPT_VERSION,
            "model": dict(environment),
            "methods": list(methods),
            "anonymous_aliases": ["A", "B"],
            "frames_per_sequence": 17,
            "images_per_request": 34,
            "orders_per_pair": 2,
            "order_seed": seed,
            "bootstrap_unit": "pair",
            "bootstrap_repeats": bootstrap_repeats,
            "no_combined_mllm_metric": True,
        },
        "completion": {
            "discovered_complete_pairs": len(pairs),
            "expected_judgments": 2 * len(pairs),
            "valid_judgments": valid_judgments,
            "failed_or_missing_judgments": 2 * len(pairs) - valid_judgments,
            "failed_requests": failures,
            "discovery_issues": [
                {
                    "subset": issue.subset,
                    "name": issue.name,
                    "error_type": issue.error_type,
                    "error": issue.message,
                }
                for issue in issues
            ],
        },
        "scores": summarize_rows(
            pair_rows,
            methods,
            subsets,
            bootstrap_repeats,
            seed,
        ),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)


def _discovery_error_record(
    issue: DiscoveryIssue,
    mapping: Mapping[str, str],
    order_index: int,
    environment: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "record_type": "judgment",
        "key": {
            "subset": issue.subset,
            "name": issue.name,
            "order": order_index,
            "prompt_version": PROMPT_VERSION,
        },
        "created_at": _utc_now(),
        "mapping": {
            "sequence_A_method": mapping["A"],
            "sequence_B_method": mapping["B"],
        },
        "status": "error",
        "error_type": issue.error_type,
        "error": issue.message,
        "environment": dict(environment),
    }


def main() -> None:
    args = parse_args()
    if args.expected_frames != 17:
        raise ValueError("this fixed protocol requires exactly 17 frames")
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")

    raw_path = args.raw_output or os.path.join(
        args.out_root, "qwen3vl_raw.jsonl"
    )
    per_pair_path = args.per_pair_output or os.path.join(
        args.out_root, "qwen3vl_per_pair.csv"
    )
    summary_path = args.summary_output or os.path.join(
        args.out_root, "qwen3vl_summary.json"
    )
    pairs, issues = discover_pairs(
        args.out_root,
        args.methods,
        args.subsets,
        args.expected_frames,
        args.pair_names,
    )
    if args.limit:
        pairs = pairs[: args.limit]
    print(
        f"Discovered {len(pairs)} complete pairs and "
        f"{len(issues)} incomplete/invalid pairs.",
        flush=True,
    )

    environment = collect_environment(args.model_path)
    records = load_jsonl(raw_path)
    selected = select_effective_records(records)

    if not args.summarize_only:
        active_pair_keys = {(pair.subset, pair.name) for pair in pairs}
        for issue in issues:
            if args.limit and (issue.subset, issue.name) not in active_pair_keys:
                continue
            for order_index in (1, 2):
                key = (
                    issue.subset,
                    issue.name,
                    order_index,
                    PROMPT_VERSION,
                )
                mapping = order_mapping(
                    args.methods,
                    args.seed,
                    issue.subset,
                    issue.name,
                    order_index,
                )
                if (
                    key in selected
                    and is_valid_record(selected[key])
                    and record_matches_mapping(selected[key], mapping)
                ):
                    continue
                record = _discovery_error_record(
                    issue,
                    mapping,
                    order_index,
                    environment,
                )
                append_jsonl(raw_path, record)
                records.append(record)
                selected[key] = record

        pending = []
        for pair in pairs:
            for order_index in (1, 2):
                key = (
                    pair.subset,
                    pair.name,
                    order_index,
                    PROMPT_VERSION,
                )
                mapping = order_mapping(
                    args.methods,
                    args.seed,
                    pair.subset,
                    pair.name,
                    order_index,
                )
                if (
                    key not in selected
                    or not is_valid_record(selected[key])
                    or not record_matches_mapping(selected[key], mapping)
                ):
                    pending.append((pair, order_index))
        print(
            f"{len(pending)} judgments pending; "
            f"{2 * len(pairs) - len(pending)} resume hits.",
            flush=True,
        )
        if pending:
            model, processor = load_model_and_processor(
                args.model_path,
                args.num_gpus,
                args.max_memory_per_gpu,
            )
            environment = {
                **environment,
                "hf_device_map": {
                    str(key): str(value)
                    for key, value in model.hf_device_map.items()
                },
            }
            for completed, (pair, order_index) in enumerate(pending, start=1):
                mapping = order_mapping(
                    args.methods,
                    args.seed,
                    pair.subset,
                    pair.name,
                    order_index,
                )
                print(
                    f"[{completed}/{len(pending)}] "
                    f"{pair.subset}/{pair.name} order={order_index}",
                    flush=True,
                )
                try:
                    record = evaluate_order(
                        pair,
                        mapping,
                        order_index,
                        model,
                        processor,
                        environment,
                        args.max_new_tokens,
                        args.num_gpus,
                    )
                except Exception as error:
                    record = _base_record(
                        pair,
                        mapping,
                        order_index,
                        environment,
                        args.max_new_tokens,
                    )
                    record.update(
                        {
                            "status": "error",
                            "error_type": "unexpected_error",
                            "error": f"{type(error).__name__}: {error}",
                            "traceback": traceback.format_exc(),
                        }
                    )
                append_jsonl(raw_path, record)
                records.append(record)
                selected[record_key(record)] = record
                print(f"  status={record['status']}", flush=True)

    pair_rows = build_per_pair_rows(
        pairs,
        records,
        args.methods,
        seed=args.seed,
    )
    write_per_pair_csv(per_pair_path, pair_rows)
    write_summary(
        summary_path,
        pair_rows,
        records,
        pairs,
        issues,
        args.methods,
        args.subsets,
        args.bootstrap_repeats,
        args.seed,
        environment,
    )
    print(f"Saved {raw_path}", flush=True)
    print(f"Saved {per_pair_path}", flush=True)
    print(f"Saved {summary_path}", flush=True)


if __name__ == "__main__":
    main()
