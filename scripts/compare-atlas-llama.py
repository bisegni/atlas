#!/usr/bin/env python3
"""Run matched Atlas and llama.cpp benchmarks and generate a comparison report.

This script is intentionally strict: it records every command, preserves raw
stdout/stderr, normalizes supported JSON output, and refuses to claim a valid
comparison when required workload fields do not match.

Atlas is expected to expose the matched benchmark JSON contract described in:
    docs/atlas-engineering.md (section 6, Performance instrumentation)

Until that CLI is implemented, pass --atlas-command with a shell-style command
template. Supported placeholders are documented in --help.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable


@dataclass
class NormalizedSample:
    engine: str
    workload_id: str
    prompt_tokens: int
    decode_tokens: int
    prefill_ms: float | None
    decode_ms: float | None
    prefill_tok_s: float | None
    decode_tok_s: float | None
    host_wall_ms: float | None
    source: dict[str, Any]


class ComparisonError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_command(command: list[str], cwd: Path | None, env: dict[str, str] | None = None) -> tuple[int, str, str, float]:
    started = time.perf_counter()
    process = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return process.returncode, process.stdout, process.stderr, elapsed_ms


def parse_json_stream(text: str) -> list[Any]:
    """Parse a JSON document, JSON array, or JSONL mixed with diagnostic text."""
    stripped = text.strip()
    if not stripped:
        return []

    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        pass

    records: list[Any] = []
    for line in stripped.splitlines():
        line = line.strip()
        if not line or not (line.startswith("{") or line.startswith("[")):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            records.extend(parsed)
        else:
            records.append(parsed)
    return records


def first_number(record: dict[str, Any], keys: Iterable[str]) -> float | None:
    for key in keys:
        value: Any = record
        found = True
        for component in key.split("."):
            if not isinstance(value, dict) or component not in value:
                found = False
                break
            value = value[component]
        if found and isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def first_int(record: dict[str, Any], keys: Iterable[str]) -> int | None:
    value = first_number(record, keys)
    return int(value) if value is not None else None


def normalize_atlas(records: list[Any], workload_id: str, expected_prompt: int, expected_decode: int) -> list[NormalizedSample]:
    samples: list[NormalizedSample] = []
    for item in records:
        if not isinstance(item, dict):
            continue

        prompt_tokens = first_int(item, ["prompt_tokens", "workload.prompt_tokens"])
        decode_tokens = first_int(item, [
            "measured_generated_tokens",
            "decode_tokens",
            "generated_tokens",
            "workload.decode_tokens",
        ])

        if prompt_tokens is None or decode_tokens is None:
            continue

        prefill_ms = first_number(item, ["prefill_time_ms", "timing.prefill_ms", "prefill_ms"])
        decode_ms = first_number(item, ["decode_time_ms", "timing.decode_ms", "decode_ms"])
        prefill_tok_s = first_number(item, ["prefill_tokens_per_second", "prefill_tok_s"])
        decode_tok_s = first_number(item, ["decode_tokens_per_second", "decode_tok_s"])
        host_wall_ms = first_number(item, ["host_wall_time_ms", "timing.host_ms", "host_ms"])

        if prefill_tok_s is None and prefill_ms and prefill_ms > 0:
            prefill_tok_s = prompt_tokens / (prefill_ms / 1000.0)
        if decode_tok_s is None and decode_ms and decode_ms > 0:
            decode_tok_s = decode_tokens / (decode_ms / 1000.0)

        samples.append(NormalizedSample(
            engine="atlas",
            workload_id=workload_id,
            prompt_tokens=prompt_tokens,
            decode_tokens=decode_tokens,
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            prefill_tok_s=prefill_tok_s,
            decode_tok_s=decode_tok_s,
            host_wall_ms=host_wall_ms,
            source=item,
        ))

    if not samples:
        raise ComparisonError(
            "Atlas output did not contain a supported matched-benchmark JSON record. "
            "Implement the documented contract or supply an Atlas command that emits it."
        )

    validate_workload(samples, expected_prompt, expected_decode, "Atlas")
    return samples


def normalize_llama(records: list[Any], workload_id: str, expected_prompt: int, expected_decode: int) -> list[NormalizedSample]:
    samples: list[NormalizedSample] = []

    for item in records:
        if not isinstance(item, dict):
            continue

        # llama-bench JSON fields have changed across releases. Accept common
        # variants while retaining the complete source record in the artifact.
        prompt_tokens = first_int(item, ["n_prompt", "n_prompt_tokens", "prompt_tokens", "pp"])
        decode_tokens = first_int(item, ["n_gen", "n_gen_tokens", "decode_tokens", "tg"])
        test = item.get("test")

        # Some llama-bench records represent pp and tg as separate tests.
        if prompt_tokens is None and isinstance(test, str) and test.startswith("pp"):
            suffix = "".join(ch for ch in test[2:] if ch.isdigit())
            prompt_tokens = int(suffix) if suffix else expected_prompt
        if decode_tokens is None and isinstance(test, str) and test.startswith("tg"):
            suffix = "".join(ch for ch in test[2:] if ch.isdigit())
            decode_tokens = int(suffix) if suffix else expected_decode

        # A combined invocation may emit separate pp and tg records. Keep them;
        # merge_llama_samples() combines matching phase statistics later.
        prompt_tokens = prompt_tokens if prompt_tokens is not None else 0
        decode_tokens = decode_tokens if decode_tokens is not None else 0

        prefill_tok_s = first_number(item, ["prefill_tok_s", "pp_avg", "tokens_per_second", "avg_ts"])
        decode_tok_s = first_number(item, ["decode_tok_s", "tg_avg", "tokens_per_second", "avg_ts"])
        prefill_ms = first_number(item, ["prefill_ms", "pp_ms", "time_ms"])
        decode_ms = first_number(item, ["decode_ms", "tg_ms", "time_ms"])

        is_pp = prompt_tokens > 0 and decode_tokens == 0
        is_tg = decode_tokens > 0 and prompt_tokens == 0
        if isinstance(test, str):
            is_pp = is_pp or test.startswith("pp")
            is_tg = is_tg or test.startswith("tg")

        throughput = first_number(item, ["tokens_per_second", "avg_ts"])
        if is_pp:
            prefill_tok_s = prefill_tok_s or throughput
            if prefill_ms is None and prefill_tok_s and prefill_tok_s > 0:
                prefill_ms = prompt_tokens / prefill_tok_s * 1000.0
            decode_tok_s = None
            decode_ms = None
        elif is_tg:
            decode_tok_s = decode_tok_s or throughput
            if decode_ms is None and decode_tok_s and decode_tok_s > 0:
                decode_ms = decode_tokens / decode_tok_s * 1000.0
            prefill_tok_s = None
            prefill_ms = None

        samples.append(NormalizedSample(
            engine="llama.cpp",
            workload_id=workload_id,
            prompt_tokens=prompt_tokens,
            decode_tokens=decode_tokens,
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            prefill_tok_s=prefill_tok_s,
            decode_tok_s=decode_tok_s,
            host_wall_ms=None,
            source=item,
        ))

    if not samples:
        raise ComparisonError("llama-bench output did not contain parseable JSON records")

    # Do not require every individual llama record to carry both dimensions;
    # llama-bench commonly emits independent pp and tg rows.
    pp_exists = any(s.prompt_tokens == expected_prompt and s.prefill_tok_s is not None for s in samples)
    tg_exists = any(s.decode_tokens == expected_decode and s.decode_tok_s is not None for s in samples)
    if expected_prompt > 0 and not pp_exists:
        raise ComparisonError(f"llama.cpp output has no pp{expected_prompt} result")
    if expected_decode > 0 and not tg_exists:
        raise ComparisonError(f"llama.cpp output has no tg{expected_decode} result")
    return samples


def validate_workload(samples: list[NormalizedSample], prompt_tokens: int, decode_tokens: int, engine: str) -> None:
    mismatches = [
        (sample.prompt_tokens, sample.decode_tokens)
        for sample in samples
        if sample.prompt_tokens != prompt_tokens or sample.decode_tokens != decode_tokens
    ]
    if mismatches:
        raise ComparisonError(
            f"{engine} workload mismatch: expected pp={prompt_tokens}, tg={decode_tokens}; "
            f"observed {mismatches[:5]}"
        )


def values(samples: list[NormalizedSample], field: str) -> list[float]:
    result: list[float] = []
    for sample in samples:
        value = getattr(sample, field)
        if value is not None and math.isfinite(value):
            result.append(float(value))
    return result


def summary(numbers: list[float]) -> dict[str, float | int | None]:
    if not numbers:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None, "stdev": None}
    return {
        "count": len(numbers),
        "min": min(numbers),
        "max": max(numbers),
        "mean": statistics.fmean(numbers),
        "median": statistics.median(numbers),
        "stdev": statistics.stdev(numbers) if len(numbers) > 1 else 0.0,
    }


def llama_phase_summary(samples: list[NormalizedSample], field: str, expected_count: int) -> dict[str, float | int | None]:
    filtered = []
    for sample in samples:
        value = getattr(sample, field)
        if value is None:
            continue
        if field.startswith("prefill") and sample.prompt_tokens != expected_count:
            continue
        if field.startswith("decode") and sample.decode_tokens != expected_count:
            continue
        filtered.append(float(value))
    return summary(filtered)


def ratio(atlas_value: float | None, llama_value: float | None) -> float | None:
    if atlas_value is None or llama_value is None or llama_value == 0:
        return None
    return atlas_value / llama_value


def format_number(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def build_markdown(report: dict[str, Any]) -> str:
    comparison = report["comparison"]
    atlas = report["summaries"]["atlas"]
    llama = report["summaries"]["llama_cpp"]

    lines = [
        "# Atlas vs llama.cpp matched benchmark",
        "",
        f"- Model SHA-256: `{report['identity']['model_sha256']}`",
        f"- Workload: pp{report['workload']['prompt_tokens']} + tg{report['workload']['decode_tokens']}",
        f"- Valid matched workload: `{str(report['valid_matched_workload']).lower()}`",
        "",
        "| Metric | Atlas median | llama.cpp median | Atlas/llama ratio |",
        "|---|---:|---:|---:|",
    ]

    rows = [
        ("Prefill tok/s", atlas["prefill_tok_s"]["median"], llama["prefill_tok_s"]["median"], comparison["prefill_speed_ratio"]),
        ("Decode tok/s", atlas["decode_tok_s"]["median"], llama["decode_tok_s"]["median"], comparison["decode_speed_ratio"]),
        ("Prefill ms", atlas["prefill_ms"]["median"], llama["prefill_ms"]["median"], comparison["prefill_time_ratio"]),
        ("Decode ms", atlas["decode_ms"]["median"], llama["decode_ms"]["median"], comparison["decode_time_ratio"]),
    ]
    for label, a, l, r in rows:
        lines.append(f"| {label} | {format_number(a)} | {format_number(l)} | {format_number(r)}× |")

    lines.extend(["", "## Diagnosis", ""])
    diagnoses = report.get("diagnosis", [])
    if diagnoses:
        lines.extend(f"- {item}" for item in diagnoses)
    else:
        lines.append("- Insufficient normalized metrics for an automatic diagnosis.")

    lines.extend(["", "## Commands", "", "### Atlas", "", "```text", " ".join(report["commands"]["atlas"]), "```", "", "### llama.cpp", "", "```text", " ".join(report["commands"]["llama_cpp"]), "```", ""])
    return "\n".join(lines)


def command_from_template(template: str, replacements: dict[str, str]) -> list[str]:
    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace("{" + key + "}", value)
    return shlex.split(rendered)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Automatically compare matched Atlas and llama.cpp workloads.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, type=Path, help="Exact GGUF file used by both engines")
    parser.add_argument("--atlas-model", help="Atlas model ID/path passed to the Atlas CLI; defaults to --model")
    parser.add_argument("--atlas-bin", default="target/release/atlas-cli", help="Atlas CLI executable")
    parser.add_argument("--llama-bench", required=True, type=Path, help="Path to llama-bench")
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--kv-cache-type", default="q4_0")
    parser.add_argument("--workload-id")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/atlas-vs-llama"))
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument(
        "--atlas-command",
        default=(
            "{atlas_bin} benchmark matched --model {atlas_model} "
            "--prompt-tokens {prompt_tokens} --decode-tokens {decode_tokens} "
            "--warmup-runs {warmup_runs} --runs {runs} --greedy "
            "--kv-cache-type {kv_cache_type} --output-format json"
        ),
        help=(
            "Atlas command template. Placeholders: {atlas_bin}, {atlas_model}, "
            "{model}, {prompt_tokens}, {decode_tokens}, {warmup_runs}, {runs}, "
            "{kv_cache_type}, {output_dir}."
        ),
    )
    parser.add_argument("--llama-extra", default="", help="Additional arguments appended to llama-bench")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    model = args.model.resolve()
    llama_bench = args.llama_bench.resolve()
    if not model.is_file():
        raise ComparisonError(f"Model does not exist: {model}")
    if not llama_bench.is_file():
        raise ComparisonError(f"llama-bench does not exist: {llama_bench}")
    if args.prompt_tokens < 0 or args.decode_tokens < 0 or args.runs < 1:
        raise ComparisonError("Token counts must be non-negative and runs must be positive")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    workload_id = args.workload_id or f"pp{args.prompt_tokens}_tg{args.decode_tokens}"

    replacements = {
        "atlas_bin": args.atlas_bin,
        "atlas_model": args.atlas_model or str(model),
        "model": str(model),
        "prompt_tokens": str(args.prompt_tokens),
        "decode_tokens": str(args.decode_tokens),
        "warmup_runs": str(args.warmup_runs),
        "runs": str(args.runs),
        "kv_cache_type": args.kv_cache_type,
        "output_dir": str(output_dir),
    }
    atlas_command = command_from_template(args.atlas_command, replacements)
    llama_command = [
        str(llama_bench),
        "-m", str(model),
        "-p", str(args.prompt_tokens),
        "-n", str(args.decode_tokens),
        "-r", str(args.runs),
        "-o", "json",
    ] + shlex.split(args.llama_extra)

    print("Atlas command:", shlex.join(atlas_command))
    print("llama.cpp command:", shlex.join(llama_command))
    if args.dry_run:
        return 0

    # Alternate startup order using a deterministic choice based on the workload.
    engines = ["atlas", "llama"] if (args.prompt_tokens + args.decode_tokens) % 2 == 0 else ["llama", "atlas"]
    raw: dict[str, dict[str, Any]] = {}

    for engine in engines:
        command = atlas_command if engine == "atlas" else llama_command
        code, stdout, stderr, wall_ms = run_command(command, args.cwd)
        raw[engine] = {
            "return_code": code,
            "stdout": stdout,
            "stderr": stderr,
            "wrapper_wall_ms": wall_ms,
            "command": command,
        }
        (output_dir / f"{engine}.stdout.txt").write_text(stdout, encoding="utf-8")
        (output_dir / f"{engine}.stderr.txt").write_text(stderr, encoding="utf-8")
        if code != 0:
            raise ComparisonError(
                f"{engine} command failed with exit code {code}. "
                f"See {output_dir / (engine + '.stderr.txt')}"
            )

    atlas_records = parse_json_stream(raw["atlas"]["stdout"])
    llama_records = parse_json_stream(raw["llama"]["stdout"])

    atlas_samples = normalize_atlas(atlas_records, workload_id, args.prompt_tokens, args.decode_tokens)
    llama_samples = normalize_llama(llama_records, workload_id, args.prompt_tokens, args.decode_tokens)

    atlas_summary = {
        field: summary(values(atlas_samples, field))
        for field in ["prefill_ms", "decode_ms", "prefill_tok_s", "decode_tok_s", "host_wall_ms"]
    }
    llama_summary = {
        "prefill_ms": llama_phase_summary(llama_samples, "prefill_ms", args.prompt_tokens),
        "decode_ms": llama_phase_summary(llama_samples, "decode_ms", args.decode_tokens),
        "prefill_tok_s": llama_phase_summary(llama_samples, "prefill_tok_s", args.prompt_tokens),
        "decode_tok_s": llama_phase_summary(llama_samples, "decode_tok_s", args.decode_tokens),
        "host_wall_ms": summary([]),
    }

    comparison = {
        "prefill_speed_ratio": ratio(atlas_summary["prefill_tok_s"]["median"], llama_summary["prefill_tok_s"]["median"]),
        "decode_speed_ratio": ratio(atlas_summary["decode_tok_s"]["median"], llama_summary["decode_tok_s"]["median"]),
        "prefill_time_ratio": ratio(atlas_summary["prefill_ms"]["median"], llama_summary["prefill_ms"]["median"]),
        "decode_time_ratio": ratio(atlas_summary["decode_ms"]["median"], llama_summary["decode_ms"]["median"]),
    }

    diagnosis: list[str] = []
    prefill_gap = comparison["prefill_time_ratio"]
    decode_gap = comparison["decode_time_ratio"]
    if prefill_gap is not None:
        diagnosis.append(f"Atlas prefill takes {prefill_gap:.2f}× the llama.cpp time.")
    if decode_gap is not None:
        diagnosis.append(f"Atlas decode takes {decode_gap:.2f}× the llama.cpp time.")
    if prefill_gap and decode_gap:
        if prefill_gap > decode_gap * 1.2:
            diagnosis.append("The larger relative loss is in prompt processing; prioritize batch projection, chunking, and prefill scheduling.")
        elif decode_gap > prefill_gap * 1.2:
            diagnosis.append("The larger relative loss is in token generation; prioritize matvec kernels, fusion, synchronization, and LM-head cost.")
        else:
            diagnosis.append("Prefill and decode have similar relative gaps; investigate shared projection kernels, command submission, and memory behavior.")

    report = {
        "schema_version": 1,
        "valid_matched_workload": True,
        "identity": {
            "model_path": str(model),
            "model_bytes": model.stat().st_size,
            "model_sha256": sha256_file(model),
            "host": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "workload": {
            "id": workload_id,
            "prompt_tokens": args.prompt_tokens,
            "decode_tokens": args.decode_tokens,
            "runs": args.runs,
            "warmup_runs": args.warmup_runs,
            "kv_cache_type": args.kv_cache_type,
        },
        "commands": {
            "atlas": atlas_command,
            "llama_cpp": llama_command,
        },
        "summaries": {
            "atlas": atlas_summary,
            "llama_cpp": llama_summary,
        },
        "comparison": comparison,
        "diagnosis": diagnosis,
        "samples": {
            "atlas": [asdict(sample) for sample in atlas_samples],
            "llama_cpp": [asdict(sample) for sample in llama_samples],
        },
        "wrapper": {
            "atlas_wall_ms": raw["atlas"]["wrapper_wall_ms"],
            "llama_cpp_wall_ms": raw["llama"]["wrapper_wall_ms"],
            "execution_order": engines,
        },
    }

    json_path = output_dir / "comparison.json"
    md_path = output_dir / "comparison.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(build_markdown(report), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    for item in diagnosis:
        print("-", item)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ComparisonError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
