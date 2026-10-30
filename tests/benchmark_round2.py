#!/usr/bin/env python3
"""Interleaved, correctness-checked round-two CLI benchmark."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

import numpy as np


CASES = [
    {"name": "default", "sr": 96000, "n_fft": 8192, "hop_length": 128,
     "n_mels": 256, "f_min": 0.0, "f_max": 22050.0},
    {"name": "small_mels", "sr": 96000, "n_fft": 8192, "hop_length": 128,
     "n_mels": 64, "f_min": 0.0, "f_max": 22050.0},
    {"name": "large_fft", "sr": 96000, "n_fft": 16384, "hop_length": 128,
     "n_mels": 256, "f_min": 0.0, "f_max": 22050.0},
    {"name": "narrow_band", "sr": 96000, "n_fft": 16384, "hop_length": 128,
     "n_mels": 256, "f_min": 300.0, "f_max": 8000.0},
]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_program(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("program must use NAME=/absolute/path")
    name, path_text = value.split("=", 1)
    path = Path(path_text).expanduser().resolve()
    if not name or not path.is_absolute() or not path.is_file():
        raise argparse.ArgumentTypeError(f"invalid program: {value}")
    return name, path


def command_for(program, pcm_path, output_path, params):
    command = [str(program), str(pcm_path), str(output_path)]
    for key in ("sr", "n_fft", "hop_length", "n_mels", "f_min", "f_max"):
        command.extend([f"--{key}", str(params[key])])
    return command


def invoke(program, pcm_path, output_path, params):
    command = command_for(program, pcm_path, output_path, params)
    output_path.unlink(missing_ok=True)
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    values = np.fromfile(output_path, dtype="<f4")
    return values, command, completed.stderr


def paired_summary(samples, baseline_name, names):
    result = {}
    for name in names:
        ratios = [a / b for a, b in zip(samples[baseline_name], samples[name])]
        median = float(statistics.median(ratios))
        result[name] = {
            "baseline_over_program_median": median,
            "paired_ratio_mad": float(statistics.median(abs(x - median) for x in ratios)),
            "paired_ratios": ratios,
        }
    return result


def rotation_order(names, index):
    cycle, offset = divmod(index, len(names))
    order = names[offset:] + names[:offset]
    return order if cycle % 2 == 0 else list(reversed(order))


def affinity_snapshot():
    try:
        return sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", action="append", required=True, type=parse_program)
    parser.add_argument("--pcm", type=Path)
    parser.add_argument("--suite", choices=("current-long", "current-short"), required=True)
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--case", action="append", choices=[case["name"] for case in CASES])
    parser.add_argument("--baseline", default="optimized")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()

    if args.runs < 1 or args.warmups < 0:
        parser.error("--runs must be positive and --warmups nonnegative")
    if args.suite == "current-long" and args.pcm is None:
        parser.error("--pcm is required for current-long")
    programs = dict(args.program)
    if len(programs) != len(args.program):
        parser.error("program names must be unique")
    if args.baseline not in programs:
        parser.error(f"baseline program {args.baseline!r} was not provided")
    cases = [c for c in CASES if args.case is None or c["name"] in args.case]
    if not cases:
        parser.error("no cases selected")

    args.work_dir = args.work_dir.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    report = {
        "suite": args.suite,
        "runs": args.runs,
        "warmups": args.warmups,
        "seed": args.seed if args.suite == "current-short" else None,
        "tolerance": {"rtol": args.rtol, "atol": args.atol},
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "cpu_affinity": affinity_snapshot(),
            "MELSPEC_THREADS": os.environ.get("MELSPEC_THREADS"),
            "TMPDIR": os.environ.get("TMPDIR"),
            "programs": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in programs.items()
            },
        },
        "cases": [],
    }

    for case_index, case in enumerate(cases):
        params = {key: value for key, value in case.items() if key != "name"}
        case_dir = args.work_dir / case["name"]
        case_dir.mkdir(parents=True, exist_ok=True)
        if args.suite == "current-short":
            frames = 128
            sample_count = (frames - 1) * params["hop_length"] + params["n_fft"]
            pcm = rng.integers(-32768, 32768, size=sample_count, dtype=np.int16)
            pcm_path = case_dir / "input.pcm"
            pcm.astype("<i2").tofile(pcm_path)
        else:
            pcm_path = args.pcm.expanduser().resolve()
            if not pcm_path.is_file():
                raise RuntimeError(f"PCM does not exist: {pcm_path}")
            size_bytes = pcm_path.stat().st_size
            if size_bytes % 2:
                raise RuntimeError("PCM byte count is not divisible by int16 size")
            sample_count = size_bytes // 2
            frames = max(0, 1 + (sample_count - params["n_fft"]) // params["hop_length"])
            if frames == 0:
                raise RuntimeError(f"PCM is shorter than {case['name']} n_fft")
        input_hash = sha256_file(pcm_path)

        output_paths = {name: case_dir / f"{name}.bin" for name in programs}
        preflight = {}
        preflight_argv = {}
        for name, program in programs.items():
            values, argv, _ = invoke(program, pcm_path, output_paths[name], params)
            if values.size != params["n_mels"] * frames:
                raise RuntimeError(
                    f"{name} {case['name']}: expected {params['n_mels'] * frames} "
                    f"values, found {values.size}"
                )
            if not np.all(np.isfinite(values)):
                raise RuntimeError(f"{name} {case['name']}: preflight output is non-finite")
            preflight[name] = values
            preflight_argv[name] = argv
        canonical = preflight[args.baseline].astype(np.float64)
        consistency = {}
        for name, values in preflight.items():
            diff = np.abs(values.astype(np.float64) - canonical)
            passed = bool(np.allclose(values, canonical, rtol=args.rtol, atol=args.atol))
            consistency[name] = {
                "matches_baseline": passed,
                "max_abs_error": float(diff.max(initial=0.0)),
            }
            if not passed:
                raise RuntimeError(
                    f"{name} disagrees with {args.baseline} on {case['name']}: "
                    f"max_abs={diff.max(initial=0.0):.8g}"
                )

        names = list(programs)
        samples = {name: [] for name in names}
        timed_hashes = {name: [] for name in names}
        timing_argv = {name: preflight_argv[name] for name in names}
        warmup_order = []
        for i in range(args.warmups):
            order = rotation_order(names, i)
            warmup_order.append(order)
            for name in order:
                output_paths[name].unlink(missing_ok=True)
                subprocess.run(
                    timing_argv[name], check=True, capture_output=True, text=True
                )
                values = np.fromfile(output_paths[name], dtype="<f4")
                if not np.array_equal(values, preflight[name]):
                    raise RuntimeError(f"{name} warmup output changed on {case['name']}")

        round_orders = []
        for run in range(args.runs):
            order = rotation_order(names, run)
            round_orders.append(order)
            for name in order:
                output_paths[name].unlink(missing_ok=True)
                start = time.perf_counter()
                subprocess.run(
                    timing_argv[name], check=True, capture_output=True, text=True
                )
                elapsed = time.perf_counter() - start
                values = np.fromfile(output_paths[name], dtype="<f4")
                if values.size != preflight[name].size or not np.allclose(
                    values, preflight[name], rtol=args.rtol, atol=args.atol
                ):
                    raise RuntimeError(f"{name} output changed on {case['name']} run {run}")
                samples[name].append(elapsed)
                timed_hashes[name].append(sha256_file(output_paths[name]))

        timings = {}
        for name, values in samples.items():
            timings[name] = {
                "argv": timing_argv[name],
                "program_sha256": report["environment"]["programs"][name]["sha256"],
                "samples_seconds": values,
                "median_seconds": float(statistics.median(values)),
                "min_seconds": float(min(values)),
                "output_sha256_each_run": timed_hashes[name],
            }
        case_result = {
            "name": case["name"],
            "parameters": params,
            "frames": frames,
            "input_samples": sample_count,
            "input_path": str(pcm_path),
            "input_sha256": input_hash,
            "baseline_program": args.baseline,
            "consistency": consistency,
            "preflight_argv": preflight_argv,
            "warmup_order": warmup_order,
            "round_orders": round_orders,
            "timings": timings,
            "paired_ratios": paired_summary(samples, args.baseline, names),
        }
        report["cases"].append(case_result)
        print(
            f"PASS {case['name']} "
            + ", ".join(
                f"{name} median={timings[name]['median_seconds']:.6f}s "
                f"min={timings[name]['min_seconds']:.6f}s"
                for name in names
            ),
            flush=True,
        )

    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
