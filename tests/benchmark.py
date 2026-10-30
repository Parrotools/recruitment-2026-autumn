#!/usr/bin/env python3
"""Small, reproducible end-to-end benchmark for the mel spectrogram binaries."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time

import numpy as np


FRAME_COUNT = 128
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
        raise argparse.ArgumentTypeError("program must use NAME=PATH")
    name, path_text = value.split("=", 1)
    if not name or not path_text:
        raise argparse.ArgumentTypeError("program must use NAME=PATH")
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"program does not exist: {path}")
    return name, path


def invoke(program, pcm_path, output_path, params):
    command = [str(program), str(pcm_path), str(output_path)]
    for key in ("sr", "n_fft", "hop_length", "n_mels", "f_min", "f_max"):
        command.extend([f"--{key}", str(params[key])])
    subprocess.run(command, check=True, capture_output=True, text=True)
    values = np.fromfile(output_path, dtype="<f4")
    expected_count = params["n_mels"] * FRAME_COUNT
    if values.size != expected_count:
        raise RuntimeError(
            f"{program}: {expected_count} output values expected, got {values.size}"
        )
    return values.reshape(params["n_mels"], FRAME_COUNT)


def compare_outputs(outputs, canonical_name, rtol, atol):
    canonical = outputs[canonical_name].astype(np.float64)
    checks = {}
    for name, result in outputs.items():
        values = result.astype(np.float64)
        max_abs = float(np.max(np.abs(values - canonical), initial=0.0))
        passed = bool(np.allclose(values, canonical, rtol=rtol, atol=atol))
        checks[name] = {"matches_canonical": passed, "max_abs_error": max_abs}
        if not passed:
            raise RuntimeError(
                f"{name} disagrees with {canonical_name}: max_abs_error={max_abs:.8g}"
            )
    return checks


def benchmark_program(program, pcm_path, params, runs, directory):
    output_path = directory / "result.bin"
    times = []
    first_output = None
    argv = [str(program), str(pcm_path), str(output_path)]
    for key in ("sr", "n_fft", "hop_length", "n_mels", "f_min", "f_max"):
        argv.extend([f"--{key}", str(params[key])])
    for run in range(runs):
        start = time.perf_counter()
        subprocess.run(argv, check=True, capture_output=True, text=True)
        times.append(time.perf_counter() - start)
        if run == 0:
            values = np.fromfile(output_path, dtype="<f4")
            if values.size != params["n_mels"] * FRAME_COUNT:
                raise RuntimeError(f"{program}: unexpected output element count")
            first_output = values.reshape(params["n_mels"], FRAME_COUNT)
    return {
        "argv": argv,
        "min_seconds": min(times),
        "median_seconds": float(np.median(times)),
        "runs_seconds": times,
    }, first_output


def environment_snapshot(programs):
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": affinity,
        "MELSPEC_THREADS": os.environ.get("MELSPEC_THREADS"),
        "TMPDIR": os.environ.get("TMPDIR"),
        "programs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in programs.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", action="append", required=True, type=parse_program)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()
    if args.runs <= 0:
        parser.error("--runs must be positive")
    programs = dict(args.program)
    if len(programs) != len(args.program):
        parser.error("program names must be unique")
    if "original" not in programs or "optimized" not in programs:
        parser.error("provide both --program original=... and optimized=...")
    canonical_name = "reference" if "reference" in programs else "original"

    rng = np.random.default_rng(args.seed)
    report = {
        "seed": args.seed,
        "frame_count": FRAME_COUNT,
        "runs": args.runs,
        "tolerance": {"rtol": args.rtol, "atol": args.atol},
        "environment": environment_snapshot(programs),
        "cases": [],
    }

    with tempfile.TemporaryDirectory(prefix="melspec-benchmark-") as temp_dir:
        root = Path(temp_dir)
        for case in CASES:
            params = {key: value for key, value in case.items() if key != "name"}
            sample_count = (
                (FRAME_COUNT - 1) * params["hop_length"] + params["n_fft"]
            )
            pcm = rng.integers(-32768, 32768, size=sample_count, dtype=np.int16)
            pcm_path = root / f"{case['name']}.pcm"
            pcm.astype("<i2").tofile(pcm_path)
            input_digest = sha256_file(pcm_path)

            outputs = {}
            program_dirs = {}
            for name, program in programs.items():
                program_dir = root / f"{case['name']}-{name}"
                program_dir.mkdir()
                program_dirs[name] = program_dir
                outputs[name] = invoke(
                    program, pcm_path, program_dir / "check.bin", params
                )

            consistency = compare_outputs(
                outputs, canonical_name, args.rtol, args.atol
            )
            timing = {}
            for name, program in programs.items():
                timing[name], timed_output = benchmark_program(
                    program, pcm_path, params, args.runs, program_dirs[name]
                )
                if not np.array_equal(timed_output, outputs[name]):
                    raise RuntimeError(
                        f"{name} changed output between consistency and timing runs"
                    )
            case_result = {
                "name": case["name"],
                "parameters": params,
                "frames": FRAME_COUNT,
                "input_samples": sample_count,
                "input_sha256": input_digest,
                "canonical_program": canonical_name,
                "consistency": consistency,
                "timings": timing,
            }
            report["cases"].append(case_result)
            print(
                f"PASS {case['name']} "
                + ", ".join(
                    f"{name} min={timing[name]['min_seconds']:.6f}s "
                    f"median={timing[name]['median_seconds']:.6f}s"
                    for name in programs
                )
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
