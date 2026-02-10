#!/usr/bin/env python3
"""
Fault Injection Campaign Orchestrator for SPEC2017 Benchmarks

This script orchestrates a fault injection campaign using QEMU plugins.
It profiles the program once, then systematically injects bit flips and records results.

Workflow:
1. Run profiler once to get instruction profile (CSV with addresses and execution counts)
2. Generate sampling plan based on instructions and their execution counts
3. Run fault injector thousands of times with different parameters
4. Compare outputs with baseline and classify results

Sampling Strategy:
- For each instruction, we sample executions across the timeline (early, middle, late phases)
- We sample bits uniformly across the register width (64-bit for RISC-V) to avoid bias
- We use statistical sampling to estimate sensitivity with confidence intervals
"""

import os
import sys
import subprocess
import re
import csv
import argparse
import time
import random
import hashlib
import shlex
import multiprocessing
import tempfile
import shutil
import signal
import fcntl
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import lzma

# Global lock for CSV writing in parallel mode
CSV_LOCK = multiprocessing.Lock()

# Configuration - Default paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_QEMU_PATH = os.path.join(SCRIPT_DIR, "qemu-riscv64")
DEFAULT_PLUGINS_DIR = os.path.join(SCRIPT_DIR, "qemu_pin")
DEFAULT_BENCHMARKS_DIR = os.path.join(SCRIPT_DIR, "benchmarks")

# SPEC2017 installation path (for specdiff and reference outputs)
DEFAULT_SPEC_PATH = DEFAULT_BENCHMARKS_DIR

# Crash signals that indicate a real program crash (not normal exit)
# These are the signals that QEMU propagates when the emulated program crashes
CRASH_SIGNALS = {
    signal.SIGSEGV,  # Segmentation fault (11)
    signal.SIGBUS,   # Bus error (7)
    signal.SIGFPE,   # Floating point exception (8)
    signal.SIGILL,   # Illegal instruction (4)
    signal.SIGABRT,  # Abort (6)
    signal.SIGTRAP,  # Trap (5)
    signal.SIGSYS,   # Bad system call (31)
}

# SPEC2017 Benchmark configurations
# Format: benchmark_name -> {dir, binary, args, stdin, spec_num, outputs, specdiff_opts}
# outputs: list of (generated_file, reference_file) tuples
# specdiff_opts: additional options for specdiff (tolerances, etc.)
BENCHMARK_CONFIG = {
    # NOTE: SPEC runs bwaves 4 times (bwaves_1 through bwaves_4).
    # This config runs only bwaves_1 for faster fault injection testing.
    # To match full SPEC validation, you would need to run all 4.
    "bwaves": {
        "dir": "503.bwaves_r/run/run_base_refrate_riscv.0000",
        "binary": "../run_base_refrate_riscv.0000/bwaves_r_base.riscv",
        "args": ["bwaves_1"],
        "stdin": "bwaves_1.in",
        "spec_num": "503.bwaves_r",
        "stdout_file": "bwaves_1.out",  # SPEC redirects stdout to this file
        "outputs": [
            ("bwaves_1.out", "bwaves_1.out"),
        ],
        "specdiff_opts": ["--abstol", "1e-16", "--reltol", "0.015"]
    },
    "cactuBSSN": {
        "dir": "507.cactuBSSN_r/run/run_base_refrate_riscv.0000",
        "binary": "../run_base_refrate_riscv.0000/cactusBSSN_r_base.riscv",
        "args": ["spec_ref.par"],
        "stdin": None,
        "spec_num": "507.cactuBSSN_r",
        "stdout_file": "spec_ref.out",  # SPEC redirects stdout to this file
        "outputs": [
            ("gxx.xl", "gxx.xl"),
            ("gxy.xl", "gxy.xl"),
            ("spec_ref.out", "spec_ref.out"),
        ],
        "specdiff_opts": ["--abstol", "5e-13", "--floatcompare"]
    },
    "namd": {
        "dir": "508.namd_r/run/run_base_refrate_riscv.0000",
        "binary": "../run_base_refrate_riscv.0000/namd_r_base.riscv",
        "args": ["--input", "apoa1.input", "--output", "apoa1.ref.output", "--iterations", "65"],
        "stdin": None,
        "spec_num": "508.namd_r",
        "outputs": [
            ("apoa1.ref.output", "apoa1.ref.output"),
        ],
        "specdiff_opts": ["--abstol", "5e-05"]
    },
    "parest": {
        "dir": "510.parest_r/run/run_base_refrate_riscv.0000",
        "binary": "../run_base_refrate_riscv.0000/parest_r_base.riscv",
        "args": ["ref.prm"],
        "stdin": None,
        "spec_num": "510.parest_r",
        "outputs": [
            ("log", "log"),
            ("statistics", "statistics"),
        ],
        "specdiff_opts": ["--abstol", "1e-03", "--floatcompare"]
    },
    # NOTE: POV-Ray produces SPEC-benchmark.tga as output.
    # The imagevalidate binary compares it against SPEC-benchmark.org.tga reference
    # and produces imagevalidate_SPEC-benchmark.tga.out with SSIM similarity scores.
    # For fault injection, you MUST run imagevalidate after POV-Ray rendering.
    "povray": {
        "dir": "511.povray_r/run/run_base_refrate_riscv.0000",
        "binary": "../run_base_refrate_riscv.0000/povray_r_base.riscv",
        "args": ["SPEC-benchmark-ref.ini"],
        "stdin": None,
        "spec_num": "511.povray_r",
        "needs_special_validation": True,
        "validation_binary": "../../exe/imagevalidate_511_base.riscv",
        "validation_args": ["SPEC-benchmark.tga", "../../data/refrate/compare/SPEC-benchmark.org.tga"],
        "validation_output": "imagevalidate_SPEC-benchmark.tga.out",
        "outputs": [
            ("imagevalidate_SPEC-benchmark.tga.out", "imagevalidate_SPEC-benchmark.tga.out"),
        ],
        "specdiff_opts": ["--reltol", "0.06"]
    },
    "lbm": {
        "dir": "519.lbm_r/run/run_base_refrate_riscv.0000",
        "binary": "../run_base_refrate_riscv.0000/lbm_r_base.riscv",
        "args": ["3000", "reference.dat", "0", "0", "100_100_130_ldc.of"],
        "stdin": None,
        "spec_num": "519.lbm_r",
        "stdout_file": "lbm.out",  # SPEC redirects stdout to this file
        "outputs": [
            ("lbm.out", "lbm.out"),
        ],
        "specdiff_opts": ["--abstol", "1e-07"]
    },
    # NOTE: WRF requires special validation using the 'diffwrf' binary.
    # The main output is wrfout_d01_* (binary). diffwrf compares it against
    # wrf_reference_01 and produces diffwrf_output_01.txt with PASSED/FAILED lines.
    # IMPORTANT: There is NO rsl.out.0000 reference file! Only diffwrf_output_01.txt exists.
    # For fault injection, you MUST run diffwrf after the benchmark to produce the validation output.
    # Post-run command: ./diffwrf_521_base.riscv wrfout_d01_2000-01-24_20_00_00 wrf_reference_01
    "wrf": {
        "dir": "521.wrf_r/run/run_base_refrate_riscv.0000",
        "binary": "../run_base_refrate_riscv.0000/wrf_r_base.riscv",
        "args": [],
        "stdin": None,
        "spec_num": "521.wrf_r",
        "needs_special_validation": True,  # Requires running diffwrf binary post-benchmark
        "validation_binary": "diffwrf_521_base.riscv",
        "validation_args": ["wrfout_d01_2000-01-24_20_00_00", "wrf_reference_01"],
        "outputs": [
            ("diffwrf_output_01.txt", "diffwrf_output_01.txt"),
        ],
        "specdiff_opts": ["--cw"]  # Collapse whitespace per compare.cmd
    },
    # NOTE: Blender produces sh3_no_char_0849.tga as output.
    # The imagevalidate binary compares it against sh3_no_char_0849.org.tga reference
    # and produces imagevalidate_sh3_no_char_0849.out with SSIM similarity scores.
    # For fault injection, you MUST run imagevalidate after Blender rendering.
    "blender": {
        "dir": "526.blender_r/run/run_base_refrate_riscv.0000",
        "binary": "../run_base_refrate_riscv.0000/blender_r_base.riscv",
        "args": ["sh3_no_char.blend", "--render-output", "sh3_no_char_", "--threads", "1", "-b", "-F", "RAWTGA", "-s", "849", "-e", "849", "-a"],
        "stdin": None,
        "spec_num": "526.blender_r",
        "needs_special_validation": True,
        "validation_binary": "../../exe/imagevalidate_526_base.riscv",
        "validation_args": ["-avg", "-threshold", "0.75", "-maxthreshold", "0.01", 
                           "sh3_no_char_0849.tga", "../../data/refrate/compare/sh3_no_char_0849.org.tga"],
        "validation_output": "imagevalidate_sh3_no_char_0849.out",
        "outputs": [
            ("imagevalidate_sh3_no_char_0849.out", "imagevalidate_sh3_no_char_0849.out"),
        ],
        "specdiff_opts": ["--reltol", "0.05"]  # per compare.cmd
    },
    # NOTE: CAM4 requires special validation using the 'cam4_validate' binary.
    # The main output is h0.nc (netCDF). cam4_validate compares it against
    # h0_ctrl.nc (in data/refrate/compare/) and produces cam4_validate.txt with "PASS: N points."
    # For fault injection, you MUST run cam4_validate after the benchmark.
    # Validation binary: ../exe/cam4_validate_527_base.riscv
    # Post-run command: ./cam4_validate_527_base.riscv 10 0.0035 <path_to_h0_ctrl.nc> h0.nc
    # Reference: data/refrate/compare/h0_ctrl.nc
    "cam4": {
        "dir": "527.cam4_r/run/run_base_refrate_riscv.0000",
        "binary": "../run_base_refrate_riscv.0000/cam4_r_base.riscv",
        "args": [],
        "stdin": None,
        "spec_num": "527.cam4_r",
        "needs_special_validation": True,  # Requires running cam4_validate binary post-benchmark
        "validation_binary": "../../exe/cam4_validate_527_base.riscv",
        "validation_args": ["10", "0.0035", "../../data/refrate/compare/h0_ctrl.nc", "h0.nc"],
        "validation_output": "cam4_validate.txt",
        "outputs": [
            ("cam4_validate.txt", "cam4_validate.txt"),
        ],
        "specdiff_opts": ["--cw"]  # Collapse whitespace per compare.cmd
    },
    # NOTE: ImageMagick produces refrate_output.tga as output.
    # The imagevalidate binary compares it against refrate_expected.tga reference
    # and produces refrate_validate.out with similarity scores.
    # For fault injection, you MUST run imagevalidate after the convert operation.
    "imagick": {
        "dir": "538.imagick_r/run/run_base_refrate_riscv.0000",
        "binary": "../run_base_refrate_riscv.0000/imagick_r_base.riscv",
        "args": ["-limit", "disk", "0", "refrate_input.tga", "-edge", "41", "-resample", "181%", "-emboss", "31", "-colorspace", "YUV", "-mean-shift", "19x19+15%", "-resize", "30%", "refrate_output.tga"],
        "stdin": None,
        "spec_num": "538.imagick_r",
        "needs_special_validation": True,
        "validation_binary": "../../exe/imagevalidate_538_base.riscv",
        "validation_args": ["-avg", "-threshold", "0.95", "-maxthreshold", "0.001", 
                           "refrate_output.tga", "../../data/refrate/compare/refrate_expected.tga"],
        "validation_output": "refrate_validate.out",
        "outputs": [
            ("refrate_validate.out", "refrate_validate.out"),
        ],
        "specdiff_opts": ["--reltol", "0.01"]
    },
    "nab": {
        "dir": "544.nab_r/run/run_base_refrate_riscv.0000",
        "binary": "../run_base_refrate_riscv.0000/nab_r_base.riscv",
        "args": ["1am0", "1122214447", "122"],
        "stdin": None,
        "spec_num": "544.nab_r",
        "stdout_file": "1am0.out",  # SPEC redirects stdout to this file
        "outputs": [
            ("1am0.out", "1am0.out"),
        ],
        "specdiff_opts": ["--reltol", "0.01", "--skipreltol", "2"]
    },
    "fotonik3d": {
        "dir": "549.fotonik3d_r/run/run_base_refrate_riscv.0000",
        "binary": "../run_base_refrate_riscv.0000/fotonik3d_r_base.riscv",
        "args": [],
        "stdin": None,
        "spec_num": "549.fotonik3d_r",
        # OBJ.dat.xz must be decompressed before running (see inputgen.cmd)
        "compressed_inputs": [("OBJ.dat.xz", "OBJ.dat")],
        "outputs": [
            ("pscyee.out", "pscyee.out"),
        ],
        "specdiff_opts": ["--abstol", "1e-27", "--reltol", "1e-10", "--obiwan", "--floatcompare"]
    },
    "roms": {
        "dir": "554.roms_r/run/run_base_refrate_riscv.0000",
        "binary": "../run_base_refrate_riscv.0000/roms_r_base.riscv",
        "args": [],
        "stdin": "ocean_benchmark2.in.x",
        "spec_num": "554.roms_r",
        "stdout_file": "ocean_benchmark2.log",  # SPEC redirects stdout to this file
        "outputs": [
            ("ocean_benchmark2.log", "ocean_benchmark2.log"),
        ],
        "specdiff_opts": ["--abstol", "1e-07", "--reltol", "1e-07"]
    },
}


# Test workload overrides: only the fields that differ from refrate.
# These use the same compiled binaries but with smaller/faster inputs.
# The working directory will be set up by copying test input files.
BENCHMARK_CONFIG_TEST = {
    "bwaves": {
        "dir": "503.bwaves_r/run/run_base_test_riscv.0000",
        "args": ["bwaves_1"],
        "stdin": "bwaves_1.in",
        "stdout_file": "bwaves_1.out",
        "outputs": [("bwaves_1.out", "bwaves_1.out")],
        "specdiff_opts": ["--abstol", "1e-16", "--reltol", "0.015"],
        "test_inputs": "503.bwaves_r/data/test/input",
    },
    "cactuBSSN": {
        "dir": "507.cactuBSSN_r/run/run_base_test_riscv.0000",
        "args": ["spec_test.par"],
        "stdout_file": "spec_test.out",
        "outputs": [
            ("gxx.xl", "gxx.xl"),
            ("gxy.xl", "gxy.xl"),
            ("spec_test.out", "spec_test.out"),
        ],
        "specdiff_opts": ["--abstol", "5e-13", "--floatcompare"],
        "test_inputs": "507.cactuBSSN_r/data/test/input",
    },
    "namd": {
        "dir": "508.namd_r/run/run_base_test_riscv.0000",
        "args": ["--input", "apoa1.input", "--output", "apoa1.test.output", "--iterations", "1"],
        "outputs": [("apoa1.test.output", "apoa1.test.output")],
        "specdiff_opts": ["--abstol", "5e-05"],
        "test_inputs": "508.namd_r/data/test/input",
        "extra_inputs": ["508.namd_r/data/all/input"],
    },
    "parest": {
        "dir": "510.parest_r/run/run_base_test_riscv.0000",
        "args": ["test.prm"],
        "outputs": [("log", "log"), ("statistics", "statistics")],
        "specdiff_opts": ["--abstol", "1e-03", "--floatcompare"],
        "test_inputs": "510.parest_r/data/test/input",
    },
    "povray": {
        "dir": "511.povray_r/run/run_base_test_riscv.0000",
        "args": ["SPEC-benchmark-test.ini"],
        "needs_special_validation": True,
        "validation_binary": "../../exe/imagevalidate_511_base.riscv",
        "validation_args": ["SPEC-benchmark.tga", "../../data/test/compare/SPEC-benchmark.org.tga"],
        "validation_output": "imagevalidate_SPEC-benchmark.tga.out",
        "outputs": [("imagevalidate_SPEC-benchmark.tga.out", "imagevalidate_SPEC-benchmark.tga.out")],
        "specdiff_opts": ["--reltol", "0.06"],
        "test_inputs": "511.povray_r/data/test/input",
        "extra_inputs": ["511.povray_r/data/all/input"],
    },
    "lbm": {
        "dir": "519.lbm_r/run/run_base_test_riscv.0000",
        "args": ["20", "reference.dat", "0", "1", "100_100_130_cf_a.of"],
        "stdout_file": "lbm.out",
        "outputs": [("lbm.out", "lbm.out")],
        "specdiff_opts": ["--abstol", "1e-07"],
        "test_inputs": "519.lbm_r/data/test/input",
    },
    "wrf": {
        "dir": "521.wrf_r/run/run_base_test_riscv.0000",
        "args": [],
        "needs_special_validation": True,
        "validation_binary": "diffwrf_521_base.riscv",
        "validation_args": ["wrfout_d01_2000-01-24_12_10_00", "wrf_reference_01"],
        "validation_output": "diffwrf_output_01.txt",
        "outputs": [("diffwrf_output_01.txt", "diffwrf_output_01.txt")],
        "specdiff_opts": ["--cw"],
        "test_inputs": "521.wrf_r/data/test/input",
        "extra_inputs": ["521.wrf_r/data/all/input"],
    },
    "blender": {
        "dir": "526.blender_r/run/run_base_test_riscv.0000",
        "args": ["cube.blend", "--render-output", "cube_", "--threads", "1", "-b", "-F", "RAWTGA", "-s", "1", "-e", "1", "-a"],
        "needs_special_validation": True,
        "validation_binary": "../../exe/imagevalidate_526_base.riscv",
        "validation_args": ["-avg", "-threshold", "0.75", "-maxthreshold", "0.01",
                           "cube_0001.tga", "../../data/test/compare/cube_0001.org.tga"],
        "validation_output": "imagevalidate_cube_0001.out",
        "outputs": [("imagevalidate_cube_0001.out", "imagevalidate_cube_0001.out")],
        "specdiff_opts": ["--reltol", "0.05"],
        "test_inputs": "526.blender_r/data/test/input",
    },
    "cam4": {
        "dir": "527.cam4_r/run/run_base_test_riscv.0000",
        "args": [],
        "needs_special_validation": True,
        "validation_binary": "../../exe/cam4_validate_527_base.riscv",
        "validation_args": ["10", "0.0035", "../../data/test/compare/h0_ctrl.nc", "h0.nc"],
        "validation_output": "cam4_validate.txt",
        "outputs": [("cam4_validate.txt", "cam4_validate.txt")],
        "specdiff_opts": ["--cw"],
        "test_inputs": "527.cam4_r/data/test/input",
        "extra_inputs": ["527.cam4_r/data/all/input"],
    },
    "imagick": {
        "dir": "538.imagick_r/run/run_base_test_riscv.0000",
        "args": ["-limit", "disk", "0", "test_input.tga", "-shear", "25", "-resize", "640x480",
                 "-negate", "-alpha", "Off", "test_output.tga"],
        "needs_special_validation": True,
        "validation_binary": "../../exe/imagevalidate_538_base.riscv",
        "validation_args": ["-avg", "-threshold", "0.95", "-maxthreshold", "0.001",
                           "test_output.tga", "../../data/test/compare/test_expected.tga"],
        "validation_output": "test_validate.out",
        "outputs": [("test_validate.out", "test_validate.out")],
        "specdiff_opts": ["--reltol", "0.01"],
        "test_inputs": "538.imagick_r/data/test/input",
    },
    "nab": {
        "dir": "544.nab_r/run/run_base_test_riscv.0000",
        "args": ["hkrdenq", "1930344093", "1000"],
        "stdout_file": "hkrdenq.out",
        "outputs": [("hkrdenq.out", "hkrdenq.out")],
        "specdiff_opts": ["--reltol", "0.01", "--skipreltol", "2"],
        "test_inputs": "544.nab_r/data/test/input",
    },
    "fotonik3d": {
        "dir": "549.fotonik3d_r/run/run_base_test_riscv.0000",
        "args": [],
        "compressed_inputs": [("OBJ.dat.xz", "OBJ.dat")],
        "outputs": [("pscyee.out", "pscyee.out")],
        "specdiff_opts": ["--abstol", "1e-27", "--reltol", "1e-10", "--obiwan", "--floatcompare"],
        "test_inputs": "549.fotonik3d_r/data/test/input",
    },
    "roms": {
        "dir": "554.roms_r/run/run_base_test_riscv.0000",
        "args": [],
        "stdin": "ocean_benchmark0.in.x",
        "stdout_file": "ocean_benchmark0.log",
        "outputs": [("ocean_benchmark0.log", "ocean_benchmark0.log")],
        "specdiff_opts": ["--abstol", "1e-07", "--reltol", "1e-07"],
        "test_inputs": "554.roms_r/data/test/input",
        "extra_inputs": ["554.roms_r/data/all/input"],
    },
}

# Active workload - set by CLI flag, used by all functions
ACTIVE_WORKLOAD = "refrate"


def get_effective_config(benchmark: str) -> dict:
    """Get the effective benchmark config, merging test overrides if active."""
    if benchmark not in BENCHMARK_CONFIG:
        raise ValueError(f"Unknown benchmark: {benchmark}. Available: {list(BENCHMARK_CONFIG.keys())}")
    config = dict(BENCHMARK_CONFIG[benchmark])  # shallow copy
    if ACTIVE_WORKLOAD == "test" and benchmark in BENCHMARK_CONFIG_TEST:
        config.update(BENCHMARK_CONFIG_TEST[benchmark])
    return config


def setup_test_run_dir(benchmark: str, benchmarks_dir: str) -> str:
    """
    Set up a test run directory by copying test input files into the
    existing refrate run directory structure.
    
    Creates: <spec_num>/run/run_base_test_riscv.0000/ with test input files
    
    Returns the relative dir path (like the 'dir' field in config).
    """
    config = BENCHMARK_CONFIG[benchmark]
    test_overrides = BENCHMARK_CONFIG_TEST.get(benchmark, {})
    spec_num = config["spec_num"]
    
    # Create test run directory alongside the refrate one
    test_run_dir = os.path.join(benchmarks_dir, spec_num, "run", "run_base_test_riscv.0000")
    
    if os.path.exists(test_run_dir):
        # Already set up
        return f"{spec_num}/run/run_base_test_riscv.0000"
    
    os.makedirs(test_run_dir, exist_ok=True)
    
    # Copy test input files
    test_inputs_rel = test_overrides.get("test_inputs", "")
    if test_inputs_rel:
        test_inputs_dir = os.path.join(benchmarks_dir, test_inputs_rel)
        if os.path.exists(test_inputs_dir):
            for item in os.listdir(test_inputs_dir):
                src = os.path.join(test_inputs_dir, item)
                dst = os.path.join(test_run_dir, item)
                if os.path.isdir(src):
                    if not os.path.exists(dst):
                        shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
    
    # Copy extra shared input files (data/all/input/)
    for extra_rel in test_overrides.get("extra_inputs", []):
        extra_dir = os.path.join(benchmarks_dir, extra_rel)
        if os.path.exists(extra_dir):
            for item in os.listdir(extra_dir):
                src = os.path.join(extra_dir, item)
                dst = os.path.join(test_run_dir, item)
                if not os.path.exists(dst):
                    if os.path.isdir(src):
                        shutil.copytree(src, dst)
                    else:
                        shutil.copy2(src, dst)
    
    # For WRF: also copy the compare reference file needed by diffwrf
    if benchmark == "wrf":
        wrf_compare = os.path.join(benchmarks_dir, spec_num, "data", "test", "compare")
        if os.path.exists(wrf_compare):
            for item in os.listdir(wrf_compare):
                src = os.path.join(wrf_compare, item)
                dst = os.path.join(test_run_dir, item)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
    
    # Copy the validation binary into test run dir if it's a local path (like diffwrf for WRF)
    refrate_dir = os.path.join(benchmarks_dir, config["dir"])
    val_binary = config.get("validation_binary", "")
    if val_binary and not val_binary.startswith("../"):
        src = os.path.join(refrate_dir, val_binary)
        dst = os.path.join(test_run_dir, val_binary)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
    
    print(f"  Set up test run directory: {test_run_dir}")
    return f"{spec_num}/run/run_base_test_riscv.0000"


@dataclass
class InstructionInfo:
    """Information about a profiled instruction."""
    offset: int  # Virtual address
    disassembly: str
    routine: str
    dest_register: str
    has_dest_reg: bool
    execution_count: int


@dataclass
class FaultResult:
    """Result of a fault injection experiment."""
    offset: int
    execution: int
    bit: int
    outcome: str  # 'same', 'sdc', 'crash', 'timeout', 'hang', 'error'
    return_code: Optional[int] = None
    stdout_hash: Optional[str] = None
    stderr_hash: Optional[str] = None
    fault_injected: bool = False
    output_dir: Optional[str] = None  # Directory containing output files for specdiff
    is_signal_crash: bool = False  # True if terminated by a crash signal (SIGSEGV, etc.)


@dataclass 
class InstructionSummary:
    """Summary statistics for an instruction at a specific execution."""
    offset: int
    instruction_number: int
    disassembly: str
    routine: str
    execution_tested: int
    same_result: int = 0
    sdc: int = 0  # Silent Data Corruption
    crashes: int = 0
    timeouts: int = 0
    hangs: int = 0
    errors: int = 0


class BenchmarkRunner:
    """Handles running benchmarks with QEMU and plugins."""
    
    def __init__(self, qemu_path: str, plugins_dir: str, benchmarks_dir: str):
        self.qemu_path = qemu_path
        self.plugins_dir = plugins_dir
        self.benchmarks_dir = benchmarks_dir
        self.profiler_plugin = os.path.join(plugins_dir, "profiler.so")
        self.fault_plugin = os.path.join(plugins_dir, "fault_injector.so")
        
    def get_benchmark_config(self, benchmark: str) -> dict:
        """Get configuration for a benchmark, with workload overrides applied."""
        return get_effective_config(benchmark)
    
    def get_work_dir(self, benchmark: str) -> str:
        """Get the working directory for a benchmark."""
        config = self.get_benchmark_config(benchmark)
        return os.path.join(self.benchmarks_dir, config["dir"])
    
    def run_baseline(self, benchmark: str, timeout: int) -> Tuple[int, str, str]:
        """
        Run the benchmark without any plugins to get baseline output.
        
        Returns: (return_code, stdout, stderr)
        """
        config = self.get_benchmark_config(benchmark)
        work_dir = self.get_work_dir(benchmark)
        
        cmd = [self.qemu_path, config["binary"]] + config["args"]
        
        print(f"Running baseline for {benchmark}...")
        print(f"  Working dir: {work_dir}")
        print(f"  Command: {' '.join(cmd)}")
        print(f"  Timeout: {timeout}s")
        
        stdin_file = None
        if config["stdin"]:
            stdin_path = os.path.join(work_dir, config["stdin"])
            if os.path.exists(stdin_path):
                stdin_file = open(stdin_path, 'r')
        
        proc = None
        try:
            # Use Popen for better control over process termination
            proc = subprocess.Popen(
                cmd,
                cwd=work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=stdin_file,
                text=True,
                preexec_fn=os.setsid  # Create new process group for clean kill
            )
            
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
                return proc.returncode, stdout, stderr
            except subprocess.TimeoutExpired:
                print(f"  Baseline timed out after {timeout}s - killing process group...")
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()  # Reap the zombie
                return -1, "", "TIMEOUT"
                
        except Exception as e:
            print(f"  Error running baseline: {e}")
            if proc and proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except:
                    pass
            return -2, "", str(e)
        finally:
            if stdin_file:
                stdin_file.close()
    
    def run_profiler(self, benchmark: str, output_dir: str, timeout: int) -> Tuple[Optional[str], int, str, str]:
        """
        Run the profiler plugin to generate instruction profile.
        Also captures program output to use as baseline (avoids running benchmark twice).
        
        Returns: Tuple of (profile_path, return_code, stdout, stderr)
                 profile_path is None if profiling failed
        """
        config = self.get_benchmark_config(benchmark)
        work_dir = self.get_work_dir(benchmark)
        
        # The profiler writes to "instruction_profile.csv" in the working directory
        profile_output = os.path.join(work_dir, "instruction_profile.csv")
        
        # Build QEMU command with profiler plugin
        cmd = [
            self.qemu_path,
            "-plugin", self.profiler_plugin,
            "-d", "plugin",
            config["binary"]
        ] + config["args"]
        
        # Decompress any compressed input files (e.g., fotonik3d's OBJ.dat.xz -> OBJ.dat)
        for compressed, decompressed in config.get("compressed_inputs", []):
            decompressed_path = os.path.join(work_dir, decompressed)
            compressed_path = os.path.join(work_dir, compressed)
            if not os.path.exists(decompressed_path) and os.path.exists(compressed_path):
                print(f"  Decompressing {compressed} -> {decompressed}")
                try:
                    with lzma.open(compressed_path) as f_in:
                        with open(decompressed_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                except Exception as e:
                    print(f"  ERROR: Failed to decompress {compressed}: {e}")
                    return None, -3, "", f"Decompression failed: {e}"
        
        print(f"Running profiler for {benchmark} (also capturing baseline output)...")
        print(f"  Working dir: {work_dir}")
        print(f"  Command: {' '.join(cmd)}")
        print(f"  Profile output: {profile_output}")
        print(f"  Timeout: {timeout}s")
        
        stdin_file = None
        if config["stdin"]:
            stdin_path = os.path.join(work_dir, config["stdin"])
            if os.path.exists(stdin_path):
                stdin_file = open(stdin_path, 'r')
        
        proc = None
        try:
            # Use Popen for better control over process termination
            proc = subprocess.Popen(
                cmd,
                cwd=work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=stdin_file,
                text=True,
                preexec_fn=os.setsid  # Create new process group for clean kill
            )
            
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
                
                if proc.returncode != 0:
                    print(f"  Profiler stderr (first 500 chars): {stderr[:500]}")
                
                # Copy profile to campaign output directory
                dest_profile = os.path.join(output_dir, f"{benchmark}_profile.csv")
                if os.path.exists(profile_output):
                    shutil.copy(profile_output, dest_profile)
                    print(f"  Profile copied to: {dest_profile}")
                    return dest_profile, proc.returncode, stdout, stderr
                else:
                    print(f"  ERROR: Profile not generated at {profile_output}")
                    return None, proc.returncode, stdout, stderr
                    
            except subprocess.TimeoutExpired:
                print(f"  Profiler timed out after {timeout}s - killing process group...")
                # Kill the entire process group to ensure all children die
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()  # Reap the zombie
                return None, -1, "", "TIMEOUT"
                
        except Exception as e:
            print(f"  Error running profiler: {e}")
            # Try to kill any leftover process
            if proc and proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except:
                    pass
            return None, -2, "", str(e)
        finally:
            if stdin_file:
                stdin_file.close()
    
    def run_fault_injection(self, benchmark: str, address: int, bit: int, 
                           exec_count: int, timeout: int,
                           worker_id: int = 0,
                           keep_output: bool = True) -> FaultResult:
        """
        Run a single fault injection experiment.
        
        Args:
            benchmark: Name of the benchmark
            address: Virtual address of instruction to target
            bit: Bit position to flip (0-63 for RISC-V)
            exec_count: Which execution of the instruction to inject fault at
            timeout: Timeout in seconds
            worker_id: Unique ID for parallel workers to avoid file conflicts
            keep_output: If True, keep output dir for specdiff comparison
            
        Returns: FaultResult with outcome classification
        """
        config = self.get_benchmark_config(benchmark)
        base_work_dir = self.get_work_dir(benchmark)
        
        # Create a unique temp directory for this worker to avoid race conditions
        # when running in parallel - each worker needs isolated output files
        temp_dir = tempfile.mkdtemp(prefix=f"fault_inject_{worker_id}_")
        
        result = FaultResult(offset=address, execution=exec_count, bit=bit, outcome='error')
        result.output_dir = temp_dir  # Store for later specdiff comparison
        proc = None
        stdin_file = None
        cleanup_needed = True
        
        try:
            # The fault_result.out will be created in temp_dir (cwd)
            # Build plugin argument string
            plugin_args = f"{self.fault_plugin},addr=0x{address:x},bit={bit},count={exec_count}"
            
            # Use absolute path to binary since we're in temp dir
            binary_abs = os.path.abspath(os.path.join(base_work_dir, config["binary"]))
            
            cmd = [
                self.qemu_path,
                "-plugin", plugin_args,
                "-d", "plugin",
                binary_abs
            ] + config["args"]
            
            # Copy any required input files to temp dir
            if config["stdin"]:
                stdin_src = os.path.join(base_work_dir, config["stdin"])
                if os.path.exists(stdin_src):
                    stdin_dst = os.path.join(temp_dir, config["stdin"])
                    shutil.copy(stdin_src, stdin_dst)
                    stdin_file = open(stdin_dst, 'r')
            
            # Copy other required input files and directories from work_dir
            # This is needed for benchmarks that read input files from cwd
            # e.g., nab needs the "1am0/" directory with .pdb and .prm files
            for fname in os.listdir(base_work_dir):
                src = os.path.join(base_work_dir, fname)
                dst = os.path.join(temp_dir, fname)
                
                if os.path.exists(dst):
                    continue  # Already copied
                    
                try:
                    if os.path.isfile(src):
                        size = os.path.getsize(src)
                        # Copy files < 100MB that look like inputs
                        if size < 100 * 1024 * 1024:
                            shutil.copy(src, dst)
                    elif os.path.isdir(src):
                        # Copy directories (needed for nab's 1am0/ directory, etc.)
                        # Skip large directories and build/exe directories
                        if fname not in ('build', 'exe', 'Docs', 'Spec', 'src', '__pycache__'):
                            dir_size = sum(
                                os.path.getsize(os.path.join(dirpath, f))
                                for dirpath, _, filenames in os.walk(src)
                                for f in filenames
                            )
                            # Copy directories < 50MB
                            if dir_size < 50 * 1024 * 1024:
                                shutil.copytree(src, dst)
                except Exception:
                    pass  # Ignore copy errors for non-essential files
            
            # Decompress any compressed input files in temp_dir
            # (e.g., fotonik3d's OBJ.dat.xz -> OBJ.dat)
            for compressed, decompressed in config.get("compressed_inputs", []):
                compressed_path = os.path.join(temp_dir, compressed)
                decompressed_path = os.path.join(temp_dir, decompressed)
                if os.path.exists(compressed_path) and not os.path.exists(decompressed_path):
                    try:
                        with lzma.open(compressed_path) as f_in:
                            with open(decompressed_path, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                    except Exception:
                        pass  # Will likely cause benchmark to fail, classified as crash
            
            # Use Popen for better control over process termination
            proc = subprocess.Popen(
                cmd,
                cwd=temp_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=stdin_file,
                text=True,
                preexec_fn=os.setsid  # Create new process group for clean kill
            )
            
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
                result.return_code = proc.returncode
                result.stdout_hash = hashlib.md5(stdout.encode()).hexdigest()
                result.stderr_hash = hashlib.md5(stderr.encode()).hexdigest()
                
                # CRITICAL: Write stdout to file if the benchmark expects it.
                # Many SPEC benchmarks (bwaves, lbm, nab, roms, cactuBSSN) create
                # their validated output files via stdout redirection (> file.out).
                # Since we capture stdout via PIPE, we must write it to the file
                # that specdiff will later look for.
                stdout_file = config.get("stdout_file")
                if stdout_file and stdout:
                    stdout_path = os.path.join(temp_dir, stdout_file)
                    try:
                        with open(stdout_path, 'w') as sf:
                            sf.write(stdout)
                    except Exception:
                        pass  # Non-fatal: specdiff will report missing_output
                
                # Detect if the program terminated due to a crash signal
                # In Python, subprocess returns negative signal number for signal termination
                # e.g., SIGSEGV (11) is returned as -11
                if result.return_code is not None and result.return_code < 0:
                    crash_signal = -result.return_code
                    if crash_signal in CRASH_SIGNALS:
                        result.is_signal_crash = True
                
                # Check if fault was actually injected by reading fault_result.out
                fault_result_file = os.path.join(temp_dir, "fault_result.out")
                if os.path.exists(fault_result_file):
                    with open(fault_result_file, 'r') as f:
                        content = f.read()
                        result.fault_injected = "FAULT_INJECTED" in content
                
                result.outcome = 'completed'
                
                # Keep output dir for specdiff if requested and program completed
                if keep_output:
                    cleanup_needed = False
                
            except subprocess.TimeoutExpired:
                # Kill the entire process group to ensure all children die
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()  # Reap the zombie
                result.outcome = 'timeout'
                
        except Exception as e:
            result.outcome = 'error'
            # Try to kill any leftover process
            if proc and proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except:
                    pass
                    
        finally:
            if stdin_file:
                stdin_file.close()
            # Clean up temp directory only if not needed for comparison
            if cleanup_needed:
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    result.output_dir = None
                except:
                    pass
        
        return result


def parse_profile_csv(profile_path: str) -> List[InstructionInfo]:
    """
    Parse the profiler output CSV file.
    
    Format: offset,disassembly,routine,dest_register,has_dest_reg,execution_count
    Note: disassembly is quoted and may contain commas
    """
    instructions = []
    
    with open(profile_path, 'r') as f:
        for line in f:
            # Skip comments and empty lines
            if line.startswith('#') or not line.strip():
                continue
            
            line = line.strip()
            
            # Parse the line - format is:
            # 0xADDR,"disassembly",routine,dest_reg,has_dest(0/1),exec_count
            # The disassembly is quoted to handle commas
            
            try:
                # Find the quoted disassembly
                first_comma = line.index(',')
                offset_str = line[:first_comma]
                rest = line[first_comma + 1:]
                
                # Check if disassembly is quoted
                if rest.startswith('"'):
                    # Find the closing quote
                    end_quote = rest.index('"', 1)
                    disassembly = rest[1:end_quote]
                    rest = rest[end_quote + 2:]  # Skip closing quote and comma
                else:
                    # Disassembly not quoted, parse normally
                    parts = rest.split(',')
                    disassembly = parts[0]
                    rest = ','.join(parts[1:])
                
                # Now parse the remaining: routine,dest_reg,has_dest,exec_count
                parts = rest.split(',')
                if len(parts) >= 4:
                    routine = parts[0]
                    dest_register = parts[1]
                    has_dest_reg = parts[2] == '1'
                    execution_count = int(parts[3])
                    
                    offset = int(offset_str, 16) if offset_str.startswith('0x') else int(offset_str)
                    
                    info = InstructionInfo(
                        offset=offset,
                        disassembly=disassembly,
                        routine=routine,
                        dest_register=dest_register,
                        has_dest_reg=has_dest_reg,
                        execution_count=execution_count
                    )
                    
                    # Only include instructions with destination registers and executions
                    if info.has_dest_reg and info.execution_count > 0:
                        instructions.append(info)
                        
            except (ValueError, IndexError) as e:
                # Skip malformed lines
                continue
    
    return instructions


def classify_result(baseline_retcode: int, baseline_stdout_hash: str, 
                   baseline_stderr_hash: str, fault_result: FaultResult) -> str:
    """
    DEPRECATED: Old classification function using stdout hashing.
    Use classify_result_specdiff() instead for proper SPEC benchmark comparison.
    
    Kept for backwards compatibility when specdiff is not available.
    """
    if fault_result.outcome == 'timeout':
        return 'timeout'
    
    if fault_result.outcome == 'error':
        return 'error'
    
    if fault_result.return_code is None:
        return 'error'
    
    # Check for crash signals first (SIGSEGV, SIGILL, etc.)
    if fault_result.is_signal_crash:
        return 'crash'
    
    # Check output hashes as fallback
    if (fault_result.stdout_hash == baseline_stdout_hash and 
        fault_result.stderr_hash == baseline_stderr_hash):
        return 'same'
    else:
        return 'sdc'  # Silent Data Corruption - different output


def run_special_validation(benchmark: str, output_dir: str, qemu_path: str) -> Tuple[bool, str]:
    """
    Run special validation binary for benchmarks that require it.
    
    Some SPEC benchmarks use custom validation binaries (imagevalidate, diffwrf,
    cam4_validate) to compare outputs against reference. This function runs
    those binaries to produce the validation output file that specdiff will compare.
    
    Supported benchmarks:
    - WRF: runs diffwrf to compare binary output files
    - CAM4: runs cam4_validate to compare netCDF files  
    - POV-Ray, Blender, ImageMagick: run imagevalidate for image comparison
    
    Args:
        benchmark: Name of the benchmark
        output_dir: Directory containing the benchmark output files
        qemu_path: Path to QEMU binary for running RISC-V validators
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    config = get_effective_config(benchmark)
    
    # Check if this benchmark needs special validation
    if not config.get("needs_special_validation", False):
        return True, "No special validation needed"
    
    validation_binary = config.get("validation_binary")
    validation_args = list(config.get("validation_args", []))  # Make a copy
    
    if not validation_binary:
        return False, "needs_special_validation=True but no validation_binary specified"
    
    # Build absolute path to validation binary
    # Use refrate dir for binary location (binaries are always from refrate build)
    refrate_work_dir = os.path.join(DEFAULT_BENCHMARKS_DIR, BENCHMARK_CONFIG[benchmark]["dir"])
    validator_path = os.path.abspath(os.path.join(refrate_work_dir, validation_binary))
    
    if not os.path.exists(validator_path):
        return False, f"Validation binary not found: {validator_path}"
    
    compare_subdir = "test" if ACTIVE_WORKLOAD == "test" else "refrate"
    
    # Handle benchmark-specific setup
    if benchmark == "wrf":
        # WRF: need wrf_reference_01 file for comparison
        wrf_ref = os.path.join(DEFAULT_BENCHMARKS_DIR, config["spec_num"],
                               "data", compare_subdir, "compare", "wrf_reference_01")
        wrf_ref_dst = os.path.join(output_dir, "wrf_reference_01")
        if os.path.exists(wrf_ref) and not os.path.exists(wrf_ref_dst):
            try:
                shutil.copy(wrf_ref, wrf_ref_dst)
            except Exception as e:
                return False, f"Failed to copy WRF reference: {e}"
    
    elif benchmark == "cam4":
        # CAM4: use absolute path for h0_ctrl.nc reference
        h0_ctrl_path = os.path.join(DEFAULT_BENCHMARKS_DIR, config["spec_num"], 
                                     "data", compare_subdir, "compare", "h0_ctrl.nc")
        if not os.path.exists(h0_ctrl_path):
            return False, f"CAM4 reference not found: {h0_ctrl_path}"
        validation_args = ["10", "0.0035", h0_ctrl_path, "h0.nc"]
    
    elif benchmark in ("povray", "blender", "imagick"):
        # Image validation: need to resolve relative paths in validation_args
        # The args contain paths like "../data/refrate/compare/xxx.tga" or "../data/test/compare/xxx.tga"
        # These are relative to the refrate run dir (where binaries live)
        resolved_args = []
        for arg in validation_args:
            if arg.startswith("../"):
                # Resolve relative path to absolute (relative to refrate work dir)
                abs_path = os.path.abspath(os.path.join(refrate_work_dir, arg))
                resolved_args.append(abs_path)
            else:
                resolved_args.append(arg)
        validation_args = resolved_args
    
    # Build command: run validator through QEMU
    cmd = [qemu_path, validator_path] + validation_args
    
    # Get output file name from config
    validation_output = config.get("validation_output", "validation.out")
    
    # Before running the validator, check if the primary output file exists.
    # For image benchmarks: the .tga file. For WRF: wrfout_d01_*. For CAM4: h0.nc.
    # If the primary output doesn't exist, the benchmark crashed before producing it.
    primary_outputs = {
        "povray": "SPEC-benchmark.tga",
        "blender": None,  # varies by test/ref, check validation_args
        "imagick": None,  # varies by test/ref, check validation_args
        "wrf": None,  # checked via validation_args
        "cam4": "h0.nc",
    }
    primary_file = primary_outputs.get(benchmark)
    if primary_file is None:
        # Try to get primary output from first non-flag validation arg
        for arg in validation_args:
            if not arg.startswith("-") and not arg.startswith("/") and not arg.replace(".", "").replace("e", "").replace("-", "").isdigit():
                primary_file = arg
                break
    
    if primary_file and not os.path.isabs(primary_file):
        primary_path = os.path.join(output_dir, primary_file)
        if not os.path.exists(primary_path):
            return False, f"Primary output missing: {primary_file} (benchmark likely crashed)"
    
    try:
        with open(os.path.join(output_dir, validation_output), 'w') as outfile:
            result = subprocess.run(
                cmd,
                cwd=output_dir,
                stdout=outfile,
                stderr=subprocess.PIPE,
                text=True,
                timeout=300  # 5 minute timeout for validation
            )
        
        # Validation binary ran. Even if it returned non-zero, the validation
        # output file was written and specdiff can compare it against reference.
        # A non-zero return code from the validator does NOT mean "crash" — it
        # may mean the images differ (which is SDC). Let specdiff decide.
        if result.returncode != 0:
            return True, f"Validation completed with rc={result.returncode} (output written for specdiff)"
        
        return True, f"Validation output written to {validation_output}"
        
    except subprocess.TimeoutExpired:
        return False, "Validation binary timed out"
    except Exception as e:
        return False, f"Error running validation: {str(e)}"


def run_specdiff(benchmark: str, output_dir: str, spec_path: str, 
                  qemu_path: str = None) -> Tuple[str, str]:
    """
    Run SPEC's specdiff tool to compare output files against golden reference.
    
    For benchmarks that require special validation (WRF, CAM4), this function
    first runs the validation binary to produce the comparison output.
    
    Args:
        benchmark: Name of the benchmark
        output_dir: Directory containing the generated output files
        spec_path: Path to SPEC2017 installation
        qemu_path: Path to QEMU binary (needed for special validation)
        
    Returns:
        Tuple of (result_code: str, details: str)
        result_code is one of:
          - 'match': All outputs match within tolerance (specdiff confirmed)
          - 'differ': Specdiff ran and found actual differences (true SDC)
          - 'missing_output': Output files were not generated (likely crash)
          - 'error': Could not run specdiff (tools missing, etc.)
    """
    config = get_effective_config(benchmark)
    
    # Run special validation first if needed (WRF, CAM4, image benchmarks)
    if config.get("needs_special_validation", False):
        if qemu_path is None:
            qemu_path = DEFAULT_QEMU_PATH
        success, msg = run_special_validation(benchmark, output_dir, qemu_path)
        if not success:
            # Special validation failed.
            # If the primary output file is missing, the benchmark crashed.
            # Otherwise, there's an infrastructure error.
            if "Primary output missing" in msg:
                return 'missing_output', f"Special validation failed: {msg}"
            else:
                # Validation binary itself failed to run (not found, timeout, etc.)
                # This is an infrastructure error, not a benchmark crash or SDC.
                return 'error', f"Special validation failed: {msg}"
    
    spec_num = config.get("spec_num")
    outputs = config.get("outputs", [])
    specdiff_opts = config.get("specdiff_opts", [])
    
    if not outputs:
        # No outputs to compare - assume match (rely on return code check)
        return 'match', "No outputs defined for comparison"
    
    specperl = os.path.join(DEFAULT_BENCHMARKS_DIR, "bin", "specperl")
    specdiff = os.path.join(DEFAULT_BENCHMARKS_DIR, "bin", "harness", "specdiff")
    workload_name = "test" if ACTIVE_WORKLOAD == "test" else "refrate"
    ref_output_dir = os.path.join(DEFAULT_BENCHMARKS_DIR, spec_num, "data", workload_name, "output")
    
    if not os.path.exists(specperl):
        return 'error', f"specperl not found at {specperl}"
    if not os.path.exists(specdiff):
        return 'error', f"specdiff not found at {specdiff}"
    
    details = []
    has_missing_output = False
    has_differ = False
    has_error = False
    
    for generated_file, reference_file in outputs:
        gen_path = os.path.join(output_dir, generated_file)
        ref_path = os.path.join(ref_output_dir, reference_file)
        
        if not os.path.exists(gen_path):
            # Output file missing - program likely crashed before writing it
            details.append(f"MISSING: {generated_file}")
            has_missing_output = True
            continue
            
        if not os.path.exists(ref_path):
            # Reference file missing - this is a setup error, not SDC
            details.append(f"REF_MISSING: {reference_file}")
            has_error = True
            continue
        
        # Build specdiff command
        # specdiff returns 0 if files match, non-zero otherwise
        cmd = [specperl, specdiff, "-m", "-l", "10"] + specdiff_opts + [ref_path, gen_path]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,  # specdiff should be quick
                cwd=output_dir
            )
            
            if result.returncode == 0:
                details.append(f"MATCH: {generated_file}")
            else:
                # Specdiff found actual differences - this is true SDC
                details.append(f"DIFFER: {generated_file}")
                has_differ = True
                
        except subprocess.TimeoutExpired:
            details.append(f"SPECDIFF_TIMEOUT: {generated_file}")
            has_error = True
        except Exception as e:
            details.append(f"SPECDIFF_ERROR: {generated_file}: {str(e)}")
            has_error = True
    
    detail_str = "; ".join(details)
    
    # Determine overall result:
    # - If specdiff found actual differences, it's SDC
    # - If outputs are missing (and no differ), likely a crash
    # - If there were errors running specdiff, report error
    # - Otherwise, outputs match
    if has_differ:
        return 'differ', detail_str
    elif has_missing_output:
        return 'missing_output', detail_str
    elif has_error:
        return 'error', detail_str
    else:
        return 'match', detail_str


def classify_result_specdiff(benchmark: str, fault_result: FaultResult, 
                              spec_path: str, baseline_retcode: int = 0,
                              qemu_path: str = None) -> str:
    """
    Classify fault injection result using SPEC's specdiff tool for accurate SDC detection.
    
    Categories:
    - 'same': Output matches golden reference within tolerances (fault was masked)
    - 'sdc': Silent Data Corruption - specdiff found actual differences in output content
    - 'crash': Program crashed (terminated by signal or non-zero exit code)
    - 'timeout': Program timed out
    - 'error': Error in running the experiment
    
    IMPORTANT: SDC is ONLY reported when specdiff actually runs and finds content differences.
    Missing output files are treated as crashes, not SDC.
    
    Args:
        benchmark: Name of the benchmark
        fault_result: Result from run_fault_injection
        spec_path: Path to SPEC2017 installation
        baseline_retcode: Expected return code from baseline run
        qemu_path: Path to QEMU binary (needed for special validation)
        
    Returns:
        Classification string
    """
    # Check for timeout first
    if fault_result.outcome == 'timeout':
        return 'timeout'
    
    # Check for errors in running the experiment
    if fault_result.outcome == 'error':
        return 'error'
    
    if fault_result.return_code is None:
        return 'error'
    
    # Check for crash signals (SIGSEGV, SIGILL, SIGBUS, SIGFPE, SIGABRT, etc.)
    # In Python, subprocess returns negative signal number for signal termination
    # These are REAL crashes - the program was killed by a signal.
    if fault_result.is_signal_crash:
        return 'crash'
    
    # Negative return codes indicate signal termination (crash)
    if fault_result.return_code < 0:
        return 'crash'
    
    # For non-signal exits, ALWAYS try specdiff first if we have output files.
    # A non-zero return code alone does NOT mean crash - the fault may have caused
    # an error exit but still produced partial/complete output that we can compare.
    # Only specdiff can tell us if the output is correct (same), corrupted (sdc),
    # or missing (crash).
    if not fault_result.output_dir or not os.path.exists(fault_result.output_dir):
        # No output dir at all
        if fault_result.return_code != baseline_retcode:
            return 'crash'
        return 'same'
    
    # Run specdiff to compare outputs (handles special validation for WRF/CAM4)
    specdiff_result, details = run_specdiff(benchmark, fault_result.output_dir, 
                                             spec_path, qemu_path)
    
    if specdiff_result == 'match':
        # Specdiff confirmed outputs match - fault was masked
        return 'same'
    elif specdiff_result == 'differ':
        # Specdiff found actual content differences - TRUE SDC
        return 'sdc'
    elif specdiff_result == 'missing_output':
        # Output files missing - program likely crashed before writing them.
        # But check: if return code was 0 (same as baseline), the benchmark
        # THOUGHT it completed normally but output is missing/incomplete.
        # This is still a crash from our perspective (the fault caused the
        # program to skip producing output).
        return 'crash'
    else:  # 'error'
        # Error running specdiff tools themselves - fall back to return code.
        # If return code matches baseline, assume same (conservative).
        if fault_result.return_code != baseline_retcode:
            return 'crash'
        return 'same'


def cleanup_output_dir(output_dir: Optional[str]):
    """Clean up temporary output directory after classification."""
    if output_dir and os.path.exists(output_dir):
        try:
            shutil.rmtree(output_dir, ignore_errors=True)
        except:
            pass


def generate_execution_samples(exec_count: int, num_phases: int, 
                               samples_per_phase: int) -> List[Tuple[int, str]]:
    """
    Generate execution samples across the timeline with phase labels.
    
    Divides execution into phases (e.g., early, mid-early, middle, mid-late, late)
    and samples uniformly within each phase.
    
    Returns list of (execution_number, phase_label) tuples.
    """
    if exec_count <= num_phases * samples_per_phase:
        # If we have few executions, test all of them
        return [(i, f"exec_{i}") for i in range(1, exec_count + 1)]
    
    samples = []
    phase_size = exec_count // num_phases
    
    phase_names = ['early', 'mid_early', 'middle', 'mid_late', 'late']
    if num_phases > len(phase_names):
        phase_names = [f'phase_{i}' for i in range(num_phases)]
    
    for phase_idx in range(num_phases):
        phase_start = phase_idx * phase_size + 1
        phase_end = (phase_idx + 1) * phase_size if phase_idx < num_phases - 1 else exec_count
        phase_name = phase_names[phase_idx] if phase_idx < len(phase_names) else f'phase_{phase_idx}'
        
        if phase_end - phase_start + 1 <= samples_per_phase:
            for e in range(phase_start, phase_end + 1):
                samples.append((e, phase_name))
        else:
            phase_samples = random.sample(range(phase_start, phase_end + 1), samples_per_phase)
            for e in sorted(phase_samples):
                samples.append((e, phase_name))
    
    return samples


def generate_bit_samples(register_bits: int, num_samples: int, seed: int = None) -> List[int]:
    """
    Generate uniformly distributed bit samples across the register width.
    
    Uses stratified sampling to ensure coverage across all bit positions.
    For RISC-V, register_bits is typically 64.
    """
    if num_samples >= register_bits:
        return list(range(register_bits))
    
    if seed is not None:
        random.seed(seed)
    
    # Stratified sampling
    num_strata = min(num_samples, 8)
    samples_per_stratum = num_samples // num_strata
    extra_samples = num_samples % num_strata
    
    stratum_size = register_bits // num_strata
    samples = []
    
    for stratum_idx in range(num_strata):
        stratum_start = stratum_idx * stratum_size
        stratum_end = (stratum_idx + 1) * stratum_size if stratum_idx < num_strata - 1 else register_bits
        
        n_samples = samples_per_stratum + (1 if stratum_idx < extra_samples else 0)
        
        if stratum_end - stratum_start <= n_samples:
            samples.extend(range(stratum_start, stratum_end))
        else:
            stratum_samples = random.sample(range(stratum_start, stratum_end), n_samples)
            samples.extend(stratum_samples)
    
    return sorted(samples)


def run_single_experiment(args_tuple):
    """
    Worker function for parallel fault injection.
    
    Args:
        args_tuple: Tuple of (qemu_path, plugins_dir, benchmarks_dir, benchmark, 
                              address, bit, exec_num, timeout, baseline_retcode,
                              baseline_stdout_hash, baseline_stderr_hash,
                              inst_disassembly, inst_routine, phase, worker_id,
                              spec_path, workload)
    
    Returns:
        Dict with experiment results
    """
    (qemu_path, plugins_dir, benchmarks_dir, benchmark, 
     address, bit, exec_num, timeout, baseline_retcode,
     baseline_stdout_hash, baseline_stderr_hash,
     inst_disassembly, inst_routine, phase, worker_id,
     spec_path, workload) = args_tuple
    
    # Set the global workload for this worker process
    global ACTIVE_WORKLOAD
    ACTIVE_WORKLOAD = workload
    
    # Create a runner for this process
    runner = BenchmarkRunner(qemu_path, plugins_dir, benchmarks_dir)
    
    # Run the experiment with unique worker_id to avoid file conflicts
    fault_result = runner.run_fault_injection(
        benchmark, address, bit, exec_num, timeout, worker_id=worker_id,
        keep_output=True  # Keep output for specdiff comparison
    )
    
    # Classify the result using specdiff for accurate SDC detection
    outcome = classify_result_specdiff(
        benchmark, fault_result, spec_path, baseline_retcode, qemu_path
    )
    
    # Clean up the output directory after classification
    cleanup_output_dir(fault_result.output_dir)
    
    return {
        'offset': address,
        'disassembly': inst_disassembly,
        'routine': inst_routine,
        'execution': exec_num,
        'phase': phase,
        'bit': bit,
        'outcome': outcome,
        'return_code': fault_result.return_code,
        'fault_injected': fault_result.fault_injected,
        'is_signal_crash': fault_result.is_signal_crash
    }


def run_campaign(benchmark: str, runner: BenchmarkRunner, output_dir: str,
                num_phases: int, samples_per_phase: int,
                bits_per_instruction: int, timeout: int,
                focus_routines: List[str], random_seed: int,
                max_instructions: int, skip_profiling: bool,
                profile_path: str, num_jobs: int,
                spec_path: str = DEFAULT_SPEC_PATH,
                workload: str = "refrate"):
    """
    Run a fault injection campaign for a SPEC2017 benchmark.
    
    Args:
        benchmark: Name of the benchmark to test
        runner: BenchmarkRunner instance
        output_dir: Directory to store results
        num_phases: Number of timeline phases
        samples_per_phase: Samples per phase
        bits_per_instruction: Bits to sample per instruction
        timeout: Timeout per experiment
        focus_routines: If provided, only test instructions in these routines
        random_seed: Random seed for reproducibility
        max_instructions: Limit number of instructions to test
        skip_profiling: Skip profiling if profile already exists
        profile_path: Path to existing profile file
        num_jobs: Number of parallel jobs to run (default: 1 for sequential)
        spec_path: Path to SPEC2017 installation (for specdiff comparison)
        workload: Workload size - 'refrate' (default) or 'test' (fast/small)
    """
    # Set global workload so all functions pick it up
    global ACTIVE_WORKLOAD
    ACTIVE_WORKLOAD = workload
    
    # Set up test run directory if using test workload
    if workload == "test":
        test_dir = setup_test_run_dir(benchmark, runner.benchmarks_dir)
        print(f"  Using test workload (fast inputs, test reference outputs)")
    
    random.seed(random_seed)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Fault Injection Campaign: {benchmark} (workload: {workload})")
    print(f"{'='*60}")
    
    baseline_json_path = os.path.join(output_dir, 'baseline.json')
    
    # Step 1: Run profiler (or load existing) and get baseline
    # The profiler run also captures baseline output, avoiding a separate baseline run
    print("\n[1/3] Profiling and capturing baseline...")
    
    if profile_path and os.path.exists(profile_path):
        # Using provided profile - need to load baseline from file or run baseline separately
        print(f"  Using existing profile: {profile_path}")
        instructions = parse_profile_csv(profile_path)
        
        # Check if baseline exists
        if os.path.exists(baseline_json_path):
            print(f"  Loading existing baseline from: {baseline_json_path}")
            with open(baseline_json_path, 'r') as f:
                baseline_info = json.load(f)
            baseline_retcode = baseline_info['return_code']
            baseline_stdout_hash = baseline_info['stdout_hash']
            baseline_stderr_hash = baseline_info['stderr_hash']
        else:
            print(f"  No baseline found, running baseline...")
            baseline_retcode, baseline_stdout, baseline_stderr = runner.run_baseline(benchmark, timeout)
            baseline_stdout_hash = hashlib.md5(baseline_stdout.encode()).hexdigest()
            baseline_stderr_hash = hashlib.md5(baseline_stderr.encode()).hexdigest()
            
            # Save baseline info
            baseline_info = {
                'benchmark': benchmark,
                'return_code': baseline_retcode,
                'stdout_hash': baseline_stdout_hash,
                'stderr_hash': baseline_stderr_hash,
                'stdout_len': len(baseline_stdout),
                'stderr_len': len(baseline_stderr)
            }
            with open(baseline_json_path, 'w') as f:
                json.dump(baseline_info, f, indent=2)
                
    elif skip_profiling:
        expected_profile = os.path.join(output_dir, f"{benchmark}_profile.csv")
        if os.path.exists(expected_profile):
            print(f"  Using existing profile: {expected_profile}")
            instructions = parse_profile_csv(expected_profile)
            
            # Check if baseline exists
            if os.path.exists(baseline_json_path):
                print(f"  Loading existing baseline from: {baseline_json_path}")
                with open(baseline_json_path, 'r') as f:
                    baseline_info = json.load(f)
                baseline_retcode = baseline_info['return_code']
                baseline_stdout_hash = baseline_info['stdout_hash']
                baseline_stderr_hash = baseline_info['stderr_hash']
            else:
                print(f"  ERROR: No baseline found at {baseline_json_path}")
                print(f"  Run without --skip-profiling first to generate baseline")
                return
        else:
            print(f"  ERROR: No profile found and skip_profiling is set")
            return
    else:
        # Run profiler - this also captures baseline output
        profile_file, baseline_retcode, baseline_stdout, baseline_stderr = runner.run_profiler(
            benchmark, output_dir, timeout
        )
        if not profile_file:
            print("  ERROR: Profiling failed")
            return
        instructions = parse_profile_csv(profile_file)
        
        # Compute baseline hashes
        baseline_stdout_hash = hashlib.md5(baseline_stdout.encode()).hexdigest()
        baseline_stderr_hash = hashlib.md5(baseline_stderr.encode()).hexdigest()
        
        # Save baseline info
        baseline_info = {
            'benchmark': benchmark,
            'return_code': baseline_retcode,
            'stdout_hash': baseline_stdout_hash,
            'stderr_hash': baseline_stderr_hash,
            'stdout_len': len(baseline_stdout),
            'stderr_len': len(baseline_stderr)
        }
        with open(baseline_json_path, 'w') as f:
            json.dump(baseline_info, f, indent=2)
    
    print(f"  Baseline return code: {baseline_retcode}")
    print(f"  Baseline stdout hash: {baseline_stdout_hash}")
    print(f"  Baseline stderr hash: {baseline_stderr_hash}")
    
    print(f"  Found {len(instructions)} profiled instructions with destination registers")
    
    # Filter by routine if specified
    if focus_routines:
        instructions = [i for i in instructions if i.routine in focus_routines]
        print(f"  Filtered to {len(instructions)} instructions in routines: {focus_routines}")
    
    # Randomly sample instructions if max_instructions is specified
    if max_instructions and len(instructions) > max_instructions:
        # Randomly sample instructions for better statistical representation
        total_candidates = len(instructions)
        instructions = random.sample(instructions, max_instructions)
        print(f"  Randomly sampled {max_instructions} instructions from {total_candidates} candidates")
    
    if not instructions:
        print("  ERROR: No instructions to test")
        return
    
    # Step 2: Generate experiment plan
    print("\n[2/3] Generating experiment plan...")
    experiment_plan = []
    
    for inst in instructions:
        # RISC-V uses 64-bit registers (or 32-bit for RV32)
        register_bits = 64
        
        exec_samples = generate_execution_samples(inst.execution_count, num_phases, samples_per_phase)
        bit_samples = generate_bit_samples(register_bits, bits_per_instruction, random_seed + inst.offset)
        
        for exec_num, phase in exec_samples:
            for bit in bit_samples:
                experiment_plan.append((inst, exec_num, phase, bit))
    
    total_experiments = len(experiment_plan)
    print(f"  Instructions: {len(instructions)}")
    print(f"  Phases: {num_phases} (with {samples_per_phase} samples each)")
    print(f"  Bits per instruction: {bits_per_instruction}")
    print(f"  Total experiments: {total_experiments}")
    print(f"  Parallel jobs: {num_jobs}")
    print(f"  Estimated time: {total_experiments * timeout / 3600 / num_jobs:.1f} hours (worst case)")
    
    # Save campaign config
    config = {
        'benchmark': benchmark,
        'num_phases': num_phases,
        'samples_per_phase': samples_per_phase,
        'bits_per_instruction': bits_per_instruction,
        'timeout': timeout,
        'focus_routines': focus_routines,
        'random_seed': random_seed,
        'total_instructions': len(instructions),
        'total_experiments': total_experiments
    }
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Prepare CSV output files
    results_csv = os.path.join(output_dir, 'results.csv')
    detailed_csv = os.path.join(output_dir, 'detailed_results.csv')
    summary_csv = os.path.join(output_dir, 'summary.csv')
    
    with open(results_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'instruction_offset', 'instruction_idx', 'disassembly', 'routine',
            'dest_register', 'exec_count', 'same', 'sdc', 'crashes', 
            'timeouts', 'errors', 'total_tested', 'vulnerability_rate'
        ])
    
    with open(detailed_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'instruction_offset', 'disassembly', 'routine', 'execution', 
            'phase', 'bit', 'outcome', 'return_code', 'fault_injected'
        ])
    
    # Step 3: Run experiments
    print("\n[3/3] Running fault injection experiments...")
    
    # Track statistics
    inst_stats = {}  # offset -> InstructionSummary
    phase_stats = {}  # (routine, phase) -> counts
    
    # Initialize instruction stats
    for inst in instructions:
        inst_stats[inst.offset] = InstructionSummary(
            offset=inst.offset,
            instruction_number=instructions.index(inst),
            disassembly=inst.disassembly,
            routine=inst.routine,
            execution_tested=0
        )
    
    completed = 0
    start_time = time.time()
    
    # Prepare experiment arguments for parallel execution
    # Include a unique worker_id for each experiment to avoid file conflicts
    experiment_args = []
    for exp_idx, (inst, exec_num, phase, bit) in enumerate(experiment_plan):
        experiment_args.append((
            runner.qemu_path, runner.plugins_dir, runner.benchmarks_dir,
            benchmark, inst.offset, bit, exec_num, timeout,
            baseline_retcode, baseline_stdout_hash, baseline_stderr_hash,
            inst.disassembly, inst.routine, phase,
            exp_idx,  # worker_id - unique per experiment
            spec_path,  # SPEC2017 path for specdiff comparison
            workload  # Workload type (refrate or test)
        ))
    
    if num_jobs == 1:
        # Sequential execution
        for args in experiment_args:
            result = run_single_experiment(args)
            
            # Process result
            offset = result['offset']
            outcome = result['outcome']
            phase = result['phase']
            routine = result['routine']
            
            summary = inst_stats[offset]
            summary.execution_tested += 1
            
            # Update stats
            if outcome == 'same':
                summary.same_result += 1
            elif outcome == 'sdc':
                summary.sdc += 1
            elif outcome == 'crash':
                summary.crashes += 1
            elif outcome == 'timeout':
                summary.timeouts += 1
            else:
                summary.errors += 1
            
            # Update phase stats
            phase_key = (routine, phase)
            if phase_key not in phase_stats:
                phase_stats[phase_key] = {'same': 0, 'sdc': 0, 'crash': 0, 'timeout': 0, 'error': 0, 'total': 0}
            phase_stats[phase_key]['total'] += 1
            phase_stats[phase_key][outcome if outcome in phase_stats[phase_key] else 'error'] += 1
            
            # Write detailed result
            with open(detailed_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    f'0x{offset:x}', result['disassembly'], routine,
                    result['execution'], phase, result['bit'], outcome,
                    result['return_code'], result['fault_injected']
                ])
            
            completed += 1
            
            # Progress update
            if completed % 10 == 0 or completed == total_experiments:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = (total_experiments - completed) / rate if rate > 0 else 0
                print(f"\r  Progress: {completed}/{total_experiments} ({100*completed/total_experiments:.1f}%) "
                      f"Rate: {rate:.2f}/s, ETA: {remaining/60:.1f} min", end='', flush=True)
    else:
        # Parallel execution
        print(f"  Starting {num_jobs} parallel workers...")
        
        with ProcessPoolExecutor(max_workers=num_jobs) as executor:
            # Submit all jobs
            future_to_args = {executor.submit(run_single_experiment, args): args for args in experiment_args}
            
            # Collect results as they complete
            for future in as_completed(future_to_args):
                try:
                    result = future.result()
                except Exception as e:
                    # Handle failed experiment
                    # Args format: (qemu_path[0], plugins_dir[1], benchmarks_dir[2], benchmark[3],
                    #               address[4], bit[5], exec_num[6], timeout[7], baseline_retcode[8],
                    #               baseline_stdout_hash[9], baseline_stderr_hash[10],
                    #               inst_disassembly[11], inst_routine[12], phase[13],
                    #               worker_id[14], spec_path[15], workload[16])
                    args = future_to_args[future]
                    result = {
                        'offset': args[4],        # address
                        'disassembly': args[11],  # inst_disassembly
                        'routine': args[12],      # inst_routine
                        'execution': args[6],     # exec_num
                        'phase': args[13],        # phase
                        'bit': args[5],           # bit
                        'outcome': 'error',
                        'return_code': None,
                        'fault_injected': False
                    }
                    print(f"\n  ERROR in experiment (addr=0x{args[4]:x}, exec={args[6]}, bit={args[5]}): {e}")
                
                # Process result
                offset = result['offset']
                outcome = result['outcome']
                phase = result['phase']
                routine = result['routine']
                
                summary = inst_stats[offset]
                summary.execution_tested += 1
                
                # Update stats
                if outcome == 'same':
                    summary.same_result += 1
                elif outcome == 'sdc':
                    summary.sdc += 1
                elif outcome == 'crash':
                    summary.crashes += 1
                elif outcome == 'timeout':
                    summary.timeouts += 1
                else:
                    summary.errors += 1
                
                # Update phase stats
                phase_key = (routine, phase)
                if phase_key not in phase_stats:
                    phase_stats[phase_key] = {'same': 0, 'sdc': 0, 'crash': 0, 'timeout': 0, 'error': 0, 'total': 0}
                phase_stats[phase_key]['total'] += 1
                phase_stats[phase_key][outcome if outcome in phase_stats[phase_key] else 'error'] += 1
                
                # Write detailed result
                with open(detailed_csv, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        f'0x{offset:x}', result['disassembly'], routine,
                        result['execution'], phase, result['bit'], outcome,
                        result['return_code'], result['fault_injected']
                    ])
                
                completed += 1
                
                # Progress update
                if completed % 10 == 0 or completed == total_experiments:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = (total_experiments - completed) / rate if rate > 0 else 0
                    print(f"\r  Progress: {completed}/{total_experiments} ({100*completed/total_experiments:.1f}%) "
                          f"Rate: {rate:.2f}/s, ETA: {remaining/60:.1f} min", end='', flush=True)
    
    print()  # New line after progress
    
    # Write instruction summaries
    with open(results_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        for offset, summary in sorted(inst_stats.items()):
            inst = next(i for i in instructions if i.offset == offset)
            total = summary.same_result + summary.sdc + summary.crashes + summary.timeouts + summary.errors
            vuln_rate = (summary.sdc + summary.crashes) / total if total > 0 else 0
            writer.writerow([
                f'0x{offset:x}', summary.instruction_number, summary.disassembly,
                summary.routine, inst.dest_register, inst.execution_count,
                summary.same_result, summary.sdc, summary.crashes,
                summary.timeouts, summary.errors, total, f'{vuln_rate:.4f}'
            ])
    
    # Write phase summary
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['routine', 'phase', 'total', 'same', 'sdc', 'crash', 'timeout', 'error', 'vulnerability_rate'])
        for (routine, phase), stats in sorted(phase_stats.items()):
            vuln_rate = (stats['sdc'] + stats['crash']) / stats['total'] if stats['total'] > 0 else 0
            writer.writerow([
                routine, phase, stats['total'], stats['same'], stats['sdc'],
                stats['crash'], stats['timeout'], stats['error'], f'{vuln_rate:.4f}'
            ])
    
    # Final summary
    elapsed = time.time() - start_time
    total_same = sum(s.same_result for s in inst_stats.values())
    total_sdc = sum(s.sdc for s in inst_stats.values())
    total_crashes = sum(s.crashes for s in inst_stats.values())
    total_timeouts = sum(s.timeouts for s in inst_stats.values())
    total_errors = sum(s.errors for s in inst_stats.values())
    
    print(f"\n{'='*60}")
    print(f"Campaign completed in {elapsed/60:.1f} minutes")
    print(f"{'='*60}")
    print(f"Total experiments: {completed}")
    print(f"  Same (masked):     {total_same} ({100*total_same/completed:.1f}%)")
    print(f"  SDC:               {total_sdc} ({100*total_sdc/completed:.1f}%)")
    print(f"  Crashes:           {total_crashes} ({100*total_crashes/completed:.1f}%)")
    print(f"  Timeouts:          {total_timeouts} ({100*total_timeouts/completed:.1f}%)")
    print(f"  Errors:            {total_errors} ({100*total_errors/completed:.1f}%)")
    print(f"\nResults saved to: {output_dir}")
    print(f"  - {results_csv}")
    print(f"  - {detailed_csv}")
    print(f"  - {summary_csv}")


def main():
    parser = argparse.ArgumentParser(
        description='Fault Injection Campaign for SPEC2017 Benchmarks using QEMU',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with nab benchmark
  python3 run_campaign.py -b nab --phases 3 --samples-per-phase 2 --bits 8 --max-instructions 10

  # Medium campaign with 8 parallel jobs
  python3 run_campaign.py -b nab --phases 5 --samples-per-phase 3 --bits 16 -j 8

  # Use existing profile with parallel execution
  python3 run_campaign.py -b nab --skip-profiling --profile results/nab_profile.csv -j 16

  # Focus on specific routines
  python3 run_campaign.py -b nab --focus-routines main compute

  # Full statistical campaign with max parallelism
  python3 run_campaign.py -b nab --phases 10 --samples-per-phase 5 --bits 32 -j $(nproc)

  # Quick test with test workload (fast inputs for debugging)
  python3 run_campaign.py -b nab -w test --phases 2 --samples-per-phase 1 --bits 4 --max-instructions 5 -j 10

Available benchmarks:
  bwaves, cactuBSSN, namd, parest, povray, lbm, wrf, 
  blender, cam4, imagick, nab, fotonik3d, roms
        """
    )
    
    parser.add_argument('-b', '--benchmark', required=True,
                       choices=list(BENCHMARK_CONFIG.keys()),
                       help='SPEC2017 benchmark to test')
    parser.add_argument('-o', '--output', default=None,
                       help='Output directory for results (default: results/<benchmark>)')
    parser.add_argument('--qemu', default=DEFAULT_QEMU_PATH,
                       help=f'Path to QEMU binary (default: {DEFAULT_QEMU_PATH})')
    parser.add_argument('--plugins-dir', default=DEFAULT_PLUGINS_DIR,
                       help=f'Path to QEMU plugins directory (default: {DEFAULT_PLUGINS_DIR})')
    parser.add_argument('--benchmarks-dir', default=DEFAULT_BENCHMARKS_DIR,
                       help=f'Path to SPEC benchmarks directory (default: {DEFAULT_BENCHMARKS_DIR})')
    parser.add_argument('--spec-path', default=DEFAULT_SPEC_PATH,
                       help=f'Path to SPEC2017 installation (for specdiff comparison, default: {DEFAULT_SPEC_PATH})')
    
    parser.add_argument('--phases', '-p', type=int, default=5,
                       help='Number of timeline phases (default: 5)')
    parser.add_argument('--samples-per-phase', '-n', type=int, default=3,
                       help='Execution samples per phase (default: 3)')
    parser.add_argument('--bits', type=int, default=16,
                       help='Bits to sample per instruction (default: 16)')
    parser.add_argument('--timeout', '-t', type=int, default=36000,
                       help='Timeout per run in seconds (default: 36000)')
    parser.add_argument('--focus-routines', '-r', nargs='+', default=None,
                       help='Only test instructions in these routines')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--max-instructions', type=int, default=None,
                       help='Maximum number of instructions to test')
    parser.add_argument('-j', '--jobs', type=int, default=1,
                       help='Number of parallel jobs (default: 1 for sequential)')
    parser.add_argument('--skip-profiling', action='store_true',
                       help='Skip profiling, use existing profile')
    parser.add_argument('--profile', default=None,
                       help='Path to existing profile CSV file')
    parser.add_argument('--workload', '-w', default='refrate',
                       choices=['refrate', 'test'],
                       help='Workload size: refrate (default, full) or test (fast/small for debugging)')
    
    args = parser.parse_args()
    
    # Set default output directory
    if args.output is None:
        suffix = '_test' if args.workload == 'test' else ''
        args.output = os.path.join(SCRIPT_DIR, 'results', args.benchmark + suffix)
    
    # Validate paths
    if not os.path.exists(args.qemu):
        print(f"Error: QEMU binary not found: {args.qemu}")
        sys.exit(1)
    
    profiler_plugin = os.path.join(args.plugins_dir, "profiler.so")
    fault_plugin = os.path.join(args.plugins_dir, "fault_injector.so")
    
    if not os.path.exists(profiler_plugin):
        print(f"Error: Profiler plugin not found: {profiler_plugin}")
        print("Please build the plugins first")
        sys.exit(1)
    
    if not os.path.exists(fault_plugin):
        print(f"Error: Fault injector plugin not found: {fault_plugin}")
        print("Please build the plugins first")
        sys.exit(1)
    
    if not os.path.exists(args.benchmarks_dir):
        print(f"Error: Benchmarks directory not found: {args.benchmarks_dir}")
        sys.exit(1)
    
    # Validate SPEC path for specdiff
    specperl = os.path.join(args.spec_path, "bin", "specperl")
    specdiff = os.path.join(args.spec_path, "bin", "harness", "specdiff")
    if not os.path.exists(specperl) or not os.path.exists(specdiff):
        print(f"Warning: SPEC2017 specdiff tools not found at {args.spec_path}")
        print("         SDC detection may fall back to stdout hash comparison")
    
    # Create runner and run campaign
    runner = BenchmarkRunner(args.qemu, args.plugins_dir, args.benchmarks_dir)
    
    run_campaign(
        benchmark=args.benchmark,
        runner=runner,
        output_dir=args.output,
        num_phases=args.phases,
        samples_per_phase=args.samples_per_phase,
        bits_per_instruction=args.bits,
        timeout=args.timeout,
        focus_routines=args.focus_routines,
        random_seed=args.seed,
        max_instructions=args.max_instructions,
        skip_profiling=args.skip_profiling,
        profile_path=args.profile,
        num_jobs=args.jobs,
        spec_path=args.spec_path,
        workload=args.workload
    )


if __name__ == '__main__':
    main()
