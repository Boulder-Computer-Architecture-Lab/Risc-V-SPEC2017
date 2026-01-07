# RISC-V SPEC CPU 2017 & QEMU Instrumentation

This repository contains scripts and configurations for building SPEC CPU 2017 for RISC-V, as well as QEMU plugins for profiling and fault injection. **This project references a custom version of QEMU that supports the custom `secon` and `secoff` instructions.**

## 1. SPEC CPU 2017 Setup & Compilation

Before compiling, you must copy the necessary build scripts and configuration files into your SPEC 2017 directory.

### Configuration Setup
Copy the build script and the cross-compilation config file from this repository to your SPEC 2017 installation:

```bash
# Copy the build script to the CPU folder
cp spec_riscv/build_riscv.sh SPEC2017/cpu/

# Copy the configuration file
cp spec_riscv/linux-rv64-cross.cfg SPEC2017/cpu/config/
```

### Compiling Benchmarks

To compile the benchmarks, execute the `build_riscv.sh` script located in the `SPEC2017/cpu/` directory.
```bash
./SPEC2017/cpu/build_riscv.sh [-c|--clobber] [-bm|--benchmark BENCHMARK] [-j NCPUS] [-d|--dir RISCV_PATH]
```

#### Arguments
* `-c` or `--clobber`: Clobber build.
* `-bm` or `--benchmark`: The specific benchmark suite or test to build.
* `-j`: The number of CPUs to use for the build process (parallel jobs).
* `-d` or `--dir`: The path to the RISC-V toolchain.

## 2. QEMU Setup & Execution
This section details how to build the custom QEMU plugins and run the emulator for profiling or fault injection.

### Building QEMU
Navigate to the root directory and run the build script:

```bash
./build_qemu.sh [-j NCPUS]
```
#### Arguments
* `-j`: The number of CPUs to use for the build process (parallel jobs).

### Building Plugins
Navigate to the qemu_pin directory and run the build script:
```bash
cd qemu_pin
./build_plugins.sh
cd ..
```

### Running the Profiler
To run the instruction counter plugin (insn_counter.so):
```bash
./qemu-riscv64 -plugin qemu_pin/insn_counter.so -d plugin ./benchmark.riscv
```
This will generate an `instruction_profile.csv` file in the `out` directory.

### Running the Fault Injector
To run the fault injection plugin (fault_injector.so). You can configure the injection address, bit, and count directly in the command arguments:
```bash
./qemu-riscv64 -plugin qemu_pin/fault_injector.so,addr=0x0000,bit=0,count=0 -d plugin ./benchmark.riscv
```
This will generate an `fault_result.out` file in the `out` directory.

#### Parameters:
* `addr`: The instruction address to inject the fault (e.g., 0x0000).
* `bit`: The specific bit position to flip.
* `count`: The injection count trigger.


## 3. Ubuntu SPEC2017 Image Generation for gem5
To build the gem5 Ubuntu image for SPEC 2017, navigate to the spec_gem5 directory and follow the internal build procedures:
```bash
cd spec_gem5
sudo ./build.sh [-i|--image FILE] [-m|--mount DIR] [-s|--spec-source DIR] [-d|--spec-dest DIR]
```

#### Arguments
* `-v` or `--verbose`: Enable verbose output.
* `-i` or `--image FILE`: Ubuntu RISC-V image file.
* `-m` or `--mount DIR`: Mount point directory.
* `-s` or `--spec-source DIR`: SPEC benchmarks source directory.
* `-d` or `--spec-dest DIR`: Destination inside image.

