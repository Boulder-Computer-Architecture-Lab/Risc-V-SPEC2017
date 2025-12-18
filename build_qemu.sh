#!/bin/bash

# Default number of threads
THREADS=1

# --- Parse Arguments ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -j)
            THREADS="$2"
            shift; shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [-j NUM_THREADS]"
            exit 1
            ;;
    esac
done

echo "Building QEMU with $THREADS thread(s)..."

# --- Main Build Steps ---

# Initialize submodules
git submodule update --init --recursive

# Setup build directory
cd qemu/ || { echo "Error: 'qemu' directory not found."; exit 1; }
mkdir -p build
rm -rf build/*
cd build || { echo "Error: Could not enter 'build' directory."; exit 1; }

# Configure QEMU for RISC-V 64-bit Linux User mode
../configure --enable-plugins --target-list=riscv64-linux-user

# Compile
make -j"$THREADS"

# Copy the resulting binary to the top level (assuming you want it in the script's dir)
# Note: We need to go up two levels from 'qemu/build' to get back to the start
cp qemu-riscv64 ../../

echo "Build complete. Binary 'qemu-riscv64' is in the current directory."