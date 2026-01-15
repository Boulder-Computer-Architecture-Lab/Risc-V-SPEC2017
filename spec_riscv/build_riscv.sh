#!/bin/bash

# Default values
DEFAULT_CLOBBER=false           # By default, do not clobber
DEFAULT_RISCV_PATH="/opt/riscv" # Default RISC-V installation path
DEFAULT_BENCHMARK="all"         # Default benchmark target
DEFAULT_BUILD_CPUS=16           # Default number of CPUs for building

# --- Initialize variables with default values ---
CLOBBER=$DEFAULT_CLOBBER
BENCHMARK=$DEFAULT_BENCHMARK
BUILD_CPUS=$DEFAULT_BUILD_CPUS
RISCV_DIR=$DEFAULT_RISCV_PATH   

# --- Parse Command Line Arguments ---
# We loop through all arguments passed to this script ($@)
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -c|--clobber) 
            # If --c is found enable clobber.
            CLOBBER=true
            shift # Skip -c
            ;;
        -bm|--benchmark) 
            # If --bm is found, assign the NEXT argument to BENCHMARK
            BENCHMARK="$2"
            shift # Skip -bm
            shift # Skip the value
            ;;
        -j)
            # If -j is found, assign the NEXT argument to BUILD_CPUS
            BUILD_CPUS="$2"
            shift # Skip -j
            shift # Skip the value
            ;;
        -dir|--dir)
            # If --dir is found, assign the NEXT argument to RISCV_DIR
            RISCV_DIR="$2"
            shift # Skip -dir
            shift # Skip the value
            ;;
        -h|--help)
            echo "Usage: $0 [-c|--clobber] [-bm|--benchmark BENCHMARK] [-j NCPUS] [-d|--dir RISCV_PATH]"
            echo "  -c    : Clobber build"
            echo "  -bm   : Benchmark to build (default: $DEFAULT_BENCHMARK)"
            echo "  -j    : Number of build CPUs (default: $DEFAULT_BUILD_CPUS)"
            echo "  -d    : Path to RISC-V toolchain (default: $DEFAULT_RISCV_PATH)"
            exit 0
            ;;
        *) 
            echo "Unknown option: $1"
            echo "Try '$0 --help' for usage."
            exit 1
            ;;
    esac
done

# Validate Source Directory
RISCV_BIN_DIR="$RISCV_DIR/bin"
if [ ! -d "$RISCV_BIN_DIR" ]; then
    echo "Error: The 'bin' directory was not found at '$RISCV_DIR'."
    exit 1
fi

# --- Create Temporary Directory for Links ---
# mktemp -d creates a unique temporary directory safely
TEMP_LINKS_DIR=$(mktemp -d -t riscv_build_wrapper.XXXXXX)
echo "--- Created temporary link directory: $TEMP_LINKS_DIR ---"

# --- Cleanup Trap ---
# This ensures the temp directory is removed when the script exits (success or failure)
cleanup() {
    echo "Cleaning up temporary links..."
    rm -rf "$TEMP_LINKS_DIR"
}
trap cleanup EXIT

# --- Function to create the symbolic links ---
create_links() {
    # Helper function for creating links
    # Arguments: $1 = Source Binary Name (inside RISCV_BIN_DIR), $2 = New Link Name
    link_tool() {
        local source_path="$RISCV_BIN_DIR/$1"
        local link_path="$TEMP_LINKS_DIR/$2"
        
        if [ -f "$source_path" ]; then
            # We use absolute paths for the source so the link works anywhere
            ln -sf "$source_path" "$link_path"
            # echo "Linked: $2 -> $1"
        else
            echo "Warning: Source tool '$1' not found. Skipping link '$2'."
        fi
    }

    echo "--- Creating symbolic links ---"

    # 1. Links for 'riscv64-linux-gnu' prefix
    link_tool riscv64-unknown-linux-gnu-g++ riscv64-linux-gnu-g++
    link_tool riscv64-unknown-linux-gnu-gcc riscv64-linux-gnu-gcc
    link_tool riscv64-unknown-linux-gnu-gfortran riscv64-linux-gnu-gfortran

    # 2. Links for simplified commands (gcc, g++, etc)
    link_tool riscv64-unknown-linux-gnu-g++ g++
    link_tool riscv64-unknown-linux-gnu-gcc gcc
    link_tool riscv64-unknown-linux-gnu-gfortran gfortran
    
    echo "--- Symbolic linking complete! ---"
}

echo "--- RISC-V Build Script for SPEC CPU2017 ---"

# --- Main script execution ---
echo "Setting up RISC-V environment..."
create_links

# Save the original PATH
ORIGINAL_PATH=$PATH

# Add RISCV bin directory to PATH for the current session.
export PATH=$TEMP_LINKS_DIR:$PATH

echo "Using PATH: $TEMP_LINKS_DIR..."

# Build SPEC CPU2017 benchmarks for RISC-V
echo "Building SPEC CPU2017 benchmarks for RISC-V..."
echo "Benchmark: $BENCHMARK"
echo "CPUs: $BUILD_CPUS"

if [ "$CLOBBER" = true ]; then
    echo "Clobbering previous builds..."
    ./bin/runcpu --config=linux-rv64-cross -define build_ncpus=$BUILD_CPUS --action=clobber  $BENCHMARK
else
    ./bin/runcpu --config=linux-rv64-cross -define gcc_dir=$RISCV_DIR -define build_ncpus=$BUILD_CPUS --action=build --tune=base --size=ref $BENCHMARK
fi

# Restore the original PATH
export PATH=$ORIGINAL_PATH

echo "--- RISC-V SPEC CPU2017 build process completed! ---"