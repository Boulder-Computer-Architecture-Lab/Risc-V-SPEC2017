#!/bin/bash

# Default Configuration
DEFAULT_VERBOSE=false
DEFAULT_IMG_FILE="ubuntu-riscv64.img"
DEFAULT_MOUNT_POINT="/mnt/ubuntu-riscv64"
DEFAULT_SPEC_SOURCE_DIR="/opt/SPEC2017/cpu"
DEFAULT_SPEC_DEST_DIR="/home/ubuntu/spec"

# Parse command line arguments
VERBOSE=$DEFAULT_VERBOSE
IMG_FILE=$DEFAULT_IMG_FILE
MOUNT_POINT=$DEFAULT_MOUNT_POINT
SPEC_SOURCE_DIR=$DEFAULT_SPEC_SOURCE_DIR
SPEC_DEST_DIR=$DEFAULT_SPEC_DEST_DIR

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_msg() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_usage() {
    echo -e "${BLUE}Usage:${NC} $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -v, --verbose             Enable verbose output"
    echo "  -i, --image FILE          Ubuntu RISC-V image file (default: $DEFAULT_IMG_FILE)"
    echo "  -m, --mount DIR           Mount point directory (default: $DEFAULT_MOUNT_POINT)"
    echo "  -s, --spec-source DIR     SPEC benchmarks source directory (default: $DEFAULT_SPEC_SOURCE_DIR)"
    echo "  -d, --spec-dest DIR       Destination inside image (default: $DEFAULT_SPEC_DEST_DIR)"
    echo "  -h, --help                Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 -i ${DEFAULT_IMG_FILE} -m ${DEFAULT_MOUNT_POINT} -s ${DEFAULT_SPEC_SOURCE_DIR} -d ${DEFAULT_SPEC_DEST_DIR}"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            shift 1
            ;;
        -i|--image)
            IMG_FILE="$2"
            shift 2
            ;;
        -m|--mount)
            MOUNT_POINT="$2"
            shift 2
            ;;
        -s|--spec-source)
            SPEC_SOURCE_DIR="$2"
            shift 2
            ;;
        -d|--spec-dest)
            SPEC_DEST_DIR="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Generate working copy filename
WORKING_IMG="${IMG_FILE%.img}_spec2017.img"

# Function to cleanup on exit
cleanup() {
    if mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
        print_msg "Unmounting $MOUNT_POINT..."
        sudo umount "$MOUNT_POINT"
    fi
    
    if [ -d "$MOUNT_POINT" ]; then
        print_msg "Removing mount point..."
        sudo rmdir "$MOUNT_POINT" 2>/dev/null
    fi
}

# Set trap to cleanup on script exit
trap cleanup EXIT

# Show help if requested
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    print_usage
    exit 0
fi

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then 
    print_error "Please run with sudo"
    exit 1
fi

print_msg "Configuration:"
echo "  Image file:       $IMG_FILE"
echo "  Working copy:     $WORKING_IMG"
echo "  Mount point:      $MOUNT_POINT"
echo "  SPEC source:      $SPEC_SOURCE_DIR"
echo "  SPEC destination: $SPEC_DEST_DIR"
echo ""

# Check if base image exists, if not download it.
if [ ! -f "$IMG_FILE" ]; then
    print_warning "Base ubuntu image does not exist. Downloading..."
    curl -L -o "$IMG_FILE" "https://o365coloradoedu-my.sharepoint.com/personal/viji2154_colorado_edu/_layouts/15/download.aspx?UniqueId=dc0c6b4c-1a84-4581-a66b-7fbef193c92e&Translate=false&tempauth=v1.eyJzaXRlaWQiOiIzNTU2MTQwMS05ZGRiLTQxYjgtYWM3OC0zMDYzNGNkMWEzOWIiLCJhdWQiOiIwMDAwMDAwMy0wMDAwLTBmZjEtY2UwMC0wMDAwMDAwMDAwMDAvbzM2NWNvbG9yYWRvZWR1LW15LnNoYXJlcG9pbnQuY29tQDNkZWQ4YjFiLTA3MGQtNDYyOS04MmU0LWMwYjAxOWY0NjA1NyIsImV4cCI6IjE3Njc4Mjc4MzYifQ.CiMKCXNoYXJpbmdpZBIWUk1hWXZIeWRyRVNqWDRSZlliYnloZwoKCgRzbmlkEgI0MxILCMSL3MiRkec-EAUaDjEyOC4xMzguNzUuMTk0IhRtaWNyb3NvZnQuc2hhcmVwb2ludCosaUxyM21qMkxPdG9VRUw2eXhGQ1doRUtoU1N0NGhCM0hJUld3bHMrT3hrQT0woAE4AUIQoerLX36wALAVZcDvJ3HjVkoQaGFzaGVkcHJvb2Z0b2tlbmIEdHJ1ZXJLMGguZnxtZW1iZXJzaGlwfHVybiUzYXNwbyUzYXRlbmFudGFub24jZWFlNGZmN2UtNWM4OC00MTVjLTllZDEtZTM3OTU5ZDhmYzc1egEwwgFLMCMuZnxtZW1iZXJzaGlwfHVybiUzYXNwbyUzYXRlbmFudGFub24jZWFlNGZmN2UtNWM4OC00MTVjLTllZDEtZTM3OTU5ZDhmYzc1yAEB.8-dXRz5Hd8DCEpid6XUI3lDnFJN7vjH_vaJkF1TJIWA"
    print_msg "Downloaded base ubuntu image to $IMG_FILE"

    # Verify checksum.
    checksum=$(sha256sum "$IMG_FILE" | awk '{print $1}')
    expected_checksum="b9aa00e6d0b717dac891d90b08a0ef7147a3262188e208bd63c0ec023c17f51d"
    if [ "$checksum" != "$expected_checksum" ]; then
        print_error "Checksum verification failed for downloaded image"
        rm -f "$IMG_FILE"
        exit 1
    fi

    # Change ownership of working image to the user who invoked sudo.
    if [ -n "$SUDO_USER" ]; then
        print_msg "Changing ownership of $IMG_FILE to $SUDO_USER..."
        chown "$SUDO_USER:$SUDO_USER" "$IMG_FILE"
    fi
fi

# Check if SPEC source directory exists
if [ ! -d "$SPEC_SOURCE_DIR" ]; then
    print_error "SPEC source directory '$SPEC_SOURCE_DIR' not found"
    print_usage
    exit 1
fi

# Check if benchspec/CPU directory exists
SPEC_CPU_DIR="$SPEC_SOURCE_DIR/benchspec/CPU"
if [ ! -d "$SPEC_CPU_DIR" ]; then
    print_error "SPEC CPU directory '$SPEC_CPU_DIR' not found"
    print_error "Expected structure: SPEC_SOURCE_DIR/benchspec/CPU/*/exe/*.riscv"
    exit 1
fi

# Find RISC-V executables
print_msg "Searching for RISC-V executables in $SPEC_CPU_DIR..."
RISCV_EXES=$(find "$SPEC_CPU_DIR" -type f -path "*/exe/*.riscv" 2>/dev/null)

if [ -z "$RISCV_EXES" ]; then
    print_error "No RISC-V executables found matching pattern: */exe/*.riscv"
    exit 1
fi

EXE_COUNT=$(echo "$RISCV_EXES" | wc -l)
print_msg "Found $EXE_COUNT RISC-V executable(s)"

# Create working copy of the image
print_msg "Creating working copy of image (original will remain untouched)..."
if [ -f "$WORKING_IMG" ]; then
    print_warning "Working copy already exists, removing old copy..."
    rm -f "$WORKING_IMG"
fi

cp "$IMG_FILE" "$WORKING_IMG"
if [ $? -ne 0 ]; then
    print_error "Failed to create working copy"
    exit 1
fi
print_msg "Working copy created: $WORKING_IMG"

# Create mount point if it doesn't exist
if [ ! -d "$MOUNT_POINT" ]; then
    print_msg "Creating mount point at $MOUNT_POINT..."
    mkdir -p "$MOUNT_POINT"
fi

# Find the partition offset (for partitioned images)
print_msg "Analyzing image file..."
OFFSET=$(fdisk -l "$WORKING_IMG" | grep "^${WORKING_IMG}1" | awk '{print $2}' | head -n 1)

if [ -z "$OFFSET" ]; then
    print_warning "No partition table found, mounting as raw filesystem..."
    # Mount without offset (raw filesystem image)
    print_msg "Mounting $WORKING_IMG to $MOUNT_POINT..."
    mount -o loop "$WORKING_IMG" "$MOUNT_POINT"
else
    # Calculate byte offset
    BYTE_OFFSET=$((OFFSET * 512))
    print_msg "Found partition at sector $OFFSET (offset: $BYTE_OFFSET bytes)"
    print_msg "Mounting $WORKING_IMG to $MOUNT_POINT..."
    mount -o loop,offset=$BYTE_OFFSET "$WORKING_IMG" "$MOUNT_POINT"
fi

# Check if mount was successful
if ! mountpoint -q "$MOUNT_POINT"; then
    print_error "Failed to mount image"
    rm -f "$WORKING_IMG"
    exit 1
fi

print_msg "Successfully mounted image"

# Create destination directory if it doesn't exist
FULL_DEST_PATH="$MOUNT_POINT/$SPEC_DEST_DIR"
if [ ! -d "$FULL_DEST_PATH" ]; then
    print_msg "Creating destination directory: $SPEC_DEST_DIR"
    mkdir -p "$FULL_DEST_PATH"
fi

# Copy RISC-V executables maintaining directory structure
print_msg "Copying RISC-V executables..."
COPIED_COUNT=0

while IFS= read -r exe; do
    # Copy executable directly to destination (no subdirectories)
    EXE_NAME=$(basename "$exe")
    cp "$exe" "$FULL_DEST_PATH/"
    
    if [ $? -eq 0 ]; then
        ((COPIED_COUNT++))
        if [ "$VERBOSE" = true ]; then
            print_msg "  ✓ $EXE_NAME"
        fi
    else
        print_warning "Failed to copy $exe"
    fi
done <<< "$RISCV_EXES"

if [ $COPIED_COUNT -eq 0 ]; then
    print_error "Failed to copy any executables"
    exit 1
fi

print_msg "Successfully copied $COPIED_COUNT executable(s)"

# If verbose is enabled, display additional information.
if [ "$VERBOSE" = true ]; then
    # Show disk usage
    print_msg "Disk usage of mounted image:"
    df -h "$MOUNT_POINT"

    # Show what was copied
    print_msg "Contents of $SPEC_DEST_DIR:"
    print_msg "$(du -sh "$FULL_DEST_PATH"/*)"
fi

# Sync to ensure all writes are completed
print_msg "Syncing filesystem..."
sync

print_msg "Operation completed successfully!"
print_msg "Original image:  $IMG_FILE (untouched)"
print_msg "Working image:   $WORKING_IMG (with SPEC benchmarks)"

# Change ownership of working image to the user who invoked sudo
if [ -n "$SUDO_USER" ]; then
    print_msg "Changing ownership of $WORKING_IMG to $SUDO_USER..."
    chown "$SUDO_USER:$SUDO_USER" "$WORKING_IMG"
fi