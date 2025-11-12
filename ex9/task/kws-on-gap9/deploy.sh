#!/bin/bash

if [ "$1" == "-h" ] ; then
    echo "PLATFORM: gvsoc, board"
    echo "APPL: 0 (record), 1 (read .wav)"
    echo "MFCC computation: 0 (online), 1 (precomputed)"
    exit 0
fi

export PLATFORM=$1
export APPL=$2
export MFCC=$3

# export PATH=/home/osboxes/mlonmcu_libs/linux-x64/bin:$PATH 
# export LD_LIBRARY_PATH=/home/osboxes/mlonmcu_libs/linux-x64/lib64/:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/osboxes/mlonmcu_libs/miniconda3/lib

export GAP_RISCV_GCC_TOOLCHAIN=/home/osboxes/gap_riscv_toolchain_ubuntu

source /home/osboxes/gap_sdk_private/sourceme.sh

export WAV_FILE=/home/osboxes/mlonmcu_libs/right_94de6a6a_nohash_4.wav # ORIGINAL

cd application/

cmake -B build
cmake --build build --target menuconfig
cmake --build build --target run # --verbose
