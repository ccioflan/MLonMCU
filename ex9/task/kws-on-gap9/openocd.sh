#!/bin/bash

export PATH=/home/osboxes/mlonmcu_libs/linux-x64/bin:$PATH 
export LD_LIBRARY_PATH=/home/osboxes/mlonmcu_libs/linux-x64/lib64/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/osboxes/mlonmcu_libs/miniconda3/lib

export GAP_RISCV_GCC_TOOLCHAIN=/home/osboxes/gap_riscv_toolchain_ubuntu

source /home/osboxes/gap_sdk_private/sourceme.sh

openocd -f $GAP_SDK_HOME/utils/openocd/tcl/interface/ftdi/olimex-arm-usb-ocd-h.cfg -f $GAP_SDK_HOME/utils/openocd_tools/tcl/gap9revb.tcl
