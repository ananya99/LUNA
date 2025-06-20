#!/bin/bash

source .venv/bin/activate
echo "[test.sh] Activated virtual environment"

# Get the directory of the config file from the first argument "config_dir"
config_dir=$1

name='test'
output_path=$config_dir/output_test.log
echo "[test.sh] output_log: $output_path"

export HYDRA_FULL_ERROR='1'
export name=$name
export dir=$config_dir
export output_path=$output_path

python3 /home/anagupta/luna/LUNA/main.py --config-path=$config_dir --config-name=config.yaml general.mode='test_only' general.name=$name > $output_path 2>&1 || { echo "[test.sh] main.py failed, exiting"; exit 1; }