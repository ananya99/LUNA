#!/bin/bash

# This script is used to run the LUNA experiment with a specific configuration.
# It sets the root directory, output path, and runs the main Python script with the specified configuration.
train_test_data_folder="train_test_split_1"

root_directory="/home/anagupta/luna/$train_test_data_folder"
output_path="$root_directory/output.txt"

export HYDRA_FULL_ERROR=1 
python3 /home/anagupta/luna/LUNA/main.py --config-path="$root_directory" --config-name=config.yaml > "$output_path"