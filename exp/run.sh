#!/bin/bash

# This script is used to run the LUNA experiment with a specific configuration.
# It sets the root directory, output path, and runs the main Python script with the specified configuration.

# get the current date and time combined string with underscore
date_time=$(date +%Y_%m_%d_%H_%M_%S)

# get the current directory
current_directory=$(pwd)


root_directory="/home/anagupta/luna/runs/run_$date_time"
output_path="$root_directory/output.log"

echo "Running the experiment with the following configuration:"
echo "Root directory: $root_directory"
echo "Output path: $output_path"

export HYDRA_FULL_ERROR=1 
# python3 /home/anagupta/luna/LUNA/exp/run.py $date_time
# python3 /home/anagupta/luna/LUNA/main.py --config-path="$root_directory" --config-name=config.yaml | tee "$output_path"

# use this command to directly run the python script
# python3 /home/anagupta/luna/LUNA/main.py --config-path=/home/anagupta/luna/train_test_split_1 --config-name=config.yaml | tee /home/anagupta/luna/train_test_split_1/output.log
