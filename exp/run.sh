#!/bin/bash

source .venv/bin/activate
echo "[run.sh] Activated virtual environment"

echo "[run.sh] Running create_config.py with args: $@"
# Run create_config.py with original args; if it fails, exit script
b=$(python3 /mlbio_scratch/anagupta/luna/LUNA/exp/create_config.py "$@") || { echo "create_config.py failed, exiting"; exit 1; }
echo "[run.sh] Config created at: $b/config.yaml"

# Run main.py with output from create_config.py
echo "[run.sh] Running main.py"
echo "[run.sh] Command: python3 /mlbio_scratch/anagupta/luna/LUNA/main.py --config-path=$b > $b/output.log 2>&1"
echo "[run.sh] Output log: $b/output.log"
python3 /mlbio_scratch/anagupta/luna/LUNA/main.py "--config-path=$b" > "$b/output.log" 2>&1 || { echo "main.py failed, exiting"; exit 1; }
echo "[run.sh] Done!!!"