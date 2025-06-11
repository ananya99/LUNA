#!/bin/bash

echo "[run.sh] Running create_config.py with args: $@"
# Run create_config.py with original args; if it fails, exit script
b=$(python3 /mlbio_scratch/anagupta/luna/LUNA/exp/create_config.py "$@") || { echo "create_config.py failed, exiting"; exit 1; }
echo "[run.sh] Config created at: $b"

# Run main.py with output from create_config.py
echo "[run.sh] Running main.py with args: --config-path=$b > $b/output.log"
python3 /mlbio_scratch/anagupta/luna/LUNA/main.py "--config-path=$b" > "$b/output.log"
echo "[run.sh] Done!!!"