import argparse
import subprocess
import sys

if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO
import pandas as pd

def get_free_gpus():
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
    gpu_df = pd.read_csv(StringIO(gpu_stats.decode('utf-8')), names=['memory.used', 'memory.free'], skiprows=1)
    
    gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: x.rstrip(' [MiB]'))
    gpu_df['memory.free'] = pd.to_numeric(gpu_df['memory.free'])
    
    # Sort by free memory (descending)
    sorted_gpus = gpu_df.sort_values('memory.free', ascending=False)
    return sorted_gpus.index.tolist()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=2, help='Number of GPUs to use')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    num_gpus = args.num
    if args.debug:
        num_gpus = 1
    
    free_gpus = get_free_gpus()
    # print(f"Free GPUs: {free_gpus}")
    # print(f"Using GPUs: {free_gpus[:num_gpus]}")
    
    gpus_arg = "--gpus " + " ".join(map(str, free_gpus[:num_gpus]))
    print(gpus_arg)