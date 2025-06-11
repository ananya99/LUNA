import argparse
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra import initialize, compose
from datetime import datetime
import pandas as pd
import os
import pytz

all_outputs_dir ='/mlbio_scratch/anagupta/luna/runs'

def create_and_save_config(name='', mode='baseline', debug=False, gpus=[5,6,7], data_dir='train_test_split_1'):
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H-%M-%S")
        
    if name is not None:
        output_subdir = name + ("_debug" if debug else "")
    else:
        output_subdir = mode + ("_debug" if debug else "")
    
    output_dir = os.path.join(all_outputs_dir, output_subdir, date_str + '_' + time_str)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="config", overrides=[f"hydra.run.dir={output_dir}"], return_hydra_config=True)
        os.chdir(cfg.hydra.run.dir)  # This is what @hydra.main does under the hood

    cfg.general.name = 'luna' +  '_' + date_str + '_' + time_str 

    cfg.distribute.gpus_per_node=gpus

    cfg.dataset.maximum_graph_size.train=500
    cfg.dataset.maximum_graph_size.test=5000
    cfg.train.batch_size=4
    cfg.model.hidden_dims.cell_image_embedding_dim=32
    cfg.model.hidden_dims.num_heads=16

    if debug:
        cfg.general.mock_data_for_debugging = True
        cfg.general.wandb='disabled'
        cfg.validation.check_val_every_n_epochs=1
        cfg.validation.save_model_every_n_epochs=1
        cfg.train.n_epochs=2
        cfg.model.diffusion_steps=2
    else:
        cfg.train.n_epochs=500

    cfg.dataset.gene_columns_start = 13
    cfg.dataset.gene_columns_end = 360

    cfg.general.local_saved_path = output_dir
    cfg.test.checkpoints_parent_dir = output_dir + '/checkpoints'
    cfg.test.save_dir = output_dir + '/test_results' 
    if not os.path.exists(cfg.test.save_dir):
        os.makedirs(cfg.test.save_dir)

    data_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', data_dir)
    if not os.path.exists(data_directory):
        raise FileNotFoundError(f"Data directory {data_directory} does not exist")
    
    cfg.dataset.train_data_path = data_directory + '/train_data.csv' 
    cfg.dataset.test_data_path = data_directory + '/test_data.csv' 
    
    if mode == 'baseline':
        cfg.dataset.dataset_name = 'luna_baseline'
        pass
    elif mode == 'cnn':
        cfg.model.cell_image_encoder="CNN"
        cfg.dataset.train_cell_images_path = data_directory + '/train_cell_images' 
        cfg.dataset.test_cell_images_path = data_directory + '/test_cell_images' 
    elif mode == 'dinov2':
        cfg.model.cell_image_encoder="DINOv2"
        cfg.dataset.train_cell_images_path = data_directory + '/train_cell_images' 
        cfg.dataset.test_cell_images_path = data_directory + '/test_cell_images' 
    elif mode == 'mae' or mode == 'embed':
        cfg.model.cell_image_encoder="MAE_embeddings"
        cfg.dataset.train_cell_image_embeddings_path = data_directory + '/train_embeddings.pt'
        cfg.dataset.test_cell_image_embeddings_path = data_directory + '/test_embeddings.pt'
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    if mode != 'baseline':
        cfg.dataset.dataset_name = 'luna_' + cfg.model.cell_image_encoder
    # cfg.dataset.slice_images_path = data_directory + '/slice_images' 
    
    OmegaConf.save(cfg, output_dir + '/config.yaml')
    print(f"{output_dir}")


if __name__ == "__main__":
    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='Name of the configuration, if not provided, mode and date will be used')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--mode', type=str, default='baseline', help='Mode to use, options: baseline, cnn, dinov2, mae, embed')
    parser.add_argument('--gpus', type=int, nargs='+', default=[5,6,7], help='GPUs to use, Pass as integers separated by spaces')
    parser.add_argument('--data_dir', type=str, default='train_test_split_1', help='Name of the data directory inside "/mlbio_scratch/anagupta/luna/data"')
    args = parser.parse_args()
    
    create_and_save_config(name=args.name, mode=args.mode, debug=args.debug, gpus=args.gpus, data_dir=args.data_dir)