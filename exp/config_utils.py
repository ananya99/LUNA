import argparse
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra import initialize, compose
from datetime import datetime
import pandas as pd
import os
import pytz

def update_config(cfg, output_dir, data_directory_name='train_test_split_1', name='', debug=False):
    
    now = datetime.now()
    if name == '':
        name = now.strftime("%Y-%m-%d_%H:%M:%S")
    cfg.general.name = 'luna' +  '_' + name

    cfg.distribute.gpus_per_node=[3,4]
    cfg.general.mock_data_for_debugging = True

    cfg.dataset.maximum_graph_size.train=500
    # cfg.dataset.maximum_graph_size.test=2000
    cfg.train.batch_size=4
    cfg.model.hidden_dims.cell_image_embedding_dim=32
    cfg.model.hidden_dims.num_heads=16
    # cfg.model.cell_image_encoder="DINOv2"

    # Use the setting to quickly check the model
    if debug:
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

    data_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', data_directory_name)
    if not os.path.exists(data_directory):
        raise FileNotFoundError(f"Data directory {data_directory} does not exist")
    cfg.dataset.train_data_path = data_directory + '/train_data.csv' 
    cfg.dataset.test_data_path = data_directory + '/test_data.csv' 
    # cfg.dataset.slice_images_path = data_directory + '/slice_images' 
    
    cfg.dataset.cell_image_embeddings_path = '/mlbio_scratch/wen2/scMAE/xenium_preprocessed_indomain'
    # cfg.dataset.train_cell_images_path = data_directory + '/train_cell_images' 
    # cfg.dataset.test_cell_images_path = data_directory + '/test_cell_images' 

    if cfg.dataset.cell_image_embeddings_path:
        cfg.dataset.dataset_name = 'luna_mae_embeddings'
    elif cfg.dataset.train_cell_images_path:
        cfg.dataset.dataset_name = 'luna' + '_' + cfg.model.cell_image_encoder
    else:
        cfg.dataset.dataset_name = 'luna_baseline'
    
    return cfg


if __name__ == "__main__":
    
    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='Name of the configuration')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--model', type=str, default='baseline', help='Model to use')
    args = parser.parse_args()

    initialize(config_path="../configs") 
    cfg = compose(config_name="config")
    
    # output_dir=os.path.dirname(__file__)
    # update_config(cfg, output_dir)
    
    output_dir ='/mlbio_scratch/anagupta/luna/runs'
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H-%M-%S")
    name = date_str + '_' + time_str
        
    if args.name != '':
        output_dir = os.path.join(output_dir, args.name, name)
    else:
        output_dir = os.path.join(output_dir, name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    update_config(cfg, output_dir, name=name, debug=args.debug)
    
    # Save the cfg configuration file
    OmegaConf.save(cfg, output_dir + '/config.yaml')
    print(f"Config saved to {output_dir}/config.yaml")