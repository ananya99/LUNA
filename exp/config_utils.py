import sys
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra import initialize, compose
from datetime import datetime
import pandas as pd
import os
import pytz

def update_config(cfg, output_dir, data_directory_name='train_test_split_1'):
    
    timezone = pytz.timezone('Europe/Zurich')
    now = datetime.now(timezone)
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    cfg.general.name = 'luna' + '_' + date_str + '_' + time_str

    cfg.distribute.gpus_per_node=[1,2,3,4]
    # cfg.general.wandb='disabled'
    # cfg.general.debug = True

    cfg.dataset.maximum_graph_size.train=1000
    cfg.dataset.maximum_graph_size.test=1000
    cfg.train.batch_size=4
    cfg.model.hidden_dims.cell_image_dimensions=32
    cfg.model.hidden_dims.num_heads=16

    # Use the setting to quickly check the model
    # cfg.validation.check_val_every_n_epochs=1
    # cfg.validation.save_model_every_n_epochs=1
    # cfg.train.n_epochs=4
    # cfg.model.diffusion_steps=2

    cfg.dataset.gene_columns_start = 13
    cfg.dataset.gene_columns_end = 360

    cfg.general.local_saved_path = output_dir + '/train_results'
    cfg.test.save_dir = output_dir + '/test_results' 

    data_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', data_directory_name)
    if not os.path.exists(data_directory):
        raise FileNotFoundError(f"Data directory {data_directory} does not exist")
    cfg.dataset.train_data_path = data_directory + '/train_data.csv' 
    cfg.dataset.test_data_path = data_directory + '/test_data.csv' 
    cfg.dataset.slice_images_path = data_directory + '/slice_images' 
    cfg.dataset.train_cell_images_path = data_directory + '/train_cell_images' 
    cfg.dataset.test_cell_images_path = data_directory + '/test_cell_images' 

    cfg.dataset.dataset_name = 'luna_with_cell_images' if cfg.dataset.train_cell_images_path else 'luna_without_cell_images'
    
    return cfg


if __name__ == "__main__":
    initialize(config_path="../configs") 
    cfg = compose(config_name="config")
    update_config(cfg, output_dir=os.path.dirname(__file__))