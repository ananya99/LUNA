import sys
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra import initialize, compose
from datetime import datetime
import pandas as pd
import os
import pytz

def update_config(cfg, data_directory_name='train_test_split_1'):
    
    timezone = pytz.timezone('Europe/Zurich')
    now = datetime.now(timezone)
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    cfg.general.name = 'luna' + '_' + date_str + '_' + time_str

    cfg.distribute.gpus_per_node=[1]
    # cfg.general.wandb='disabled'
    cfg.general.debug = True

    cfg.dataset.maximum_graph_size.train=1000
    cfg.dataset.maximum_graph_size.test=1000
    cfg.train.batch_size=4
    cfg.model.hidden_dims.cell_image_dimensions=256
    cfg.model.hidden_dims.num_heads=16

    # Use the setting to quickly check the model
    cfg.validation.check_val_every_n_epochs=1
    cfg.validation.save_model_every_n_epochs=1
    cfg.train.n_epochs=4
    cfg.model.diffusion_steps=2

    cfg.dataset.gene_columns_start = 13
    cfg.dataset.gene_columns_end = 360


    run_directory = '/home/anagupta/luna/runs' + '/' + date_str + '/' + time_str
    if not os.path.exists(run_directory):
        os.makedirs(run_directory)
    cfg.general.local_saved_path = run_directory + '/train_results'
    cfg.test.save_dir = run_directory + '/test_results' # Change this to the directory where you want to save the results

    data_directory = '/home/anagupta/luna/' + data_directory_name
    cfg.dataset.train_data_path = data_directory + '/train_data.csv' # Change this to the path of the train csv file
    cfg.dataset.test_data_path = data_directory + '/test_data.csv' # Change this to the path of the test csv file
    cfg.dataset.slice_images_path = data_directory + '/slice_images' # Change this to the path of the slice images
    cfg.dataset.train_cell_images_path = data_directory + '/train_cell_images' # Change this to the path of the train cell images
    cfg.dataset.test_cell_images_path = data_directory + '/test_cell_images' # Change this to the path of the test cell images

    cfg.dataset.dataset_name = 'luna_with_cell_images' if cfg.dataset.train_cell_images_path else 'luna_without_cell_images'

    # Save the cfg configuration file
    OmegaConf.save(cfg, run_directory + '/config.yaml')
    
    return cfg