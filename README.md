# LUNA

<p align="center">
<img src="https://github.com/mlbio-epfl/LUNA/blob/main/image/LUNA_Framework.png" width="1100" align="center">
</p>

LUNA is a Python package, written in PyTorch that generates tissue structures by predicting the spatial locations of individual cells based on their gene expression profiles. 

LUNA takes as an input:
- Gene expression matrix and cell coordinatesfrom Spatial transcriptomics data with section information and ideally with cell class information (.csv)

LUNA outputs:
- 2D Spatial coordinates of cells in tissue, generated de novo from gene expression data (.csv)


## Setting up LUNA

### Prepare Dataset
To train LUNA you will need to prepare the input .csv file with rows as cells and columns as features. The columns should include the 2d coordinates of cells ('X', 'Y'), the section information of cells, and gene expression matrics (preprocessed, we recommend the matrices to be log2-transformed). 


### Requirements

To begin, clone the codebase from GitHub:

```bash
git clone https://github.com/mlbio-epfl/luna.git
```

Create the conda environment
```bash
conda create -n LUNA python=3.9 numpy pandas
conda activate LUNA
```

Install cuda, pytorch, torch-geometric, lightning and other pip libraries: 
```bash
conda install nvidia/label/cuda-11.8.0::cuda-toolkitpip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install torch_geometric
pip install lightning
pip install scanpy wandb colorcet squidpy hydra-core linear_attention_transformer
```

## Generating tissue structure using LUNA

### Configuration

To run LUNA, modify the configuration file located in `/configs/experiment`. The main sections of the configuration file allow you to set up the experiment's name, mode, dataset paths, and training/test split. Below is a description of the key components:

#### General Settings
- **name**: Set the name of your experiment. For example, `'MERFISH_mouse_cortex'` is used for this dataset.
- **wandb**: Specifies the logging mode for Weights & Biases. Options are:
  - `'online'`: Log data to the cloud.
  - `'disabled'`: Turn off logging.
- **mode**: Set the run mode to either:
  - `'train_and_test'`: Train the model and test it automatically.
  - `'test_only'`: Skip training and only run the model on the test set by loading from existing checkpoints.

#### Dataset
- **dataset_name**: The name of the dataset to be used.
- **data_path**: Path to the dataset .csv file.
- **gene_columns_start** and **gene_columns_end**: Specify the start and end columns in the dataset for gene expression data.
- **coordinate_X_column_name** and **coordinate_Y_column_name**: The column names for cell coordinates (X and Y).
- **cell_type_column_name**: The column name for cell type annotations (for evaluation purpose).
- **section_column_name**: The column that holds region or section information in the dataset.
- **train_regions**: List of region slices to be used for training. You can modify the list to include specific slices from your dataset.
- **test_regions**: List of region slices to be used for testing. Add or modify based on the dataset split.
- **validation_regions**: List of region slices for validation during training (optionally).

#### Test Settings
- **checkpoints_parent_dir**: The path to the directory containing the checkpoints for testing. This is only used in `'test_only'` mode. Set it to `'null'` for `'train_and_test'` to use the default location.
- **checkpoints_name_list**: The list of specific checkpoints to test. You can:
  - Set it to `'all'` to test all checkpoints in the `checkpoints_parent_dir`.
  - Specify a list of particular checkpoints, such as `['epoch=749.ckpt']`, to test specific models.
- **save_dir**: The directory where test results will be saved. Set it to `'./'` to save in the current directory, or use `'null'` to save in `checkpoints_parent_dir`.
####

You can customize the above parameters according to your dataset and experimental setup. For example, you may adjust the `data_path`, change the `train_regions` and `test_regions` as per your specific data split, or modify the number of GPUs being used.

Once your configuration is set, LUNA will be ready to run. Simply execute the script by changing the **experiment** in `/configs/config.yaml` to the modified configuration file.





## Data Availability

|Link|Description|
|----|-----------|
|https://alleninstitute.github.io/abc_atlas_access/descriptions/Zhuang-ABCA-1.html|MERFISH Whole Mouse Brain Atlas (ABC Atlas) for Animal 1|
|https://alleninstitute.github.io/abc_atlas_access/descriptions/Zhuang-ABCA-2.html|MERFISH Whole Mouse Brain Atlas (ABC Atlas) for Animal 2|
|https://doi.brainimagelibrary.org/doi/10.35077/g.21|MERFISH Mouse Primary Motor Cortex Atlas (Brain Image Library)|
|https://singlecell.broadinstitute.org/single_cell/study/SCP1830|scRNA-seq Mouse Central Nervous System Atlas (Single Cell Portal)|
|https://singlecell.broadinstitute.org/single_cell/study/SCP2170/slide-tags-snrna-seq-on-mouse-embryonic-e14-brain|Slide-tags Datasets: Mouse Embryonic Brain (SCP2170)|
|https://singlecell.broadinstitute.org/single_cell/study/SCP2167/slide-tags-snrna-seq-on-human-prefrontal-cortex|Slide-tags Datasets: Human Brain (SCP2167)|
|https://singlecell.broadinstitute.org/single_cell/study/SCP2169/slide-tags-snrna-seq-on-human-tonsil|Slide-tags Datasets: Human Tonsil (SCP2169)|
|https://singlecell.broadinstitute.org/single_cell/study/SCP2171/slide-tags-snrna-seq-on-human-melanoma|Slide-tags Datasets: Human Melanoma (SCP2171)|






