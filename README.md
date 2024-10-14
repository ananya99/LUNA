# LUNA

<p align="center">
<img src="https://github.com/mlbio-epfl/LUNA/blob/main/image/LUNA_Framework.png" width="1100" align="center">
</p>

**LUNA** is a Python package written in PyTorch designed to generate tissue structures by predicting the spatial locations of individual cells based on their gene expression profiles. LUNA is a generative diffusion model that learns cell embeddings that capture each cell's spatial position relative to others based on their gene expression profiles. During inference, LUNA starts with random Gaussian noise and reconstructs cell locations de novo, guided by spatial priors learned during training. 

Training Input:
- A **gene expression matrix** and corresponding **cell coordinates** from spatial transcriptomics data, ideally accompanied by section information and cell class annotations. The input should be in `.csv` format.

Testing Input and Output:
- **Input**: A **gene expression matrix** for single cells in `.csv` format, which should have the same number of genes as the training data.
- **Output**: Generated **2D spatial coordinates** of cells in the tissue, based on the input gene expression data, provided in `.csv` format.

---

## Setting up LUNA

### Prepare the Dataset
To train LUNA, prepare the input `.csv` file where:
- Rows represent **cells**.
- Columns represent **features** such as:
  - **2D coordinates** of cells (`'X'`, `'Y'`).
  - **Section information** of cells.
  - **Gene expression matrix** (preprocessed, preferably log2-transformed).

---

### Installation Requirements

To begin, clone the LUNA repository from GitHub:

```bash
git clone https://github.com/mlbio-epfl/luna.git
```

Create the conda environment:
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

## Generating Tissue Structure Using LUNA

### Configuration

To run LUNA, you'll need to modify the configuration file found in `/configs/experiment`, and then run the `main.py` file. The configuration file contains all the necessary settings for your experiment, such as the experiment name, dataset paths, and training/test split. Below is a guide to the key components of the configuration:

#### General Settings
- `name`: Set the name of your experiment. For example, `'MERFISH_mouse_cortex'` works well for a mouse cortex dataset.
- `mode`: Defines the mode to run the model. Options include:
  - `'train_and_test'`: Train the model and then automatically run tests.
  - `'test_only'`: Skip training and load checkpoints for testing only.

#### Dataset Configuration
- `dataset_name`: Name of the dataset you are working with.
- `data_path`: Path to your dataset’s `.csv` file.
- `gene_columns_start` and `gene_columns_end`: Indicate the starting and ending columns for gene expression data in your dataset.
- `coordinate_X_column_name` and `coordinate_Y_column_name`: The column names for X and Y cell coordinates in your data.
- `cell_type_column_name`: Column for cell type annotations (used for evaluation).
- `section_column_name`: Column for the dataset section or region.
- `train_regions`: List of regions/slices to use for training. You can modify this list to include specific slices.
- `test_regions`: List of regions/slices to use for testing. Adjust the list to reflect your test dataset.
- `validation_regions`: List of regions used for validation during training (optional).

#### Test Settings
- `checkpoints_parent_dir`: Directory path containing the checkpoints. This setting is only used in `'test_only'` mode. If running `'train_and_test'`, set this to `'null'`.
- `checkpoints_name_list`: A list of checkpoints to test:
  - Use `'all'` to test every checkpoint in the `checkpoints_parent_dir`.
  - Specify individual checkpoints, e.g., `['epoch=749.ckpt']`, if you only want to test specific models.
- `save_dir`: Directory to save test results. Use `'./'` to save in the current directory, or set it to `'null'` to save in `checkpoints_parent_dir`.

#### Example Setup
You can customize all the above parameters based on your specific dataset and experiment. For example:
- Modify the configuration under `dataset` to fit to your data.
- Change the number of GPUs in use by specifying GPU parameters in the config.

Once your configuration is ready, execute the script. Simply change the `experiment` value in `/configs/config.yaml` to point to your updated configuration file, and LUNA will be ready to run.






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






