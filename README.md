<p>
<img src="https://github.com/mlbio-epfl/LUNA/blob/main/image/LUNA.png" width="400" align="center">
</p>

**LUNA** is a Python package written in [PyTorch](https://pytorch.org/) that predicts spatial locations of cells from gene expression profiles. It employs a diffusion-based generative model that captures the complex spatial interrelationships of cells within tissues, enabling de novo reconstruction of cell locations. The single-cell and spatial datasets should be derived from the same anatomical region or tissue type and share a common set of genes. During training, LUNA learns spatial priors over existing spatial transcriptomics data. At inference stage, LUNA generates complex tissue structures solely from gene expressions of dissociated cell

<p align="center">
<img src="https://github.com/mlbio-epfl/LUNA/blob/main/image/LUNA_Framework.png" width="1100" align="center">
</p>


Training Input:
- A **gene expression matrix** and corresponding **cell coordinates** from spatial transcriptomics data, accompanied by section information and ideally cell class annotations. The input should be in `.csv` format.

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
  - **Section information** of cells (`regions`).
  - **Gene expression matrix** (preprocessed cell by gene matrix, preferably log2-transformed).

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

To use LUNA, begin by adjusting the settings in the configuration file located at `/configs/experiment`. This file, which leverages [Hydra](https://hydra.cc/docs/intro/) for managing configurations, contains essential parameters like the experiment name, dataset paths, and training/testing splits. Run the `main.py` file to start the experiment. Here is a breakdown of the critical elements in the configuration file:

#### General Settings
- `mode`: Defines the mode to run the model. Options include:
  - `'train_and_test'`: Train the model and then automatically run tests.
  - `'test_only'`: Skip training and load checkpoints for testing only.

#### Dataset Configuration
- `dataset_name`: Name of the dataset you are working with.
- `data_path`: Path to your dataset’s `.csv` file.
- `gene_columns_start` and `gene_columns_end`: Indicate the starting and ending columns for gene expression data in your dataset.

#### Test Settings
- `checkpoints_parent_dir`: Directory path containing the checkpoints. This setting is only used in `'test_only'` mode. If running `'train_and_test'`, set this to `'null'`.
- `checkpoints_name_list`: A list of checkpoints to test:
  - Use `'all'` to test every checkpoint in the `checkpoints_parent_dir`.
  - Specify individual checkpoints, e.g., `['epoch=749.ckpt']`, if you only want to test specific models.
- `save_dir`: Directory to save test results. Use `'./'` to save in the current directory, or set it to `'null'` to save in `checkpoints_parent_dir`.

Once your configuration is ready, execute the script. Simply change the `experiment` value in `/configs/config.yaml` to point to your updated configuration file, and LUNA will be ready to run by

```python
python main.py 
```

### Example Usage

We provide a sample dataset from the [MERFISH Mouse Primary Motor Cortex Atlas](https://drive.google.com/file/d/1j5LRRQ66n8PpRKRmOhn9eRekgxuxw-fw/view?usp=drive_link). To use this dataset with LUNA, download it to your local machine, update the `data_path` in the configuration file to reflect this dataset's location, and execute `main.py` to run LUNA on this dataset.


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



***
## Contact

If you have questions, please contact the authors of the method:
- Yist YU - <tingyang.yu@epfl.ch>  
- Maria Brbi'c - <mbrbic@epfl.ch>


