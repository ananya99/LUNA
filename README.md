<p>
<img src="https://github.com/mlbio-epfl/LUNA/blob/main/image/LUNA.png" width="400" align="center">
</p>

**LUNA** is a Python package written in [PyTorch](https://pytorch.org/) that predicts spatial locations of cells from gene expression profiles. It employs a diffusion-based generative model that captures the complex spatial interrelationships of cells within tissues, enabling de novo reconstruction of cell locations. The single-cell and spatial datasets should be derived from the same anatomical region or tissue type and share a common set of genes. During training, LUNA learns spatial priors over existing spatial transcriptomics data. At inference stage, LUNA generates complex tissue structures solely from gene expressions of dissociated cell

<p align="center">
<img src="https://github.com/mlbio-epfl/LUNA/blob/main/image/LUNA_Framework.png" width="1100" align="center">
</p>

Input:
- A **gene expression matrix** and corresponding **cell coordinates** from spatial transcriptomics data, accompanied by section information and ideally cell class annotations. The files should be in `.csv` format.
- A separate **gene expression matrix** for single cells, also in `.csv` format, for testing purposes. This matrix should contain the same number of genes as the training dataset.

Output:
- **Output**: The generated **2D spatial coordinates** of cells, based on their gene expression data, provided in `.csv` format.

---

## Setting up LUNA

### Prepare the Dataset
To effectively train LUNA, organize your input `.csv` file in the following format:
- **Rows** represent **cells** and include data from both the training and testing datasets.
- **Columns** represent **features**, detailed as follows:
  - **2D Coordinates**: Use `'coord_X'` and `'coord_Y'` for spatial coordinates of cells. For cells without spatial information (i.e., test set), use zeros.
  - **Section Information** (`region`): This column should specify the regions cells are sourced from, helping differentiate between training and testing slices.
  - **Gene Expression Matrix**: Include a preprocessed cell-by-gene matrix, preferably normalized using log2 transformation.
  - **Cell Annotation** (`class`): Use this to categorize cells, aiding in the evaluation and visualization of generated results.

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

#### General 
- `mode`: Defines the mode to run the model. Options include:
  - `'train_and_test'`: Train the model and then automatically run tests.
  - `'test_only'`: Skip training and load checkpoints for testing only.

#### Dataset Configuration
- `dataset_name`: Specifies the name of the dataset you are utilizing.
- `data_path`: Provides the path to your dataset’s `.csv` file.
- `gene_columns_start` and `gene_columns_end`: Define the columns where gene expression data begins and ends within your dataset.
- `train_regions` and `test_regions`: Designate specific regions of the dataset to be used for training and testing, respectively.

#### Test
- `save_dir`: Directory to save test results. Use `'./'` to save in the current directory, or set it to `'null'` to save in `checkpoints_parent_dir`.
- `checkpoints_parent_dir`: Directory path containing the checkpoints. This setting is only used in `'test_only'` mode. If running `'train_and_test'`, set this to `'null'`.
- `checkpoints_name_list`: A list of checkpoints to test:
  - Use `'all'` to test every checkpoint in the `checkpoints_parent_dir`.
  - Specify individual checkpoints, e.g., `['epoch=749.ckpt']`, if you only want to test specific models.

Once your configuration is ready, execute the script. Simply change the `experiment` value in `/configs/config.yaml` to point to your updated configuration file, and LUNA will be ready to run by

```python
python main.py 
```

### Example Usage

We provide a sample dataset from the [MERFISH Mouse Primary Motor Cortex Atlas](https://drive.google.com/file/d/1j5LRRQ66n8PpRKRmOhn9eRekgxuxw-fw/view?usp=drive_link) [1]. To use this dataset with LUNA, download it to your local machine, update the `data_path` in the configuration file to reflect this dataset's location, and execute `main.py` to run LUNA on this dataset.


## Reference
[1] Zhang, Meng, et al. "Spatially Resolved Cell Atlas of the Mouse Primary Motor Cortex by MERFISH." Nature 598.7879 (2021): 137-143.

***
## Contact

If you have questions, please contact the authors of the method:
- Yist YU - <tingyang.yu@epfl.ch>  
- Maria Brbi'c - <mbrbic@epfl.ch>


