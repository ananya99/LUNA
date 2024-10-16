<p>
<img src="https://github.com/mlbio-epfl/LUNA/blob/main/image/LUNA.png" width="400" align="center">
</p>

**LUNA** is a Python package written in [PyTorch](https://pytorch.org/) that predicts spatial locations of cells from gene expression profiles. It employs a diffusion-based generative model that captures the complex spatial interrelationships of cells within tissues, enabling de novo reconstruction of cell locations. The single-cell and spatial datasets should be derived from the same anatomical region or tissue type and share a common set of genes. During training, LUNA learns spatial priors over existing spatial transcriptomics data. At inference stage, LUNA generates complex tissue structures solely from gene expressions of dissociated cells.

<p align="center">
<img src="https://github.com/mlbio-epfl/LUNA/blob/main/image/LUNA_Framework.png" width="1100" align="center">
</p>

Input:
- A **gene expression matrix** along with the corresponding **cell coordinates** from spatial transcriptomics data, accompanied by section information. The files should be in `.csv` format, and is used for model training.
- A **gene expression matrix** for single cells lacking spatial information, accompanied ideally by cell class annotations (for visualization purpose). This matrix should contain the same number of genes as the training dataset. The files should be in `.csv` format, and is used for model inference. 

Output:
- The generated **2D spatial coordinates** of cells, based on their gene expression data, provided in `.csv` format.

---

## Setting up LUNA

### Prepare the Dataset
To effectively train LUNA, organize your input `.csv` files in the following format:
- **Rows** represent **cells**.
- **Columns** represent **features**, detailed as follows:
  - **2D Coordinates**: Use `'coord_X'` and `'coord_Y'` for spatial coordinates of cells. For cells without spatial information (i.e., test set), use zeros.
  - **Section Information** (`cell_section`): This column should specify the section cells are sourced from. Cells from the same section (slice) with be grouped as one input sample. 
  - **Gene Expression Matrix**: Include a preprocessed cell-by-gene matrix, preferably normalized using log2 transformation.
  - **Cell Annotation** (`cell_class`): Use this to categorize cells, aiding in the evaluation and visualization of generated results.

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
***
## Generating Tissue Structure Using LUNA

`/example/MERFISH_mouse_cortex.ipynb` has an example of running LUNA, and results evaluation.

### Configuration

To use LUNA, begin by adjusting the settings in the configuration file located at `/configs/experiment`. This file, which leverages [Hydra](https://hydra.cc/docs/intro/) for managing configurations, contains essential parameters like the experiment name, dataset paths, and training/testing splits. Run the `main.py` file to start the experiment. Here is a breakdown of the critical elements in the configuration file:

#### Dataset
- `dataset_name`: Specifies the name of the dataset you are utilizing.
- `data_path_train`: Provides the path to your train dataset’s `.csv` file.
- `data_path_inference`: Provides the path to your inference dataset’s `.csv` file.
- `gene_columns_start` and `gene_columns_end`: Define the columns where gene expression data begins and ends within your dataset (train dataset and inference dataset should have the same number of genes and gene columns should be ordered the same).

#### Test
- `save_dir`: Directory to save test results. Use `'./'` to save in the current codebase directory.

Once your configuration is ready, execute the script. Simply change the `experiment` value in `/configs/config.yaml` to point to your updated configuration file, and LUNA will be ready to run by

```python
python main.py 
```

### Example Usage

We provide a sample dataset from the [MERFISH Mouse Primary Motor Cortex Atlas](https://drive.google.com/file/d/1YP1s_dERAUh7vXUjMRSvuFJllGBX8tYr/view?usp=drive_link) [1]. To use this dataset with LUNA, download it to your local machine, you can either follow the instruction in `/example/MERFISH_mouse_cortex.ipynb` file OR simply update the `data_path` in the configuration file to reflect this dataset's location, and execute `main.py` to run LUNA on this dataset.

## Contact

If you have questions, please contact the authors of the method:
- Yist YU - <tingyang.yu@epfl.ch>  
- Maria Brbic - <mbrbic@epfl.ch>

## Reference
[1] Zhang, Meng, et al. "Spatially Resolved Cell Atlas of the Mouse Primary Motor Cortex by MERFISH." Nature 598.7879 (2021): 137-143.


