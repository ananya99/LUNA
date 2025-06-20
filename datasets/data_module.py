from typing import Dict
import numpy as np
import omegaconf
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from utils.data.abstract_datatype import (
    AbstractDataModule,
    AbstractDatasetInfos,
    Statistics,
)
from utils.data.load import (
    character_to_int,
    detect_nan_rows,
    position_normalize,
    standardise_dataframe_colnames
)
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
import tarfile
import os
from PIL import Image
from io import BytesIO
import concurrent.futures


class Dataset(InMemoryDataset):
    def __init__(
        self,
        split: int,
        input_data: pd.DataFrame,
        cell_images = None,
        root: str = None,
        transform: callable = None,
        pre_transform: callable = None,
        pre_filter: callable = None,
        cfg: omegaconf = None,
    ) -> None:
        super().__init__(root, transform, pre_transform, pre_filter)
        self.split = split
        self.name = cfg.dataset.dataset_name
        self.input_data = input_data
        self.cell_images = cell_images
        self.num_cell_class = len(input_data["cell_class"].unique())
        self.maximum_graph_size = cfg.dataset.maximum_graph_size[split]
        self.cfg = cfg
        
        self._data, self.slices = Data(), {}

        # Dataset processing pipeline
        self.process_data()
        self.process_slices()

    def process_data(self) -> None:
        self.input_data = self.input_data.sort_values("cell_section", ignore_index=False)
        gene_names = self.filter_genes()

        # Normalize coordinates
        self.input_data = position_normalize(self.input_data)
        (
            positions,
            node_features,
            cell_class,
            cell_class_decoder,
        ) = self._convert_data_to_tensors(gene_names)

        # Clean NaN rows
        nan_rows = detect_nan_rows(positions)
        # print the number of nan_rows
        print("[DEBUG] nan_rows", nan_rows.sum())
        cell_id = torch.tensor(self.input_data.index.values)
        clean_positions, clean_node_features, clean_cell_class, clean_cell_id = self._clean_data(
            positions, node_features, cell_class, cell_id, nan_rows
        )

        # Update data attributes
        self._update_data_attributes(
            clean_cell_id,
            clean_positions,
            clean_node_features,
            clean_cell_class,
            gene_names,
            cell_class_decoder,
            cell_images=self.cell_images,
        )

    def _convert_data_to_tensors(self, gene_names: list):
        positions = torch.tensor(self.input_data[["coord_X", "coord_Y"]].values).float()
        node_features = torch.tensor(self.input_data[gene_names].values).float()
        cell_class = self.input_data["cell_class"]
        unique_class = sorted(list(cell_class.unique()))
        cell_class, cell_class_decoder = character_to_int(
            list(cell_class.values), unique_class
        )
        return positions, node_features, torch.tensor(cell_class), cell_class_decoder

    def _clean_data(self, positions, node_features, cell_class, cell_id, nan_rows):
        clean_positions = positions[~nan_rows]
        clean_node_features = node_features[~nan_rows]
        clean_cell_class = cell_class[~nan_rows]
        clean_cell_id = cell_id[~nan_rows]
        return clean_positions, clean_node_features, clean_cell_class, clean_cell_id

    def _update_data_attributes(
        self,
        clean_cell_id,
        clean_positions,
        clean_node_features,
        clean_cell_class,
        gene_names,
        cell_class_decoder,
        cell_images=None,
    ):        
        # if not self.input_data.index.is_numeric():
        #     self.input_data.index = pd.to_numeric(self.input_data.index, errors='coerce').fillna(0).astype(int)

        cell_ID = clean_cell_id

        self._data.positions = clean_positions
        self._data.node_features = clean_node_features
        self._data.cell_class = clean_cell_class
        self._data.cell_ID = cell_ID
        self._data.cell_images = cell_images if cell_images is not None else None

        num_cell_to_region_mapping_dict = self._create_region_mapping_dict()
        self.statistics = Statistics(
            num_cell_class=self.num_cell_class,
            num_genes=len(gene_names),
            cell_class_decoder=cell_class_decoder,
            num_cell_to_region_mapping_dict=num_cell_to_region_mapping_dict,
        )

    def _create_region_mapping_dict(self):
        num_cell_to_region_mapping_dict = (
            self.input_data.groupby("cell_section").size().to_dict()
        )
        return {v: k for k, v in num_cell_to_region_mapping_dict.items()}

    def filter_genes(self) -> list:
        gene_columns_start = self.cfg.dataset.gene_columns_start
        gene_columns_end = self.cfg.dataset.gene_columns_end
        gene_names = list(self.input_data.columns[gene_columns_start:gene_columns_end])
        gene_names.sort()
        return gene_names

    def process_slices(self) -> None:
        slice_indices = self._generate_slice_indices()
        slice_ = torch.tensor(slice_indices, dtype=torch.float32)

        self.slices = {
            k: slice_
            for k in [
                "node_features",
                "positions",
                "cell_class",
                "cell_ID",
                "cell_images",
            ]
        }

    def _generate_slice_indices(self):
        slices = self.input_data["cell_section"].values
        current_slice, slice_start, slice_ = slices[0], 0, []

        for i in range(1, len(slices)):
            # Check for slice change or end of slices array
            if slices[i] != current_slice or i == len(slices) - 1:
                # Determine the end of the current slice
                slice_end = i + 1 if i == len(slices) - 1 else i

                # Apply different logic based on whether maximum_graph_size is set
                if self.maximum_graph_size is None:
                    slice_.extend([slice_start, slice_end])
                else:
                    # Generate indices with step size of maximum_graph_size
                    new_indices = np.arange(slice_start, slice_end, self.maximum_graph_size).astype(int)
                    slice_.extend(new_indices)
                    slice_.append(slice_end)

                # Update the current slice and start for the next one
                current_slice, slice_start = slices[i], i      

        print(self.split)
        print(self.maximum_graph_size)
        print(np.unique(slice_))

        # Slice cell_images based on the generated indices
        if self.cell_images is not None:
            sliced_cell_images = []
            for start, end in zip(slice_[:-1], slice_[1:]):
                sliced_cell_images.append(self.cell_images[start:end])
            self._data.sliced_cell_images = sliced_cell_images

        return np.unique(slice_)


class DataModule(AbstractDataModule):
    def __init__(self, cfg):
        train_data, train_num_cells = self.data_loading(cfg, 'train')
        test_data, test_num_cells = self.data_loading(cfg, 'test')
        
        if cfg.dataset.train_cell_image_embeddings_path is not None and cfg.dataset.train_cell_images_path is not None:
            raise ValueError("Both train_cell_image_embeddings_path and train_cell_images_path are provided. Please provide only one.")
        
        train_cell_images_or_embeddings = None
        test_cell_images_or_embeddings = None
        
        if cfg.dataset.train_cell_image_embeddings_path is not None:
            train_cell_images_or_embeddings = self.load_cell_image_embeddings(cfg, 'train', train_num_cells)
            test_cell_images_or_embeddings = self.load_cell_image_embeddings(cfg, 'test', test_num_cells)
        
        if cfg.dataset.train_cell_images_path is not None:
            train_cell_images_or_embeddings = self.load_cell_images(cfg, 'train', train_num_cells)
            test_cell_images_or_embeddings = self.load_cell_images(cfg, 'test', test_num_cells)
            
        self.train_dataset = self._initialize_dataset("train", train_data, train_cell_images_or_embeddings, cfg)
        self.test_dataset = self._initialize_dataset("test", test_data, test_cell_images_or_embeddings, cfg)

        if cfg.dataset.validation_data_path:
            validation_data = self.data_loading(cfg, 'validation')
            self.validation_dataset = self._initialize_dataset("validation", validation_data, cfg)
        else:
            self.validation_dataset = None

        self.statistics = {
            "train": self.train_dataset.statistics,
            "validation": self.validation_dataset.statistics if self.validation_dataset else None,
            "test": self.test_dataset.statistics,
            "cell_images_size": self.train_dataset.cell_images[0].shape if self.train_dataset.cell_images is not None and len(self.train_dataset.cell_images)>0 else None,
        }
        super().__init__(
            cfg,
            train_dataset=self.train_dataset,
            val_dataset=self.validation_dataset if self.validation_dataset else None,
            test_dataset=self.test_dataset,
        )
        
    def process_cell_images_or_embeddings(self, cell_images_or_embeddings, data):
        if isinstance(cell_images_or_embeddings, dict):
            print("cell_images_or_embeddings is a dict")
            cell_ids = data["cell_id"].values
            present_mask = np.isin(cell_ids, list(cell_images_or_embeddings.keys()))
            if not present_mask.all():
                missing = cell_ids[~present_mask]
                print(f"[WARNING] {len(missing)} cell_ids missing in image dict. Ignoring them.")
            filtered_ids = cell_ids[present_mask]
            try:
                cell_images_tensor = torch.stack([cell_images_or_embeddings[cid] for cid in filtered_ids])
                data = data[present_mask].reset_index(drop=True)
            except Exception as e:
                raise ValueError(f"Error stacking tensors for IDs: {filtered_ids[:5]}...") from e

        elif isinstance(cell_images_or_embeddings, torch.Tensor):
            print("cell_images_or_embeddings is a tensor (mock images or embeddings)")
            cell_images_tensor = cell_images_or_embeddings
            data = data 
        else:
            raise ValueError(f"Invalid cell images type: {type(cell_images_or_embeddings)}")

        return cell_images_tensor, data

    def _initialize_dataset(self, split, data, cell_images_or_embeddings, cfg):
        if cfg.model.cell_image_encoder is None:
            return Dataset(split=split, input_data=data, cfg=cfg)
        
        cell_images_tensor, data = self.process_cell_images_or_embeddings(cell_images_or_embeddings, data)
        return Dataset(split=split, input_data=data, cell_images=cell_images_tensor, cfg=cfg)

    def collate(self, batch):
        return self._create_batch(batch)

    def _create_batch(self, batch):
        batch_data = Batch()
        batch_data.node_features = torch.cat(
            [data.node_features for data in batch], dim=0
        )
        batch_data.positions = torch.cat([data.positions for data in batch], dim=0)
        batch_data.cell_images = torch.cat(
            [data.cell_images for data in batch], dim=0
        ) if hasattr(batch[0], "cell_images") else None
        batch_data.cell_class = torch.cat([data.cell_class for data in batch], dim=0)
        batch_data.cell_ID = torch.cat([data.cell_ID for data in batch], dim=0)

        batch_data.batch = torch.tensor(
            [
                i
                for i, data in enumerate(batch)
                for _ in range(data.node_features.size(0))
            ],
            dtype=torch.long,
        )
        return batch_data

    def data_loading(self, cfg: omegaconf.DictConfig, split) -> pd.DataFrame:
        if split == 'train':
            data_path = cfg.dataset.train_data_path
        elif split == 'validation':
            data_path = cfg.dataset.validation_data_path
        else:
            data_path = cfg.dataset.test_data_path
        data = pd.read_csv(f"{data_path}")
        print(f"[INFO] Loaded {split} data from {data_path}")
        
        # remove duplicate "cell_id" rows
        data = data.drop_duplicates(subset=['cell_id'])
        
        # Ensure that the data contains the necessary columns, if not call standardise_dataframe_colnames:
        # data = standardise_dataframe_colnames(data)
        assert all(column in data.columns for column in ['coord_X', 'coord_Y', 'cell_section', 'cell_class'])
        
        return data, data.shape[0]
    
    def load_cell_images(self, cfg: omegaconf.DictConfig, split, num_cells) -> Dict[str, torch.Tensor]:
        """
        Load the cell images from the specified path or generate mock images in mock_data_for_debugging mode.
        Args:
            cfg: Configuration object containing dataset paths.
            split: The split of the dataset ('train', 'validation', 'test').
        Returns:
            torch.Tensor: Array of loaded or mock cell images.
        """
        
        # Generate mock images (128x128 zero images) for all cells in mock_data_for_debugging mode
        if cfg.general.mock_data_for_debugging:
            print(f"[INFO] mock_data_for_debugging mode enabled. Generating {num_cells} mock images.")
            return torch.zeros((num_cells, 128, 128), dtype=torch.float32)
        
        if split == 'train':
            cell_images_path = cfg.dataset.train_cell_images_path
        elif split == 'validation':
            cell_images_path = cfg.dataset.validation_cell_images_path
        else:
            cell_images_path = cfg.dataset.test_cell_images_path
            
        print(f"[INFO] Loading {split} cell images from {cell_images_path}")
        
        all_images = {}

        # Traverse all .tar files and extract .npy images into a dict
        for root, _, files in os.walk(cell_images_path):
            for file in files:
                if file.endswith('.tar'):
                    tar_file_path = os.path.join(root, file)
                    with tarfile.open(tar_file_path, 'r') as tar:
                        for tarinfo in tar:
                            if tarinfo.name.endswith('.npy'):
                                cell_id = os.path.basename(tarinfo.name).replace('.npy', '')
                                file_obj = tar.extractfile(tarinfo)
                                if file_obj is not None:
                                    image_data = file_obj.read()
                                    np_image = np.load(BytesIO(image_data)).astype(np.float32)
                                    tensor_image = torch.from_numpy(np_image)
                                    all_images[cell_id] = tensor_image

                    print(f"[INFO] Finished extracting from {tar_file_path}, total images so far: {len(all_images)}")

        print(f"[INFO] Finished loading {len(all_images)} images for split '{split}'.")
        return all_images
    
    def load_cell_image_embeddings(self, cfg: omegaconf.DictConfig, split, num_cells) -> Dict[str, torch.Tensor]:
        """
        Load the cell image embeddings from the specified path or generate mock embeddings in mock_data_for_debugging mode.
        Args:
            cfg: Configuration object containing dataset paths.
            split: The split of the dataset ('train', 'validation', 'test').
        Returns:
            torch.Tensor: Array of loaded or mock cell image embeddings.
        """
        
        if cfg.general.mock_data_for_debugging:
            print(f"[INFO] mock_data_for_debugging mode enabled. Generating {num_cells} mock cell image embeddings.")
            return torch.zeros((num_cells, 768), dtype=torch.float32)
        
        if split == 'train':
            cell_image_embeddings_path = cfg.dataset.train_cell_image_embeddings_path
        elif split == 'validation':
            cell_image_embeddings_path = cfg.dataset.validation_cell_image_embeddings_path
        else:
            cell_image_embeddings_path = cfg.dataset.test_cell_image_embeddings_path
        
        # Load entire preprocessed dict
        print(f"[INFO] Loading preprocessed {split} embeddings from: {cell_image_embeddings_path}")
        embeddings_dict = torch.load(cell_image_embeddings_path, map_location="cpu")
            
        print(f"[INFO] Loaded all {split} cell image embeddings from {cell_image_embeddings_path}")
        return embeddings_dict


class Infos(AbstractDatasetInfos):
    """
    Class for storing information about the MERFISH dataset.

    This class encapsulates various statistics and configurations specific to the
    MERFISH dataset, aiding in dataset handling and model training processes.

    Attributes:
        datamodule: Instance of the data module associated with MERFISH data.
        cfg: Configuration object containing dataset and model parameters.
    """

    def __init__(self, datamodule, cfg):
        self.input_dims = {}
        self.output_dims = {}
        self.name = cfg.dataset.dataset_name
        self.num_cell_class = datamodule.statistics["train"].num_cell_class
        self.num_genes = datamodule.statistics["train"].num_genes
        self.cell_class_decoder = {}
        self.num_cell_to_region_mapping_dict = {}
        self.cell_class_decoder = datamodule.statistics["test"].cell_class_decoder
        self.num_cell_to_region_mapping_dict = datamodule.statistics[
            "test"
        ].num_cell_to_region_mapping_dict
        self.input_dims["node_features_dimensions"] = self.num_genes
        self.input_dims["cell_image_size"] = datamodule.statistics["cell_images_size"]
        self.input_dims["diffusion_time_dimensions"] = 1
        self.output_dims["node_features_dimensions"] = self.num_genes
        self.output_dims["diffusion_time_dimensions"] = 0
