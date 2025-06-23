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
import random



class Dataset(InMemoryDataset):
    def __init__(
        self,
        split: int,
        input_data: pd.DataFrame,
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
        clean_positions, clean_node_features, clean_cell_class = self._clean_data(
            positions, node_features, cell_class, nan_rows
        )

        # Update data attributes
        self._update_data_attributes(
            clean_positions,
            clean_node_features,
            clean_cell_class,
            gene_names,
            cell_class_decoder,
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

    def _clean_data(self, positions, node_features, cell_class, nan_rows):
        clean_positions = positions[~nan_rows]
        clean_node_features = node_features[~nan_rows]
        clean_cell_class = cell_class[~nan_rows]
        return clean_positions, clean_node_features, clean_cell_class

    def _update_data_attributes(
        self,
        clean_positions,
        clean_node_features,
        clean_cell_class,
        gene_names,
        cell_class_decoder,
    ):
        cell_ID = torch.tensor(self.input_data.index)

        self._data.positions = clean_positions
        self._data.node_features = clean_node_features
        self._data.cell_class = clean_cell_class
        self._data.cell_ID = cell_ID

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

        return np.unique(slice_)


class DataModule(AbstractDataModule):
    def __init__(self, cfg):
        train_data = self.data_loading(cfg, 'train')
        test_data = self.data_loading(cfg, 'test')
        self.train_dataset = self._initialize_dataset("train", train_data, cfg) # 这个里面会有process data的步骤，对数据点的坐标进行归一化。
        self.test_dataset = self._initialize_dataset("test", test_data, cfg)

        if cfg.dataset.validation_data_path:
            validation_data = self.data_loading(cfg, 'validation')
            self.validation_dataset = self._initialize_dataset("validation", validation_data, cfg)
        else:
            self.validation_dataset = None

        self.statistics = {
            "train": self.train_dataset.statistics,
            "validation": self.validation_dataset.statistics if self.validation_dataset else None,
            "test": self.test_dataset.statistics,
        }

        ########################## TODO
        self.sub_window_schedule = {0.5:0.1, 0.6: 0.1, 0.7: 0.1, 0.8: 0.1, 0.9: 0.1, 1.0: 0.5}
        self.Ks = list(self.sub_window_schedule.keys())
        self.Ks_probs = list(self.sub_window_schedule.values())
        #########################

        super().__init__(
            cfg,
            train_dataset=self.train_dataset,
            val_dataset=self.validation_dataset if self.validation_dataset else None,
            test_dataset=self.test_dataset,
        )

    def _subwindow_sampling(self, one_data_sample, K):
        if K <= 0 or K > 1:
            raise ValueError("The range of K must be between 0 and 1.")

        points = one_data_sample.positions
        node_features = one_data_sample.node_features
        cell_class = one_data_sample.cell_class
        cell_ID = one_data_sample.cell_ID


        x_left = torch.rand(1).item() * (1 - K) - 0.5
        y_left = torch.rand(1).item() * (1 - K) - 0.5

        x_interval = [x_left, x_left + K]
        y_interval = [y_left, y_left + K]

        mask_x = (points[:, 0] >= x_left) & (points[:, 0] <= x_left + K)
        mask_y = (points[:, 1] >= y_left) & (points[:, 1] <= y_left + K)
        mask = mask_x & mask_y

        sampled_points = points[mask]
        sampled_node_features = node_features[mask]
        cell_class = cell_class[mask]
        cell_ID = cell_ID[mask]

        if sampled_points.shape[0] == 0:
            raise ValueError('The number of sampled points is 0.')

        normalized_points = (sampled_points - torch.tensor([x_left, y_left])) / K - 0.5

        return sampled_node_features, normalized_points, cell_class, cell_ID


    def _sub_window_data_augmentation(self,batched_data):
        K_selected = random.choices(self.Ks, weights=self.Ks_probs, k=len(batched_data))
        for i in range(len(batched_data)):
            if K_selected[i] == 1.0:
                continue
            batched_data[i].node_features, batched_data[i].positions,batched_data[i].cell_class, batched_data[i].cell_ID = self._subwindow_sampling(batched_data[i], K_selected[i])
        return batched_data



    def _initialize_dataset(self, split, data, cfg):
        return Dataset(split=split, input_data=data, cfg=cfg)

    def collate(self, batch):
        return self._create_batch(batch)

    def _create_batch(self, batch):
        batch_data = Batch()
        ##################TODO
        if self.split == 'train':
            batch = self._sub_window_data_augmentation(batch)

        #############################
        batch_data.node_features = torch.cat([data.node_features for data in batch], dim=0)
        batch_data.positions = torch.cat([data.positions for data in batch], dim=0)
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
        self.split = split
        if split == 'train':
            data_path = cfg.dataset.train_data_path
        elif split == 'validation':
            data_path = cfg.dataset.validation_data_path
        else:
            data_path = cfg.dataset.test_data_path
        data = pd.read_csv(f"{data_path}", index_col=0)
        
        # Ensure that the data contains the necessary columns, if not call standardise_dataframe_colnames:
        data = standardise_dataframe_colnames(data)
        assert all(column in data.columns for column in ['coord_X', 'coord_Y', 'cell_section', 'cell_class'])
        
        return data


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
        self.input_dims["diffusion_time_dimensions"] = 1
        self.output_dims["node_features_dimensions"] = self.num_genes
        self.output_dims["diffusion_time_dimensions"] = 0      