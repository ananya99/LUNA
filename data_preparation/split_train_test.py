import argparse
from io import BytesIO
import shutil
import tarfile
import numpy as np
import torch
import pandas as pd
import os

def create_split_mask_based_on_donors(data_path, test_donors):
    data = pd.read_csv(data_path)
    train_mask = ~data['donor'].isin(test_donors)
    test_mask = data['donor'].isin(test_donors)
    del data
    return train_mask, test_mask

def create_split_mask_based_on_cell_sections(data_path, test_cell_section_suffixes):
    data = pd.read_csv(data_path)
    train_mask = ~data['cell_section'].str.contains('|'.join(test_cell_section_suffixes))
    test_mask = data['cell_section'].str.contains('|'.join(test_cell_section_suffixes))
    del data
    return train_mask, test_mask

def prepare_data(data_path, train_mask, test_mask, embedding_file_path = None, cell_images_path = None, slice_images_path = None, output_path = None):
    """
    Prepare the data for training and testing

    Args:
        data_path (str): Path to the data csv file
        embedding_file_path (str, optional): Path to the cell embeddings file. Defaults to None.
        cell_images_path (str, optional): Path to the cell images tar file. Defaults to None.
        slice_images_path (str, optional): Path to the slice images tar file. Defaults to None.
        output_path (str, optional): Path to the output directory. Defaults to None.
        train_mask (pd.Series, optional): Mask for the training data. Defaults to None.
        test_mask (pd.Series, optional): Mask for the testing data. Defaults to None.
    """
    data = pd.read_csv(data_path)
    
    donors = {
        "TgCRND8_2_5",
        "TgCRND8_5_7",
        "TgCRND8_17_9",
        "wildtype_2_5",
        "wildtype_5_7",
        "wildtype_13_4"}
    
    train_data = data[train_mask]
    test_data = data[test_mask]
    
    os.makedirs(output_path, exist_ok=True)
    
    # save train and test data to csv
    train_data.rename(columns={'cell_id': 'original_cell_id'}, inplace=True)
    train_data.loc[:, 'cell_id'] = train_data.index
    train_data.to_csv(os.path.join(output_path, "train_data.csv"), index=False)
    print(f"Saved {len(train_data)} training data to {output_path}")
    
    test_data.rename(columns={'cell_id': 'original_cell_id'}, inplace=True)
    test_data.loc[:, 'cell_id'] = test_data.index
    test_data.to_csv(os.path.join(output_path, "test_data.csv"), index=False)
    print(f"Saved {len(test_data)} testing data to {output_path}")
    
    cell_to_donor = dict(zip(data['cell_id'], data['donor']))
    
    if embedding_file_path is not None:
        embeddings = torch.load(embedding_file_path)

        train_embeddings = {cid: embeddings[cid] for cid in train_data['original_cell_id'] if cid in embeddings}
        test_embeddings = {cid: embeddings[cid] for cid in test_data['original_cell_id'] if cid in embeddings}
        
        torch.save(train_embeddings, os.path.join(output_path, "train_embeddings.pt"))
        print(f"Saved {len(train_embeddings)} train embeddings to {output_path}")
        torch.save(test_embeddings, os.path.join(output_path, "test_embeddings.pt"))
        print(f"Saved {len(test_embeddings)} test embeddings to {output_path}")
        
    if cell_images_path is not None:        
        train_tar_path = os.path.join(output_path, "train_cell_images.tar")
        test_tar_path = os.path.join(output_path, "test_cell_images.tar")
        
        with tarfile.open(cell_images_path, "r") as merged_tar, \
            tarfile.open(train_tar_path, "w") as train_tar, \
            tarfile.open(test_tar_path, "w") as test_tar:

            n_train, n_test = 0, 0
            skipped_cells = []
            for member in merged_tar.getmembers():
                if member.isfile() and member.name.endswith(".npy"):
                    cell_id = os.path.splitext(os.path.basename(member.name))[0]

                    file_obj = merged_tar.extractfile(member)
                    if file_obj is None:
                        continue  # skip if file couldn't be read

                    if cell_id in train_data['original_cell_id'].values:
                        train_tar.addfile(member, file_obj)
                        n_train += 1
                    elif cell_id in test_data['original_cell_id'].values:
                        test_tar.addfile(member, file_obj)
                        n_test += 1
                    else:
                        # print(f"Skipping cell {cell_id} from {member.name} not found in train or test data")
                        skipped_cells.append(cell_id)
            print(f"Skipped {len(skipped_cells)} cells not found in train or test data: {skipped_cells[:10]}")
        print(f"Saved {n_train} training cell images to {train_tar_path}")
        print(f"Saved {n_test} testing cell images to {test_tar_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='Absolute path to the sliced data directory')
    parser.add_argument('--output_dir', type=str, help='Absolute path to the output directory')
    args = parser.parse_args()

    data_dir = '/mlbio_scratch/anagupta/xenium_preprocessed/' if args.data_dir is None else args.data_dir
    output_dir = '/mlbio_scratch/anagupta/luna/data/train_test_split_3N2D_1D_2' if args.output_dir is None else args.output_dir
    
    # data_dir = '/mlbio_scratch/anagupta/xenium_preprocessed/sliced_data' if args.data_dir is None else args.data_dir
    # output_dir = '/mlbio_scratch/anagupta/luna/sliced_data/train_test_split_3L_1R' if args.output_dir is None else args.output_dir

    data_path = os.path.join(data_dir, '10xgenomics_alzheimers_disease_mouse_data_shuffled.csv')
    # data_path = os.path.join(data_dir, 'sliced_data.csv')
    embedding_file_path = os.path.join(data_dir, 'cell_embeddings.pt')
    cell_images_path = os.path.join(data_dir, 'cell_images.tar')
    
    # Strategy 1: Split based on donors(One TgCRND8 donor is in the test set, rest are in the train set)
    test_donors = {"TgCRND8_5_7"}
    train_mask, test_mask = create_split_mask_based_on_donors(data_path, test_donors)
    
    # Strategy 2: Split based on donors(all the Normal donors are in the train set, and all the TgCRND8(Diseased) donors are in the test set)
    # test_donors = {"TgCRND8_5_7", "TgCRND8_2_5", "TgCRND8_17_9"}
    # train_mask, test_mask = create_split_mask_based_on_donors(data_path, test_donors)
    
    # Strategy 3: Split based on cell sections(all the rightmost slices are in the test set, so that test shapes remain unseen)
    # [__1, __2, __3, __4, 
    # __5, __6, __7, __8, 
    # __9, __10, __11, __12, 
    # __13, __14, __15, __16]
    # test_cell_section_suffixes = {"__4", "__8", "__12", "__16"}
    # train_mask, test_mask = create_split_mask_based_on_cell_sections(data_path, test_cell_section_suffixes)

    # TODO: Add an argument to specify how to split the data
    prepare_data(data_path, train_mask, test_mask, embedding_file_path=embedding_file_path, cell_images_path=cell_images_path, output_path=output_dir)

if __name__ == "__main__":
    main()