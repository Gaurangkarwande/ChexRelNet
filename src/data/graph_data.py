import ast, os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms
from torch_geometric.data import Data

class BipartiteCXR(Data):
    def __init__(self, X, y, bbox, cur_cxr_path, prev_cxr_path, roi, disease):
        super().__init__(x=X, y=y)
        self.bbox = bbox
        self.meta = {
            'cur_cxr_path': cur_cxr_path,
            'prev_cxr_path': prev_cxr_path,
            'roi': roi,
            'disease': disease
            }


class GraphDataset(data.Dataset):
    def __init__(self, regions, comparison_file, metadata_file, bbox_file, adjacency_file, image_data_dir, image_size, labelset, transform=None):
        """
        Args:
            comparison_file (pd.dataframe): dataframe table containing image names, disease severity category label, and other metadata
            image_dir (string): directory containing all of the image files
            transform (callable, optional): optional transform to be applied on a sample
        """
        self.bboxlist = regions
        self.image_paths = pd.read_csv(metadata_file, usecols=['dicom_id','image_full_path'], index_col=['dicom_id'])
        self.comparison_file = comparison_file
        self.bbox_df = pd.read_csv(bbox_file, sep='\t', index_col=['image_id']).convert_dtypes()
        self.bbox_df.dropna(inplace=True)
        self.transform = transform
        self.image_size = image_size
        self.edge_index = self.get_adjacency_coo(adjacency_file)
        self.labelset = labelset
        if self.transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        self.image_data_dir = image_data_dir

    def __len__(self):
        #print('In ComparisionDataset len')
        return len(self.comparison_file)

    def __getitem__(self, idx):
        #print('In ComparisionDataset getitem')
        item = self.comparison_file.iloc[idx]
        #print(item)
        current_image_id = item['current_image_id']
        previous_image_id = item['previous_image_id']

        current_regions = self.bbox_df.loc[current_image_id]
        previous_regions = self.bbox_df.loc[previous_image_id]

        current_image_path = os.path.join(self.image_data_dir, self.image_paths.loc[current_image_id].image_full_path)
        previous_image_path = os.path.join(self.image_data_dir, self.image_paths.loc[previous_image_id].image_full_path)

        current_image = cv2.imread(current_image_path, cv2.IMREAD_GRAYSCALE)
        current_image = current_image.squeeze()
        current_image = Image.fromarray(current_image/np.max(current_image))

        previous_image = cv2.imread(previous_image_path, cv2.IMREAD_GRAYSCALE)
        previous_image = previous_image.squeeze()
        previous_image = Image.fromarray(previous_image/np.max(previous_image))

        previous_cxr, prev_image = self.construct_cxr(previous_image_id, previous_image, previous_regions, self.transform)
        current_cxr, cur_image = self.construct_cxr(current_image_id, current_image, current_regions, self.transform)
        combined_cxr = torch.vstack((previous_cxr, current_cxr))
        
        label = item['comparison']
        y = self.labelset.index(label)
        bbox = torch.tensor(self.bboxlist.index(item['bbox']) + len(self.bboxlist))
        del current_regions; del previous_regions; del current_cxr; del previous_cxr; del previous_image; del current_image
        return BipartiteCXR(combined_cxr, y, bbox, current_image_path, previous_image_path, item['bbox'], item['label_name']), prev_image, cur_image
        #[batch_size x Data]

    def construct_cxr(self, image_id, image, bbox_info, transform=None):
        cxr = []
        for bbox_name in self.bboxlist:
            try:
                bbox = ast.literal_eval(bbox_info[bbox_name])
                cropped = image.crop(bbox)
                if transform is not None:
                    cropped = transform(cropped)
            except ValueError:
                print(image_id, bbox_name)
                raise ValueError
            cxr.append(cropped)         # num_regions x channels x height x width
        if self.transform is not None:
            image = self.transform(image)
        return torch.stack(cxr, dim=0), image #Data(x=torch.stack(cxr, dim=0), edge_index=self.edge_index)
    
    def get_adjacency_coo(self, adjacency_file):
        num_regions = len(self.bboxlist)
        adjacency_matrix = pd.read_csv(adjacency_file, index_col=['bbox'])
        adjacency_matrix = adjacency_matrix.loc[self.bboxlist][self.bboxlist].to_numpy()
        source = []
        destination = []

        for i in range(adjacency_matrix.shape[0]):
            for j in range(adjacency_matrix.shape[1]):
                if adjacency_matrix[i,j]:
                    # previous cxr edges
                    source.append(i)
                    destination.append(j)
                    source.append(j)            #to account for non directed edges
                    destination.append(i)

                    # current cxr edges
                    source.append(i+num_regions)
                    destination.append(j+num_regions)
                    source.append(j+num_regions)            #to account for non directed edges
                    destination.append(i+num_regions)
            #directed edges from previous cxr to current cxr
            source.append(i)
            destination.append(i+num_regions)
        coo_adjacency = torch.stack([torch.as_tensor(source), torch.as_tensor(destination)])
        return coo_adjacency