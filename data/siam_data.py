import ast, os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from torch.utils import data
from torchvision import transforms

class SiamDataset(data.Dataset):
    def __init__(self, comparison_file, metadata_file, image_size, labelset, transform=None, local=False):
        """
        Args:
            comparison_file (pd.dataframe): dataframe table containing image names, disease severity category label, and other metadata
            image_dir (string): directory containing all of the image files
            transform (callable, optional): optional transform to be applied on a sample
        """
        self.image_paths = pd.read_csv(metadata_file, usecols=['dicom_id','image_full_path'], index_col=['dicom_id'])
        self.comparison_file = comparison_file
        self.transform = transform
        self.image_size = image_size
        self.labelset = labelset
        if self.transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        self.local = local

    def __len__(self):
        #print('In ComparisionDataset len')
        return len(self.comparison_file)

    def __getitem__(self, idx):
        #print('In ComparisionDataset getitem')
        image_data_dir = '/home/gaurang/CXR/data/mimic/physionet.org/files/mimic-cxr-jpg/2.0.0/files'
        item = self.comparison_file.iloc[idx]
        #print(item)
        current_image_id = item['current_image_id']
        previous_image_id = item['previous_image_id']

        current_image_path = os.path.join(image_data_dir, self.image_paths.loc[current_image_id].image_full_path)
        previous_image_path = os.path.join(image_data_dir, self.image_paths.loc[previous_image_id].image_full_path)

        current_image = cv2.imread(current_image_path, cv2.IMREAD_GRAYSCALE)
        current_image = current_image.squeeze()
        current_image = Image.fromarray(current_image/np.max(current_image))

        previous_image = cv2.imread(previous_image_path, cv2.IMREAD_GRAYSCALE)
        previous_image = previous_image.squeeze()
        previous_image = Image.fromarray(previous_image/np.max(previous_image))

        if self.local:
            bbox_current = ast.literal_eval(item['bbox_coord_orig_subject'])
            bbox_previous = ast.literal_eval(item['bbox_coord_orig_object'])
            current_image = current_image.crop(bbox_current)
            previous_image = previous_image.crop(bbox_previous)
        
        label = item['comparison']
        y = self.labelset.index(label)
        meta = {
                'cur_cxr_id': current_image_id,
                'prev_cxr_id': previous_image_id,
                'cur_cxr_path': current_image_path,
                'prev_cxr_path': previous_image_path,
                'roi': item['bbox'],
                'disease': item['label_name']
                }
        return current_image, previous_image, y, meta
        #[batch_size x Data]