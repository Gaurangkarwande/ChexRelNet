import torch
import json
import os
import pandas as pd
from collections import Counter

class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the 
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """
    def __init__(
        self, optimizer, patience=4, min_lr=1e-6, factor=0.3
    ):
        """
        new_lr = old_lr * factor

        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode='min',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
                verbose=True
            )

    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)
    
    def get_final_lr(self):
        return self.lr_scheduler.state_dict()


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=11, min_delta=1e-5):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
    
def save_checkpoint(state, directory, file_name):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    checkpoint_file = os.path.join(directory, file_name + '.pth')
    torch.save(state, checkpoint_file)

def load_config(config_file: str):
    with open(config_file) as f:
        config = json.load(f)
    return config

def balance_dataset(df):
    # df_no_change = df[df['comparison'] == 'no change']
    # num_no_change = len(df_no_change)
    df_improved = df[df['comparison'] == 'improved']
    num_improved = len(df_improved)
    df_worsened = df[df['comparison'] == 'worsened']
    num_worsened = len(df_worsened)
    num_samples = min(num_improved, num_worsened)
    new_df = pd.concat([ df_improved.sample(num_samples), df_worsened.sample(num_samples)], axis=0)
    return new_df

def get_regions(df, diseases):
    n = 7
    a = Counter(df[df.label_name.isin(diseases)].bbox)
    a = a.most_common()
    return [a[i][0] for i in range(min(n, len(a)))]

def split_dataset(file_path,bbox_file_path, splits, label_list=None, diseases=None):
    # regions = None
    bbox_df = pd.read_csv(bbox_file_path, sep='\t').convert_dtypes()
    comp_df = pd.read_csv(file_path, sep='\t').convert_dtypes()
    if diseases:
        regions = get_regions(comp_df, diseases)
        comp_df = comp_df[comp_df['bbox'].isin(regions) & comp_df['label_name'].isin(diseases)]
    bbox_df.dropna(inplace=True)
    comp_df.drop_duplicates(subset='current_image_id', keep="last", inplace=True)
    print(f'\nSelected regions are: {regions}\n')
    bbox_pid = set(bbox_df['image_id'])
    comp_pid = set(comp_df['current_image_id']).union(set(comp_df['previous_image_id']))
    bbox_pid = bbox_pid.intersection(comp_pid)
    comp_df = comp_df[comp_df['current_image_id'].isin(bbox_pid) & comp_df['previous_image_id'].isin(bbox_pid)]
    # Keep specific labels
    if label_list: comp_df = comp_df[comp_df['comparison'].isin(label_list)]

    train_split = pd.read_csv(splits[0])
    valid_split = pd.read_csv(splits[1])
    test_split = pd.read_csv(splits[2])

    pid = set(list(train_split['dicom_id'].unique()))
    train = balance_dataset(comp_df[comp_df['current_image_id'].isin(pid)])

    pid = set(list(valid_split['dicom_id'].unique()))
    dev = balance_dataset(comp_df[comp_df['current_image_id'].isin(pid)])

    pid = set(list(test_split['dicom_id'].unique()))
    test = balance_dataset(comp_df[comp_df['current_image_id'].isin(pid)])
    
    print(Counter(train['comparison']))
    print(Counter(dev['comparison']))
    print(Counter(test['comparison']))
    return train, dev, test, regions