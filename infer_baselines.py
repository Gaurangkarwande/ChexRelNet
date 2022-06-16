import argparse
import gc
import pandas as pd
import torch
import logging
import json
import numpy as np
from data.siam_data import SiamDataset
from torchvision import transforms
from models.Baselines import SiameseModel

from utils import load_config, split_dataset

gc.collect()
torch.cuda.empty_cache()

use_amp =True

pathology_labels = {
                        0: 'lung opacity',
                        1:  'pleural effusion',
                        2:  'atelectasis',
                        3:  'enlarged cardiac silhouette',
                        4:  'pulmonary edema/hazy opacity',
                        5:  'pneumothorax', 
                        6:  'consolidation',
                        7:  'fluid overload/heart failure', 
                        8:  'pneumonia'
                    }

def setup_data_paths(env_config_file: str):
    global mimic_data_dir, image_path_file, h5_file, comparison_file, bbox_file_224, bbox_file_orig, adjacency_file, training_splits
    with open(env_config_file) as f:
        paths = json.load(f)
    mimic_data_dir = paths['MIMIC_DATA_DIR']
    image_path_file = paths['IMAGE_PATH_FILE']
    h5_file = paths['H5_FILE']
    comparison_file = paths['COMPARISON_FILE']
    bbox_file_orig = paths['BBOX_FILE_ORIG']
    bbox_file_224 = paths['BBOX_FILE_224']
    adjacency_file = paths['ADJACENCY_FILE']
    training_splits = paths['TRAINING_SPLITS']


def infer(dataloader, model, criterion, device):
    loss_epoch = correct = total = 0
    model.eval()
    ground_truth = []
    prediction = []
    disease = []
    roi = [] 
    cur_cxr_path = []
    prev_cxr_path = []
    with torch.no_grad():
        for batch in dataloader:
            prev_image, cur_image, y, meta = batch
            cur_image = cur_image.to(device)
            prev_image = prev_image.to(device)
            y = y.to(device)
            pred = model(prev_image, cur_image)
            _, pred_classes = pred.max(1)
            loss = criterion(pred, y)
            loss_epoch += float(loss.item())
            total += y.size(0)
            correct += (pred_classes == y).sum().float()
            prediction += pred_classes.tolist()
            ground_truth += y.tolist()
            disease += meta['disease']
            roi += meta['roi']
            cur_cxr_path += meta['current_cxr_path']
            prev_cxr_path += meta['prev_cxr_path']

        accuracy = correct/total
        avg_loss = loss_epoch/len(dataloader)
        infer_df = pd.DataFrame(list(zip(roi, disease, ground_truth, prediction, cur_cxr_path, prev_cxr_path)), columns=['ROI', 'disease','ground_truth','prediction', 'cur_cxr', 'prev_cxr'])
        infer_df['result'] = infer_df['ground_truth'] == infer_df['prediction'] 
    return avg_loss, accuracy, infer_df



def main(env_config_file:str, fpath_model: str, model_type:str, model_config_file:str, train_disease_ids:list, infer_disease_ids: list):
    assert model_type == 'graph'
    seed = 345
    np.random.seed(seed)
    torch.manual_seed(seed)
    setup_data_paths(env_config_file)
    config = load_config(model_config_file)

    train_disease_str = ''
    for id in train_disease_ids:
        train_disease_str += str(id)

    logging.basicConfig(filename=f'./inference/cxr_{train_disease_str}_{model_type}_infer_{infer_disease_ids}.log', format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    diseases = []
    for id in train_disease_ids:
        diseases.append(pathology_labels[id])
    
    for_diseases = []
    for id in infer_disease_ids:
        for_diseases.append(pathology_labels[id])

    print(f'\n\n{model_type} model trained on disease/anatomical finding: {diseases}, computing inference on {for_diseases}\n\n')
    logging.info(f'\n\n{model_type} model trained on disease/anatomical finding: {diseases}, computing inference on {for_diseases}\n\n')


    print(config)
    logging.info(config)

    num_classes = config['n_classes']
    input_size = 224
    local = model_type == 'local'
    if num_classes == 3:
        label_list = ['worsened', 'improved', 'no change']
    else:
        label_list = ['worsened', 'improved']
    logging.info('\n')
    logging.info(label_list)
    logging.info('\n')
    train_file, valid_file, test_file, regions = split_dataset(comparison_file, bbox_file_orig, training_splits, label_list, set(for_diseases))
    print(f'\nSelected regions are: {regions}\n')
    logging.info(f'Selected regions are: {regions}\n')
    logging.info(
        f'Train {len(train_file)}, Validation {len(valid_file)}, Test {len(test_file)}')

    data_transforms = transforms.Compose([
        transforms.Resize(size=(input_size, input_size),
                          interpolation=transforms.functional.InterpolationMode.NEAREST),
        transforms.CenterCrop(input_size),
        lambda x: np.expand_dims(x, axis=-1),
        transforms.ToTensor(),
    ])

    
    print('Preparing Datasets')

    test_dataset = SiamDataset(test_file, image_path_file, bbox_file_orig , adjacency_file, mimic_data_dir, input_size, label_list, data_transforms, local=local)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config['test_batch_size'], shuffle=False, num_workers=8)

    print('Create Model')
    model = SiameseModel(config)

    print('Setup criterion')
    criterion = torch.nn.CrossEntropyLoss()

    print('Check CUDA')
    train_on_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if(train_on_gpu):
        print('Training on GPU.')
    else:
        print('No GPU available, training on CPU.')

    print('Load checkpoint')
    checkpoint = torch.load(fpath_model)
    model.to(device)
    model.load_state_dict(checkpoint['model'])
    del checkpoint

    print('Started Testing')
    test_loss, test_acc, infer_df = infer(test_dataloader, model, criterion, device)

    print(f'Test Loss= {test_loss}, Test Acc= {test_acc}')
    logging.info(f'Test Loss= {test_loss}, Test Acc= {test_acc}')
    infer_df.to_csv(f'./inference/cxr_{train_disease_str}_{model_type}_{num_classes}_infer_{infer_disease_ids}.csv')
    print(f'Stored inference results at ./inference/cxr_{train_disease_str}_{model_type}_{num_classes}_{seed}_infer_{test_acc}.csv')

def parse_args():
    parser = argparse.ArgumentParser(description="Baselines Inference Arguments")
    parser.add_argument("--fpath_env_setup", type=str, required=True,
                        help="Filepath to environment config file")
    parser.add_argument("--model_type", type=str, required=True,
                        help="should be 'graph'")
    parser.add_argument("--fpath_model", type=str, required=True,
                        help="Filepath to trained model")
    parser.add_argument("--fpath_model_config", type=str, required=True,
                        help="Filepath to model config file")
    parser.add_argument("--train_disease_ids", type=str, required=True,
                        help="Disease ids on which the model is trained on")
    parser.add_argument("--infer_disease_ids", type=str, required=True,
                        default="012345678", help="Disease ids on which the model is trained on")


if __name__ == '__main__':
    args = parse_args()
    train_disease_ids = [int(id) for id in args.train_disease_ids]
    infer_disease_ids = [int(id) for id in args.infer_disease_ids]
    main(args.fpath_env_setup, args.model_type, args.fpath_model, args.fpath_model_config, train_disease_ids, infer_disease_ids)