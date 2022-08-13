import time
import argparse
import gc
import pandas as pd
import torch
import logging
import json
import numpy as np
from matplotlib import pyplot as plt
from data.siam_data import SiamDataset
from torchvision import transforms
from models.Baselines import SiameseModel
from torch.cuda.amp import GradScaler, autocast

from utils import EarlyStopping, LRScheduler, save_checkpoint, load_config, split_dataset

gc.collect()
torch.cuda.empty_cache()

use_amp =True

scaler = GradScaler(enabled=use_amp)
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


def train(training_dataloader, model, criterion, optimizer, device, batches_to_accumulate):
    start = time.time()
    loss_epoch = correct = total = 0
    model.train()
    N = len(training_dataloader)
    for batch_idx, batch in enumerate(training_dataloader):
        prev_image, cur_image, y, meta = batch
        cur_image = cur_image.to(device)
        prev_image = prev_image.to(device)
        y = y.to(device)
        with autocast(enabled=use_amp):
            pred = model(prev_image, cur_image)
            loss = criterion(pred, y)
            loss = loss/batches_to_accumulate
        scaler.scale(loss).backward()
        
        if ((batch_idx+1) % batches_to_accumulate == 0) or (batch_idx+1 == len(training_dataloader)):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        _, pred_classes = pred.max(1)
        total += y.size(0)
        correct += float(pred_classes.eq(y).sum().item())
        loss_epoch += float(loss.item())
    time_for_epoch = time.time() - start
    accuracy = correct/total
    avg_loss = loss_epoch/len(training_dataloader)
    return avg_loss, accuracy, time_for_epoch


def test(dataloader, model, criterion, device):
    loss_epoch = correct = total = 0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            prev_image, cur_image, y, meta = batch
            cur_image = cur_image.to(device)
            prev_image = prev_image.to(device)
            y = y.to(device)

            pred = model(prev_image, cur_image)

            _, pred_classes = pred.max(1)
            loss = criterion(pred, y)
            total += y.size(0)
            correct += float(pred_classes.eq(y).sum().item())
            loss_epoch += float(loss.item())
        accuracy = correct/total
        avg_loss = loss_epoch/len(dataloader)
    return avg_loss, accuracy

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

def main(env_config_file:str, model_type:str, model_config_file:str, disease_ids:list):
    start = time.time()
    seed = 345
    np.random.seed(seed)
    torch.manual_seed(seed)
    setup_data_paths(env_config_file)
    config = load_config(model_config_file)

    disease_str = ''
    for id in disease_ids:
        disease_str += str(id)
    logging.basicConfig(filename=f'./logs/{model_type}/cxr_{disease_str}_{model_type}_{seed}.log', format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    logging.info('\n************************************\n')

    diseases = []
    for id in disease_ids:
        diseases.append(pathology_labels[id])

    print(f'\n\nFor disease/anatomical finding: {diseases} and model_type: {model_type} and seed {seed}\n\n')
    logging.info(f'\n\nFor disease/anatomical finding: {diseases} and model_type: {model_type} and seed {seed}\n\n')

    if config['train_batch_size'] < 32:
        batches_to_accumulate = 4
    else:
        batches_to_accumulate = 1

    print(config)
    logging.info(config)

    ngpu = 1
    num_epochs = config['num_epochs']
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
    train_file, valid_file, test_file, regions = split_dataset(comparison_file, bbox_file_orig, training_splits, label_list, set(diseases))
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

    training_dataset = SiamDataset(train_file, image_path_file, bbox_file_orig, adjacency_file, mimic_data_dir, input_size, label_list, data_transforms, local=local)
    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=config['train_batch_size'], shuffle=True, num_workers=8)

    validation_dataset = SiamDataset(valid_file, image_path_file, bbox_file_orig, adjacency_file, mimic_data_dir, input_size, label_list, data_transforms, local=local)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=config['test_batch_size'], shuffle=True, num_workers=8)

    test_dataset = SiamDataset(test_file, image_path_file, bbox_file_orig , adjacency_file, mimic_data_dir, input_size, label_list, data_transforms, local=local)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config['test_batch_size'], shuffle=False, num_workers=8)

    print('Create Model')
    model = SiameseModel(config)
    

    print('Setup criterion and optimizer')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    lr_scheduler = LRScheduler(optimizer)
    early_stopping = EarlyStopping()

    print('Check CUDA')
    train_on_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if(train_on_gpu):
        print('Training on GPU.')
    else:
        print('No GPU available, training on CPU.')

    print('***** Training *****')
    logging.info('Started Training')

    best_valid_acc = 0
    model.to(device)
    train_history_loss = []
    train_history_acc = []
    val_history_loss = []
    val_history_acc = []
    for epoch in range(num_epochs):
        train_loss, train_acc, train_time = train(
            training_dataloader, model, criterion, optimizer, device, batches_to_accumulate)
        valid_loss, valid_acc = test(
            validation_dataloader, model, criterion, device)

        print(f'Epoch {epoch}: Train Loss= {train_loss:.3f}, Train Acc= {train_acc:.3f} \t Valid Loss= {valid_loss:.3f}, Valid Acc= {valid_acc:.3f} \t Time Taken={train_time:.2f} s')
        logging.info(
            f'Epoch {epoch}: Train Loss= {train_loss:.3f}, Train Acc= {train_acc:.3f} \t Valid Loss= {valid_loss:.3f}, Valid Acc= {valid_acc:.3f} \t Time Taken={train_time:.2f} s')
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            checkpoint = {
                        'epoch': epoch, 
                        'model': model.state_dict(), 
                        'criterion': criterion.state_dict(), 
                        'optimizer': optimizer.state_dict(),
                        'scaler': scaler.state_dict(),
                        'best_acc': best_valid_acc
                        }
            save_checkpoint(checkpoint, directory='./model_checkpoint', file_name=f'best_checkpoint_{disease_str}_{model_type}_{seed}')
            logging.info(f'Checkpoint saved at Epoch {epoch}')

        lr_scheduler(valid_loss)
        early_stopping(valid_loss)

        #save losses for learning curves
        
        train_history_loss.append(train_loss)
        val_history_loss.append(valid_loss)
        train_history_acc.append(train_acc)
        val_history_acc.append(valid_acc)
        if early_stopping.early_stop:
            break
    del model; del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    logging.info(f'Final scheduler state {lr_scheduler.get_final_lr()}\n')
    print('***** Testing ********')  #
    # save curves
    plt.plot(range(len(train_history_loss)),train_history_loss, label="Training")
    plt.plot(range(len(val_history_loss)),val_history_loss, label="Validation")
    plt.legend()
    plt.title(f"Loss Curves:{diseases}")
    plt.savefig(f'curves/{model_type}/loss_curves_classes_{num_classes}_D_{disease_str}_M_{model_type}_{seed}.jpg', bbox_inches='tight', dpi=150)
    plt.close()

    plt.plot(range(len(train_history_acc)),train_history_acc, label="Training")
    plt.plot(range(len(val_history_acc)),val_history_acc, label="Validation")
    plt.legend()
    plt.title(f"Accuracy Curves: {diseases}")
    plt.savefig(f'curves/{model_type}/acc_curves_classes_{num_classes}_D_{disease_str}_M_{model_type}_{seed}.jpg', bbox_inches='tight', dpi=150)
    plt.close()

    PATH = f"./model_checkpoint/{model_type}/best_checkpoint_{disease_str}_{model_type}_{seed}.pth"
    checkpoint = torch.load(PATH)
    model = SiameseModel(config)
    model.to(device)
    model.load_state_dict(checkpoint['model'])
    del checkpoint
    test_loss, test_acc, infer_df = infer(test_dataloader, model, criterion, device)
    print(f'Test Loss= {test_loss}, Test Acc= {test_acc}')
    logging.info(f'Test Loss= {test_loss}, Test Acc= {test_acc}')
    diff = time.time() - start
    logging.info(f'Total time taken= {str(diff)} s')
    print(f'Total time taken= {str(diff)} s')
    infer_df.to_csv(f'./inference/{model_type}/cxr_{disease_str}_{model_type}_{num_classes}_{seed}_infer_{test_acc}.csv')
    print(f'Stored inference results at ./inference/{model_type}/cxr_{disease_str}_{model_type}_{num_classes}_{seed}_infer_{test_acc}.csv')



def parse_args():
    parser = argparse.ArgumentParser(description="Baselines Training Arguments")
    parser.add_argument("--fpath_env_setup", type=str, required=True,
                        help="Filepath to environment config file")
    parser.add_argument("--model_type", type=str, required=True,
                        help="should be one of 'local' or 'global'")
    parser.add_argument("--fpath_model_config", type=str, required=True,
                        help="Filepath to model config file")
    parser.add_argument("--disease_ids", type=str, required=True,
                        help="Disease ids for which to train the model")
    

if __name__ == '__main__':
    args = parse_args()
    disease_ids = [int(id) for id in args.disease_ids]
    main(args.fpath_env_setup, args.model_type, args.fpath_model_config, disease_ids)
