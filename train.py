import warnings
warnings.filterwarnings('ignore')

from loss import LabelSmoothingLoss, FocalLoss, F1Loss
from model import CreateModel
from dataAugmentation import dataAugmentation
from dataPreprocessing import dataPreProcessing
from dataset import CustomDataset, StratifiedSampler
from metric import get_acc_score, get_f1_score
from transfrom import get_transform

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

import numpy as np
import pandas as pd
import random
import time
import gc

import os
from box import Box
import wandb

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_criterion(config):
    # loss 설정
    if config.loss == 'cel':
        criterion = nn.CrossEntropyLoss()
    elif config.loss == 'labelsmoothing':
        criterion = LabelSmoothingLoss(classes=config.num_classes, smoothing = config.smoothing, dim=-1)
    elif config.loss == 'focal':
        criterion = FocalLoss(weight = config.weight, gamma=2.0, reduction='mean')
    elif config.loss == 'f1':
        criterion = F1Loss(classes=config.num_classes, epsilon=1e-7)
    else:
        print('not loss')
    return criterion


def get_dataloader(trn_df: pd.DataFrame, val_df: pd.DataFrame, config, transform, model_name: str, mode: bool = True, stratify: bool = True):
    if mode:
        if stratify:
            trn_dataset = CustomDataset(
                df=trn_df,
                config=config,
                transform=transform[model_name],
                mode=True,
            )

            sampler = StratifiedSampler(
                y=np.array(trn_dataset.split_targets),
                batch_size=config.batch_size,
                shuffle=True,
            )

            train_loader = DataLoader(
                trn_dataset,
                num_workers=config.num_workers,
                batch_sampler=sampler,
            )

            val_dataset = CustomDataset(
                df=val_df,
                config=config,
                transform=transform[model_name],
                mode=True,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                shuffle=False,
            )

        else:

            trn_dataset = CustomDataset(
                df=trn_df,
                config=config,
                transform=transform[model_name],
                mode=True,
            )

            train_loader = DataLoader(
                trn_dataset,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                shuffle=True,
            )

            val_dataset = CustomDataset(
                df=val_df,
                config=config,
                transform=transform[model_name],
                mode=True,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                shuffle=False,
            )

        return train_loader, val_loader

    else:
        submission_dataset = CustomDataset(
            df=trn_df,
            config=config,
            transform=transform[model_name],
            mode=False,

        )

        submission_loader = DataLoader(
            submission_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=False,
        )

        return submission_loader


def model_train(model, optimizer, criterion, data_loader, device):
    model.train()

    train_loss = 0
    real_pred_li = []
    label_pred_li = []

    for images, targets in data_loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()

        benign_outputs = model(images)
        loss = criterion(benign_outputs, targets)
        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        predicted = benign_outputs.argmax(dim=-1)

        label_pred_li.extend(predicted.detach().cpu().numpy())
        real_pred_li.extend(targets.cpu().numpy())

    train_loss /= len(data_loader)
    train_acc = get_acc_score(y_true=real_pred_li, y_pred=label_pred_li)
    train_fi_score = get_f1_score(y_true=real_pred_li, y_pred=label_pred_li)

    return train_loss, train_acc, train_fi_score


def model_eval(model, criterion, data_loader, device):
    model.eval()

    val_loss = 0
    real_pred_li = []
    label_pred_li = []

    with torch.no_grad():
        for images, targets in data_loader:
            images, targets = images.to(device), targets.to(device)

            benign_outputs = model(images)
            loss = criterion(benign_outputs, targets)

            val_loss += loss.item()

            predicted = benign_outputs.argmax(dim=-1)

            label_pred_li.extend(predicted.cpu().numpy())
            real_pred_li.extend(targets.cpu().numpy())

    val_loss /= len(data_loader)
    val_acc = get_acc_score(y_true=real_pred_li, y_pred=label_pred_li)
    val_fi_score = get_f1_score(y_true=real_pred_li, y_pred=label_pred_li)

    return val_loss, val_acc, val_fi_score

def train(model, optimizer, criterion, train_loader, val_loader, scheduler, config, fold_num, device):
    besf_f1 = 0
    for epoch in range(1, config.epochs + 1):
        epoch_start_time = time.time()

        train_loss, train_acc, train_fi_score = model_train(
            model = model,
            optimizer = optimizer,
            criterion = criterion,
            data_loader = train_loader,
            device = device,
        )

        val_loss, val_acc, val_fi_score, = model_eval(
            model = model,
            criterion = criterion,
            data_loader = val_loader,
            device = device,
        )

        now_lr = get_lr(optimizer = optimizer)

        epoch_end_time = time.time()

        print(f'{fold_num}fold, epoch: {epoch}, lr: {now_lr}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, train_f1: {train_fi_score:.4f}, \
        val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, val_fi: {val_fi_score:.4f}, 학습시간: {epoch_end_time - epoch_start_time} \n')

        wandb.log({
            'train_loss' : train_loss,
            'train_acc' : train_acc,
            'train_f1' : train_fi_score,
            'val_loss' : val_loss,
            'val_acc' : val_acc,
            'val_fi' : val_fi_score,
        })

        scheduler.step(val_loss)

        if besf_f1 < val_fi_score:
            besf_f1 = val_fi_score
            torch.save(model.state_dict(), os.path.join(config.model_dir, f'{fold_num}fold_{config.model_name}.pt'))
            print(val_fi_score, '모델 저장')

# 데이터 스플릿
def get_val_idx(df : pd.DataFrame, target_col : str):
    skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 22)
    for trn_idx, val_idx in skf.split(df, df[target_col]):
        yield val_idx

def main(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = get_transform(config = config)

    swap_gender_li = [["001498-1", "female"], ["004432", "female"], ["005223", "female"],
                      ['006359', 'male'], ['006360', 'male'], ['006361', 'male'], ['006362', 'male'],
                      ['006363', 'male'], ['006364', 'male'], ]
    swap_mask_li = ['000020', '004418', '005227']

    df = pd.read_csv(os.path.join(config.train_data_dir, config.train_data_name))
    data_pre_processing = dataPreProcessing(df=df, swap_mask_li=swap_mask_li, swap_gender_li=swap_gender_li)

    pre_df = data_pre_processing.make_pre_df()
    train_df = data_pre_processing.make_train_df(df=pre_df, image_dir=config.train_image_dir)
    
    # face_crop_image 데이터 생성
    if not os.path.isdir(config.train_facecrop_image_dir):
        data_augmentation = dataAugmentation(pre_df = pre_df, train_df = train_df)
        data_augmentation.make_face_crop_image(config = config)
            
    facecrop_train_df = data_pre_processing.make_train_df(df=pre_df, image_dir=config.train_facecrop_image_dir)
    
    # 모델 저장 폴더 생성
    if not os.path.isdir(config.model_dir):
        os.mkdir(config.model_dir)

    all_idx_li = pre_df.index.tolist()
    val_idx_li = get_val_idx(df=pre_df, target_col=config.data_split_col)

    total_start_time = time.time()

    for fold_num in range(1, config.oof + 1):
        fold_start_time = time.time()

        # trn, val 데이터 셋
        val_idx = next(val_idx_li)
        trn_idx = list(set(all_idx_li) - set(val_idx.tolist()))

        val_id_df = pre_df.iloc[val_idx, :]
        trn_id_df = pre_df.iloc[trn_idx, :]

        val_df = train_df.set_index('id').loc[list(set(val_id_df['id'].tolist()) & set(train_df['id'].tolist())),
                 :].reset_index(drop=True)
        trn_df = train_df.set_index('id').loc[list(set(trn_id_df['id'].tolist()) & set(train_df['id'].tolist())),
                 :].reset_index(drop=True)

        if config.facecrop_data:
            facecrop_trn_df = facecrop_train_df.set_index('id').loc[
                              list(set(trn_id_df['id'].tolist()) & set(facecrop_train_df['id'].tolist())),
                              :].reset_index(drop=True)
            trn_df = pd.concat([trn_df, facecrop_trn_df]).reset_index(drop=True)

        # 모델 정의
        train_loader, val_loader = get_dataloader(trn_df=trn_df, val_df=val_df, config = config, transform = transform, model_name=config.model_name, mode=True, stratify=False)
        model = CreateModel(config=config, pretrained=True, Multi_Sample_Dropout=True).to(device)
        criterion = get_criterion(config=config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, eps=1e-09, patience=3)

        train(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            scheduler=scheduler,
            fold_num = fold_num,
            config=config,
            device=device,
        )

        fold_end_time = time.time()

        print(f'{fold_num}fold 훈련 시간: {fold_end_time - fold_start_time} \n')

        # 메모리 정리
        gc.collect()
        torch.cuda.empty_cache()

    total_end_time = time.time()
    print(f'총 훈련 시간: {total_end_time - total_start_time}')

def get_pred_li(model, data_loader, device):
    model.eval()
    real_pred_li = []
    label_pred_li = []
    ensemble_pred_li = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            output = model(images)

            label = output.argmax(dim=-1)
            label_pred_li.extend(label.cpu().numpy())

            ensemble_label = output.softmax(1)
            ensemble_pred_li.append(ensemble_label.cpu().numpy())

            real_pred_li.extend(targets.cpu().numpy())

    return label_pred_li, np.concatenate(ensemble_pred_li), real_pred_li

def pred(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = get_transform(config=config)

    swap_gender_li = [["001498-1", "female"], ["004432", "female"], ["005223", "female"],
                      ['006359', 'male'], ['006360', 'male'], ['006361', 'male'], ['006362', 'male'],
                      ['006363', 'male'], ['006364', 'male'], ]
    swap_mask_li = ['000020', '004418', '005227']

    df = pd.read_csv(os.path.join(config.train_data_dir, config.train_data_name))
    submission = pd.read_csv(os.path.join(config.submission_data_dir, config.submission_data_name))

    data_pre_processing = dataPreProcessing(df=df, swap_mask_li=swap_mask_li, swap_gender_li=swap_gender_li)

    pre_df = data_pre_processing.make_pre_df()
    train_df = data_pre_processing.make_train_df(df=pre_df, image_dir=config.train_image_dir)

    all_idx_li = pre_df.index.tolist()
    val_idx_li = get_val_idx(df=pre_df, target_col=config.data_split_col)

    real_labels = []
    pred_labels = []
    idx_li = []

    submission_oof = np.zeros((submission.shape[0], config.num_classes))

    total_start_time = time.time()

    for fold_num in range(1, config.oof + 1):
        fold_start_time = time.time()

        # val
        val_idx = next(val_idx_li)
        trn_idx = list(set(all_idx_li) - set(val_idx.tolist()))

        val_id_df = pre_df.iloc[val_idx, :]
        trn_id_df = pre_df.iloc[trn_idx, :]

        val_df = train_df.set_index('id').loc[list(set(val_id_df['id'].tolist()) & set(train_df['id'].tolist())),
                 :].reset_index(drop=True)
        trn_df = train_df.set_index('id').loc[list(set(trn_id_df['id'].tolist()) & set(train_df['id'].tolist())),
                 :].reset_index(drop=True)

        train_loader, val_loader = get_dataloader(trn_df=trn_df, val_df=val_df, config=config, transform=transform,
                                                  model_name=config.model_name, mode=True, stratify=False)
        submission_loader = get_dataloader(trn_df=submission, val_df=None, config=config, transform=transform,
                                           model_name=config.model_name, mode=False, stratify=False)

        model = CreateModel(config=config, pretrained=True, Multi_Sample_Dropout=True).to(device)
        model.load_state_dict(torch.load(os.path.join(config.model_dir, f'{fold_num}fold_{config.model_name}.pt')))
        val_label_pred_li, val_ensemble_pred_li, real_pred_li = get_pred_li(model=model, data_loader=val_loader,
                                                                            device=device)

        submission_label_pred_li, submission_ensemble_pred_li, _ = get_pred_li(model=model, data_loader=submission_loader, device=device)

        real_labels += real_pred_li
        pred_labels += val_label_pred_li
        idx_li += val_df['idx'].tolist()

        submission_oof += submission_ensemble_pred_li / config.oof

        fold_end_time = time.time()

        _acc = get_acc_score(y_true=real_pred_li, y_pred=val_label_pred_li)
        _f1_score = get_f1_score(y_true=real_pred_li, y_pred=val_label_pred_li)
        print(f'{fold_num}fold 훈련 시간: {fold_end_time - fold_start_time}, acc: {_acc}, f1_score: {_f1_score} \n')

    total_end_time = time.time()
    print(f'총 훈련 시간: {total_end_time - total_start_time}')

    train_f1 = get_f1_score(y_true=real_labels, y_pred=pred_labels)
    train_acc = get_acc_score(y_true=real_labels, y_pred=pred_labels)
    print(f'train fi : {train_f1:.4f}, train acc: {train_acc:.4f} \n')

    submission['ans'] = submission_oof.argmax(1)
    submission.to_csv(config.file_name, index=False)
    print('Done!')


if __name__ == '__main__':

    config = {
        'train_data_name': 'train.csv',
        'train_data_dir': '/opt/ml/input/data/train',
        'train_image_dir': '/opt/ml/input/data/train/images',
        'train_facecrop_image_dir': '/opt/ml/input/data/train/facecrop_images',

        'submission_data_name': 'info.csv',
        'submission_data_dir': '/opt/ml/input/data/eval',
        'submission_image_dir': '/opt/ml/input/data/eval/images',
        'submission_save_dir': '/opt/ml/submission',

        'model_dir': '/opt/ml/model',
        'model_name' : 'final',
        'file_name' : 'final.csv',

        'facecrop_data' : True,

        'num_classes': 18,

        'num_workers': 4,
        'epochs': 20,
        'batch_size': 64,
        'lr': 9e-05,
        'image_size': [384, 384],
        'image_normal_mean': [0.5, 0.5, 0.5],
        'image_normal_std': [0.2, 0.2, 0.2],
        'target_col' : 'labels',
        'split_target_col' : 'labels',
        'timm_model_name': 'regnetx_004',
        'loss': 'cel',

        'seed': 22,
        'data_split_col': 'cv_taget_col',
        'oof': 5,

    }

    config = Box(config)

    run = wandb.init(project="p-stage-level1", entity="seongbeom", name = config.model_name)
    wandb.config.update({
        'num_workers': config.num_workers,
        'epochs': config.epochs,
        'batch_size': config.batch_size,
        'lr': config.lr,
        'image_size': config.image_size,
        'image_normal_mean': config.image_normal_mean,
        'image_normal_std': config.image_normal_std,
        'target_col' : config.target_col,
        'timm_model_name': config.timm_model_name,
        'loss': config.loss,
    })

    main(config = config)
    pred(config = config)