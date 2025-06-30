import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)


import os
import torch
from torch import nn, optim
from GR_MRL import GR_MRL
from Data.VectorBase import VectorBase
from Data.road_data_provider import *
from Model.TSFormer.TSmodel import *
from Data.VectorBase import *
from utils import My_Loss
from tqdm import tqdm
import time


def collate_fn(batch):
    def pad_sequences_2d(sequences, padding_idx=-1):
        max_length = max(len(seq) for seq in sequences)
        padded = torch.full((len(sequences), max_length), padding_idx, dtype=torch.long).cuda()
        for i, seq in enumerate(sequences):
            padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long).cuda()
        return padded
    
    ref = [item['ref'] for item in batch]
    ref = pad_sequences_2d(ref)

    for i in range(len(batch)):
        batch[i]['ref'] = ref[i]

    return batch


def exp_ft(cfg, logger=None):
    
    debug  = cfg['debug']
    
    device = cfg['device']

    temp, dataset_trg = cfg['dataset_src_trg'].split('_')
    
    dataset_src = ''.join(temp.split('-'))

    flag = 'source_train'

    # init model and vectorbase
    model = GR_MRL(cfg, mode=flag).cuda()
    model.update_mode(flag)

    # init exp settings
    bs = cfg['flag'][flag]['batch_size']
    lr = cfg['flag'][flag]['lr']

    criterion = My_Loss()

    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)

    opt = optim.Adam(trained_parameters, lr=lr)

    # time_now = time.time()
    
    # init dataset and dataloader
    source_dataset = PromptDataset(cfg, dataset_src, 'src')
    source_dataloader = DataLoader(source_dataset, 
                                    batch_size = bs, 
                                    shuffle = True, 
                                    drop_last=False,
                                    collate_fn=collate_fn)

    # source train
    for epoch in tqdm(range(cfg['flag'][flag]['epoch'])):
        train_loss = []

        for batch in source_dataloader:
            output = model(batch)
            output = output.float()
            label = [item['label'] for item in batch]
            label = torch.stack(label, dim=0).float().cuda()

            loss = criterion(output, label)
            loss.backward()
            opt.step()

            train_loss.append(loss.item())

            logger.info("{} Train Epoch: {} | Loss: {:.3f}".format(dataset_trg, epoch + 1, loss.item()))
        
        train_loss = np.average(train_loss)

        logger.info("{} Source Train Epoch: {} | Loss: {:.3f}".format(dataset_src, epoch + 1, train_loss))


    # target finetune
    flag = 'target_train'

    model.update_mode(flag)

    bs = cfg['flag'][flag]['batch_size']
    lr = cfg['flag'][flag]['lr']

    target_dataset = PromptDataset(cfg, dataset_src, 'trg')
    target_dataloader = DataLoader(target_dataset, 
                                    batch_size = bs, 
                                    shuffle = True, 
                                    drop_last=False,
                                    collate_fn=collate_fn)


    for epoch in tqdm(range(cfg['flag'][flag]['epoch'])):
        train_loss = []
        # epoch_time = time.time()
        for batch in target_dataloader:
            
            output = model(batch)
            output = output.float()
            label = [item['label'] for item in batch]
            label = torch.stack(label, dim=0).cuda()

            loss = criterion(output, label)
            loss.backward()
            opt.step()

            train_loss.append(loss.item())

        train_loss = np.average(train_loss)

        logger.info("{} Target Train Epoch: {} | Loss: {:.3f}".format(dataset_trg, epoch + 1, train_loss))
            
    # inference  
    flag = 'test'   
    
    model.update_mode(flag)
    model.test()

    test_dataset = PromptDataset(cfg, dataset_src, 'test')
    test_dataloader = DataLoader(test_dataset, 
                                batch_size = bs, 
                                shuffle = False, 
                                drop_last=False,
                                collate_fn=collate_fn)

    pred = np.zeros(len(test_dataset), cfg['pre_num'])
    truth = np.zeros(len(test_dataset), cfg['pre_num'])

    cur = 0
    for batch in test_dataloader:
        output = model(batch)
        output = output.float()
        label = [item['label'] for item in batch]
        label = torch.stack(label, dim=0).float().cuda()

        pred[cur:cur+output.shape[0], :] = output
        truth[cur:cur+output.shape[0], :] = label

        cur += output.shape[0]

    MSE, RMSE, MAE, MAPE = calc_metric(pred, truth)

    logger.info("{} Test: MSE: {:.3f}, RMSE: {:.3f}, MAE: {:.3f}, MAPE: {:.3f}".format(dataset_trg, MSE, RMSE, MAE, MAPE))
        


        

   