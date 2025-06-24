import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)


import os
import torch
from torch import nn, optim
from GR_MRL import GR_MRL
from config import cfg
from Data.VectorBase import Vectorbase
from Data.road_data_provider import *
from Model.TSFormer.TSmodel import *
from Data.VectorBase import *
from config import cfg
from utils import Feq_Loss
from tqdm import tqdm
import time

def exp_main(logger=None):
    device = cfg['device']

    temp, dataset_trg = cfg['dataset_src_trg'].split('_')
    dataset_src = ''.join(temp.split('-'))

    flag = 'source_train'

    # init model and vectorbase
    model = GR_MRL(flag)

    # init exp settings
    bs = cfg['flag'][flag]['batch_size']
    lr = cfg[flag]['lr']

    criterion = My_Loss()

    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)

    opt = optim.Adam(trained_parameters, lr=lr)

    # time_now = time.time()
    
    # init dataset and dataloader
    source_dataset = PromptDataset(dataset_src, 'src')
    source_dataloader = DataLoader(source_dataset, batch_size = bs, shuffle = True, drop_last=False)

    # source train
    for epoch in tqdm(cfg[flag]['epoch']):
        train_loss = []
        # epoch_time = time.time()

        for batch in source_dataloader:
            output = model(batch)
            label = batch['truth']

            loss = criterion(output, label)
            loss.backward()
            opt.step()

            train_loss.append(loss.item())
        
        train_loss = np.average(train_loss)

        logger.info("{} Source Train Epoch: {0} | Loss: {1:.3f}".format(dataset_src, epoch + 1, train_loss))




    # target finetune
    flag = 'target train'

    model.update_mode(flag)

    bs = cfg['flag'][flag]['batch_size']
    lr = cfg[flag]['lr']

    target_dataset = PromptDataset(dataset_src, 'trg')
    target_dataloader = DataLoader(target_dataset, batch_size = bs, shuffle = True, drop_last=False)

    for epoch in tqdm(cfg[flag]['epoch']):
        train_loss = []
        # epoch_time = time.time()
        for batch in target_dataloader:
            
            output = model(batch)
            label = batch['truth']

            loss = criterion(output, label)
            loss.backward()
            opt.step()

            train_loss.append(loss.item())

        train_loss = np.average(train_loss)

        logger.info("{} Target Train Epoch: {0} | Loss: {1:.3f}".format(dataset_trg, epoch + 1, train_loss))
            
    # inference  
    flag = 'test'   
    
    model.update_mode(flag)
    model.test()

    test_dataset = PromptDataset(dataset_src, 'test')
    test_dataloader = DataLoader(test_dataset, batch_size = bs, shuffle = False, drop_last=False)

    pred = np.zeros(len(test_dataset), cfg['pre_num'])
    truth = np.zeros(len(test_dataset), cfg['pre_num'])

    cur = 0
    for batch in test_dataloader:
        output = model(batch)
        label = batch['truth']

        pred[cur:cur+output.shape[0], :] = output
        truth[cur:cur+output.shape[0], :] = label

        cur += output.shape[0]

    MSE, RMSE, MAE, MAPE = calc_metric(pred, truth)

    logger.info("{} Test: MSE: {1:.3f}, RMSE: {1:.3f}, MAE: {1:.3f}, MAPE: {1:.3f}".format(dataset_trg, MSE, RMSE, MAE, MAPE))
        


        

   