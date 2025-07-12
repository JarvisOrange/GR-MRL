import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)


import os
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR  
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
    wd = cfg['flag'][flag]['weight_decay']

    criterion = My_Loss()

    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)

    opt = optim.AdamW(trained_parameters, lr=lr, weight_decay=wd)
    
    # Add learning rate scheduler for source training
    scheduler = CosineAnnealingLR(opt, T_max=cfg['flag'][flag]['epoch'], eta_min=lr * 0.1)

    # time_now = time.time()
    
    # init dataset and dataloader
    source_dataset = PromptDataset(cfg, dataset_src, 'src')
    source_dataloader = DataLoader(source_dataset, 
                                    batch_size = bs, 
                                    shuffle = True, 
                                    drop_last=False,
                                    collate_fn=collate_fn)

    # source train
    EPOCH = cfg['flag'][flag]['epoch']
    if debug: EPOCH = 1
    for epoch in tqdm(range(EPOCH)):
        train_loss = []
        for i, batch in enumerate(source_dataloader):
            if debug and i == 10: break
            output = model(batch)
            output = output.float()
            label = [item['label'] for item in batch]
            label = torch.stack(label, dim=0).float().cuda()

            loss = criterion(output, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            train_loss.append(loss.item())

            if i%100 == 0:
                torch.cuda.empty_cache()
                logger.info("Source Train Epoch: {} {}/{} | Loss: {:.3f}".format(epoch, i, len(source_dataloader), np.average(train_loss)))

        
        train_loss = np.average(train_loss)
        
        # Step the scheduler at the end of each epoch
        scheduler.step()

        logger.info("{} Source Train Epoch: {} | Loss: {:.3f} | LR: {:.6f}".format(dataset_src, epoch, train_loss, scheduler.get_last_lr()[0]))


    # target finetune
    flag = 'target_train'

    model.update_mode(flag)
    model.eval()

    bs = cfg['flag'][flag]['batch_size']
    if debug: bs=1
    lr = cfg['flag'][flag]['lr']
    wd = cfg['flag'][flag]['weight_decay']

    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)

    opt = optim.AdamW(trained_parameters, lr=lr, weight_decay=wd)

    target_dataset = PromptDataset(cfg, dataset_src, 'trg')
    target_dataloader = DataLoader(target_dataset, 
                                    batch_size = bs, 
                                    shuffle = True, 
                                    drop_last=False,
                                    collate_fn=collate_fn)

    EPOCH = cfg['flag'][flag]['epoch']
    if debug: EPOCH = 1
    for epoch in tqdm(range(EPOCH)):
        train_loss = []
        # epoch_time = time.time()
        for i, batch in enumerate(target_dataloader):
            if debug and i == 10: break

            output = model(batch)
            output = output.float()
            label = [item['label'] for item in batch]
            label = torch.stack(label, dim=0).cuda()

            loss = criterion(output, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            train_loss.append(loss.item())

            if i%100 == 0:
                torch.cuda.empty_cache()
                logger.info("Target Train Epoch: {} {}/{} | Loss: {:.3f}".format(epoch, i, len(target_dataloader), np.average(train_loss)))

        train_loss = np.average(train_loss)

        logger.info("{} Target Train Epoch: {} | Loss: {:.3f}".format(dataset_trg, epoch, train_loss))
            
    # inference  
    flag = 'test'   
    
    model.update_mode(flag)
    model.eval()
    torch.cuda.empty_cache()

    bs = cfg['flag'][flag]['batch_size']

    test_dataset = PromptDataset(cfg, dataset_src, 'test')
    test_dataloader = DataLoader(test_dataset, 
                                batch_size = bs, 
                                shuffle = False, 
                                drop_last=False,
                                collate_fn=collate_fn)

    pred = np.zeros((len(test_dataset), cfg['pre_num']))
    truth = np.zeros((len(test_dataset), cfg['pre_num']))

    cur = 0
    for i, batch in enumerate(test_dataloader):
        if debug and i==3: break
        with torch.no_grad():
            output = model(batch)
            output = output.float()

            bs = output.shape[0]

            label = [item['label'] for item in batch]
            label = torch.stack(label, dim=0)

            pred[cur:cur+bs, :] = output.detach().cpu().numpy()
            truth[cur:cur+bs, :] = label.cpu().numpy()

        cur += output.shape[0]

    MSE, RMSE, MAE, _ = calc_metric(pred, truth)

    logger.info("{} Test: MSE: {:.3f}, RMSE: {:.3f}, MAE: {:.3f}".format(dataset_trg, MSE, RMSE, MAE))
        


        

   