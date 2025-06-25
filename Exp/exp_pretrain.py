import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import *
import time
from pathlib import Path
from tqdm import tqdm
import sys
from Model.TSFormer.TSmodel import TSFormer  
sys.path.append('./Model/TSFormer')

from Data.road_data_provider import *





def exp_pretrain(cfg, logger=None):
    debug = cfg['debug']

    set_seed(cfg['seed'])

    device = cfg['device']
    
    temp, _ = cfg["dataset_src_trg"].split('_')
    dataset_src = ''.join(temp.split('-'))
    model_dir = './Save/pretrain_model/{}/'.format(dataset_src)
    ensure_dir(model_dir)
    
    provider = RoadDataProvider(cfg, flag='pretrain',logger=logger)
    
    train_dataloader, val_dataloader, test_dataloader = provider.generate_pretrain_dataloader()

    model = TSFormer(cfg['TSFormer']).to(device)
    model.mode = 'pretrain'
    opt = optim.Adam(model.parameters(),lr = cfg['flag']['pretrain']['lr'])
    loss_fn = nn.MSELoss(reduction = 'mean')

    logger.info('pretrain model has {} parameters'.format(count_parameters(model)))
    
    best_loss = 9999999999999.0

    epochs = cfg['flag']['pretrain']['epoch']


    for i in range(epochs):
        time_1 = time.time()

        ########################################
        #   train
        for batch in train_dataloader:
            total_loss = []
            total_mae = []
            total_mse = []
            total_rmse = []
            total_mape = []
            model.train()
           
            # input : [B, l, 7] -> [B, 7, l]
            x = batch.permute(0, 2, 1).to(device)
            
            out_masked_tokens, label_masked_tokens = model(x)
            
            # only the masked patch is loss target 
            loss = loss_fn(out_masked_tokens, label_masked_tokens)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            # unmask
            means = torch.unsqueeze(batch[:, 0, 5], dim=1).cuda()
            stds = torch.unsqueeze(batch[:, 0, 6], dim=1).cuda()
            
            unnorm_out, unnorm_label = unnorm(out_masked_tokens, means, stds), unnorm(label_masked_tokens,means,stds)
            
            MSE,RMSE,MAE,MAPE = calc_metric(unnorm_out, unnorm_label)
            
            total_mse.append(MSE.cpu().detach().numpy())
            total_rmse.append(RMSE.cpu().detach().numpy())
            total_mae.append(MAE.cpu().detach().numpy())
            total_mape.append(MAPE.cpu().detach().numpy())

            total_loss.append(loss.item())


            if debug: break
        
        logger.info('Epochs {}/{} Start'.format(i, epochs))
        logger.info('* Training MSE : {:.5f}, RMSE : {:.5f}, MAE : {:.5f}, MAPE: {:.5f}, normed MSE : {:.5f}.'.format(np.mean(total_mse), np.mean(total_rmse), np.mean(total_mae),np.mean(total_mape),np.mean(total_loss)))
        
        ############################################ 
        # validation
        for batch in val_dataloader:
            total_loss = []
            total_mae = []
            total_mse = []
            total_rmse = []
            total_mape = []
            model.eval()
            
            # input : [B,  l, 7] -> [B, 7, l]
            x = batch.permute(0, 2, 1).to(device)
            
            out_masked_tokens, label_masked_tokens = model(x)
            
            # only the masked patch is loss target 
            loss = loss_fn(out_masked_tokens, label_masked_tokens)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            # unmask
            means = torch.unsqueeze(batch[:, 0, 5], dim=1).cuda()
            stds = torch.unsqueeze(batch[:, 0, 6], dim=1).cuda()

            unnorm_out, unnorm_label = unnorm(out_masked_tokens, means, stds), unnorm(label_masked_tokens,means,stds)
            
            MSE,RMSE,MAE,MAPE = calc_metric(unnorm_out, unnorm_label)
            
            total_mse.append(MSE.cpu().detach().numpy())
            total_rmse.append(RMSE.cpu().detach().numpy())
            total_mae.append(MAE.cpu().detach().numpy())
            total_mape.append(MAPE.cpu().detach().numpy())

            total_loss.append(loss.item())


            if debug: break

        logger.info('** Validation MSE : {:.5f}, RMSE : {:.5f}, MAE : {:.5f}, MAPE: {:.5f}.'.format(np.mean(total_mse), np.mean(total_rmse), np.mean(total_mae),np.mean(total_mape)))
        
        mae_loss = np.mean(total_mae)
        if(mae_loss < best_loss):
            best_loss = mae_loss
            torch.save(model.state_dict(), model_dir  + 'best_model.pt')
            logger.info('Best model Saved at Epoch {}'.format(i))
        
        
    #####################################
    # test
        for batch in test_dataloader:
            total_loss = []
            total_mae = []
            total_mse = []
            total_rmse = []
            total_mape = []
            model.eval()
            
            # input : [B, l, 7] -> [B, 7, l]
            x = batch.permute(0, 2, 1).to(device)
            
            out_masked_tokens, label_masked_tokens = model(x)
            
            # only the masked patch is loss target 
            loss = loss_fn(out_masked_tokens, label_masked_tokens)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            # unmask
            means = torch.unsqueeze(batch[:, 0, 5], dim=1).cuda()
            stds = torch.unsqueeze(batch[:, 0, 6], dim=1).cuda()

            unnorm_out, unnorm_label = unnorm(out_masked_tokens, means, stds), unnorm(label_masked_tokens,means,stds)
            
            MSE,RMSE,MAE,MAPE = calc_metric(unnorm_out, unnorm_label)
            
            total_mse.append(MSE.cpu().detach().numpy())
            total_rmse.append(RMSE.cpu().detach().numpy())
            total_mae.append(MAE.cpu().detach().numpy())
            total_mape.append(MAPE.cpu().detach().numpy())

            total_loss.append(loss.item())

        
            if debug: break

        logger.info('*** Test MSE : {:.5f}, RMSE : {:.5f}, MAE : {:.5f}, MAPE: {:.5f}.'.format(np.mean(total_mse), np.mean(total_rmse), np.mean(total_mae),np.mean(total_mape)))
        

        logger.info('Epochs {}/{} Ends :)'.format(i, epochs))
        logger.info('This epoch costs {:.5}s'.format(time.time()-time_1))
        
        if debug: exit(0)
