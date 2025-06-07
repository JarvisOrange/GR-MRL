from config import cfg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import *
import time
from pathlib import Path
from tqdm import tqdm
import sys
from Model import TSFormer  
sys.path.append('./Model/TSFormer')

from Data.road_dataset import *


set_seed()

def pretrain(logger=None):

    device = cfg['device']
    
    temp, _ = cfg['dataset_src_trg'].split('_')
    dataset_src = ''.join(temp[0].split('-'))
    model_path = Path('./save/my_pretrain_model/{}/'.format(dataset_src))
    ensure_dir(model_path)
    
    provider = RoadDataProvider(cfg, flag='pretrain',logger=logger)
    
    train_dataloader, val_dataloader, test_dataloader = provider.generate_dataloader()

    model = TSFormer(cfg['TSFromer']).to(device)
    model.mode = 'pretrain'
    opt = optim.Adam(model.parameters(),lr = cfg['flag']['pretrain']['lr'])
    loss_fn = nn.MSELoss(reduction = 'mean')

    logger.info('pretrain model has {} parameters'.format(count_parameters(model)))
    
    best_loss = 9999999999999.0
    best_model = None

    epochs = cfg['flag']['pretrain']['epoch']

    ########################################
    #  train

    for i in range(epochs):
        time_1 = time.time()
        for batch in tqdm(train_dataloader):
            total_loss = []
            total_mae = []
            total_mse = []
            total_rmse = []
            total_mape = []
            model.train()
           
            # input : [B, N, l, 7] -> [B, N, 7, l]
            x = x.permute(0,1,3,2).to(device)
            
            out_masked_tokens, label_masked_tokens = model(x)
            
            # only the masked patch is loss target 
            loss = loss_fn(out_masked_tokens, label_masked_tokens)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            # unmask
            means, stds = torch.squeeze(batch[:, 0, 0, 5]), torch.squeeze(batch[:, 0, 0, 6])
            unnorm_out, unnorm_label = unnorm(out_masked_tokens, means, stds), unnorm(label_masked_tokens,means,stds)
            
            MSE,RMSE,MAE,MAPE = calc_metric(unnorm_out, unnorm_label)
            
            total_mse.append(MSE.cpu().detach().numpy())
            total_rmse.append(RMSE.cpu().detach().numpy())
            total_mae.append(MAE.cpu().detach().numpy())
            total_mape.append(MAPE.cpu().detach().numpy())

            total_loss.append(loss.item())

        logger.info('Epochs {}/{}'.format(i, epochs))
        logger.info('in training   Unnormed MSE : {:.5f}, RMSE : {:.5f}, MAE : {:.5f}, MAPE: {:.5f}, normed MSE : {:.5f}.'.format(np.mean(total_mse), np.mean(total_rmse), np.mean(total_mae),np.mean(total_mape),np.mean(total_loss)))
        
        ############################################ 
        # validation
        for batch in tqdm(val_dataloader):
            total_loss = []
            total_mae = []
            total_mse = []
            total_rmse = []
            total_mape = []
            model.eval()
            
            # input : [B, N, l, 7] -> [B, N, 7, l]
            x = x.permute(0,1,3,2).to(device)
            
            out_masked_tokens, label_masked_tokens = model(x)
            
            # only the masked patch is loss target 
            loss = loss_fn(out_masked_tokens, label_masked_tokens)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            # unmask
            means, stds = torch.squeeze(batch[:, 0, 0, 5]), torch.squeeze(batch[:, 0, 0, 6])
            unnorm_out, unnorm_label = unnorm(out_masked_tokens, means, stds), unnorm(label_masked_tokens,means,stds)
            
            MSE,RMSE,MAE,MAPE = calc_metric(unnorm_out, unnorm_label)
            
            total_mse.append(MSE.cpu().detach().numpy())
            total_rmse.append(RMSE.cpu().detach().numpy())
            total_mae.append(MAE.cpu().detach().numpy())
            total_mape.append(MAPE.cpu().detach().numpy())

            total_loss.append(loss.item())

        logger.info('Epochs {}/{}'.format(i, epochs))
        logger.info('in validation Unnormed MSE : {:.5f}, RMSE : {:.5f}, MAE : {:.5f}, MAPE: {:.5f}.'.format(np.mean(total_mse), np.mean(total_rmse), np.mean(total_mae),np.mean(total_mape)))
        
        mae_loss = np.mean(total_mae)
        if(mae_loss < best_loss):
            best_model = model
            best_loss = mae_loss
            torch.save(model.state_dict(), model_path / 'best_model.pt')
            logger.info('Best model. Saved.')
        logger.info('this epoch costs {:.5}s'.format(time.time()-time_1))
        
    #####################################
    # test
        for batch in tqdm(test_dataloader):
            total_loss = []
            total_mae = []
            total_mse = []
            total_rmse = []
            total_mape = []
            model.eval()
            
            # input : [B, N, l, 7] -> [B, N, 7, l]
            x = x.permute(0,1,3,2).to(device)
            
            out_masked_tokens, label_masked_tokens = model(x)
            
            # only the masked patch is loss target 
            loss = loss_fn(out_masked_tokens, label_masked_tokens)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            # unmask
            means, stds = torch.squeeze(batch[:, 0, 0, 5]), torch.squeeze(batch[:, 0, 0, 6])
            unnorm_out, unnorm_label = unnorm(out_masked_tokens, means, stds), unnorm(label_masked_tokens,means,stds)
            
            MSE,RMSE,MAE,MAPE = calc_metric(unnorm_out, unnorm_label)
            
            total_mse.append(MSE.cpu().detach().numpy())
            total_rmse.append(RMSE.cpu().detach().numpy())
            total_mae.append(MAE.cpu().detach().numpy())
            total_mape.append(MAPE.cpu().detach().numpy())

            total_loss.append(loss.item())

        logger.info('Epochs {}/{}'.format(i, epochs))
        logger.info('in test Unnormed MSE : {:.5f}, RMSE : {:.5f}, MAE : {:.5f}, MAPE: {:.5f}.'.format(np.mean(total_mse), np.mean(total_rmse), np.mean(total_mae),np.mean(total_mape)))