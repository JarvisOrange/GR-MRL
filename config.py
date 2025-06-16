cfg = {
    # 手动设置 tag
    # 0: try
    # 1: train
    
    'exp_tag': 0,

    'device': 'cuda: 0',



    'flag':  {
        'pretrain': {
                'batch_size': 4,  # [B, N, 288, 2]
                'epoch': 1000,
                'lr' : 0.0001,
                'train_val_test':[0.7, 0.1, 0.2]
        },
        'source_train':{
            'batch_size': 16,
            'source_train_epochs': 1000,
            'source_train_lr': 0.0001,
        },

        'target_train':{
            'batch_size': 16,
            'target_train_epochs': 1000,
            'target_train_lr': 0.0001,
        },

        'test':{
            'batch_size': 16
        },
    },


    #dataset information
    'dataset_info': {
        'L':{
            'nodes': 207,
            'edges': 1722,
            'interval': 5,
            'time_step': 34272,
            'mean': '58.2749',
            'std':'13.128',
            'start_time': '2017-05-01 00:00:00',
            'end_time': '2017-06-30 24:00:00'
        },
        'B':{
            'nodes': 325,
            'edges': 2694,
            'interval': 5,
            'time_step': 52116,
            'mean': 61.7768,
            'std':9.2852,
            'start_time': '2017-01-01 00:00:00',
            'end_time': '2017-06-30 24:00:00'
        },
        'C':{
            'nodes': 524,
            'edges': 1120,
            'interval': 10,
            'time_step': 17280,
            'mean': 29.0235,
            'std':9.662,
            'start_time': '2018-01-01 00:00:00',
            'end_time': '2018-04-30 24:00:00'
        },
        'S':{
            'nodes': 627,
            'edges': 4845,
            'interval': 10,
            'time_step': 17280,
            'mean': 31.0092,
            'std':10.9694,
            'start_time': '2018-01-01 00:00:00',
            'end_time': '2018-04-30 24:00:00'
        }
    },

    

    

    # 数据集设置
    # B: PEMS_BAY
    # L: MetrLA
    # C Chengdu
    # S: Shenzhen
    # dataset name order: BLCS
    # exmaple: B L C_S: B L C -> S

    'dataset_src_trg': 'B-L-C_S',

    'laplacian_dim': 64,

    'road_cluster_k': 16,

    'time_cluster_k': 32,

    'retrieve_k': 1,

    'his_num': 288, # 历史时间步数
    'pre_num': 12, # 预测时间步数

    'target_day': 3,


    'TSFormer': {
        'patch_size':12,
        'in_channel':1,
        'out_channel':128, #输出的维度

        'dropout':0.1,
        'window_size': 288,
        'mask_size':24,
        'mask_ratio':0.75,
        'n_layer':4,
    },

    'GR_MRL': {
        'time_expert': 32,
        'road_expert': 16,
        'expert_dim': 64
    },

    'LLM_path': './LLM/glm-4-9b',


    

    # 训练设置

    'drop_last': True,



    'validate_epoch': 10,
    'patience': 5,
    'lr': 1e-5,
    'dropout': 0.2,
    'batch_size': 32,

}
