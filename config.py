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

    'his_num': 72, # 历史时间步数
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

    'epoch_recons': 200,
    'epoch_cluster': 500,

    'epoch_supervise': 300,

    'epoch_domain_adapt': 1000,
    'outer_adapt_itr': 200,
    'inner_dis_itr': 5000 * 4,
    'inner_gen_itr': 1000,

 
    # 域分类损失权重
    'disc_weight': 100,
    # rank loss 的权重
    'rank_weight': 50,
    # 正交损失权重
    # 'or_weight': 200,
    'or_weight': 5,

    # 各项 cost 的权重
    'w_time': 0.33,
    'w_speed': 0.33,
    'w_hidden': 0.33,

    'validate_epoch': 10,
    'patience': 5,
    'lr': 1e-5,
    'dropout': 0.2,
    'batch_size': 32,

    # preference train
    'pref_train_ckpt': [100, 500, 1000, 5000, 10000, 50000, 100000],

    # cluster train (废弃)
    'num_cluster': 10,

    # cluster GCN
    'local_size': 50,
    'sample_cnum': 3, # 每个输入样本采样的簇个数

    # finetune
    'finetune_ratio': [0.1, 0.2, 0.5, 1, 2, 5, 10, 20],


    # 模型设置
    'num_hops': 6,
    'num_layers': 4,
    'dim_gat_hidden': 128,

    # 边的属性
    'dim_edge': 4,

    # 'num_time': 96,
    'num_time': 24,
    'num_type': 16,

    # 考虑对部分道路属性做 embedding
    'dim_time': 32,
    'dim_type': 32,
    'dim_biway': 16,
    'dim_islink': 16,

    'dim_gat_heads': 8,
    'dim_node_feature': 62,
    'dim_con_feature': 59, # 连续值的特征
    'dim_predict_hidden': 32,
    'dim_predict_output': 2,
}
