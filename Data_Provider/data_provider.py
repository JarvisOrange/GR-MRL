from torch.utils.data import DataLoader

from Data_Provider.dataset_factory import (
    METRLA_Dataset,
    PEMS_Dataset,
)

from Node_Embedding.DeepWalk import DeepWalk


dataset_dict = {
    "PEMS_BAY": PEMS_Dataset,
    "METR_LA": METRLA_Dataset,
}


def data_provide(args, flag):
    Data = dataset_dict[args.dataset]

    if flag == "test":
        flag_train=False

        shuffle_flag = False
        drop_last = True
        batch_size = 1  # bsz=1 for evaluation
        batch_size = args.bs
        
        flag_train=True
    else:
        flag_train=True

        shuffle_flag = True
        drop_last = True
        batch_size = args.bs  # bsz for train and valid
        
        
    
    data_set = Data(
                'METR_LA',
                train=flag_train,
    )
    
    adj_mx = data_set.get_adj_mx()
    node_embedding_method = DeepWalk(adj_mx)
    node_embedding_method.train()

    a, b, c = data_set.get_dataloader()
    

    
    

    

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        # num_workers=args.num_workers, # not very stable
        drop_last=drop_last,
    )

    return data_set, data_loader
