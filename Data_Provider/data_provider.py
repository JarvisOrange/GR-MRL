from torch.utils.data import DataLoader

from Data_Provider.dataset_factory import (
    LA_Dataset,
    PEMS_Dataset,
    CD_Dataset,
    SZ_Dataset,
)

from Node_Embedding.DeepWalk import DeepWalk


dataset_dict = {
    "PEMS": PEMS_Dataset,
    "LA": LA_Dataset,
    'CD': CD_Dataset,
    'SZ': SZ_Dataset,
}


def data_provide(cfg, flag):
    Data = dataset_dict[cfg['dataset_source']]

    if flag == "test":
        flag_train=False
        shuffle = False
        drop_last = True
        batch_size = 1  # bsz=1 for evaluation
    else:
        flag_train=True
        shuffle = True
        drop_last = True
        batch_size = cfg['batch_size']  # bsz for train and valid
        

    data_set = Data(dataset=cfg['dataset_source'], train=flag_train)

    laplace_embed = data_set.get_laplace_matrix()
    deep_walk_embed = data_set.get_deep_walk_matrix()
    space_syntax_embed = data_set.get_space_syntax_matrix()

    a, b, c = data_set.get_dataloader()

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle,
        # num_workers=args.num_workers, # not very stable
        drop_last=drop_last,
    )

    return data_set, data_loader

def get_node_embed():
    pass