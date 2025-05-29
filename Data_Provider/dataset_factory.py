import os
import numpy as np
import pandas as pd
from numpy import copy
from torch.utils.data import Dataset, DataLoader
from utils import *
from GR_MRL.logger import get_logger
from GR_MRL.Scaler import *


import warnings

warnings.filterwarnings('ignore')

dataset_path_dict = {
    "BAY": 'pems-bay',
    "LA": 'metr-la',
    'CD': 'chengdu_m',
    'SZ': 'shenzhen',
}


class ListDataset(Dataset):
    def __init__(self, data):
        """
        data: 必须是一个 list
        """
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
    
class ClusterDataset()
    

class LA_Dataset(Dataset):
    def __init__(self, 
                dataset='LA',
                root_path='./raw_data/',
                train=True,
                split_rate=[0.7, 0.1, 0.2],  # train, eval, test
                batch_size=64, 
                return_single_feature=False,
                saved_model=True):

        # self.return_sing_feature = False

        # self.data_path = root_path + dataset + '/'
        
        # self.config_file_name = self.data_path + '/' +'config'

        # self.config = ConfigParser(dataset, self.config_file_name, saved_model, train)

        # self._logger = get_logger(self.config)

        # self.dataset = dataset

        # #train test valid
        # self.train_rate, self.eval_rate, self.test_rate = split_rate
        # self.shuffle_dataset = True
        

        # self.batch_size = batch_size
        # self.input_window = self.config.get('input_window', 72)
        # self.output_window = self.config.get('output_window', 12)
        # self.scaler_type = self.config.get('scaler_type', 'normal')

        # self.num_workers = self.config.get('num_workers', 4)

        # self.data = None # initialize

        # #get attribute from config
        # self.geo_file = self.config.get('geo_file', self.dataset)
        # self.rel_file = self.config.get('rel_file', self.dataset)

        # self.weight_col = self.config.get('weight_col', '')
        # self.data_col = self.config.get('data_col', '')
        # self.ext_col = self.config.get('ext_col', '')
        # self.geo_file = self.config.get('geo_file', self.dataset)
        # self.rel_file = self.config.get('rel_file', self.dataset)
        
        # self.feature_name = {'X': 'float', 'y': 'float'}
        # self.ext_file = self.config.get('ext_file', self.dataset)
        # self.output_dim = self.config.get('output_dim', 1)
        # self.time_intervals = self.config.get('time_intervals', 300)  # s
        # self.init_weight_inf_or_zero = self.config.get('init_weight_inf_or_zero', 'inf')
        # self.set_weight_link_or_dist = self.config.get('set_weight_link_or_dist', 'dist')
        # self.bidir_adj_mx = self.config.get('bidir_adj_mx', False)
        # self.calculate_weight_adj = self.config.get('calculate_weight_adj', False)
        # self.weight_adj_epsilon = self.config.get('weight_adj_epsilon', 0.1)
        # self.distance_inverse = self.config.get('distance_inverse', False)
        # self.pad_with_last_sample = self.config.get('pad_with_last_sample', False)


        # self.data_files = self.config.get('data_files', self.dataset) #设置多个数据集


        # self.cache_file_name = os.path.join('./cache/dataset_cache/',
        #                                     'traffic_state_{}.npz'.format(self.dataset))
        # self.cache_file_folder = './cache/dataset_cache/'

        # ensure_dir(self.cache_file_folder)

        # self.cache_dataset = True if os.path.exists(self.cache_file_name) else False 

        
        # self.saved_cache_dataset = True


        # # use function
        self.data_path = root_path  + '/' + dataset_path_dict[dataset] + '/'
        self.__load_rel__()





        #self._load_dyna_()



    def __load_rel__(self):
        relfile = pd.read_csv(self.data_path + 'matrix.npy')
        self._logger.info('set_weight_link_or_dist: {}'.format(self.set_weight_link_or_dist))
        self._logger.info('init_weight_inf_or_zero: {}'.format(self.init_weight_inf_or_zero))
        if self.weight_col != '':  # 根据weight_col确认权重列
            if isinstance(self.weight_col, list):
                if len(self.weight_col) != 1:
                    raise ValueError('`weight_col` parameter must be only one column!')
                self.weight_col = self.weight_col[0]
            self.distance_df = relfile[~relfile[self.weight_col].isna()][[
                'origin_id', 'destination_id', self.weight_col]]
        else:
            if len(relfile.columns) > 5 or len(relfile.columns) < 4:  # properties不只一列，且未指定weight_col，报错
                raise ValueError("Don't know which column to be loaded! Please set `weight_col` parameter!")
            elif len(relfile.columns) == 4:  # 4列说明没有properties列，那就是rel文件中有的代表相邻，否则不相邻
                self.calculate_weight_adj = False
                self.set_weight_link_or_dist = 'link'
                self.init_weight_inf_or_zero = 'zero'
                self.distance_df = relfile[['origin_id', 'destination_id']]
            else:  # len(relfile.columns) == 5, properties只有一列，那就默认这一列是权重列
                self.weight_col = relfile.columns[-1]
                self.distance_df = relfile[~relfile[self.weight_col].isna()][[
                    'origin_id', 'destination_id', self.weight_col]]
        # 把数据转换成矩阵的形式
        self.adj_mx = np.zeros((len(self.geo_ids), len(self.geo_ids)), dtype=np.float32)
        if self.init_weight_inf_or_zero.lower() == 'inf' and self.set_weight_link_or_dist.lower() != 'link':
            self.adj_mx[:] = np.inf
        for row in self.distance_df.values:
            if row[0] not in self.geo_to_ind or row[1] not in self.geo_to_ind:
                continue
            if self.set_weight_link_or_dist.lower() == 'dist':  # 保留原始的距离数值
                self.adj_mx[self.geo_to_ind[row[0]], self.geo_to_ind[row[1]]] = row[2]
                if self.bidir_adj_mx:
                    self.adj_mx[self.geo_to_ind[row[1]], self.geo_to_ind[row[0]]] = row[2]
            else:  # self.set_weight_link_or_dist.lower()=='link' 只保留01的邻接性
                self.adj_mx[self.geo_to_ind[row[0]], self.geo_to_ind[row[1]]] = 1
                if self.bidir_adj_mx:
                    self.adj_mx[self.geo_to_ind[row[1]], self.geo_to_ind[row[0]]] = 1
        self._logger.info("Loaded file " + self.rel_file + '.rel, shape=' + str(self.adj_mx.shape))
        # 计算权重
        if self.distance_inverse and self.set_weight_link_or_dist.lower() != 'link':
            self._distance_inverse()
        elif self.calculate_weight_adj and self.set_weight_link_or_dist.lower() != 'link':
            self._calculate_adjacency_matrix()

    def _distance_inverse(self):
        self._logger.info("Start Calculate the weight by _distance_inverse!")
        self.adj_mx = 1 / self.adj_mx
        self.adj_mx[np.isinf(self.adj_mx)] = 1

    def _calculate_adjacency_matrix(self):
        """
        使用带有阈值的高斯核计算邻接矩阵的权重，如果有其他的计算方法，可以覆盖这个函数,
        公式为：$ w_{ij} = \exp \left(- \\frac{d_{ij}^{2}}{\sigma^{2}} \\right) $, $\sigma$ 是方差,
        小于阈值`weight_adj_epsilon`的值设为0：$  w_{ij}[w_{ij}<\epsilon]=0 $

        Returns:
            np.ndarray: self.adj_mx, N*N的邻接矩阵
        """
        self._logger.info("Start Calculate the weight by Gauss kernel!")
        distances = self.adj_mx[~np.isinf(self.adj_mx)].flatten()
        std = distances.std()
        self.adj_mx = np.exp(-np.square(self.adj_mx / std))
        self.adj_mx[self.adj_mx < self.weight_adj_epsilon] = 0

    def _load_dyna_(self, filename):
         
        """
        加载.dyna文件，格式[dyna_id, type, time, entity_id, properties(若干列)],
        .geo文件中的id顺序应该跟.dyna中一致,
        其中全局参数`data_col`用于指定需要加载的数据的列，不设置则默认全部加载

        Args:
            filename(str): 数据文件名，不包含后缀

        Returns:
            np.ndarray: 数据数组, 3d-array: (len_time, num_nodes, f eature_dim)
        """
         
        self._logger.info("Loading file " + filename + '.dyna')
        dynafile = pd.read_csv(self.data_path + filename + '.dyna')
        if self.data_col != '':  # 根据指定的列加载数据集
            if isinstance(self.data_col, list):
                data_col = self.data_col.copy()
            else:  # str
                data_col = [self.data_col].copy()
            data_col.insert(0, 'time')
            data_col.insert(1, 'entity_id')
            dynafile = dynafile[data_col]
        else:  # 不指定则加载所有列
            dynafile = dynafile[dynafile.columns[2:]]  # 从time列开始所有列
        # 求时间序列
        self.timesolts = list(dynafile['time'][:int(dynafile.shape[0] / len(self.geo_ids))])
        
        if not dynafile['time'].isna().any():  # 时间没有空值
            self.timesolts = list(map(lambda x: x.replace('T', ' ').replace('Z', ''), self.timesolts))
            self.timesolts = np.array(self.timesolts, dtype='datetime64[ns]')
            
        # 转3-d数组
        feature_dim = len(dynafile.columns) - 2
        df = dynafile[dynafile.columns[-feature_dim:]]
        len_time = len(self.timesolts)
        data = []
        for i in range(0, df.shape[0], len_time):
            data.append(df[i:i + len_time].values)
        
        data = data[:-1]
        # 这里需要注意，最后一个时间点的数据可能不完整，所以要去掉
        data = np.array(data, dtype=np.float32)  # (len(self.geo_ids), len_time, feature_dim)
        self._logger.info("Loaded file " + filename + '.dyna' + ', shape=' + str(data.shape))
        return data

        

    def _generate_data(self):
        """
        加载多个dataset
        加载数据文件(.dyna/.grid/.od/.gridod)和外部数据(.ext)，且将二者融合，以X，y的形式返回

        Returns:
            tuple: tuple contains:
                x(np.ndarray): 模型输入数据，(num_samples, input_length, ..., feature_dim) \n
                y(np.ndarray): 模型输出数据，(num_samples, output_length, ..., feature_dim)
        """
        # 处理多数据文件问题
        if isinstance(self.data_files, list):
            data_files = self.data_files.copy()
        else:  # str
            data_files = [self.data_files].copy()
        # 加载外部数据
        
        x_list, y_list = [], []
        for filename in data_files:
            df = self._load_dyna_(filename)  # (len_time, ..., feature_dim)
            
            x, y = self._generate_input_data(df)
            # x: (num_samples, input_length, ..., input_dim)
            # y: (num_samples, output_length, ...asd, output_dim)
            x_list.append(x)
            y_list.append(y)
        x = np.concatenate(x_list)
        y = np.concatenate(y_list)
        self._logger.info("Dataset created")
        self._logger.info("x shape: " + str(x.shape) + ", y shape: " + str(y.shape))
        return x, y

    def _generate_input_data(self, df):
        """
        根据全局参数`input_window`和`output_window`切分输入，产生模型需要的张量输入，
        即使用过去`input_window`长度的时间序列去预测未来`output_window`长度的时间序列

        Args:
            df(np.ndarray): 数据数组，shape: (len_time, ..., feature_dim)

        Returns:
            tuple: tuple contains:
                x(np.ndarray): 模型输入数据，(epoch_size, input_length, ..., feature_dim) \n
                y(np.ndarray): 模型输出数据，(epoch_size, output_length, ..., feature_dim)
        """
        num_samples = df.shape[0]
        # 预测用的过去时间窗口长度 取决于self.input_window
        x_offsets = np.sort(np.concatenate((np.arange(-self.input_window + 1, 1, 1),)))
        # 未来时间窗口长度 取决于self.output_window
        y_offsets = np.sort(np.arange(1, self.output_window + 1, 1))

        x, y = [], []
        min_t = abs(min(x_offsets))
        max_t = abs(num_samples - abs(max(y_offsets)))
        for t in range(min_t, max_t):
            x_t = df[t + x_offsets, ...]
            y_t = df[t + y_offsets, ...]
            x.append(x_t)
            y.append(y_t)
        x = np.stack(x, axis=0)
        y = np.stack(y, axis=0)
        return x, y

   

    def _generate_train_val_test(self):
        """
        加载数据集，并划分训练集、测试集、验证集，并缓存数据集

        Returns:
            tuple: tuple contains:
                x_train: (num_samples, input_length, ..., feature_dim) \n
                y_train: (num_samples, input_length, ..., feature_dim) \n
                x_val: (num_samples, input_length, ..., feature_dim) \n
                y_val: (num_samples, input_length, ..., feature_dim) \n
                x_test: (num_samples, input_length, ..., feature_dim) \n
                y_test: (num_samples, input_length, ..., feature_dim)
        """
        x, y = self._generate_data()
        return self._split_train_val_test(x, y)

    def _load_cache_train_val_test(self):
        """
        加载之前缓存好的训练集、测试集、验证集

        Returns:
            tuple: tuple contains:
                x_train: (num_samples, input_length, ..., feature_dim) \n
                y_train: (num_samples, input_length, ..., feature_dim) \n
                x_val: (num_samples, input_length, ..., feature_dim) \n
                y_val: (num_samples, input_length, ..., feature_dim) \n
                x_test: (num_samples, input_length, ..., feature_dim) \n
                y_test: (num_samples, input_length, ..., feature_dim)
        """
        self._logger.info('Loading ' + self.cache_file_name)
        cat_data = np.load(self.cache_file_name)
        x_train = cat_data['x_train']
        y_train = cat_data['y_train']
        x_test = cat_data['x_test']
        y_test = cat_data['y_test']
        x_val = cat_data['x_val']
        y_val = cat_data['y_val']
        self._logger.info("train\t" + "x: " + str(x_train.shape) + ", y: " + str(y_train.shape))
        self._logger.info("eval\t" + "x: " + str(x_val.shape) + ", y: " + str(y_val.shape))
        self._logger.info("test\t" + "x: " + str(x_test.shape) + ", y: " + str(y_test.shape))
        return x_train, y_train, x_val, y_val, x_test, y_test

    def _split_train_val_test(self, x, y):
        """
        划分训练集、测试集、验证集，并缓存数据集

        Args:
            x(np.ndarray): 输入数据 (num_samples, input_length, ..., feature_dim)
            y(np.ndarray): 输出数据 (num_samples, input_length, ..., feature_dim)

        Returns:
            tuple: tuple contains:
                x_train: (num_samples, input_length, ..., feature_dim) \n
                y_train: (num_samples, input_length, ..., feature_dim) \n
                x_val: (num_samples, input_length, ..., feature_dim) \n
                y_val: (num_samples, input_length, ..., feature_dim) \n
                x_test: (num_samples, input_length, ..., feature_dim) \n
                y_test: (num_samples, input_length, ..., feature_dim)
        """
        test_rate = 1 - self.train_rate - self.eval_rate
        num_samples = x.shape[0]
        num_test = round(num_samples * test_rate)
        num_train = round(num_samples * self.train_rate)
        num_val = num_samples - num_test - num_train

        # train
        x_train, y_train = x[:num_train], y[:num_train]
        # val
        x_val, y_val = x[num_train: num_train + num_val], y[num_train: num_train + num_val]
        # test
        x_test, y_test = x[-num_test:], y[-num_test:]
        self._logger.info("train\t" + "x: " + str(x_train.shape) + ", y: " + str(y_train.shape))
        self._logger.info("eval\t" + "x: " + str(x_val.shape) + ", y: " + str(y_val.shape))
        self._logger.info("test\t" + "x: " + str(x_test.shape) + ", y: " + str(y_test.shape))

        if self.saved_cache_dataset:
            if not os.path.exists(self.cache_file_folder):
                os.mkdir(self.cache_file_folder)

            np.savez_compressed(
                self.cache_file_name,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                x_val=x_val,
                y_val=y_val,
            )
            self._logger.info('Saved at ' + self.cache_file_name)
        return x_train, y_train, x_val, y_val, x_test, y_test

    def _get_scalar(self, scaler_type, x_train, y_train):
        """
        根据全局参数`scaler_type`选择数据归一化方法

        Args:
            x_train: 训练数据X
            y_train: 训练数据y

        Returns:
            Scaler: 归一化对象
        """
        if scaler_type == "normal":
            scaler = NormalScaler(maxx=max(x_train.max(), y_train.max()))
            self._logger.info('NormalScaler max: ' + str(scaler.max))
        elif scaler_type == "standard":
            scaler = StandardScaler(mean=x_train.mean(), std=x_train.std())
            self._logger.info('StandardScaler mean: ' + str(scaler.mean) + ', std: ' + str(scaler.std))
        elif scaler_type == "minmax01":
            scaler = MinMax01Scaler(
                maxx=max(x_train.max(), y_train.max()), minn=min(x_train.min(), y_train.min()))
            self._logger.info('MinMax01Scaler max: ' + str(scaler.max) + ', min: ' + str(scaler.min))
        elif scaler_type == "minmax11":
            scaler = MinMax11Scaler(
                maxx=max(x_train.max(), y_train.max()), minn=min(x_train.min(), y_train.min()))
            self._logger.info('MinMax11Scaler max: ' + str(scaler.max) + ', min: ' + str(scaler.min))
        elif scaler_type == "log":
            scaler = LogScaler()
            self._logger.info('LogScaler')
        elif scaler_type == "none":
            scaler = NoneScaler()
            self._logger.info('NoneScaler')
        else:
            raise ValueError('Scaler type error!')
        return scaler

    def get_dataloader(self):
        """
        返回数据的DataLoader，包括训练数据、测试数据、验证数据

        Returns:
            tuple: tuple contains:
                train_dataloader: Dataloader composed of Batch (class) \n
                eval_dataloader: Dataloader composed of Batch (class) \n
                test_dataloader: Dataloader composed of Batch (class)
        """
        # 加载数据集
        x_train, y_train, x_val, y_val, x_test, y_test = [], [], [], [], [], []
        if self.data is None:
            self.data = {}
            if self.cache_dataset and os.path.exists(self.cache_file_name):
                x_train, y_train, x_val, y_val, x_test, y_test = self._load_cache_train_val_test()
            else:
                x_train, y_train, x_val, y_val, x_test, y_test = self._generate_train_val_test()
        
        # 数据归一化
        self.feature_dim = x_train.shape[-1]
        self.ext_dim = self.feature_dim - self.output_dim
        self.scaler = self._get_scalar(self.scaler_type,
                                       x_train[..., :self.output_dim], y_train[..., :self.output_dim])
        
        x_train[..., :self.output_dim] = self.scaler.transform(x_train[..., :self.output_dim])
        y_train[..., :self.output_dim] = self.scaler.transform(y_train[..., :self.output_dim])
        x_val[..., :self.output_dim] = self.scaler.transform(x_val[..., :self.output_dim])
        y_val[..., :self.output_dim] = self.scaler.transform(y_val[..., :self.output_dim])
        x_test[..., :self.output_dim] = self.scaler.transform(x_test[..., :self.output_dim])
        y_test[..., :self.output_dim] = self.scaler.transform(y_test[..., :self.output_dim])
        
        # 把训练集的X和y聚合在一起成为list，测试集验证集同理
        # x_train/y_train: (num_samples, input_length, ..., feature_dim)
        # train_data(list): train_data[i]是一个元组，由x_train[i]和y_train[i]组成
        train_data = list(zip(x_train, y_train))
        eval_data = list(zip(x_val, y_val))
        test_data = list(zip(x_test, y_test))
        
        # 转Dataloader
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            self.generate_dataloader(train_data, eval_data, test_data, self.feature_name, 
                                self.batch_size, self.num_workers, pad_with_last_sample=self.pad_with_last_sample)
        self.num_batches = len(self.train_dataloader)
        return self.train_dataloader, self.eval_dataloader, self.test_dataloader

    

    def generate_dataloader(self, train_data, eval_data, test_data, feature_name,
                        batch_size, num_workers, 
                        pad_with_last_sample=False):
        """
        create dataloader(train/test/eval)

        Args:
            train_data(list of input): 训练数据，data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
            eval_data(list of input): 验证数据，data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
            test_data(list of input): 测试数据，data 中每个元素是模型单次的输入，input 是一个 list，里面存放单次输入和 target
            feature_name(dict): 描述上面 input 每个元素对应的特征名, 应保证len(feature_name) = len(input)
            batch_size(int): batch_size
            num_workers(int): num_workers
            shuffle(bool): shuffle
            pad_with_last_sample(bool): 对于若最后一个 batch 不满足 batch_size的情况，是否进行补齐（使用最后一个元素反复填充补齐）。

        Returns:
            tuple: tuple contains:
                train_dataloader: Dataloader composed of Batch (class) \n
                eval_dataloader: Dataloader composed of Batch (class) \n
                test_dataloader: Dataloader composed of Batch (class)
        """
        if pad_with_last_sample:
            num_padding = (batch_size - (len(train_data) % batch_size)) % batch_size
            data_padding = np.repeat(train_data[-1:], num_padding, axis=0)
            train_data = np.concatenate([train_data, data_padding], axis=0)

            num_padding = (batch_size - (len(eval_data) % batch_size)) % batch_size
            data_padding = np.repeat(eval_data[-1:], num_padding, axis=0)
            eval_data = np.concatenate([eval_data, data_padding], axis=0)

            num_padding = (batch_size - (len(test_data) % batch_size)) % batch_size
            data_padding = np.repeat(test_data[-1:], num_padding, axis=0)
            test_data = np.concatenate([test_data, data_padding], axis=0)

        train_dataset = ListDataset(train_data)
        eval_dataset = ListDataset(eval_data)
        test_dataset = ListDataset(test_data)

        def collator(indices):
            batch = Batch(feature_name)
            for item in indices:
                batch.append(copy.deepcopy(item))
            return batch

        shuffle = self.shuffle_dataset
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                    num_workers=num_workers,
                                    shuffle=shuffle)
        eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=batch_size,
                                    num_workers=num_workers, 
                                    shuffle=shuffle)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                    num_workers=num_workers,
                                    shuffle=False)
        return train_dataloader, eval_dataloader, test_dataloader


    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.geo_to_ind)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def get_adj_mx(self):
        return self.adj_mx



class BAY_Dataset(Dataset):
    def __init__(self, dataset_name='METRLA',
                train=True,
                train_rate=0.7,
                val_rate=0.1,
                test_rate=0.2,
                batch_size=64, 
                root_path='./Raw_Dataset',
                saved_model=True):
        
        self.config_file_name = root_path + '/' + dataset_name + '/' +'config.json'

        self.config = ConfigParser(dataset_name, self.config_file_name, saved_model, train)

        self.dataset_name = dataset_name
        self.train_rate = train_rate
        self.val_rate = val_rate

        self.geo_file = self.config.get('geo_file', self.dataset)
        self.rel_file = self.config.get('rel_file', self.dataset)

        self.__load_geo__()
        self.__load_rel__()


    def __load_geo__(self):
        """
        加载.geo文件，格式[geo_id, type, coordinates, properties(若干列)]

        """
        
        geofile = pd.read_csv(self.root_path + self.geo_file + '.geo')
        self.geo_ids = list(geofile['geo_id'])
        self.num_nodes = len(self.geo_ids)
        self.geo_to_ind = {}
        self.ind_to_geo = {}
        for index, idx in enumerate(self.geo_ids):
            self.geo_to_ind[idx] = index
            self.ind_to_geo[index] = idx
        self._logger.info("Loaded file " + self.geo_file + '.geo' + ', num_nodes=' + str(len(self.geo_ids)))

    def __load_rel__(self):
        relfile = pd.read_csv(self.data_path + self.rel_file + '.rel')
        self._logger.info('set_weight_link_or_dist: {}'.format(self.set_weight_link_or_dist))
        self._logger.info('init_weight_inf_or_zero: {}'.format(self.init_weight_inf_or_zero))
        if self.weight_col != '':  # 根据weight_col确认权重列
            if isinstance(self.weight_col, list):
                if len(self.weight_col) != 1:
                    raise ValueError('`weight_col` parameter must be only one column!')
                self.weight_col = self.weight_col[0]
            self.distance_df = relfile[~relfile[self.weight_col].isna()][[
                'origin_id', 'destination_id', self.weight_col]]
        else:
            if len(relfile.columns) > 5 or len(relfile.columns) < 4:  # properties不只一列，且未指定weight_col，报错
                raise ValueError("Don't know which column to be loaded! Please set `weight_col` parameter!")
            elif len(relfile.columns) == 4:  # 4列说明没有properties列，那就是rel文件中有的代表相邻，否则不相邻
                self.calculate_weight_adj = False
                self.set_weight_link_or_dist = 'link'
                self.init_weight_inf_or_zero = 'zero'
                self.distance_df = relfile[['origin_id', 'destination_id']]
            else:  # len(relfile.columns) == 5, properties只有一列，那就默认这一列是权重列
                self.weight_col = relfile.columns[-1]
                self.distance_df = relfile[~relfile[self.weight_col].isna()][[
                    'origin_id', 'destination_id', self.weight_col]]
        # 把数据转换成矩阵的形式
        self.adj_mx = np.zeros((len(self.geo_ids), len(self.geo_ids)), dtype=np.float32)
        if self.init_weight_inf_or_zero.lower() == 'inf' and self.set_weight_link_or_dist.lower() != 'link':
            self.adj_mx[:] = np.inf
        for row in self.distance_df.values:
            if row[0] not in self.geo_to_ind or row[1] not in self.geo_to_ind:
                continue
            if self.set_weight_link_or_dist.lower() == 'dist':  # 保留原始的距离数值
                self.adj_mx[self.geo_to_ind[row[0]], self.geo_to_ind[row[1]]] = row[2]
                if self.bidir_adj_mx:
                    self.adj_mx[self.geo_to_ind[row[1]], self.geo_to_ind[row[0]]] = row[2]
            else:  # self.set_weight_link_or_dist.lower()=='link' 只保留01的邻接性
                self.adj_mx[self.geo_to_ind[row[0]], self.geo_to_ind[row[1]]] = 1
                if self.bidir_adj_mx:
                    self.adj_mx[self.geo_to_ind[row[1]], self.geo_to_ind[row[0]]] = 1
        self._logger.info("Loaded file " + self.rel_file + '.rel, shape=' + str(self.adj_mx.shape))
        # 计算权重
        if self.distance_inverse and self.set_weight_link_or_dist.lower() != 'link':
            self._distance_inverse()
        elif self.calculate_weight_adj and self.set_weight_link_or_dist.lower() != 'link':
            self._calculate_adjacency_matrix()
        


    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

class CD_Dataset(Dataset):
    def __init__(self, dataset_name='CD',
                train=True,
                train_rate=0.7,
                val_rate=0.1,
                test_rate=0.2,
                batch_size=64, 
                root_path='./Raw_Dataset',
                saved_model=True):
        
        self.config_file_name = root_path + '/' + dataset_name + '/' +'config.json'

        self.config = ConfigParser(dataset_name, self.config_file_name, saved_model, train)

        self.dataset_name = dataset_name
        self.train_rate = train_rate
        self.val_rate = val_rate

        self.geo_file = self.config.get('geo_file', self.dataset)
        self.rel_file = self.config.get('rel_file', self.dataset)

        self.__load_geo__()
        self.__load_rel__()


    def __load_geo__(self):
        """
        加载.geo文件，格式[geo_id, type, coordinates, properties(若干列)]

        """
        
        geofile = pd.read_csv(self.root_path + self.geo_file + '.geo')
        self.geo_ids = list(geofile['geo_id'])
        self.num_nodes = len(self.geo_ids)
        self.geo_to_ind = {}
        self.ind_to_geo = {}
        for index, idx in enumerate(self.geo_ids):
            self.geo_to_ind[idx] = index
            self.ind_to_geo[index] = idx
        self._logger.info("Loaded file " + self.geo_file + '.geo' + ', num_nodes=' + str(len(self.geo_ids)))

    def __load_rel__(self):
        relfile = pd.read_csv(self.data_path + self.rel_file + '.rel')
        self._logger.info('set_weight_link_or_dist: {}'.format(self.set_weight_link_or_dist))
        self._logger.info('init_weight_inf_or_zero: {}'.format(self.init_weight_inf_or_zero))
        if self.weight_col != '':  # 根据weight_col确认权重列
            if isinstance(self.weight_col, list):
                if len(self.weight_col) != 1:
                    raise ValueError('`weight_col` parameter must be only one column!')
                self.weight_col = self.weight_col[0]
            self.distance_df = relfile[~relfile[self.weight_col].isna()][[
                'origin_id', 'destination_id', self.weight_col]]
        else:
            if len(relfile.columns) > 5 or len(relfile.columns) < 4:  # properties不只一列，且未指定weight_col，报错
                raise ValueError("Don't know which column to be loaded! Please set `weight_col` parameter!")
            elif len(relfile.columns) == 4:  # 4列说明没有properties列，那就是rel文件中有的代表相邻，否则不相邻
                self.calculate_weight_adj = False
                self.set_weight_link_or_dist = 'link'
                self.init_weight_inf_or_zero = 'zero'
                self.distance_df = relfile[['origin_id', 'destination_id']]
            else:  # len(relfile.columns) == 5, properties只有一列，那就默认这一列是权重列
                self.weight_col = relfile.columns[-1]
                self.distance_df = relfile[~relfile[self.weight_col].isna()][[
                    'origin_id', 'destination_id', self.weight_col]]
        # 把数据转换成矩阵的形式
        self.adj_mx = np.zeros((len(self.geo_ids), len(self.geo_ids)), dtype=np.float32)
        if self.init_weight_inf_or_zero.lower() == 'inf' and self.set_weight_link_or_dist.lower() != 'link':
            self.adj_mx[:] = np.inf
        for row in self.distance_df.values:
            if row[0] not in self.geo_to_ind or row[1] not in self.geo_to_ind:
                continue
            if self.set_weight_link_or_dist.lower() == 'dist':  # 保留原始的距离数值
                self.adj_mx[self.geo_to_ind[row[0]], self.geo_to_ind[row[1]]] = row[2]
                if self.bidir_adj_mx:
                    self.adj_mx[self.geo_to_ind[row[1]], self.geo_to_ind[row[0]]] = row[2]
            else:  # self.set_weight_link_or_dist.lower()=='link' 只保留01的邻接性
                self.adj_mx[self.geo_to_ind[row[0]], self.geo_to_ind[row[1]]] = 1
                if self.bidir_adj_mx:
                    self.adj_mx[self.geo_to_ind[row[1]], self.geo_to_ind[row[0]]] = 1
        self._logger.info("Loaded file " + self.rel_file + '.rel, shape=' + str(self.adj_mx.shape))
        # 计算权重
        if self.distance_inverse and self.set_weight_link_or_dist.lower() != 'link':
            self._distance_inverse()
        elif self.calculate_weight_adj and self.set_weight_link_or_dist.lower() != 'link':
            self._calculate_adjacency_matrix()
        


    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class SZ_Dataset(Dataset):
    def __init__(self, dataset_name='CD',
                train=True,
                train_rate=0.7,
                val_rate=0.1,
                test_rate=0.2,
                batch_size=64, 
                root_path='./Raw_Dataset',
                saved_model=True):
        
        self.config_file_name = root_path + '/' + dataset_name + '/' +'config.json'

        self.config = ConfigParser(dataset_name, self.config_file_name, saved_model, train)

        self.dataset_name = dataset_name
        self.train_rate = train_rate
        self.val_rate = val_rate

        self.geo_file = self.config.get('geo_file', self.dataset)
        self.rel_file = self.config.get('rel_file', self.dataset)

        self.__load_geo__()
        self.__load_rel__()


    def __load_geo__(self):
        """
        加载.geo文件，格式[geo_id, type, coordinates, properties(若干列)]

        """
        
        geofile = pd.read_csv(self.root_path + self.geo_file + '.geo')
        self.geo_ids = list(geofile['geo_id'])
        self.num_nodes = len(self.geo_ids)
        self.geo_to_ind = {}
        self.ind_to_geo = {}
        for index, idx in enumerate(self.geo_ids):
            self.geo_to_ind[idx] = index
            self.ind_to_geo[index] = idx
        self._logger.info("Loaded file " + self.geo_file + '.geo' + ', num_nodes=' + str(len(self.geo_ids)))

    def __load_rel__(self):
        relfile = pd.read_csv(self.data_path + self.rel_file + '.rel')
        self._logger.info('set_weight_link_or_dist: {}'.format(self.set_weight_link_or_dist))
        self._logger.info('init_weight_inf_or_zero: {}'.format(self.init_weight_inf_or_zero))
        if self.weight_col != '':  # 根据weight_col确认权重列
            if isinstance(self.weight_col, list):
                if len(self.weight_col) != 1:
                    raise ValueError('`weight_col` parameter must be only one column!')
                self.weight_col = self.weight_col[0]
            self.distance_df = relfile[~relfile[self.weight_col].isna()][[
                'origin_id', 'destination_id', self.weight_col]]
        else:
            if len(relfile.columns) > 5 or len(relfile.columns) < 4:  # properties不只一列，且未指定weight_col，报错
                raise ValueError("Don't know which column to be loaded! Please set `weight_col` parameter!")
            elif len(relfile.columns) == 4:  # 4列说明没有properties列，那就是rel文件中有的代表相邻，否则不相邻
                self.calculate_weight_adj = False
                self.set_weight_link_or_dist = 'link'
                self.init_weight_inf_or_zero = 'zero'
                self.distance_df = relfile[['origin_id', 'destination_id']]
            else:  # len(relfile.columns) == 5, properties只有一列，那就默认这一列是权重列
                self.weight_col = relfile.columns[-1]
                self.distance_df = relfile[~relfile[self.weight_col].isna()][[
                    'origin_id', 'destination_id', self.weight_col]]
        # 把数据转换成矩阵的形式
        self.adj_mx = np.zeros((len(self.geo_ids), len(self.geo_ids)), dtype=np.float32)
        if self.init_weight_inf_or_zero.lower() == 'inf' and self.set_weight_link_or_dist.lower() != 'link':
            self.adj_mx[:] = np.inf
        for row in self.distance_df.values:
            if row[0] not in self.geo_to_ind or row[1] not in self.geo_to_ind:
                continue
            if self.set_weight_link_or_dist.lower() == 'dist':  # 保留原始的距离数值
                self.adj_mx[self.geo_to_ind[row[0]], self.geo_to_ind[row[1]]] = row[2]
                if self.bidir_adj_mx:
                    self.adj_mx[self.geo_to_ind[row[1]], self.geo_to_ind[row[0]]] = row[2]
            else:  # self.set_weight_link_or_dist.lower()=='link' 只保留01的邻接性
                self.adj_mx[self.geo_to_ind[row[0]], self.geo_to_ind[row[1]]] = 1
                if self.bidir_adj_mx:
                    self.adj_mx[self.geo_to_ind[row[1]], self.geo_to_ind[row[0]]] = 1
        self._logger.info("Loaded file " + self.rel_file + '.rel, shape=' + str(self.adj_mx.shape))
        # 计算权重
        if self.distance_inverse and self.set_weight_link_or_dist.lower() != 'link':
            self._distance_inverse()
        elif self.calculate_weight_adj and self.set_weight_link_or_dist.lower() != 'link':
            self._calculate_adjacency_matrix()
        


    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)