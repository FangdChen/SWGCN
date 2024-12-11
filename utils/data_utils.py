import gc
import os
import pickle

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.utils.data as data
from utils import helper


def convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo()  # coo_matrix: data,[(row,col)],shape
    i = torch.LongTensor(np.array([coo.row, coo.col]))
    v = torch.from_numpy(coo.data).float()
    return torch.sparse.FloatTensor(i, v, coo.shape)

def load_all(dataset):
    prexdir = 'data/'+dataset+'/'
    behaviors = ['pv', 'fav', 'cart', 'buy'] if dataset!='beibei' else ['pv', 'cart', 'buy']

    train_path = prexdir + 'train_'
    train_mats = list()
    for i in range(len(behaviors)):
        behavior = behaviors[i]
        path = train_path + behavior
        with open(path, 'rb') as fs:
            mat = (pickle.load(fs) != 0).astype(np.float)  # 有交互即设置为1.
            # print(mat.data)
        train_mats.append(mat)

    # test set without negative
    test_data = pd.read_csv(prexdir + 'test.txt', sep='\t', header=None, names=['user', 'item'], usecols=[1, 2])
    test_data = np.array(test_data.values)  # [ [user,item],..., [user, item] ]m

    validation_data = pd.read_csv(prexdir + 'validation.txt', sep='\t', header=None, names=['user', 'item'], usecols=[1, 2])
    validation_data = np.array(validation_data.values)  # [ [user,item],..., [user, item] ]，此处全为pos_item

    n_user, n_item = train_mats[0].shape

    return train_mats, test_data, validation_data, n_user, n_item

class TrainData(data.Dataset):
    def __init__(self, n_user, n_item, n_neg, train_mats, features=None, n_train_user=0, n_train_sample=0, is_training=False):
        super(TrainData, self).__init__()
        self.features = features  # [ [user,item],..., [user, item] ] 样式的train data与test data
        self.n_user = n_user
        self.n_item = n_item
        self.n_train_user = n_train_user
        self.n_train_sample = n_train_sample
        self.n_neg = n_neg
        self.n_behavior = len(train_mats)
        self.train_mats = train_mats
        self.is_training = is_training
        self.features_fill = []

        self.ground_truth = self._get_label()

        # 手动清空内存
        del features
        del train_mats
        gc.collect()

    def _get_label(self):
        if self.is_training:
            ground_truth = self.train_mats[0]
            for i in range(1, len(self.train_mats)):
                ground_truth += self.train_mats[i]
            ground_truth = (ground_truth != 0).astype(np.float32)
            ground_truth = ground_truth.todok()
        else:
            ground_truth = self.train_mats[-1]  # 设置buy行为交互为pos instance，其他行为均为neg instance
            ground_truth = ground_truth.todok()

        return ground_truth
    
    def generate_train_data(self, start_behavior, epoch, dataset, cache_flag):
        self.features = []
        predix = 'data/'+dataset+'/cache/'
        helper.ensureDir(predix)
        if start_behavior == self.n_behavior-1:
            train_data_path = predix + 'OB-train_epoch_{}.data'.format(epoch)
        else:
            train_data_path = predix + 'MB-train_epoch_{}.data'.format(epoch)
        if cache_flag and os.path.exists(train_data_path):
            with open(train_data_path, 'rb') as f:
                self.features = pickle.load(f)
        else:
            num = self.n_user
            np.random.seed(42 + epoch)
            users = np.random.permutation(num)[:self.n_train_user]
            num = len(users)

            for i in range(start_behavior, self.n_behavior):
                step = 10000
                total_step = num // step + 1
                for s in range(total_step):
                    step_users = users[s*step:(s+1)*step]
                    temp_label = self.train_mats[i][step_users].toarray()
                    def user_neg_sample(u):
                        pos_pool = np.reshape(np.argwhere(temp_label[u] != 0), [-1])
                        n_train_sample = min(self.n_train_sample, len(pos_pool))
                        if n_train_sample == 0:
                            np.random.seed(42+ epoch)
                            pos_items = [np.random.choice(self.n_item)]
                            neg_items = [pos_items[0]]
                        else:
                            np.random.seed(42+ epoch)
                            pos_items = np.random.permutation(pos_pool)[:n_train_sample].repeat(self.n_neg)  # 正样本个数与负样本个数要匹配
                            neg_pool = np.reshape(np.argwhere(temp_label[u] == 0), [-1])
                            np.random.seed(42 + epoch)  # 没问题
                            neg_items = np.random.permutation(neg_pool)[:n_train_sample * self.n_neg]
                        feature = [[step_users[u], pos_item, neg_item] for pos_item, neg_item in zip(pos_items, neg_items)]
                        return feature

                    features = [sample for u in range(len(step_users)) for sample in user_neg_sample(u)]
                    self.features.extend(features)

            if cache_flag:
                with open(train_data_path, 'wb') as f:
                    pickle.dump(self.features, f)

    def __len__(self):
        return len(self.features) if self.is_training else len(self.features_fill)

    def __getitem__(self, idx):
        features = self.features if self.is_training else self.features_fill

        user = features[idx][0]
        pos_item = features[idx][1]
        neg_item = features[idx][2] if self.is_training else features[idx][1]

        return user, pos_item, neg_item

class TestDataset(data.Dataset):
    def __init__(self, test_data, train_mask, task='test'):
        self.train_mask = train_mask
        self.test_data = test_data
        self.task = task
        self.n_user, self.n_item = train_mask.shape[0], train_mask.shape[1]
        self.ground_truth = self._get_ground_truth()

    def _get_ground_truth(self):
        row = np.array(self.test_data[:, 0])
        col = np.array(self.test_data[:, 1])
        values = np.ones(len(row), dtype=float)
        ground_truth = sp.csr_matrix((values,(row,col)),shape=(self.n_user, self.n_item))
        return ground_truth

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, index):
        user = self.test_data[index, 0]
        return user, torch.from_numpy(self.ground_truth[user].toarray()).squeeze(), torch.from_numpy(self.train_mask[user].toarray()).float().squeeze()



if __name__ == '__main__':
    train_mats, test_data, validation_data, n_user, n_item = load_all('test')
    test_dataset = TestDataset(test_data, train_mats[-1])
