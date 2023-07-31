
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
import warnings
import re
import numpy as np
import pandas as pd
import joblib
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random
import math
from sklearn.preprocessing import LabelBinarizer
import os


def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

random_seed(777)


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

data_path = os.path.dirname(os.path.realpath('__file__'))+"/"+"XUprotbert_feature.csv"

Rna_data = pd.read_csv(data_path, header = None)

def split_features_labels(data):
    features = data.iloc[:, :-1]
    labels = data.iloc[:, -1]
    return features.values, labels.values

Rna_features, labels = split_features_labels(Rna_data)
Rna_features = Rna_features.astype(np.float32)
Rna_features = torch.from_numpy(Rna_features)


import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.manifold import TSNE
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup, AutoConfig
import seaborn as sns
from torch.autograd import Variable



def cal_score(pred, label):

    pred = np.argmax(pred, axis=1)

    try:
        AUC = roc_auc_score(label, pred, average='micro')
    except:
        AUC = 0

    # Pre = precision_score(label, pred)
    ACC = accuracy_score(label, pred)
    Macro_Pre = precision_score(label, pred, average='macro')
    Macro_recall = recall_score(label, pred, average='macro')
    Macro_F1score = f1_score(label, pred, average='macro')

    print("Model score ---    ACC:{0:.3f}       Macro_Pre:{1:.3f}       Macro_recall:{2:.3f}      Macro_F1score:{3:.3f}   AUC:{4:.3f}".format(ACC, Macro_Pre, Macro_recall, Macro_F1score, AUC))

    return ACC


class ProteinDataset(Dataset):
    def __init__(self, Rna_features, labels, device):
        super(ProteinDataset, self).__init__()

        self.Rna_feature = Rna_features
        self.label_list = labels

    def __getitem__(self, index):
        rna_feature = self.Rna_feature[index]
        label = self.label_list[index]

        return rna_feature, label

    def __len__(self):

        return len(self.label_list)


def fit(model, train_loader, optimizer, criterion, device):
    model.train()

    pred_list = []
    label_list = []

    for rna_feature, label in train_loader:
        rna_feature = rna_feature.to(device)
        label = label.to(device)

        pred = model(rna_feature)
        #         pred = pred.squeeze()
        loss = criterion(pred, label)

        optimizer.zero_grad()
        #         model.zero_grad()
        loss.backward()  # 反向传播
        optimizer.step()

        pred_list.extend(pred.squeeze().cpu().detach().numpy())  # extend 在已存在的列表中添加新的列表内容
        label_list.extend(label.squeeze().cpu().detach().numpy())  # .squeeze() 移除数组中维度为1的维度


    score = cal_score(pred_list, label_list)

    return score


def validate(model, val_loader, device):
    model.eval()

    pred_list = []
    label_list = []

    for rna_feature, label in val_loader:
        rna_feature = rna_feature.to(device)
        label = label.to(device)

        pred = model(rna_feature)

        pred_list.extend(pred.squeeze().cpu().detach().numpy())
        label_list.extend(label.squeeze().cpu().detach().numpy())

    score = cal_score(pred_list, label_list)

    return score, pred_list, label_list


class nnNetwork(nn.Module):
    def __init__(self, input_dim=768, num_class=5):
        super(nnNetwork, self).__init__()

        self.fc = nn.Linear(input_dim, 512)
        self.classifier = nn.Linear(512, num_class)

    def forward(self, x):
        #         print(x.shape) [128, 768]
        x = self.fc(x)
        x = self.classifier(x)
        out = F.softmax(x, dim=1)
        # print(out.shape)

        return out



import copy
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold

warnings.filterwarnings('ignore')

X_train = Rna_features
Y_train = labels

# 10折交叉验证
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)  # 42

# 把每一折的验证预测结果和真实标签都拿到
valid_pred = []
valid_label = []

independ_pred = []
independ_label = []

for index, (train_idx, val_idx) in enumerate(skf.split(X_train, Y_train)):

    # print(train_idx.shape, val_idx.shape)

    print('**' * 10, '第', index + 1, '折', 'ing....', '**' * 10)

    x_train, x_valid = X_train[train_idx], X_train[val_idx]
    y_train, y_valid = Y_train[train_idx], Y_train[val_idx]

    train_dataset = ProteinDataset(x_train, y_train, device)
    valid_dataset = ProteinDataset(x_valid, y_valid, device)

    #     test_dataset = ProteinDataset(proBer_test_seq, testX, testY, device)

    #     test_loader = DataLoader(test_dataset,
    #                              batch_size=12,
    #                              shuffle=False,
    #                              drop_last=True,
    #                              num_workers=4)

    train_loader = DataLoader(train_dataset,
                              batch_size=128,
                              shuffle=True)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=128,
                              shuffle=False)

    # 模型
    model = nnNetwork().to(device)

    #     model = CNetwork().to(device)

    # 优化算法
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=5e-05)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01) #  , momentum=0.9,weight_decay=1e-4
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.0012, weight_decay=1e-4)  #lr=0.001  1e-4  5e-05   lr=0.00122   lr=0.0018  结果稍微好点

    #   optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-05)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)

    #   optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-05)
    # optimizer = torch.optim.Adamax(model.parameters(), lr=0.01, weight_decay=5e-05)
    # optimizer = torch.optim.NAdam(model.parameters(), lr=0.01, weight_decay=5e-05)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.0012, weight_decay=5e-05)

    # 损失函数
    criterion = nn.CrossEntropyLoss()
    #   criterion = WeightedFocalLoss()

    best_val_score = float('-inf')
    last_improve = 0
    best_model = None

    for epoch in range(20):
        train_score = fit(model, train_loader, optimizer, criterion, device)
        val_score, _, _ = validate(model, valid_loader, device)

        if val_score > best_val_score:
            best_val_score = val_score
            best_model = copy.deepcopy(model)
            last_improve = epoch
            improve = '*'
        else:

            improve = ''

        print(
            f'Epoch: {epoch} Train Score: {train_score}, Valid Score: {val_score} {improve} '
        )

    model = best_model

    # print(model.state_dict().keys())  # 输出模型参数名称

    #   保存模型参数到路径"./data/model_parameter.pkl"
    #   torch.save(model.state_dict(), "./DeepGlu_"+str(index)+".pkl")

    print(f"=============end!!!!================")
    print("train")
    train_score, _, _ = validate(model, train_loader, device)
    print("valid")
    valid_score, pred_list, label_list = validate(model, valid_loader, device)

    valid_pred.extend(pred_list)
    valid_label.extend(label_list)

#     print("independ test")
#     independ_score, test_pred_list, test_label_list  = validate(model, test_loader, device)
#     independ_pred.extend(test_pred_list)
#     independ_label.extend(test_label_list)

print("******************************************训练集10折交叉验证的结果**********************************************")

print("cross_valid_score")
cross_valid_score = cal_score(valid_pred, valid_label)

# 保存计算10cross_valid平均AUC的pkl
# joblib.dump(valid_pred, '')
# joblib.dump(valid_label, )


# print("independ_test_score")
# independ_score = cal_score(independ_pred,independ_label)
