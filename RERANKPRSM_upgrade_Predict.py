import argparse


import numpy as np
import torch
import torch.nn as nn
import joblib
import RERANKPRSM_upgrade_train
from RERANKPRSM_upgrade_train import hidden_size
from RERANKPRSM_upgrade_train import ResNeXt,ResNeXtBlock
from RERANKPRSM_upgrade_train import input_size,num_blocks,cardinality,num_classes
np.set_printoptions(threshold=np.inf)


def input1(name,i):
    a,b= RERANKPRSM_upgrade_train.inputfile_np(name)
    m1=joblib.load(filename="saved_model5/LR.pkl")
    m2 = joblib.load(filename="saved_model5/DecTree.pkl")
    m3 = joblib.load(filename="saved_model5/svm.pkl")
    m4 = joblib.load(filename="saved_model5/xgboost.pkl")
    LR_pred=m1.predict(a).reshape(-1,1)
    x1 = m2.predict_proba(a)[:, 1].reshape(-1, 1)
    x2 = m3.predict_proba(a)[:, 0].reshape(-1, 1)
    x3 = m4.predict_proba(a)[:, 1].reshape(-1, 1)
    X = np.concatenate((a, LR_pred, x1, x2, x3), axis=1)
    Y=torch.tensor(b).long()
    return torch.tensor(X),Y

def predict(filename,i):

    # model=TenLayerNet(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)   #导入模型框架
    # model = ResidualNetwork(input_size = 11, hidden_size= hidden_size, num_classes=1, num_blocks=9)

    model = ResNeXt(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes, num_blocks=num_blocks, cardinality=cardinality)

    x, y = input1(filename,i)
    model = joblib.load(filename="saved_model5/DP.pkl")

    x = x.to(torch.float32)
    score = model(x)
    score = score.detach().numpy()

    # e=RERANKPRSM_upgrade_train.getevalue(filename)
    # e = e.reshape(-1, 1)
    # print('皮尔逊系数',np.corrcoef(score.ravel(),e)[0,1])
    # print(score)
    # scores = list()
    # for xx in score:
    #     scores.append(xx[1] - xx[0])
    # score = torch.tensor(scores)

    score = score.reshape(-1, 1)
    # print(np.log2(e))
    # score= score*5 - np.log10(e)

    # print(score)
    y = y.reshape(-1, 1)
    score_label = np.concatenate((score, y), axis=1)
    toprank = sorted(score_label, key=lambda x: x[0], reverse=True)
    ppp = np.array(toprank)
    fdr = 0
    target = 1
    decoy = 0
    top=0
    nixu=list()
    size=len(ppp)
    # fenshu=ppp[:,0]
    # zong=0
    # zong10=0
    # zong20=0

    # for i in fenshu:
    #     zong=zong+i
    # for i in range(int(len(fenshu)/10)):
    #     zong10=zong10+fenshu[i]
    # for i in range(int(len(fenshu)/5)):
    #     zong20 = zong20 + fenshu[i]
    #
    # print("总分数",zong)
    # print("10分数",zong10,"占比",zong10/zong)
    # print("20分数",zong20,"占比",zong20/zong)

    for i in ppp:
        k = 0
        for j in i:
            if (k == 1):
                if j == 1.0:
                    target = target + 1
                    fdr = decoy / (target)
                    nixu.append(fdr)
                else:
                    decoy = decoy + 1
                    fdr = decoy / (target)
                    nixu.append(fdr)
            else:
                k = k + 1
    nixu.reverse()
    for i in nixu:
        if i <= 0.01:
            top=size-1-nixu.index(i)

            break

    print(top)
    return str(top)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        x=x.to(torch.float32)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--arg1', type=str, help='参数1')
    # args = parser.parse_args()
    # #
    f=open('result.txt', 'a')





    f.close()


