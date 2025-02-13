import subprocess
import xml.dom.minidom
import random

from torch import optim
from torch.utils.data import Dataset
import joblib
import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
import math

seed = 8256
# seed=random.randint(1, 10000)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.set_printoptions(threshold=np.inf)
# 超参数设置
input_size = 8 # 输入特征大小：11
hidden_size = 64  # 隐藏层大小
learning_rate = 0.001  # 学习率
num_classes = 1  # 类别数（0-9）
num_epochs = 50 # 训练轮数
batch_size = 100  # 批大小
num_blocks = 9
cardinality = 8
trainfile="input1.6.2/train/FB_CB_1_ms2_toppic_prsm.xml"
flag=5
# trainfile2="input1.6.2/train/20150927_BM_sort_2E6_CD19_highCD10_techrep01_ms2_toppic_prsm.xml"
# trainfile3="input1.6.2/train/20150927_BM_sort_2E6_CD19_highCD10_techrep02_ms2_toppic_prsm.xml"
# trainfile4="input1.6.2/train/20150927_BM_sort_2E6_CD19_highCD10_techrep03_ms2_toppic_prsm.xml"
# trainfile5="input1.6.2/train/20150927_BM_sort_2E6_CD19_highCD10_techrep04_ms2_toppic_prsm.xml"
trainfile2="input1.6.2/train/train1/20150927_BM_sort_2E6_CD19_highCD10_techrep01_ms2_toppic_prsm.xml"
trainfile3="input1.6.2/train/train1/20150927_BM_sort_2E6_CD19_highCD10_techrep02_ms2_toppic_prsm.xml"
trainfile4="input1.6.2/train/train1/20150927_BM_sort_2E6_CD19_highCD10_techrep03_ms2_toppic_prsm.xml"
trainfile5="input1.6.2/train/train1/20150927_BM_sort_2E6_CD19_highCD10_techrep04_ms2_toppic_prsm.xml"
trainfile6="input1.6.2/train/FB_CB_2_ms2_toppic_prsm.xml"
trainfile7="input1.6.2/train/FB_CB_4_ms2_toppic_prsm.xml"
trainfile9="input1.6.2/train/20150930_TDQ_BC_Memory_techrep_01_ms2_toppic_prsm.xml"
trainfile10="input1.6.2/train/20150930_TDQ_BC_Memory_techrep_02_ms2_toppic_prsm.xml"
trainfile11="input1.6.2/train/20150930_TDQ_BC_Memory_techrep_03_ms2_toppic_prsm.xml"
trainfile12="input1.6.2/train/20150930_TDQ_BC_Memory_techrep_04_ms2_toppic_prsm.xml"
trainfile13="input1.6.2/train/20150930_TDQ_BC_Memory_techrep_05_ms2_toppic_prsm.xml"

def getevalue(name):
    dom = xml.dom.minidom.parse(name)
    e_value = dom.getElementsByTagName("e_value")  # 获取节点列表
    c=list()
    for i in range(len(e_value)):  # 打印节点数据
        c.append(float(e_value[i].firstChild.data))
    return np.array(c)

def getevalue_and_fdr(name):
    dom = xml.dom.minidom.parse(name)
    e_value = dom.getElementsByTagName("e_value")  # 获取节点列表
    fdr = dom.getElementsByTagName("fdr")  # 获取节点列表
    c=list()
    for i in range(len(e_value)):  # 打印节点数据
        c.append([float(e_value[i].firstChild.data),float(fdr[i].firstChild.data)])

    return np.array(c)

def inputfile_np_withSeq(name):
    dom = xml.dom.minidom.parse(name)
    match_peak_num = dom.getElementsByTagName("match_peak_num")  # 匹配峰数
    match_fragment_num = dom.getElementsByTagName("match_fragment_num")  # 匹配离子数
    norm_match_fragment_num = dom.getElementsByTagName("norm_match_fragment_num")  # 匹配离子数
    p_value = dom.getElementsByTagName("p_value")  # 获取节点列表
    # e_value= dom.getElementsByTagName("e_value")
    # frac_feature_score = dom.getElementsByTagName("frac_feature_score")
    # variable_ptm_num = dom.getElementsByTagName("variable_ptm_num")
    # unexpected_ptm_num = dom.getElementsByTagName("unexpected_ptm_num")
    e_value = dom.getElementsByTagName("e_value")  # 获取节点列表
    ori_prec_mass = dom.getElementsByTagName("ori_prec_mass")
    seq_name = dom.getElementsByTagName("seq_name")  # 获取节点列表
    prsm_id=dom.getElementsByTagName("prsm_id")
    # spectrum_id = dom.getElementsByTagName("spectrum_id")  # 获取节点列表
    target=0
    decoy=0
    # for i in range(len(spectrum_id)):
    #     print(spectrum_id[i].firstChild.data)
    logits = list()
    label = list()
    prsmid=list()
    for i in range(len(ori_prec_mass)):  # 打印节点数据
        pe = math.log10(float(e_value[i].firstChild.data))
        logits.append([
            float(match_peak_num[i].firstChild.data) / 34,
            float(match_fragment_num[i].firstChild.data) / 24,
            float(norm_match_fragment_num[i].firstChild.data),
            float(p_value[i].firstChild.data),
            # float(sample_feature_inte[i].firstChild.data),
            # float(frac_feature_score[i].firstChild.data),
            # float(variable_ptm_num[i].firstChild.data),     #单加这个好像没问题
            # float(ori_prec_mass[i].firstChild.data),
            # float(unexpected_ptm_num[i].firstChild.data),
            # pe,
            # math.log10(float(e_value[i].firstChild.data))
        ])
        prsmid.append(int(prsm_id[i].firstChild.data))
    #float(ori_prec_mass[i].firstChild.data)/17000,
    #     # logits.append([float(ori_prec_mass[i].firstChild.data) / 10000, float(match_peak_num[i].firstChild.data) / 34,
    #     #                float(match_fragment_num[i].firstChild.data) / 24])
        if "DECOY_" in seq_name[i].firstChild.data:
            label.append(0)
            decoy=decoy+1
        else:
            label.append(1)
            target=target+1
    print("filename:",name,"target:" ,target,"   decoy:",decoy)
    return np.array(logits), np.array(label),np.array(prsmid)
def inputfile_np(name):
    dom = xml.dom.minidom.parse(name)
    match_peak_num = dom.getElementsByTagName("match_peak_num")  # 匹配峰数
    match_fragment_num = dom.getElementsByTagName("match_fragment_num")  # 匹配离子数
    norm_match_fragment_num = dom.getElementsByTagName("norm_match_fragment_num")  # 匹配离子数
    p_value = dom.getElementsByTagName("p_value")  # 获取节点列表
    # e_value= dom.getElementsByTagName("e_value")
    frac_feature_score = dom.getElementsByTagName("frac_feature_score")
    variable_ptm_num = dom.getElementsByTagName("variable_ptm_num")
    unexpected_ptm_num = dom.getElementsByTagName("unexpected_ptm_num")
    e_value = dom.getElementsByTagName("e_value")  # 获取节点列表
    ori_prec_mass = dom.getElementsByTagName("ori_prec_mass")
    seq_name = dom.getElementsByTagName("seq_name")  # 获取节点列表
    spectrum_id = dom.getElementsByTagName("spectrum_id")  # 获取节点列表
    target=0
    decoy=0
    # for i in range(len(spectrum_id)):
    #     print(spectrum_id[i].firstChild.data)
    logits = list()
    label = list()
    for i in range(len(ori_prec_mass)):  # 打印节点数据
        pe = math.log10(float(e_value[i].firstChild.data))
        logits.append([
            float(match_peak_num[i].firstChild.data) / 34,
            float(match_fragment_num[i].firstChild.data) / 24,
            float(norm_match_fragment_num[i].firstChild.data),
            float(p_value[i].firstChild.data),
            # float(sample_feature_inte[i].firstChild.data),
            # float(frac_feature_score[i].firstChild.data),
            # float(variable_ptm_num[i].firstChild.data),     #单加这个好像没问题
            # float(ori_prec_mass[i].firstChild.data),
            # float(unexpected_ptm_num[i].firstChild.data),
            # pe,
            # math.log10(float(e_value[i].firstChild.data))
        ])
    #float(ori_prec_mass[i].firstChild.data)/17000,
    #     # logits.append([float(ori_prec_mass[i].firstChild.data) / 10000, float(match_peak_num[i].firstChild.data) / 34,
    #     #                float(match_fragment_num[i].firstChild.data) / 24])
        if "DECOY_" in seq_name[i].firstChild.data:
            label.append(0)
            decoy=decoy+1
        else:
            label.append(1)
            target=target+1
    print("filename:",name,"target:" ,target,"   decoy:",decoy)
    return np.array(logits), np.array(label)

def input1(name):
    a,b=inputfile_np(name)
    m1=joblib.load(filename="saved_model5/"+"LR.pkl")
    m2 = joblib.load(filename="saved_model5/"+"DecTree.pkl")
    m3 = joblib.load(filename="saved_model5/"+"svm.pkl")
    m4 = joblib.load(filename="saved_model5/"+"xgboost.pkl")
    LR_pred=m1.predict(a).reshape(-1,1)
    x1 = m2.predict_proba(a)[:, 1].reshape(-1, 1)
    x2 = m3.predict_proba(a)[:, 0].reshape(-1, 1)
    x3 = m4.predict_proba(a)[:, 1].reshape(-1, 1)
    X = np.concatenate((a, LR_pred, x1, x2, x3), axis=1)
    # X=np.concatenate((a, LR_pred,m2.predict_proba(a),m3.predict_proba(a),m4.predict_proba(a)), axis=1)
    # X = np.concatenate((a, LR_pred, m2.predict_proba(a), m3.predict_proba(a)), axis=1)
    Y=torch.tensor(b).long()

    return torch.tensor(X),Y
class myDataset(Dataset):
    def __init__(self):
        # # 创建5*2的数据集
        # self.data = torch.tensor([[1, 2], [3, 4], [2, 1], [3, 4], [4, 5]])
        # # 5个数据的标签
        # self.label = torch.tensor([0, 1, 0, 1, 2])
        x,y=input1(trainfile)
        if flag==1:
            x2, y2 = input1(trainfile2)
            x = np.concatenate((x, x2), axis=0)
            y = np.concatenate((y, y2), axis=0)
        if flag==2:
            x2, y2 = input1(trainfile2)
            x3, y3 = input1(trainfile3)
            x = np.concatenate((x, x2,x3), axis=0)
            y = np.concatenate((y, y2,y3), axis=0)
        if flag==3:
            x2, y2 = input1(trainfile2)
            x3, y3 = input1(trainfile3)
            x4, y4 = input1(trainfile4)
            x = np.concatenate((x, x2,x3,x4), axis=0)
            y = np.concatenate((y, y2,y3,y4), axis=0)
        if flag==4:
            x2, y2 = input1(trainfile2)
            x3, y3 = input1(trainfile3)
            x4, y4 = input1(trainfile4)
            x5, y5 = input1(trainfile5)
            x = np.concatenate((x, x2,x3,x4,x5), axis=0)
            y = np.concatenate((y, y2,y3,y4,y5), axis=0)
        if flag==5:
            # x2, y2 = input1(trainfile2)
            # x3, y3 = input1(trainfile3)
            # x4, y4 = input1(trainfile4)
            # x5, y5 = input1(trainfile5)
            # x6, y6 = input1(trainfile6)
            # x7, y7 = input1(trainfile7)
            # x = np.concatenate((x, x2,x3,x4,x5,x6,x7), axis=0)
            # y = np.concatenate((y, y2,y3,y4,y5,y6,y7), axis=0)
            x2, y2 = input1(trainfile2)
            x3, y3 = input1(trainfile3)
            x4, y4 = input1(trainfile4)
            x5, y5 = input1(trainfile5)
            x6, y6 = input1(trainfile6)
            x7, y7 = input1(trainfile7)
            x9, y9 = input1(trainfile9)
            x10, y10 = input1(trainfile10)
            x11, y11 = input1(trainfile11)
            x12, y12 = input1(trainfile12)
            x13, y13 = input1(trainfile13)
            x = np.concatenate((x,x2, x3, x4, x5,x6,x7,x9,x10,x11,x12,x13), axis=0)
            y = np.concatenate((y,y2, y3, y4, y5,y6,y7,y9,y10,y11,y12,y13), axis=0)
        self.data, self.label = x, y

    # 根据索引获取data和label
    def __getitem__(self, index):
        return self.data[index], self.label[index]  # 以元组的形式返回

    # 获取数据集的大小
    def __len__(self):
        return len(self.data)
# 定义神经网络模型

class ResidualBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(output_size, output_size)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out += residual  # Add residual connection
        return out



class ResNeXtBlock(nn.Module):
    def __init__(self, in_features, out_features, cardinality=32):
        super(ResNeXtBlock, self).__init__()
        self.cardinality = cardinality
        self.group_width = out_features // self.cardinality

        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.fc3 = nn.Linear(out_features, out_features)

        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
            )

    def forward(self, x):
        residual = x

        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)

        out += self.shortcut(residual)
        return out
# input_size = 10 # 输入特征大小：11
# hidden_size = 64  # 隐藏层大小
# learning_rate = 0.001  # 学习率
# num_classes = 1  # 类别数（0-9）
# num_epochs = 30  # 训练轮数
# batch_size = 100  # 批大小
# num_blocks = 9
# cardinality = 8
class ResNeXt(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_blocks, cardinality=8):
        super(ResNeXt, self).__init__()
        self.fc_input = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()

        self.residual_blocks = nn.ModuleList([
            ResNeXtBlock(hidden_size, hidden_size, cardinality) for _ in range(num_blocks)
        ])

        self.fc_output = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.fc_input(x)
        x = self.relu(x)

        for block in self.residual_blocks:
            x = block(x)

        x = self.fc_output(x)
        return torch.sigmoid(x)

# Create an instance of the ResNeXt model


# model = ResNeXt(input_size, hidden_size, num_classes, num_blocks, cardinality)


def train_model(sfilename):
    x, y = inputfile_np(sfilename)
    if flag == 1:
        x2, y2 = inputfile_np(trainfile2)
        x = np.concatenate((x, x2), axis=0)
        y = np.concatenate((y, y2), axis=0)
    if flag == 2:
        x2, y2 = inputfile_np(trainfile2)
        x3, y3 = inputfile_np(trainfile3)
        x = np.concatenate((x, x2, x3), axis=0)
        y = np.concatenate((y, y2, y3), axis=0)
    if flag == 3:
        x2, y2 = inputfile_np(trainfile2)
        x3, y3 = inputfile_np(trainfile3)
        x4, y4 = inputfile_np(trainfile4)
        x = np.concatenate((x, x2, x3, x4), axis=0)
        y = np.concatenate((y, y2, y3, y4), axis=0)
    if flag == 4:
        x2, y2 = inputfile_np(trainfile2)
        x3, y3 = inputfile_np(trainfile3)
        x4, y4 = inputfile_np(trainfile4)
        x5, y5 = inputfile_np(trainfile5)
        x = np.concatenate((x, x2, x3, x4, x5), axis=0)
        y = np.concatenate((y, y2, y3, y4, y5), axis=0)
    if flag == 5:
        # x2, y2 = inputfile_np(trainfile2)
        # x3, y3 = inputfile_np(trainfile3)
        # x4, y4 = inputfile_np(trainfile4)
        # x5, y5 = inputfile_np(trainfile5)
        # x6, y6 = inputfile_np(trainfile6)
        # x7, y7 = inputfile_np(trainfile7)
        # x = np.concatenate((x, x2, x3, x4, x5, x6, x7), axis=0)
        # y = np.concatenate((y, y2, y3, y4, y5, y6, y7), axis=0)

        x2, y2 = inputfile_np(trainfile2)
        x3, y3 = inputfile_np(trainfile3)
        x4, y4 = inputfile_np(trainfile4)
        x5, y5 = inputfile_np(trainfile5)
        x6, y6 = inputfile_np(trainfile6)
        x7, y7 = inputfile_np(trainfile7)
        x9, y9 = inputfile_np(trainfile9)
        x10, y10 = inputfile_np(trainfile10)
        x11, y11 = inputfile_np(trainfile11)
        x12, y12 = inputfile_np(trainfile12)
        x13, y13 = inputfile_np(trainfile13)
        x = np.concatenate((x,x2, x3, x4, x5, x6,x7,x9, x10, x11, x12, x13), axis=0)
        y = np.concatenate((y,y2, y3, y4, y5, y6,y7,y9, y10, y11, y12, y13), axis=0)
    print("使用小模型数据集",sfilename)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=1024)  # 数据集分割
    # 训练模型
    model = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=160, silent=True)
    model.fit(X_train, y_train)
    # ycx, ycy = inputfile_np("input/RM_20190804_TCells_CD19CD20_F3F4_01_ms2_toppic_prsm.xml")
    # # 对测试集进行预测
    # # yc_pred2 = model.predict(ycx)
    joblib.dump(model, "saved_model5/"+"xgboost.pkl")
    print("保存xgboost")
    y = y.ravel()
    tree_clf = DecisionTreeClassifier(max_depth=4)
    # 构建决策树
    tree_clf.fit(X_train, y_train)
    joblib.dump(tree_clf, "saved_model5/"+"DecTree.pkl")
    print("保存决策树")
    svm = SVC(kernel='poly', C=0.1, gamma=1,  probability=True,max_iter=10000)
    # 训练模型
    svm.fit(X_train, y_train)
    joblib.dump(svm, "saved_model5/"+"svm.pkl")
    print("保存SVM")
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    joblib.dump(linreg, "saved_model5/"+"LR.pkl")
    print("保存线性回归")
    data = myDataset()
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
    # 创建模型实例

    # model =TenLayerNet(input_size, hidden_size, num_classes)
    # model = ResidualNetwork(input_size, hidden_size, num_classes, num_blocks)



    model = ResNeXt(input_size, hidden_size, num_classes, num_blocks, cardinality)
    # 定义损失函数和优化器
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
    #criterion =nn.HingeEmbeddingLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 训练模型

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for inputs, labels in train_loader:  # Iterate over your dataloader
            optimizer.zero_grad()  # Zero the gradients
            labels = labels.reshape(-1, 1)

            outputs = model(inputs)  # Forward pass

            outputs = outputs.to(torch.float32)
            labels = labels.to(torch.float32)
            # print(outputs)
            # print(outputs.shape)
            # print(labels.shape)

            loss = criterion(outputs, labels)  # Compute loss

            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {epoch_loss:.4f}")
    joblib.dump(model, "saved_model5/"+"DP.pkl")
    print("保存DP模型")
    f = open('result.txt', 'a')
    f.write('\n')
    f.write("随机数"+str(seed))
    f.close()
    # args = ['python', 'RERANKPRSM_upgrade_Predict.py', '--arg1', str(i)]
    # subprocess.run(args)
if __name__ == '__main__':
    # for i in range(2):
    print( "随机数"+str(seed))
    train_model(trainfile)
    print("模型训练完毕！")
