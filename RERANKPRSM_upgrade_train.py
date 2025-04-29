import subprocess
import xml.dom.minidom
import random
import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
import math
from torch import optim
from torch.utils.data import Dataset
import joblib

from AdaptiveRankLoss import AdaptiveRankLoss

seed = 8256
# seed=random.randint(1, 10000)
torch.manual_seed(seed)
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
        self.fc_output = nn.Linear(hidden_size, 1)
        # 添加可学习的gamma参数，初始值为0.2，并约束非负
        self.gamma = nn.Parameter(torch.tensor(0.2), requires_grad=True)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.fc_input(x)
        x = self.relu(x)
        for block in self.residual_blocks:
            x = block(x)
        return torch.sigmoid(self.fc_output(x)).squeeze(-1)

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

    # 初始化训练数据
    data = myDataset()
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)

    # 初始化last epoch数据
    all_y = []
    for _, y in train_loader:
        all_y.extend(y.numpy())
    torch.rand(len(all_y)).float().cpu()
    # last_epoch_y_t = torch.tensor(all_y).float().cpu()

    # 初始化模型
    model = ResNeXt(input_size, hidden_size, num_classes, num_blocks, cardinality).cpu()

    # 配置优化器
    optimizer = optim.Adam([
        {'params': [p for p in model.parameters() if p is not model.gamma], 'lr': 0.001},
        {'params': [model.gamma], 'lr': 1e-5}
    ])

    # 实例化 AdaptiveRankLoss 类 (使用默认参数)
    criterion = AdaptiveRankLoss()  # 你也可以在这里传入自定义参数，如 criterion = AdaptiveRankLoss(huber_delta=0.8)

    # 使用所有初始标签初始化损失函数的历史状态
    criterion.initialize_history(all_y)  # # 确保 all_initial_labels 是列表、Numpy数组或Tensor

    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        epoch_y_pred = []
        epoch_y_t = []

        for inputs, labels in train_loader:
            inputs = inputs.float().cpu()
            labels = labels.float().cpu()

            optimizer.zero_grad()
            outputs = model(inputs)

            # --- 修改这里 ---
            # 使用实例化的 criterion 对象计算损失
            loss = criterion(
                current_scores=outputs,
                current_labels=labels,
                margin=model.gamma  # 直接传递 gamma 值
            )
            # --- 修改结束 ---

            loss.backward()
            optimizer.step()

            # 约束gamma非负
            model.gamma.data = torch.clamp(model.gamma.data, min=0.0)

            # 收集数据
            epoch_y_pred.extend(outputs.detach().cpu().numpy())
            epoch_y_t.extend(labels.detach().cpu().numpy())

        # --- 在 epoch 循环的末尾 ---
        all_preds_tensor = torch.tensor(epoch_y_pred).float().cpu()  # 确保是 Tensor
        all_labels_tensor = torch.tensor(epoch_y_t).float().cpu()  # 确保是 Tensor

        # 调用新方法更新损失函数的内部状态
        criterion.update_history(all_preds_tensor, all_labels_tensor)
        current_auc = roc_auc_score(all_labels_tensor.numpy(), all_preds_tensor.numpy())  # 计算 AUC

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] AUC: {current_auc:.4f}, Gamma: {model.gamma.item():.4f}, Loss: {loss.item():.4f}")

    joblib.dump(model, "saved_model5/" + "DP.pkl")
    print("保存DP模型")
    f = open('result.txt', 'a')
    f.write('\n')
    f.write("随机数" + str(seed))
    f.close()
    # args = ['python', 'RERANKPRSM_upgrade_Predict.py', '--arg1', str(i)]
    # subprocess.run(args)

if __name__ == '__main__':
    # for i in range(2):
    print( "随机数"+str(seed))
    train_model(trainfile)
    print("模型训练完毕！")
