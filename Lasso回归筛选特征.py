import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.preprocessing import StandardScaler
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from sklearn.linear_model import Lasso
data = pd.read_excel(r"D:\0文献整理\网络入侵检测\KDD99\KDD_train.xlsx")
from statsmodels.stats.outliers_influence import variance_inflation_factor

'''计算VIF函数'''
def calculate_vif(df):
    vif = pd.DataFrame()
    vif['index'] = df.columns
    vif['VIF'] = [variance_inflation_factor(df.values,i) for i in range(df.shape[1])]
    return vif

X = data.iloc[:,0:-1]
y = data.iloc[:,-1]
lasso_model = Lasso(alpha=0.055, fit_intercept=True,
                   precompute=False, copy_X=True,
                   max_iter=10000, tol=0.0001, warm_start=False,
                   positive=False, random_state=None, selection='cyclic')
lasso_model.fit(X, y)  # X 和 y 是你的特征和目标变量
selected_features = X.columns[lasso_model.coef_ != 0]
print(selected_features)

'''计算Lasso前的VIF'''
vif_before = pd.DataFrame()
vif_before["变量"] = X.columns
vif_before["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("筛选特征前的VIF:")
print(vif_before)

'''计算筛选特征后的VIF'''
X_train_selected = X[selected_features]  # 使用筛选出的特征进行切片
vif_after = pd.DataFrame()
vif_after["变量"] = selected_features
vif_after["VIF"] = [variance_inflation_factor(X_train_selected.values, i) for i in range(X_train_selected.shape[1])]
print("筛选特征后的VIF:")
print(vif_after)

vif_before.to_excel(r'D:\0文献整理\网络入侵检测\KDD99\vif_before.xlsx',index=False)
vif_after.to_excel(r'D:\0文献整理\网络入侵检测\KDD99\vif_after_alpha0.055.xlsx',index=False)
