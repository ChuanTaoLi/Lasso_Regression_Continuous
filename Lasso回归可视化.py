import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.preprocessing import StandardScaler
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

data = pd.read_excel(r"D:\0文献整理\网络入侵检测\KDD99\KDD_train.xlsx")
X = data.iloc[:, 0:-1]
y = data.iloc[:, -1]

from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score,mean_absolute_error
def lasso_regression(data,test,predictor,pre_y,alpha):
    lassoreg=Lasso(alpha=alpha, max_iter=100000, fit_intercept=False)
    lassoreg.fit(data[predictors],data[pre_y])
    y_pred = lassoreg.predict(test[predictors])
    ret = [alpha]
    ret.append(r2_score(test[pre_y],y_pred)) # R方
    ret.append(mean_absolute_error(test[pre_y],y_pred)) # 平均绝对值误差
    ret.extend(lassoreg.coef_) # 模型系数
    return ret

Lasso(alpha=1.0, fit_intercept=True, random_state=None, selection='cyclic')

X = pd.DataFrame(X)
predictors = list(X.columns)
# print(len(predictors))
prey = "label"
alpha_lasso = np.linspace(0.0001,2,200)
col = ["alpha","r2_score","mae"] + predictors
ind = ["alpha_%.2g" % alpha_lasso[i] for i in range(0,len(alpha_lasso))]
coef_matrix_lasso = pd.DataFrame(index=ind,columns=col)
np.random.seed(123456)
index = np.random.permutation(data.shape[0])
trainindex = index[0:350]
testindex = index[350:-1]
diabete_train = data.iloc[trainindex,:]
diabete_test = data.iloc[testindex,:]
for i in range(len(alpha_lasso)):
    coef_matrix_lasso.iloc[i,] = lasso_regression(diabete_train,diabete_test,predictors,prey,alpha_lasso[i])
coef_matrix_lasso.sort_values("mae").head(5)

ploty = predictors

plt.figure(1,figsize=(14,6.8))
for ii in np.arange(len(ploty)):
    plt.plot(coef_matrix_lasso["alpha"],coef_matrix_lasso[ploty[ii]],color=plt.cm.Set1(ii/len(ploty)),label=ploty[ii])
    plt.legend(loc="upper right",ncol=3)
    plt.xlabel("Alpha",fontsize=14)
    plt.ylabel("系数",fontsize=14)
    plt.title('各特征系数关于Alpha的变化曲线',fontsize=16)
plt.savefig(r'D:\0文献整理\网络入侵检测\KDD99\各特征系数关于Alpha的变化曲线.png',dpi=600)
plt.show()

plt.figure(2,figsize=(14,6.8))
plt.plot(coef_matrix_lasso["alpha"],coef_matrix_lasso["mae"],linewidth=2,marker='.')
plt.xlabel("Alpha",fontsize=14)
plt.ylabel("绝对值误差",fontsize=14)
plt.suptitle("Lasso回归误差曲线",fontsize=16)
plt.savefig(r'D:\0文献整理\网络入侵检测\KDD99\Lasso回归误差曲线.png',dpi=600)
plt.show()
