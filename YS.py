import time
import os
import optuna
from networkx import degree_pearson_correlation_coefficient
from optuna.integration import SkoptSampler #skilearn——optimize
# import skopt
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from tensorflow._api.v2 import math
import matplotlib.pyplot as plt
from pylab import *
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
import warnings
from xgboost import XGBRegressor
warnings.filterwarnings('ignore')


org_x_train= []
org_x_test = []


def effictive_ratio(y_train,y_train_pred):
    # print('#'*20,"训练结果在误差范围内的占比(有效率）",'#'*20)
    y_train_in = []
    for i in range(int(np.array(y_train).shape[0])):
        if abs(y_train[i]-y_train_pred[i])<np.mean(y_train)*0.06:
            y_train_in.append(y_train_pred[i])
    # print('在误差范围内的数据',len(y_train_in))
    # print('在误差范围内的数据占比：%.4f'%(len(y_train_in)/len(y_train_pred)))
    data_sum = len(y_train_in)
    data_percent = (len(y_train_in)/len(y_train_pred))
    return data_percent
def epoch_time(start_time, end_time):
    elapsed_secs = end_time - start_time
    elapsed_mins = elapsed_secs / 60
    return elapsed_mins, elapsed_secs
def NormalizedData(data,random_sate):
    '''
    :param data: data
    :return: 归一化及划分数据后的数据
    '''
    xColumnList = [
        ## >>>>化学成分<<<<
        'C',
        'Si',
        'Mn',
        'P',
        'S',
        # 'Cu',
        # 'Ni',
        'Cr',
        # # 'As',
        # 'Al',
        # 'Nb',
        'V',
        # 'Ti',
        # 'Mo',
        # 'B',
        # 'W',
        # 'Ac1',
        # 'Ac3',
        # '1',
        # '2',
        # 'Co',
        'N',
        ## >>>>>轧制区<<<<<<<<<<
        'Mean extraction temperature',
        # 'Rolling thickness',
        'Roughing opening temperature',
        # 'Roughing final temperature',
        'intermediate billet thickness',
        'Finishing opening temperature',
        'Finishing rolling temperature',
        # 'Cooling inlet temperature',
        'Cooling outlet temperature',
        'Cooling rate',
    ]
    yColumnList = [
        ## >>>>>性能输出<<<<<<<<<<
        'YS',
        # 'TS',
        # 'EL',
    ]
    x = data[xColumnList]

    x_bak = x

    y = data[yColumnList]
    # 归一化
    XMinMaxScaler = StandardScaler()
    YMinMaxScaler = StandardScaler()
    x = XMinMaxScaler.fit_transform(x)
    y = YMinMaxScaler.fit_transform(y)
    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, shuffle=True,random_state=random_sate)

    global org_x_train,org_x_test


    org_x_train, org_x_test, org_y_train, org_y_test= train_test_split(x_bak, y, test_size=0.4, shuffle=True,random_state=random_sate)
    org_x_train = pd.DataFrame(org_x_train, columns=xColumnList)
    org_x_test = pd.DataFrame(org_x_test, columns=xColumnList)
    org_x_train.insert(len(org_x_train.columns), yColumnList[0], org_y_train)
    org_x_test.insert(len(org_x_test.columns), yColumnList[0], org_y_test)



    print('X_train_shape:%d,X_test_shape:%d,Y_train_shape:%d,Y_test_shape:%d' % (X_train.shape[0], X_test.shape[0],
                                                                                 y_train.shape[0], y_test.shape[0]))

    return X_train,X_test,y_train,y_test,XMinMaxScaler,YMinMaxScaler
def RandomModel(x_train,x_test,y_train,y_test,YMinMaxScaler):
    XGB_model = XGBRegressor()

    XGB_model.fit(x_train, y_train.ravel())                 # feed训练参数
    y_train_pred = XGB_model.predict(x_train)
    y_test_pred = XGB_model.predict(x_test)
    # 反归一化
    y_train = YMinMaxScaler.inverse_transform(y_train)
    y_train_pred = YMinMaxScaler.inverse_transform(y_train_pred.reshape(-1, 1))
    y_test = YMinMaxScaler.inverse_transform(y_test)
    y_test_pred = YMinMaxScaler.inverse_transform(y_test_pred.reshape(-1, 1))
    x_test1 = YMinMaxScaler.inverse_transform(x_test)
    x_train1= YMinMaxScaler.inverse_transform(x_train)
    y_train = y_train[:,0]
    y_test = y_test[:,0]
    # x_train2 = pd.DataFrame(x_train1)
    # x_test2 = pd.DataFrame(x_test1)
    # x_train1.to_csv(r'D:\硕士课题研究\硕士课题\数据预处理\2 缺失值处理\循环结果\XGB\x_train.csv', header=0, encoding="gbk")
    # x_test1.to_csv(r'D:\硕士课题研究\硕士课题\数据预处理\2 缺失值处理\循环结果\XGB\x_test.csv', header=0, encoding="gbk")
    df_train1 = pd.DataFrame(y_train)
    df_train_pred = pd.DataFrame(y_train_pred)

    org_x_train.insert(len(org_x_train.columns), 'EL_pre0', df_train_pred)
    org_x_train.to_csv(r'F:\硕士课题研究\硕士课题\数据预处理\2 缺失值处理\循环结果\XGB\YS\xgbmodel_train_with_orgx.csv', header=0, encoding="gbk")


    df_train1.to_csv(r'F:\硕士课题研究\硕士课题\数据预处理\2 缺失值处理\循环结果\XGB\YS\xgbmodel_train.csv', header=0, encoding="gbk")
    df_train_pred.to_csv(r'F:\硕士课题研究\硕士课题\数据预处理\2 缺失值处理\循环结果\XGB\YS\xgbmodel_trainpred.csv', header=0, encoding="gbk")
    df_test1 = pd.DataFrame(y_test)
    df_test_pred = pd.DataFrame(y_test_pred)
    df_test1.to_csv(r'F:\硕士课题研究\硕士课题\数据预处理\2 缺失值处理\循环结果\XGB\YS\xgbmodel_test.csv', header=0, encoding="gbk")
    df_test_pred.to_csv(r'F:\硕士课题研究\硕士课题\数据预处理\2 缺失值处理\循环结果\XGB\YS\xgbmodel_testpred.csv', header=0, encoding="gbk")

    org_x_test.insert(len(org_x_test.columns), 'EL_pre1', df_test_pred)
    org_x_test.to_csv(r'F:\硕士课题研究\硕士课题\数据预处理\2 缺失值处理\循环结果\XGB\YS\xgbmodel_test_with_orgx.csv', header=0, encoding="gbk")




    return    mean_absolute_error(y_train, y_train_pred),mean_absolute_error(y_test, y_test_pred), \
              math.sqrt(mean_squared_error(y_train, y_train_pred)),math.sqrt(mean_squared_error(y_test, y_test_pred)), \
              effictive_ratio(y_train, y_train_pred),effictive_ratio(y_test, y_test_pred), \
              r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred),
def model_optuna(X_train,X_test,y_train,y_test,YMinMaxScaler):
    def objective(trial):
        param = {
            # 'lambda': trial.suggest_loguniform('lambda',                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            1e-3, 10.0),
            # 'reg_alpha': trial.suggest_loguniform('reg_alpha', 1, 10.0),
            # 'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.1,0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            # 'subsample': trial.suggest_loguniform('subsample', 0.1,1),
            'learning_rate': trial.suggest_loguniform('learning_rate',
                                                      1e-3, 0.2),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 1,20),
            # 'max_depth': trial.suggest_categorical('max_depth', [5, 7, 9, 11, 13, 15, 17, 20]),
            # 'random_state': trial.suggest_categorical('random_state',(10,10, 100)),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
        }
        model = XGBRegressor(**param)
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_train1 = YMinMaxScaler.inverse_transform(y_train.reshape(-1, 1))
        y_train_pred= YMinMaxScaler.inverse_transform(y_train_pred.reshape(-1, 1))
        y_test1 = YMinMaxScaler.inverse_transform(y_test)
        y_test_pred = YMinMaxScaler.inverse_transform(y_test_pred.reshape(-1, 1))
        rmse = mean_squared_error(y_test1, y_test_pred, squared=False)
        return rmse

    optuna.logging.set_verbosity(optuna.logging.ERROR) # 关闭打印迭代过程
    # 模型训练
    algo1 = SkoptSampler(skopt_kwargs={'base_estimator': 'GP',  # 选择高斯过程
                                       'n_initial_points': 30,  # 初始观测点10个
                                       'acq_func': 'EI',} ) # 选择的采集函数为EI，期望增量
    algo2 = optuna.samplers.TPESampler(n_startup_trials=30, n_ei_candidates=24)  # 默认最开始有10个观测值，每一次计算采集函数随机抽取24组参数组合
    study = optuna.create_study(direction='minimize',sampler = algo2)
    n_trials=300
    study.optimize(objective, n_trials=n_trials,show_progress_bar=True)
    print('Number of finished trials:', len(study.trials))
    print("------------------------------------------------")
    print('Best trial:', study.best_trial.params)
    print("------------------------------------------------")
    print( 'best score:', study.best_trial.values)
    print("------------------------------------------------")
    print(study.trials_dataframe())
    print("------------------------------------------------")

    best_param = study.best_trial.params

    XGB_model = XGBRegressor(
        n_estimators=best_param['n_estimators'],
        max_depth=best_param['max_depth'],
        learning_rate=best_param['learning_rate'],
        min_child_weight=best_param['min_child_weight'],
        # subsample=best_param['subsample'],
        # colsample_bytree=best_param['colsample_bytree'],
        # reg_alpha=best_param['reg_alpha'],
        objective='reg:squarederror',)
    XGB_model.fit(X_train, y_train.ravel())  # feed训练参数
    joblib.dump(XGB_model, '../xgb.m')
    y_train_pred = XGB_model.predict(X_train)
    y_test_pred = XGB_model.predict(X_test)
    # 反归一化
    y_train = YMinMaxScaler.inverse_transform(y_train)
    y_train_pred = YMinMaxScaler.inverse_transform(y_train_pred.reshape(-1, 1))
    y_test = YMinMaxScaler.inverse_transform(y_test)
    y_test_pred = YMinMaxScaler.inverse_transform(y_test_pred.reshape(-1, 1))

    df_train1 = pd.DataFrame(y_train)
    df_train_pred = pd.DataFrame(y_train_pred)
    df_train1.to_csv(r'F:\硕士课题研究\硕士课题\数据预处理\2 缺失值处理\循环结果\XGB\YS\xgbtpe_train.csv', header=0, encoding="gbk")
    df_train_pred.to_csv(r'F:\硕士课题研究\硕士课题\数据预处理\2 缺失值处理\循环结果\XGB\YS\xgbtpe_trainpred.csv', header=0, encoding="gbk")

    org_x_train.insert(len(org_x_train.columns), 'EL_pre2', df_train_pred)
    org_x_train.to_csv(r'F:\硕士课题研究\硕士课题\数据预处理\2 缺失值处理\循环结果\XGB\YS\xgbtpe_train_with_orgx.csv', header=0, encoding="gbk")

    df_train_pred.to_csv(r'F:\硕士课题研究\硕士课题\数据预处理\2 缺失值处理\循环结果\XGB\YS\xgbtpe_trainpred.csv', header=0, encoding="gbk")
    df_test1 = pd.DataFrame(y_test)
    df_test_pred = pd.DataFrame(y_test_pred)
    df_test1.to_csv(r'F:\硕士课题研究\硕士课题\数据预处理\2 缺失值处理\循环结果\XGB\YS\xgbtpe_test.csv', header=0, encoding="gbk")
    df_test_pred.to_csv(r'F:\硕士课题研究\硕士课题\数据预处理\2 缺失值处理\循环结果\XGB\YS\xgbtpe_testpred.csv', header=0, encoding="gbk")

    org_x_test.insert(len(org_x_test.columns), 'EL_pre3', df_test_pred)
    org_x_test.to_csv(r'F:\硕士课题研究\硕士课题\数据预处理\2 缺失值处理\循环结果\XGB\YS\xgbtpe_test_with_orgx.csv', header=0, encoding="gbk")
    return best_param, \
           mean_absolute_error(y_train, y_train_pred),mean_absolute_error(y_test, y_test_pred), \
           math.sqrt(mean_squared_error(y_train, y_train_pred)),math.sqrt(mean_squared_error(y_test, y_test_pred)), \
           effictive_ratio(y_train, y_train_pred),effictive_ratio(y_test, y_test_pred), \
           r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred),

def main():
    list = []
    list0 = []
    for random_sate in range(21, 31, 10):
        data = pd.read_csv(r'F:\硕士课题研究\硕士课题\数据预处理\2 缺失值处理\DATA_JOM.csv', header=0, encoding="gbk")
        # x_train, x_test, y_train, y_test, XMinMaxScaler, YMinMaxScaler = NormalizedData(data, random_sate)
        x_train, x_test, y_train, y_test, XMinMaxScaler, YMinMaxScaler = NormalizedData(data, random_sate)
        # x = model_optuna(x_train, x_test, y_train, y_test, YMinMaxScaler)
        # list.append(x)
        x0 = RandomModel(x_train, x_test, y_train, y_test, YMinMaxScaler)
        list0.append(x0)
    # a = np.array(list)
    # df = pd.DataFrame(a)
    # df.to_csv(r'D:\硕士课题研究\硕士课题\数据预处理\2 缺失值处理\循环结果\XGB\YS\DATA_xgb_optuna_GP_el.csv', header=0, encoding="gbk")
    a0 = np.array(list0)
    df0 = pd.DataFrame(a0)
    df0.to_csv(r'F:\硕士课题研究\硕士课题\数据预处理\2 缺失值处理\DATA_111.csv', header=0, encoding="gbk")

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print('运行时间: %f 秒' % (end - start))
