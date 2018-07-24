# encoding=utf-8
import GMM_model as model
import pandas as pd 
import numpy as np 
import os 
import time 
import pickle 



# 众包数据
def load_data(filename):
    df = pd.read_csv(filename, encoding='gbk')
    clr_df = df.dropna(axis=0, how='any')
    tp_worker = clr_df.drop_duplicates(subset=['related_worker_id'], keep='first')['related_worker_id'].tolist()
    print('region workers ', len(tp_worker))
    n_classes = len(tp_worker)

    data = []
    for worker in tp_worker:
        df_worker = clr_df[clr_df['related_worker_id']==worker]
        for i in range(len(df_worker)):
            col = []
            # col.append(df_worker.iloc[i]['business_type'])
            col.append(df_worker.iloc[i]['related_region_id'])
            col.append(df_worker.iloc[i]['latitude'])
            col.append(df_worker.iloc[i]['longitude'])
            # col.append(df_worker.iloc[i]['access_type'])
            # col.append(df_worker.iloc[i]['wband_rate'])
            # col.append(df_worker.iloc[i]['action_type'])
            # col.append(df_worker.iloc[i]['gatewayType'])
            col.append(worker)
            data.append(col)
    n_columns = len(col)
    
    ret_data = np.mat(data).T
    ret_data = ret_data.flatten().A
    ret_data = ret_data.reshape(n_columns, -1)
    print(ret_data.shape)

    return ret_data, n_classes, tp_worker



# 加载训练好的参数
def load_parameters(filename):
    with open(filename, 'rb') as f:
        K, phi, mu, sigma = pickle.load(f)
    return K, phi, mu, sigma


def train(fileName, save_pb):
    input_data, K, tp_worker = load_data(fileName)

    gmm=model.GMM(K)                 #指定高斯函数的个数
    phi,mu,sigma,his=gmm.train(input_data[:-1, :])

    gmm.save_parameters(save_pb)
    print(phi,mu,sigma)


def predict(fileName, save_pb):
    input_data, n_classes, tp_worker = load_data(fileName)
    k, phi, mu, sigma = load_parameters(save_pb)
    gmm = model.GMM(K=k, phi=phi, mu=mu, sigma=sigma)

    print('tp_work ', tp_worker)
    errnum = 0
    for i in range(input_data.shape[1]):
        index = gmm.predict(np.mat(input_data[:-1,i]).T)
        if tp_worker[index] != input_data[-1,i]:
            errnum += 1
            print('predict worker {}, true worker {}'.format(tp_worker[index], input_data[-1,i]))
    print('error rate: ', errnum/input_data.shape[1])


if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] ="0"
    start_time = time.time()
    fileName = './train_500.csv'
    save_pb = './pb/sleep.pb'

    # train data
    # train(fileName, save_pb)

    # predict 
    predict(fileName, save_pb)

    end_time = time.time()
    print('time consuming ', end_time - start_time)
