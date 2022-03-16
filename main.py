import pandas as pd
import lightgbm as lgb
import numpy as np


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def get_data(path):
    return pd.read_csv(path)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    user1Highw = get_data("sample_data/user_0001/user_0001_highway.csv")
    print(user1Highw.head())
    print(user1Highw.info())

    data = np.random.rand(500, 10)  # 500 entities, each contains 10 features
    label = np.random.randint(2, size=500)  # binary target
    train_data = lgb.Dataset(data, label=label)

    param = {'num_leaves': 31, 'objective': 'binary'}
    param['metric'] = 'auc'

    num_round = 10
    # bst = lgb.train(param, train_data, num_round, valid_sets=[validation_data])
    bst = lgb.train(param, train_data, num_round)
    bst.save_model('model.txt')

    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
