import pandas as pd
import lightgbm as lgb
import numpy as np
import torch

from tcn_model import TCN
from torch import nn
from torch import optim
import random


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def get_data(path):
    d = pd.read_csv(path)
    return d.drop(d.columns[0], axis=1)

def create_data_frames():
    data = []

    for i in range(1, 6):
        tempdata = get_data("sample_data/user_000"+ str(i) + "/user_000" + str(i) + "_highway.csv")
        list_df = np.array_split(tempdata, 5)
        for j in range(0, len(list_df)):
            data.append((list_df[j], i))

    return data

def getPositive(currLabel, data, dataFrame):
    foundPositive = False
    while not foundPositive:
        randomFrame, newLabel = random.choice(data)
        #If dataframe is positive but it is not the same
        if newLabel == currLabel and not randomFrame.equals(dataFrame):
            return randomFrame


def getNegative(currLabel, data):
    foundNegative = False
    while not foundNegative:
        randomFrame, newLabel = random.choice(data)
        if newLabel != currLabel:
            return randomFrame


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    data = create_data_frames()

    model = TCN(38, True, 200, 32, 16,'25,25,25,25,25,25,25,25', 19)
    # model = TCN(39, False, 200, 5, 16, '32,32,32,32,32,32,32,32', 15)
    criterion = nn.TripletMarginLoss(margin=1, p=2)
    optimizer = optim.Adam(model.parameters(), lr=4E-4, weight_decay=0.975, amsgrad=True)
    epochs = 2
    #Train for n epochs
    for i in range(epochs):
        #Loop through each data point
        for df, j in data:
            currLabel = j
            #Get random
            positive = getPositive(currLabel, data, df)
            negative = getNegative(currLabel, data)

            optimizer.zero_grad()
            #input data
            temp = torch.tensor(df.to_numpy())
            temp2 = torch.reshape(temp, (200, 1, 38))
            model = model.double()
            output, d = model(temp2, df.to_numpy())

            loss = criterion(output, positive, negative)
            loss.backward()
            optimizer.step()

    PATH = './trainedTCNmodel.pth'
    torch.save(model.state_dict(), PATH)

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
