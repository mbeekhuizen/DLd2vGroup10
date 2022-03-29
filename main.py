import pandas as pd
import lightgbm as lgb
import numpy as np
import torch

from tcn_model import TCN
from torch import nn
from torch import optim
import random
import sys
import pickle as pkl
import copy


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def get_data(path):
    d = pd.read_csv(path)
    return d.drop(d.columns[0], axis=1)


def create_data_frames():
    data = []

    for i in range(1, 6):
        tempdata = get_data("sample_data/user_000" + str(i) + "/user_000" + str(i) + "_highway.csv")
        list_df = np.array_split(tempdata, 5)
        for j in range(0, len(list_df)):
            data.append((list_df[j], i))

    return data


def create_data_frame_suburban():
    data = []

    for i in range(1, 6):
        tempdata = get_data("sample_data/user_000" + str(i) + "/user_000" + str(i) + "_suburban.csv")
        list_df = np.array_split(tempdata, 5)
        for j in range(0, len(list_df)):
            data.append((list_df[j], i))

    return data

# data of driver 3 in tutorial is 323, using Gaussian noise on the samples for data augmentation
# samples for driver 3,  from 323 increases to 1000
def create_data_frame_tutorial():
    data = []

    for i in range(1, 6):
        tempdata = get_data("sample_data/user_000" + str(i) + "/user_000" + str(i) + "_tutorial.csv")

        if i == 3:
            list_df = np.split(tempdata, [200, 323])
            for j in range(0, len(list_df)):
                data.append((list_df[j], i))

            copy_1 = copy.deepcopy(list_df[0])
            copy_2 = copy.deepcopy(list_df[0])
            copy_3 = copy.deepcopy(list_df[0])
            copy_4 = copy.deepcopy(list_df[0])

            # gaussian noise for data augmentation
            # guassian distribution
            mu, sigma = 0, 0.1
            noise_1 = np.random.normal(mu, sigma, 7600)
            noise_2 = np.random.normal(mu, sigma, 7600)
            noise_3 = np.random.normal(mu, sigma, 7600)
            noise_4 = np.random.normal(mu, sigma, 7600)

            noice_matrix_1 = np.reshape(noise_1, (-1, 38))
            noice_matrix_2 = np.reshape(noise_2, (-1, 38))
            noice_matrix_3 = np.reshape(noise_3, (-1, 38))
            noice_matrix_4 = np.reshape(noise_4, (-1, 38))

            copy_1 = copy_1 + noice_matrix_1
            copy_2 = copy_2 + noice_matrix_2
            copy_3 = copy_3 + noice_matrix_3
            copy_4 = copy_4 + noice_matrix_4
            temp = np.split(copy_4, [77, 200])[0]

            last = np.concatenate((list_df[1], temp), axis=0)

            list_df_new = []
            list_df_new.append(list_df[0])
            list_df_new.append(copy_1)
            list_df_new.append(copy_2)
            list_df_new.append(copy_3)
            list_df_new.append(last)

            for j in range(0, len(list_df_new)):
                data.append((list_df_new[j], i))

        else:
            list_df = np.array_split(tempdata, 5)
            for j in range(0, len(list_df)):
                data.append((list_df[j], i))

    return data


def create_data_frame_urban():
    data = []

    for i in range(1, 6):
        tempdata = get_data("sample_data/user_000" + str(i) + "/user_000" + str(i) + "_urban.csv")
        list_df = np.array_split(tempdata, 5)
        for j in range(0, len(list_df)):
            data.append((list_df[j], i))

    return data

def getPositive(currLabel, data, dataFrame):
    foundPositive = False
    while not foundPositive:
        randomFrame, newLabel = random.choice(data)
        # If dataframe is positive but it is not the same
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

    if sys.argv[1] == "train":
        # model = TCN(38, True, 200, 32, 16, '25,25,25,25,25,25,25,25', 19)
        # # model = TCN(39, False, 200, 5, 16, '32,32,32,32,32,32,32,32', 15)
        # criterion = nn.TripletMarginLoss(margin=1, p=2)
        # optimizer = optim.Adam(model.parameters(), lr=4E-4, weight_decay=0.975, amsgrad=True)
        # epochs = 2
        # # Train for n epochs
        # for i in range(epochs):
        #     # Loop through each data point
        #     for df, j in data:
        #         currLabel = j
        #         # Get random
        #         positive = getPositive(currLabel, data, df)
        #         negative = getNegative(currLabel, data)
        #
        #         optimizer.zero_grad()
        #         # input data
        #         temp = torch.tensor(df.to_numpy())
        #         temp2 = torch.reshape(temp, (200, 1, 38))
        #         model = model.double()
        #         orioutput = model(temp2, df.to_numpy())
        #         posoutput = model(torch.reshape(torch.tensor(positive.to_numpy()), (200, 1, 38)), positive.to_numpy())
        #         negoutput = model(torch.reshape(torch.tensor(negative.to_numpy()), (200, 1, 38)), negative.to_numpy())
        #
        #         loss = criterion(orioutput, posoutput, negoutput)
        #         loss.backward()
        #         optimizer.step()
        #
        # #Save the trained tcn model
        # PATH = './trainedTCNmodel.pth'
        # torch.save(model.state_dict(), PATH)

        PATH = './trainedTCNmodel.pth'
        model = TCN(38, True, 200, 32, 16, '25,25,25,25,25,25,25,25', 19)
        model.load_state_dict(torch.load(PATH))
        model = model.double()
        #Setup classifier for training
        testSet = []
        testLabels = []
        for df, i in data:
            print(i)
            result = model(torch.reshape(torch.tensor(df.to_numpy()), (200, 1, 38)), df.to_numpy())
            testSet.append(result.detach().numpy().tolist())
            testLabels.append(i - 1)

        datas = np.array(testSet)
        #Pickle model results
        if sys.argv[2] == "save":
            filename = "pickle/xTrain"
            filename2 = "pickle/yTrain"
            file = open(filename, 'wb')
            file2 = open(filename2, 'wb')
            pkl.dump(datas, filename)
            pkl.dump(testLabels, filename2)


        # Setup up LGBClassifier
        train_data = lgb.Dataset(datas, label=testLabels)
        param = {'num_leaves': 31, 'num_trees': 100, 'max_depth': 12, 'feature_fraction': 0.8, 'bagging_fraction': 0.9,
                 'num_classes': 5, 'min_data_in_leaf': 1,
                 'objective': 'multiclass',
                 'metric': {'multi_logloss'},
                 }

        bst = lgb.train(param, train_data)
        #Save classifier model
        bst.save_model('./classifierModel.txt')
    # Here we evaluate the model
    else:
        # Load the tcn model
        PATH = './trainedTCNmodel.pth'
        model = TCN(38, True, 200, 32, 16, '25,25,25,25,25,25,25,25', 19)
        model.load_state_dict(torch.load(PATH))
        model = model.double()

        #Load the classifier
        bst = lgb.Booster(model_file='./classifierModel.txt')
        accuracy = 0
        for df, i in data:

            yTrain = i - 1
            result = model(torch.reshape(torch.tensor(df.to_numpy()), (200, 1, 38)), df.to_numpy())
            prediction = bst.predict(np.array([result.detach().numpy()]))
            yTest = prediction.argmax()
            print(yTest)
            if yTest == yTrain:
                print("Yay! Correct")
                accuracy += 1
        print(accuracy / 25)

    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
