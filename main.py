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
import random
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from data_extraction import create_data_frames_by_features
import copy
from datetime import datetime
random.seed(25)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def get_data(path):
    d = pd.read_csv(path)
    return d.drop(d.columns[0], axis=1)


def create_data_frames():
    data = []
    #Loop through all the users and environments
    environments = ['highway', 'suburban', 'tutorial', 'urban']
    for i in range(1, 6):
        for envir in environments:
            if i == 3 and envir == 'tutorial':
                continue
            tempdata = get_data(f"sample_data/user_000{str(i)}/user_000{str(i)}_{envir}.csv")
            list_df = np.array_split(tempdata, 5)
            for j in range(0, len(list_df)):
                data.append((list_df[j], i - 1, envir))
    random.shuffle(data)
    return data


def getPositive(currLabel, data, dataFrame, type):
    foundPositive = False
    while not foundPositive:
        randomFrame, newLabel, newtype = random.choice(data)
        # If dataframe is positive but it is not the same
        if newLabel == currLabel and type == newtype and not randomFrame.equals(dataFrame):
            return randomFrame


def getNegative(currLabel, data, type):
    foundNegative = False
    while not foundNegative:
        randomFrame, newLabel, newtype = random.choice(data)
        if newLabel != currLabel and newtype == type:
            return randomFrame

#Returns an array of k length of (xTrain, yTrain, xTest, yTest)
def kFoldCrossValidation(k, x, y):
    allFolds = []
    testLength = len(x) // k
    for i in range(0, k):
        begin = i * testLength
        end = (i+1) * testLength
        xTest = x[begin:end]
        yTest = y[begin:end]
        xTrain = np.concatenate((x[:begin], x[end:]), axis=0)
        yTrain = y[:begin] + y[end:]
        allFolds.append((xTrain, yTrain, xTest, yTest))
    return allFolds

#Trains the TCN model with 3/5 (user data) for each environment for each user. This results in 3 * 4 envir * 5 users = 60 data elements
def trainModel(data, nfeatures):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    print(torch.version.cuda)
    #Change paramater to the total number of input features

    model = TCN(nfeatures, True, 200, 32, 16, '25,25,25,25,25,25,25,25', 19, device)
    model.to(device)
    # model = TCN(39, False, 200, 5, 16, '32,32,32,32,32,32,32,32', 15)
    criterion = nn.TripletMarginLoss(margin=1, p=2)
    optimizer = optim.Adam(model.parameters(), lr=4E-4, weight_decay=0.975, amsgrad=True)
    epoch = 1
    losses = []
    epochs = []
    avgLoss = 0
    # Train for n epochs
    for i in tqdm(range(epoch)):
        # Loop through each data point
        for df, j, envir in tqdm(data):
            currLabel = j
            # Get random positive and negative example
            positive = getPositive(currLabel, data, df, envir)
            negative = getNegative(currLabel, data, envir)

            optimizer.zero_grad()
            model = model.double()

            #Get the embedding of the original input, positive and negative example
            orioutput = model(torch.reshape(torch.tensor(df.to_numpy()).to(device), (200, 1, nfeatures)), df.to_numpy())
            posoutput = model(torch.reshape(torch.tensor(positive.to_numpy()).to(device), (200, 1, nfeatures)), positive.to_numpy())
            negoutput = model(torch.reshape(torch.tensor(negative.to_numpy()).to(device), (200, 1, nfeatures)), negative.to_numpy())

            loss = criterion(orioutput, posoutput, negoutput)
            loss.backward()
            optimizer.step()

            print(loss.item())
            avgLoss += loss.item()
        avgLoss = avgLoss / len(data)
        print(f"Epoch {i}, with avg loss of {avgLoss}")
        losses.append(avgLoss)
        epochs.append(f"Ep {i}")
        avgLoss = 0
        plot = sns.lineplot(epochs, losses)
        plt.show()
        fig = plot.get_figure()
        fig.savefig(f"lossesurban{i}.png")


    #Save the trained tcn model
    PATH = './trainedTCNmodelurban.pth'
    torch.save(model.state_dict(), PATH)
    return model

#Stores the results of the tcn model when running the whole dataset on it.
def pickleResults(model, data, device, nfeatures):
    testSet = []
    testLabels = []
    for df, i, t in data:
        print(i)
        result = model(torch.reshape(torch.tensor(df.to_numpy()).to(device), (200, 1, nfeatures)), df.to_numpy())
        testSet.append(result.detach().cpu().numpy().tolist())
        testLabels.append(i)

    datas = np.array(testSet)
    # Pickle model results

    with open('pickle/xTrain.pkl', 'wb') as f:
        pkl.dump(datas, f)
    with open('pickle/yTrain.pkl', 'wb') as f:
        pkl.dump(testLabels, f)
    print("All data saved!")

# Run the model
if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Please provide if you want to train the tcn with 'tcn', train the classifier with parameter 'classifier'")
        sys.exit()

    #Select which data you want (all data or data with removed features)
    data = create_data_frames()
    #See method to see which arguments you can pass
    #data2 = create_data_frames_by_features(9)
    nfeatures = 38

    train_data = data[:15]
    test_data = data[15:]

    #Train the model
    if sys.argv[1] == 'tcn':
        model = trainModel(train_data, nfeatures)



    if sys.argv[1] == 'pickle':
        #Load the model and pickle output
        PATH = './trainedTCNmodelurban.pth'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = TCN(nfeatures, True, 200, 32, 16, '25,25,25,25,25,25,25,25', 19, device)
        model.load_state_dict(torch.load(PATH))
        model = model.double()
        model.to(device)
        pickleResults(model, data, device, nfeatures)


    if sys.argv[1] == 'test':
        #load trained classifier and testing data
        bst = lgb.Booster(model_file='./classifierModelurban.txt')
        with open('pickle/xTrain.pkl', 'rb') as f:
            xTest = pkl.load(f)[15:,-1,:]
        with open('pickle/yTrain.pkl', 'rb') as f:
            yTest = pkl.load(f)[15:]
        accuracy = 0
        for entry, label in zip(xTest, yTest):
            prediction = bst.predict(np.array([entry]))
            maxPrediction = prediction.argmax()
            if maxPrediction == label:
                print("FOund")
                accuracy += 1
        accuracy = accuracy / len(xTest)
        print(f"Total accuracy on test set = {accuracy}")


    #Trains the classifier on the trainings data + hyperparameter tuning
    if sys.argv[1] == 'classifier':
        #Retrieve model data
        with open('pickle/xTrain.pkl', 'rb') as f:
            xTrain = pkl.load(f)[:15, -1, :]
        with open('pickle/yTrain.pkl', 'rb') as f:
            yTrain = pkl.load(f)[:15]

        #Show TSNE plot
        reduction = TSNE(n_components=2).fit_transform(xTrain)
        reduction = reduction.T
        plot = sns.scatterplot(x=reduction[0], y=reduction[1], hue=yTrain, palette= sns.color_palette("hls", 5))
        plt.show()
        fig = plot.get_figure()
        fig.savefig("out.png")

        #Setup parameters for Randomized Grid Search
        maxAcc = 0
        iters = 100
        numleaves = list(range(3, 50))
        numtrees =  list(range(10, 200, 10))
        maxdepth = list(range(2,24))
        mindatainleaf = list(range(1,10))
        for i in range(iters):
            # Apply 7 fold crossvalidation on training set of 70 data points
            k = 7
            allFolds = kFoldCrossValidation(k, xTrain, yTrain)
            avgAccuracy = 0
            param = {'num_leaves': random.choice(numleaves), 'num_trees': random.choice(numtrees),
                     'max_depth': random.choice(maxdepth),
                     'num_classes': 5, 'min_data_in_leaf': random.choice(mindatainleaf), 'max_bin': 10,
                     'objective': 'multiclass', 'verbose': -1,
                     'metric': {'multi_logloss'},
                     }
            for fold in allFolds:
                #Train classifier on current fold
                classifierTrain = lgb.Dataset(fold[0], label=fold[1])
                bst = lgb.train(param, classifierTrain)

                #Evaluate classifier on current fold
                accuracy = 0

                for xTest, yTest in zip(fold[2], fold[3]):
                    prediction = bst.predict(np.array([xTest]))
                    maxPrediction = prediction.argmax()
                    if maxPrediction == yTest:
                        accuracy += 1
                accuracy = accuracy / len(fold[2])
                avgAccuracy += accuracy
            avgAccuracy = avgAccuracy / k

            #Check if achieved Accuracy is better than previous maximum accuracy
            if avgAccuracy > maxAcc:
                print(f"New iteration better then before:{avgAccuracy} better than {maxAcc}")
                print(f"Params: {param}")
                maxAcc = avgAccuracy
                # Save classifier model
                bst.save_model('./classifierModelurban.txt')
            print(f"{i}, {avgAccuracy}")

        print(f"Best avg accuracy was {maxAcc}")

