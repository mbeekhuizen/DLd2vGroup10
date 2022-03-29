import numpy as np
import pandas as pd

from tcn_model import TCN
from torch import nn
from torch import optim
import random
import sys
import pickle as pkl
import copy

# argu=0
def get_data(path):
    d = pd.read_csv(path)
    return d.drop(d.columns[0], axis=1)

# 8 features, argu=1
def get_data_with_highway_speed_acceleration_only(path):
    d = pd.read_csv(path)
    return d[['ACCELERATION', 'ACCELERATION_PEDAL', 'ACCELERATION_Z', 'SPEED', 'SPEED_LIMIT', 'SPEED_NEXT_VEHICLE',
              'SPEED_Y', 'SPEED_Z']]

# 32 features, argu=2
def get_data_without_distance_information(path):
    d = pd.read_csv(path)
    return d.drop([d.columns[0], 'DISTANCE', 'DISTANCE_TO_NEXT_INTERSECTION', 'DISTANCE_TO_NEXT_STOP_SIGNAL',
                   'DISTANCE_TO_NEXT_TRAFFIC_LIGHT_SIGNAL', 'DISTANCE_TO_NEXT_VEHICLE',
                   'DISTANCE_TO_NEXT_YIELD_SIGNAL'], axis=1)

# 34 features, argu=3
def get_data_without_lane_information(path):
    d = pd.read_csv(path)
    return d.drop([d.columns[0], 'FAST_LANE', 'FOG', 'FOG_LIGHTS',
                   'FRONT_WIPERS'], axis=1)

# 34 features, argu=4
def get_data_without_acceleration_break_peda(path):
    d = pd.read_csv(path)
    return d.drop([d.columns[0], 'ACCELERATION', 'ACCELERATION_PEDAL', 'ACCELERATION_Z',
                   'BRAKE_PEDAL'], axis=1)

# 33 features, argu=5
def get_data_without_speed(path):
    d = pd.read_csv(path)
    return d.drop([d.columns[0], 'SPEED', 'SPEED_LIMIT', 'SPEED_NEXT_VEHICLE',
                   'SPEED_Y', 'SPEED_Z'], axis=1)

# 37 features, argu=6
def get_data_without_gear_box(path):
    d = pd.read_csv(path)
    return d.drop([d.columns[0], 'GEARBOX'], axis=1)

# 35 features, argu=7
def get_data_without_acceleration(path):
    d = pd.read_csv(path)
    return d.drop([d.columns[0], 'ACCELERATION', 'ACCELERATION_PEDAL', 'ACCELERATION_Z'], axis=1)

# 36 features, argu=8
def get_data_without_SteeringWheel_roadAngle(path):
    d = pd.read_csv(path)
    return d.drop([d.columns[0], 'STEERING_WHEEL', 'ROAD_ANGLE'], axis=1)

# 36 features, argu=9
def get_data_without_Turn_indicators(path):
    d = pd.read_csv(path)
    return d.drop([d.columns[0], 'INDICATORS', 'INDICATORS_ON_INTERSECTION'], axis=1)


def create_data_frames(x):
    data = []

    for i in range(1, 6):

        if x==0:
            tempdata = get_data("sample_data/user_000" + str(i) + "/user_000" + str(i) + "_highway.csv")
        elif x==1:
            tempdata = get_data_with_highway_speed_acceleration_only("sample_data/user_000" + str(i) + "/user_000" + str(i) + "_highway.csv")
        elif x==2:
            tempdata = get_data_without_distance_information("sample_data/user_000" + str(i) + "/user_000" + str(i) + "_highway.csv")
        elif x==3:
            tempdata = get_data_without_lane_information("sample_data/user_000" + str(i) + "/user_000" + str(i) + "_highway.csv")
        elif x==4:
                tempdata = get_data_without_acceleration_break_peda("sample_data/user_000" + str(i) + "/user_000" + str(i) + "_highway.csv")
        elif x==5:
                tempdata = get_data_without_speed("sample_data/user_000" + str(i) + "/user_000" + str(i) + "_highway.csv")
        elif x==6:
                tempdata = get_data_without_gear_box("sample_data/user_000" + str(i) + "/user_000" + str(i) + "_highway.csv")
        elif x==7:
                tempdata = get_data_without_acceleration("sample_data/user_000" + str(i) + "/user_000" + str(i) + "_highway.csv")
        elif x==8:
                tempdata = get_data_without_SteeringWheel_roadAngle("sample_data/user_000" + str(i) + "/user_000" + str(i) + "_highway.csv")
        elif x==9:
                tempdata = get_data_without_Turn_indicators("sample_data/user_000" + str(i) + "/user_000" + str(i) + "_highway.csv")

        list_df = np.array_split(tempdata, 5)
        for j in range(0, len(list_df)):
            data.append((list_df[j], i))

    return data


def create_data_frame_suburban(x):
    data = []

    for i in range(1, 6):
        if x==0:
            tempdata = get_data("sample_data/user_000" + str(i) + "/user_000" + str(i) + "_suburban.csv")
        elif x==1:
            tempdata = get_data_with_highway_speed_acceleration_only("sample_data/user_000" + str(i) + "/user_000" + str(i) + "_suburban.csv")
        elif x==2:
            tempdata = get_data_without_distance_information("sample_data/user_000" + str(i) + "/user_000" + str(i) + "_suburban.csv")
        elif x==3:
            tempdata = get_data_without_lane_information("sample_data/user_000" + str(i) + "/user_000" + str(i) + "_suburban.csv")
        elif x==4:
                tempdata = get_data_without_acceleration_break_peda("sample_data/user_000" + str(i) + "/user_000" + str(i) + "_suburban.csv")
        elif x==5:
                tempdata = get_data_without_speed("sample_data/user_000" + str(i) + "/user_000" + str(i) + "_suburban.csv")
        elif x==6:
                tempdata = get_data_without_gear_box("sample_data/user_000" + str(i) + "/user_000" + str(i) + "_suburban.csv")
        elif x==7:
                tempdata = get_data_without_acceleration("sample_data/user_000" + str(i) + "/user_000" + str(i) + "_suburban.csv")
        elif x==8:
                tempdata = get_data_without_SteeringWheel_roadAngle("sample_data/user_000" + str(i) + "/user_000" + str(i) + "_suburban.csv")
        elif x==9:
                tempdata = get_data_without_Turn_indicators("sample_data/user_000" + str(i) + "/user_000" + str(i) + "_suburban.csv")

        list_df = np.array_split(tempdata, 5)
        for j in range(0, len(list_df)):
            data.append((list_df[j], i))

    return data

    return data


def create_data_frame_tutorial(x):
    data = []
    mu, sigma = 0, 0.1
    for i in range(1, 6):
        if x==0:
            # 38 features, all
            tempdata = get_data("sample_data/user_000" + str(i) + "/user_000" + str(i) + "_tutorial.csv")
            noise_1 = np.random.normal(mu, sigma, 7600)
            noise_2 = np.random.normal(mu, sigma, 7600)
            noise_3 = np.random.normal(mu, sigma, 7600)
            noise_4 = np.random.normal(mu, sigma, 7600)
            noice_matrix_1 = np.reshape(noise_1, (-1, 38))
            noice_matrix_2 = np.reshape(noise_2, (-1, 38))
            noice_matrix_3 = np.reshape(noise_3, (-1, 38))
            noice_matrix_4 = np.reshape(noise_4, (-1, 38))
        elif x==1:
            # 8 features
            tempdata = get_data_with_highway_speed_acceleration_only("sample_data/user_000" + str(i) + "/user_000" + str(i) + "_tutorial.csv")
            noise_1 = np.random.normal(mu, sigma, 1600)
            noise_2 = np.random.normal(mu, sigma, 1600)
            noise_3 = np.random.normal(mu, sigma, 1600)
            noise_4 = np.random.normal(mu, sigma, 1600)
            noice_matrix_1 = np.reshape(noise_1, (-1, 8))
            noice_matrix_2 = np.reshape(noise_2, (-1, 8))
            noice_matrix_3 = np.reshape(noise_3, (-1, 8))
            noice_matrix_4 = np.reshape(noise_4, (-1, 8))
        elif x==2:
            # 32 features
            tempdata = get_data_without_distance_information("sample_data/user_000" + str(i) + "/user_000" + str(i) + "_tutorial.csv")
            noise_1 = np.random.normal(mu, sigma, 6400)
            noise_2 = np.random.normal(mu, sigma, 6400)
            noise_3 = np.random.normal(mu, sigma, 6400)
            noise_4 = np.random.normal(mu, sigma, 6400)
            noice_matrix_1 = np.reshape(noise_1, (-1, 32))
            noice_matrix_2 = np.reshape(noise_2, (-1, 32))
            noice_matrix_3 = np.reshape(noise_3, (-1, 32))
            noice_matrix_4 = np.reshape(noise_4, (-1, 32))
        elif x==3:
            # 34
            tempdata = get_data_without_lane_information("sample_data/user_000" + str(i) + "/user_000" + str(i) + "_tutorial.csv")
            noise_1 = np.random.normal(mu, sigma, 6800)
            noise_2 = np.random.normal(mu, sigma, 6800)
            noise_3 = np.random.normal(mu, sigma, 6800)
            noise_4 = np.random.normal(mu, sigma, 6800)
            noice_matrix_1 = np.reshape(noise_1, (-1, 34))
            noice_matrix_2 = np.reshape(noise_2, (-1, 34))
            noice_matrix_3 = np.reshape(noise_3, (-1, 34))
            noice_matrix_4 = np.reshape(noise_4, (-1, 34))
        elif x==4:
            # 34
            tempdata = get_data_without_acceleration_break_peda("sample_data/user_000" + str(i) + "/user_000" + str(i) + "_tutorial.csv")
            noise_1 = np.random.normal(mu, sigma, 6800)
            noise_2 = np.random.normal(mu, sigma, 6800)
            noise_3 = np.random.normal(mu, sigma, 6800)
            noise_4 = np.random.normal(mu, sigma, 6800)
            noice_matrix_1 = np.reshape(noise_1, (-1, 34))
            noice_matrix_2 = np.reshape(noise_2, (-1, 34))
            noice_matrix_3 = np.reshape(noise_3, (-1, 34))
            noice_matrix_4 = np.reshape(noise_4, (-1, 34))
        elif x==5:
            # 33
            tempdata = get_data_without_speed("sample_data/user_000" + str(i) + "/user_000" + str(i) + "_tutorial.csv")
            noise_1 = np.random.normal(mu, sigma, 6600)
            noise_2 = np.random.normal(mu, sigma, 6600)
            noise_3 = np.random.normal(mu, sigma, 6600)
            noise_4 = np.random.normal(mu, sigma, 6600)
            noice_matrix_1 = np.reshape(noise_1, (-1, 33))
            noice_matrix_2 = np.reshape(noise_2, (-1, 33))
            noice_matrix_3 = np.reshape(noise_3, (-1, 33))
            noice_matrix_4 = np.reshape(noise_4, (-1, 33))
        elif x==6:
            # 37
            tempdata = get_data_without_gear_box("sample_data/user_000" + str(i) + "/user_000" + str(i) + "_tutorial.csv")
            noise_1 = np.random.normal(mu, sigma, 7400)
            noise_2 = np.random.normal(mu, sigma, 7400)
            noise_3 = np.random.normal(mu, sigma, 7400)
            noise_4 = np.random.normal(mu, sigma, 7400)
            noice_matrix_1 = np.reshape(noise_1, (-1, 37))
            noice_matrix_2 = np.reshape(noise_2, (-1, 37))
            noice_matrix_3 = np.reshape(noise_3, (-1, 37))
            noice_matrix_4 = np.reshape(noise_4, (-1, 37))
        elif x==7:
            # 35
            tempdata = get_data_without_acceleration("sample_data/user_000" + str(i) + "/user_000" + str(i) + "_tutorial.csv")
            noise_1 = np.random.normal(mu, sigma, 7000)
            noise_2 = np.random.normal(mu, sigma, 7000)
            noise_3 = np.random.normal(mu, sigma, 7000)
            noise_4 = np.random.normal(mu, sigma, 7000)
            noice_matrix_1 = np.reshape(noise_1, (-1, 35))
            noice_matrix_2 = np.reshape(noise_2, (-1, 35))
            noice_matrix_3 = np.reshape(noise_3, (-1, 35))
            noice_matrix_4 = np.reshape(noise_4, (-1, 35))
        elif x==8:
            # 36
            tempdata = get_data_without_SteeringWheel_roadAngle("sample_data/user_000" + str(i) + "/user_000" + str(i) + "_tutorial.csv")
            noise_1 = np.random.normal(mu, sigma, 7200)
            noise_2 = np.random.normal(mu, sigma, 7200)
            noise_3 = np.random.normal(mu, sigma, 7200)
            noise_4 = np.random.normal(mu, sigma, 7200)
            noice_matrix_1 = np.reshape(noise_1, (-1, 36))
            noice_matrix_2 = np.reshape(noise_2, (-1, 36))
            noice_matrix_3 = np.reshape(noise_3, (-1, 36))
            noice_matrix_4 = np.reshape(noise_4, (-1, 36))
        elif x==9:
            # 36
            tempdata = get_data_without_Turn_indicators("sample_data/user_000" + str(i) + "/user_000" + str(i) + "_tutorial.csv")
            noise_1 = np.random.normal(mu, sigma, 7200)
            noise_2 = np.random.normal(mu, sigma, 7200)
            noise_3 = np.random.normal(mu, sigma, 7200)
            noise_4 = np.random.normal(mu, sigma, 7200)
            noice_matrix_1 = np.reshape(noise_1, (-1, 36))
            noice_matrix_2 = np.reshape(noise_2, (-1, 36))
            noice_matrix_3 = np.reshape(noise_3, (-1, 36))
            noice_matrix_4 = np.reshape(noise_4, (-1, 36))


        if i == 3:
            list_df = np.split(tempdata, [200, 323])
            for j in range(0, len(list_df)):
                data.append((list_df[j], i))

            copy_1 = copy.deepcopy(list_df[0])
            copy_2 = copy.deepcopy(list_df[0])
            copy_3 = copy.deepcopy(list_df[0])
            copy_4 = copy.deepcopy(list_df[0])

            copy_1 = copy_1 + noice_matrix_1
            copy_2 = copy_2 + noice_matrix_2
            copy_3 = copy_3 + noice_matrix_3
            copy_4 = copy_4 + noice_matrix_4
            temp = np.split(copy_4, [77, 200])[0]

            #  np.concatenate(list_df[1], temp, axis=0)
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


def create_data_frame_urban(x):
    data = []

    for i in range(1, 6):
        if x==0:
            tempdata = get_data("sample_data/user_000" + str(i) + "/user_000" + str(i) + "_tutorial.csv")
        elif x==1:
            tempdata = get_data_with_highway_speed_acceleration_only("sample_data/user_000" + str(i) + "/user_000" + str(i) + "_tutorial.csv")
        elif x==2:
            tempdata = get_data_without_distance_information("sample_data/user_000" + str(i) + "/user_000" + str(i) + "_tutorial.csv")
        elif x==3:
            tempdata = get_data_without_lane_information("sample_data/user_000" + str(i) + "/user_000" + str(i) + "_tutorial.csv")
        elif x==4:
                tempdata = get_data_without_acceleration_break_peda("sample_data/user_000" + str(i) + "/user_000" + str(i) + "_tutorial.csv")
        elif x==5:
                tempdata = get_data_without_speed("sample_data/user_000" + str(i) + "/user_000" + str(i) + "_tutorial.csv")
        elif x==6:
                tempdata = get_data_without_gear_box("sample_data/user_000" + str(i) + "/user_000" + str(i) + "_tutorial.csv")
        elif x==7:
                tempdata = get_data_without_acceleration("sample_data/user_000" + str(i) + "/user_000" + str(i) + "_tutorial.csv")
        elif x==8:
                tempdata = get_data_without_SteeringWheel_roadAngle("sample_data/user_000" + str(i) + "/user_000" + str(i) + "_tutorial.csv")
        elif x==9:
                tempdata = get_data_without_Turn_indicators("sample_data/user_000" + str(i) + "/user_000" + str(i) + "_tutorial.csv")

        list_df = np.array_split(tempdata, 5)
        for j in range(0, len(list_df)):
            data.append((list_df[j], i))

    return data
