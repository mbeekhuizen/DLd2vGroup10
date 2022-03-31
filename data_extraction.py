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


def create_data_frames_by_features(x):
    data = []
    mu, sigma = 0, 0.1
    #Loop through all the users
    environments = ['highway', 'suburban', 'tutorial', 'urban']
    for i in range(1, 6):
        for envir in environments:
            if i == 3 and envir == 'tutorial':
                continue
            if x == 0:
                tempdata = get_data(f"sample_data/user_000{str(i)}/user_000{str(i)}_{envir}.csv")
            elif x == 1:
                tempdata = get_data_with_highway_speed_acceleration_only(
                    f"sample_data/user_000{str(i)}/user_000{str(i)}_{envir}.csv")
            elif x == 2:
                tempdata = get_data_without_distance_information(
                    f"sample_data/user_000{str(i)}/user_000{str(i)}_{envir}.csv")
            elif x == 3:
                tempdata = get_data_without_lane_information(
                    f"sample_data/user_000{str(i)}/user_000{str(i)}_{envir}.csv")
            elif x == 4:
                tempdata = get_data_without_acceleration_break_peda(
                    f"sample_data/user_000{str(i)}/user_000{str(i)}_{envir}.csv")
            elif x == 5:
                tempdata = get_data_without_speed(
                    f"sample_data/user_000{str(i)}/user_000{str(i)}_{envir}.csv")
            elif x == 6:
                tempdata = get_data_without_gear_box(
                    f"sample_data/user_000{str(i)}/user_000{str(i)}_{envir}.csv")
            elif x == 7:
                tempdata = get_data_without_acceleration(
                    f"sample_data/user_000{str(i)}/user_000{str(i)}_{envir}.csv")
            elif x == 8:
                tempdata = get_data_without_SteeringWheel_roadAngle(
                    f"sample_data/user_000{str(i)}/user_000{str(i)}_{envir}.csv")
            elif x == 9:
                tempdata = get_data_without_Turn_indicators(
                    f"sample_data/user_000{str(i)}/user_000{str(i)}_{envir}.csv")

            list_df = np.array_split(tempdata, 5)

            for j in range(0, len(list_df)):
                data.append((list_df[j], i - 1, envir))
    random.shuffle(data)

    return data


print(create_data_frames_by_features_without_driver_3_and_tutorial(2))
