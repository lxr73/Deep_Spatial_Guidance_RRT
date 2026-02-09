from __future__ import print_function
from vicon_dssdk import ViconDataStream
import copy

client = ViconDataStream.RetimingClient()

import re
import math
import time
import numpy as np
import pandas as pd
import zmq

import argparse
from importlib import import_module
import networkx as nx
from itertools import combinations
from scipy.optimize import fsolve
from itertools import permutations
from scipy.spatial.distance import cdist

def real_process(a):
    divisor = 1000
    l = len(a)
    l1 = l/3
    for i in range(l):
        if ((i+1) % 3)  == 0:
            a[i] = float(a[i])
        else:
            a[i] = float(a[i])
            a[i] = a[i] / divisor
            a[i] = int(a[i] * 100000) / 100000
    a = np.array(a)
    return a,l1

def dist1(q1, q2):
    return np.sqrt((q1[0] - q2[0])**2 + (q1[1] - q2[1])**2)

def pValueCal(con1, con2):

    a = 5.102
    b = 0.748
    c = 0.087

    d = dist1(con1[0:2], con2[0:2])

    angle = np.arctan2(con2[1] - con1[1], con2[0] - con1[0])

    theta = np.array([con1[2] - angle, con2[2] - (angle + np.pi)])

    cosValue = np.zeros(2)
    for i in range(len(theta)):
        cosValue[i] = max(np.cos(theta[i]), 0)

    p = 1 - np.exp(-(a * ((cosValue[0] + c) * (cosValue[1] + c)) / (d ** 2)) ** b)

    return p

def Eqn_GEN3(x, xx, yy, theta, n):
    global Beta

    if n == 1:
        Beta = 2
    elif n == 2:
        Beta = 20
    elif n == 3:
        Beta = 25
    elif n== 4:
        Beta = 8

    m1 = 2
    n1 = 0.4
    a = 0.24
    b = 0.12
    c = 1
    th = 10.18
    ncf = 2

    al5 = -0.2
    al4 = 0.15
    al3 = 0.35
    al2 = -0.05
    al1 = 1
    a0 = -1
    a1 = 0.5
    a2 = -1 / 6
    a3 = 1 / 24
    a4 = -1 / 120
    a5 = 1 / 720

    fn1 = Beta * (al5 / (n ** 5) + al4 / (n ** 4) + al3 / (n ** 3) + al2 / (n ** 2) + al1 / n + a0 + a1 * n + a2 * (
                n ** 2) + a3 * (n ** 3) + a4 * (n ** 4) + a5 * (n ** 5))
    fn = round(fn1, 5)

    EnergyMap = np.zeros(n)

    for i in range(n):
        CosA, SinA, A11 = GetTriangle([x[1], x[0]], [yy[i], xx[i]], theta[i] * (180 / np.pi))

        fA = max(CosA, 0) if CosA >= 0 else 0

        d = np.linalg.norm(np.array([xx[i], yy[i]]) - np.array([x[0], x[1]]))

        hisfield = m1 * fA + n1 + c * (a * b) / np.sqrt((a * CosA) ** 2 + (b * SinA) ** 2)

        EnergyMap[i] = hisfield / d ** ncf

    f = np.zeros(2)
    f[0] = np.sum(EnergyMap) - fn
    f[1] = 0

    return f


def f1(x):
    return Eqn_GEN3(x, con_data[:, 0], con_data[:, 1], con_data[:, 2], mg)

def GetTriangle(posA, posB, oriB):

    posA = np.array(posA)
    posB = np.array(posB)

    oriB_rad = np.radians(oriB)

    posC = posB + np.array([np.sin(oriB_rad), -np.cos(oriB_rad)])
    posC = np.array(posC)

    Vector1 = posA - posB
    Vector2 = posC - posB

    CosA = np.dot(Vector1, Vector2) / (np.linalg.norm(Vector1) * np.linalg.norm(Vector2))

    SinA = np.sqrt(1 - CosA ** 2)

    A11 = np.arctan2(Vector1[1], Vector1[0])

    return CosA, SinA, A11
def heading_Cal(x, xx, yy, theta, n, mg=None):
    Alpha = -25
    Beta = 634.92

    if n == 1:
        Beta = 0.6
    elif n == 2:
        Beta = 20
    elif n == 3:
        Beta = 30
    elif n == 4:
        Beta = 10

    m1 = 1.5
    n1 = 0.4
    a = 0.24
    b = 0.12
    c = 1
    th = 10.18
    ncf = 2

    al5 = -0.2
    al4 = 0.15
    al3 = 0.35
    al2 = -0.05
    al1 = 1
    a0 = -1
    a1 = 0.5
    a2 = -1 / 6
    a3 = 1 / 24
    a4 = -1 / 120
    a5 = 1 / 720

    fn = Beta * (al5 / (n ** 5) + al4 / (n ** 4) + al3 / (n ** 3) + al2 / (n ** 2) + al1 / n + a0 +
                 a1 * n + a2 * (n ** 2) + a3 * (n ** 3) + a4 * (n ** 4) + a5 * (n ** 5))

    Ex = []
    Ey = []
    theta_n = []
    EnergyMap = []

    for i in range(n):
        CosA, SinA, A11 = GetTriangle([x[1], x[0]], [yy[i], xx[i]], theta[i] * (180 / np.pi))

        if CosA < 0.20:
            fA = 0
        else:
            fA = CosA

        d = np.linalg.norm(np.array([xx[i], yy[i]]) - np.array([x[0], x[1]]))

        hisfield = m1 * fA + n1 + c * (a * b) / np.sqrt((a * CosA) ** 2 + (b * SinA) ** 2)

        theta_n_i = np.arctan2((x[1] - yy[i]), (x[0] - xx[i]))

        EnergyMap_i = hisfield / d ** ncf
        Ex_i = EnergyMap_i * np.cos(A11)
        Ey_i = EnergyMap_i * np.sin(A11)

        Ex.append(Ex_i)
        Ey.append(Ey_i)

    f = [np.sum(Ex), np.sum(Ey)]

    heading = np.arctan2(f[0], f[1]) + np.pi

    return heading


def TSP_optimizer(obser_point, robotCon):
    robotCon = np.atleast_2d(robotCon)
    obser_point = np.atleast_2d(obser_point)

    points = np.vstack([robotCon, obser_point])
    aa = points[:, :2]

    D_Tsp = cdist(aa, aa, metric='euclidean')

    n = len(aa)
    idx_permutations = permutations(range(1, n))

    min_distance = np.inf
    best_path = None

    for perm in idx_permutations:
        full_path = [0] + list(perm) + [0]
        distance = np.sum(
            np.fromiter((D_Tsp[full_path[i], full_path[i + 1]] for i in range(len(full_path) - 1)), dtype=float))

        if distance < min_distance:
            min_distance = distance
            best_path = full_path

    path_min = np.array(best_path)
    C = np.where(path_min == 0)[0][0]
    order = np.concatenate([path_min[C:], path_min[:C]])

    tsp_usepre = points[order]
    tsp_use = tsp_usepre[:2, :]
    tsp_usestore = tsp_usepre[1:, :]
    return order, tsp_use, tsp_usestore

def GroupsPerceptionPart3(robotCon, configuration, prvrad, prvnz):
    global con_data,mg
    num = configuration.shape[0]

    a = np.arange(num)
    b = np.array(list(combinations(a, 2)))

    regp = np.zeros((num, 3))
    for i in range(num):
        regp[i, 0] = configuration[i, 0]
        regp[i, 1] = configuration[i, 1]
        regp[i, 2] = prvrad

    POI = np.zeros((num, num))
    for i in range(b.shape[0]):
        POI[b[i, 0], b[i, 1]] = pValueCal(configuration[b[i, 0], :], configuration[b[i, 1], :])

    s = b[:, 0]
    t = b[:, 1]
    weights = POI[b[:, 0], b[:, 1]]

    G = nx.DiGraph()
    G.add_weighted_edges_from(zip(s, t, weights))

    for i, weight in enumerate(weights):
        if weight <= 0.81:
            G.remove_edge(s[i], t[i])

    components = list(nx.weakly_connected_components(G))

    Group = []
    for component in components:
        group = []
        for idx in component:
            group.append(configuration[idx, :])
        Group.append(group)

    para_eqn1 = 0.8
    para_eqn2 = 0.6

    range_angle = np.pi / 20
    num_group = len(Group)
    obser_point = np.zeros((num_group, 3))
    regnz = np.zeros((num_group, 3))

    len_order = 0
    while len_order == 0:
        for i in range(num_group):
            condition = np.inf
            con_data = np.array(Group[i])
            mg = len(con_data)

            while condition > 2.5:
                Current_angelPre = np.sum([c[2] for c in con_data]) / mg
                Current_Px = np.sum([c[0] for c in con_data]) / mg
                Current_Py = np.sum([c[1] for c in con_data]) / mg
                O_Current_configuration = np.zeros((mg, 3))
                for cc in range(0,mg):
                    O_Current_configuration[cc, 2] = Current_angelPre + (cc ) * (2 * np.pi / mg)
                    O_Current_configuration[cc, 0] = Current_Px + (para_eqn1 + para_eqn2) / 2 * np.cos(
                        O_Current_configuration[cc, 2])
                    O_Current_configuration[cc, 1] = Current_Py + (para_eqn1 + para_eqn2) / 2 * np.sin(
                        O_Current_configuration[cc, 2])

                O_Current_configuration = np.array(O_Current_configuration)
                robotCon = np.array(robotCon)
                O_distance = np.linalg.norm(O_Current_configuration[:, 0:2] - robotCon[0:2], axis=1)
                D_ind = np.argmin(O_distance)
                O_final_choose_Angel = O_Current_configuration[D_ind, 2]
                arr = con_data[:, 0:2]
                angle = O_final_choose_Angel - range_angle / 2 + np.random.rand(50) * range_angle
                r = np.sqrt((para_eqn1 - para_eqn2) * np.random.rand(50))
                xx = np.sum(arr[:, 0]) / mg + (para_eqn2 + r) * np.cos(angle)
                yy = np.sum(arr[:, 1]) / mg + (para_eqn2 + r) * np.sin(angle)
                x0 = np.vstack([xx, yy]).T
                len_x0 = x0.shape[0]

                for ll in range(0,len_x0):

                    solution = np.array(fsolve(f1, x0[ll, 0:2],maxfev=4000, xtol=1e-4))
                    obser_point[i, 0:2] = solution

                    if np.sqrt((obser_point[i, 0] - np.sum(con_data[:, 0]) / mg) ** 2 +
                               (obser_point[i, 1] - np.sum(con_data[:, 1]) / mg) ** 2) < 2:
                        break

                regnz[i, 0] = Current_Px
                regnz[i, 1] = Current_Py
                distance = np.sqrt((regnz[i, 0] - obser_point[i, 0]) ** 2 +
                                   (regnz[i, 1] - obser_point[i, 1]) ** 2)
                regnz[i, 2] = distance - prvnz

                obser_point[i, 2] = heading_Cal(obser_point[i, 0:2], con_data[:, 0], con_data[:, 1],
                                                con_data[:, 2], mg)

                condition = np.sqrt((obser_point[i, 0] - np.sum(con_data[:, 0]) / mg) ** 2 +
                                    (obser_point[i, 1] - np.sum(con_data[:, 1]) / mg) ** 2)

            order, tsp_use, tsp_usepre = TSP_optimizer(obser_point, robotCon)
            len_order = len(order)

    return tsp_usepre,regp,regnz


def calculate_direction(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.arctan2(dy, dx)

def interpolate_directions(path):

    directions = []
    for i in range(len(path) - 1):
        dir_angle = calculate_direction(path[i], path[i + 1])
        directions.append(dir_angle)

    interpolated_directions = [directions[0]]
    for i in range(1, len(directions)):

        start_angle = directions[i - 1]
        end_angle = directions[i]

        interpolated_angle = np.linspace(start_angle, end_angle, 2)[1]

        interpolated_directions.append(interpolated_angle)

    interpolated_directions.append(directions[-1])

    interpolated_path = []
    for i in range(len(path)):
        x, y = path[i]
        angle = interpolated_directions[i]
        interpolated_path.append([x, y, angle])

    return np.array(interpolated_path)

#dsgrrt setting

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_planner', default='dsgrrt_star',
                        help='rrt_star, irrt_star, nrrt_star, dsgrrt_star')
    parser.add_argument('-n', '--neural_net', default='pointnet2', help='none, pointnet2, unet, pointnet')
    parser.add_argument('-c', '--connect', default='bfs', help='none, bfs, astar')
    parser.add_argument('--device', default='cuda',  help='cuda, cpu')

    parser.add_argument('--step_len', type=float, default= 20 )
    parser.add_argument('--iter_max', type=int, default= 2000 )
    parser.add_argument('--clearance', type=float, default=3, help='0 for block and gap, 3 for random_2d.')
    parser.add_argument('--pc_n_points', type=int, default= 2500)
    parser.add_argument('--pc_over_sample_scale', type=int, default=5)
    parser.add_argument('--pc_sample_rate', type=float, default=0.5)
    parser.add_argument('--pc_update_cost_ratio', type=float, default=0.9)
    parser.add_argument('--connect_max_trial_attempts', type=int, default=5)

    parser.add_argument('--problem', default='random_2d', help='block, gap, random_2d')
    parser.add_argument('--result_folderpath', default='results')
    parser.add_argument('--path_len_threshold_percentage', type=float, default=0.02, help='block use only.')
    parser.add_argument('--iter_after_initial', type=int, default= 0, help='random_2d use only.')

    return parser.parse_args()



#nrrt setting
'''
def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_planner', default='nrrt_star',
                        help='rrt_star, irrt_star, nrrt_star, dsgrrt_star')
    parser.add_argument('-n', '--neural_net', default='pointnet2', help='none, pointnet2, unet, pointnet')
    parser.add_argument('-c', '--connect', default='bfs', help='none, bfs, astar')
    parser.add_argument('--device', default='cuda',  help='cuda, cpu')

    parser.add_argument('--step_len', type=float, default= 20 ) 
    parser.add_argument('--iter_max', type=int, default= 2000 )
    parser.add_argument('--clearance', type=float, default=3, help='0 for block and gap, 3 for random_2d.')
    parser.add_argument('--pc_n_points', type=int, default=2500)
    parser.add_argument('--pc_over_sample_scale', type=int, default=5)
    parser.add_argument('--pc_sample_rate', type=float, default=0.5)
    parser.add_argument('--pc_update_cost_ratio', type=float, default=0.9)
    parser.add_argument('--connect_max_trial_attempts', type=int, default=5)

    parser.add_argument('--problem', default='random_2d', help='block, gap, random_2d')
    parser.add_argument('--result_folderpath', default='results')
    parser.add_argument('--path_len_threshold_percentage', type=float, default=0.02, help='block use only.')
    parser.add_argument('--iter_after_initial', type=int, default= 0, help='random_2d use only.')

    return parser.parse_args()
'''


#irrt setting
'''
def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_planner', default='irrt_star',
                        help='rrt_star, irrt_star, nrrt_star, dsgrrt_star')
    parser.add_argument('-n', '--neural_net', default='none', help='none, pointnet2, unet, pointnet')
    parser.add_argument('-c', '--connect', default='none', help='none, bfs, astar')
    parser.add_argument('--device', default='cuda',  help='cuda, cpu')

    parser.add_argument('--step_len', type=float, default= 20 ) 
    parser.add_argument('--iter_max', type=int, default= 2000 ) 
    parser.add_argument('--clearance', type=float, default=3, help='0 for block and gap, 3 for random_2d.')
    parser.add_argument('--pc_n_points', type=int, default=2500)
    parser.add_argument('--pc_over_sample_scale', type=int, default=5)
    parser.add_argument('--pc_sample_rate', type=float, default=0.5)
    parser.add_argument('--pc_update_cost_ratio', type=float, default=0.9)
    parser.add_argument('--connect_max_trial_attempts', type=int, default=5)

    parser.add_argument('--problem', default='random_2d', help='block, gap, random_2d')
    parser.add_argument('--result_folderpath', default='results')
    parser.add_argument('--path_len_threshold_percentage', type=float, default=0.02, help='block use only.')
    parser.add_argument('--iter_after_initial', type=int, default= 0, help='random_2d use only.')

    return parser.parse_args()
'''


#rrt setting
'''
def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_planner', default='rrt_star',
                        help='rrt_star, irrt_star, nrrt_star, dsgrrt_star')
    parser.add_argument('-n', '--neural_net', default='none', help='none, pointnet2, unet, pointnet')
    parser.add_argument('-c', '--connect', default='none', help='none, bfs, astar')
    parser.add_argument('--device', default='cuda',  help='cuda, cpu')

    parser.add_argument('--step_len', type=float, default= 20 ) 
    parser.add_argument('--iter_max', type=int, default= 2000)
    parser.add_argument('--clearance', type=float, default=3, help='0 for block and gap, 3 for random_2d.')
    parser.add_argument('--pc_n_points', type=int, default=2500)
    parser.add_argument('--pc_over_sample_scale', type=int, default=5)
    parser.add_argument('--pc_sample_rate', type=float, default=0.5)
    parser.add_argument('--pc_update_cost_ratio', type=float, default=0.9)
    parser.add_argument('--connect_max_trial_attempts', type=int, default=5)

    parser.add_argument('--problem', default='random_2d', help='block, gap, random_2d')
    parser.add_argument('--result_folderpath', default='results')
    parser.add_argument('--path_len_threshold_percentage', type=float, default=0.02, help='block use only.')
    parser.add_argument('--iter_after_initial', type=int, default= 0, help='random_2d use only.')

    return parser.parse_args()
'''

def is_similar(point1, point2, threshold_x=0.85, threshold_theta=1.0):
    dx = point1[0] - point2[0]
    dy = point1[1] - point2[1]
    diff = np.sqrt(dx**2 + dy**2)
    theta_diff = abs(point1[2] - point2[2])
    return diff < threshold_x and theta_diff < threshold_theta

def remove_duplicates(tsp_usestoreold1, prev_alreatsp):
    i = 1
    while i < len(tsp_usestoreold1):
        removed = False
        for prev_point in prev_alreatsp:
            if is_similar(tsp_usestoreold1[i], prev_point):
                print(f"Duplicate found at index {i}, removing: {tsp_usestoreold1[i]}")
                tsp_usestoreold1 = np.delete(tsp_usestoreold1, i, axis=0)
                removed = True
                break
        if not removed:
            i += 1
    return tsp_usestoreold1

def main():
    global flag,all_prev_points,cnt_xunhuan
    try:
        client.Connect("localhost:801")
        # Check the version
        print('Version', client.GetVersion())
        client.SetAxisMapping(ViconDataStream.Client.AxisMapping.EForward, ViconDataStream.Client.AxisMapping.ELeft,
                              ViconDataStream.Client.AxisMapping.EUp)
        xAxis, yAxis, zAxis = client.GetAxisMapping()
        print('X Axis', xAxis, 'Y Axis', yAxis, 'Z Axis', zAxis)
        # client.SetMaximumPrediction( 10 )
        print('Maximum Prediction', client.MaximumPrediction())
        # debug_log = 'e:\\tmp\\debug_log.txt'
        # output_log = 'e:\\tmp\\output_log.txt'
        # client_log = 'e:\\tmp\\client_log.txt'
        # stream_log = 'e:\\tmp\\stream_log.txt'

        # client.SetDebugLogFile( debug_log )
        # client.SetOutputFile( output_log )
        # client.SetTimingLog( client_log, stream_log )

        while (True):
            try:
                client.UpdateFrame()
                subjectNames = client.GetSubjectNames()
                sendData = []
                for subjectName in subjectNames:
                    print(subjectName)
                    segmentNames = client.GetSegmentNames(subjectName)

                    for segmentName in segmentNames:
                        segmentChildren = client.GetSegmentChildren(subjectName, segmentName)
                        for child in segmentChildren:
                            try:
                                print(child, 'has parent', client.GetSegmentParentName(subjectName, segmentName))
                            except ViconDataStream.DataStreamException as e:
                                print('Error getting parent segment', e)
                        '''      
                        print(segmentName, 'has static translation',
                              client.GetSegmentStaticTranslation(subjectName, segmentName))
                        print(segmentName, 'has static rotation( helical )',
                              client.GetSegmentStaticRotationHelical(subjectName, segmentName))
                        print(segmentName, 'has static rotation( EulerXYZ )',
                              client.GetSegmentStaticRotationEulerXYZ(subjectName, segmentName))
                        print(segmentName, 'has static rotation( Quaternion )',
                              client.GetSegmentStaticRotationQuaternion(subjectName, segmentName))
                        print(segmentName, 'has static rotation( Matrix )',
                              client.GetSegmentStaticRotationMatrix(subjectName, segmentName))
                        '''
                        try:
                            print(segmentName, 'has static scale', client.GetSegmentStaticScale(subjectName, segmentName))
                        except ViconDataStream.DataStreamException as e:
                            print('Scale Error', e)
                        '''
                        print(segmentName, 'has global translation',
                              client.GetSegmentGlobalTranslation(subjectName, segmentName))
                        print(segmentName, 'has global rotation( helical )',
                              client.GetSegmentGlobalRotationHelical(subjectName, segmentName))
                        print(segmentName, 'has global rotation( EulerXYZ )',
                              client.GetSegmentGlobalRotationEulerXYZ(subjectName, segmentName))
                        print(segmentName, 'has global rotation( Quaternion )',
                              client.GetSegmentGlobalRotationQuaternion(subjectName, segmentName))
                        print(segmentName, 'has global rotation( Matrix )',
                              client.GetSegmentGlobalRotationMatrix(subjectName, segmentName))
                        print(segmentName, 'has local translation',
                              client.GetSegmentLocalTranslation(subjectName, segmentName))
                        print(segmentName, 'has local rotation( helical )',
                              client.GetSegmentLocalRotationHelical(subjectName, segmentName))
                        print(segmentName, 'has local rotation( EulerXYZ )',
                              client.GetSegmentLocalRotationEulerXYZ(subjectName, segmentName))
                        print(segmentName, 'has local rotation( Quaternion )',
                              client.GetSegmentLocalRotationQuaternion(subjectName, segmentName))
                        print(segmentName, 'has local rotation( Matrix )',
                              client.GetSegmentLocalRotationMatrix(subjectName, segmentName))
                        '''

                        sendData0 = str(client.GetSegmentGlobalTranslation(subjectName, segmentName)[0])
                        sendData1 = str(client.GetSegmentGlobalRotationEulerXYZ(subjectName, segmentName)[0])
                        match0 = re.search(r'\((.*?)\)', sendData0)
                        send_mess0 = sendData0[match0.start() + 1: match0.end() - 1]
                        mess0 = send_mess0.split(",")
                        match1 = re.search(r'\((.*?)\)', sendData1)
                        send_mess1 = sendData1[match1.start() + 1: match1.end() - 1]
                        mess1 = send_mess1.split(",")
                        mess = [mess0[0], mess0[1], mess1[2]]
                        sendData.extend(mess)


                data,l = real_process(sendData)
                l1 = int(l)
                data1 = data.reshape(l1,3)
                robotCon = data1[0]
                robotCon = np.array(robotCon)

                static_person = np.delete(data1,0,axis=0)
                static_person = np.array(static_person)

                print(static_person)
                for i in range(len(static_person)):
                    static_person[i][0] = round(static_person[i][0], 2)
                    static_person[i][1] = round(static_person[i][1], 2)
                    static_person[i][2] = round(static_person[i][2], 2)
                print(static_person)

                robotCon[:2] += 3.0
                print('robotCon')
                print(robotCon)
                static_person[:, :2] += 3.0

                if (flag==1):
                    t1 = time.time()

                    tsp_usestoreold2, regpold, regnzold = GroupsPerceptionPart3(robotCon, static_person, prvrad, prvnz)
                    print(cnt_xunhuan)
                    if cnt_xunhuan==0:
                        cnt_tsp = tsp_usestoreold2.shape[0] + 6
                        print(cnt_tsp)
                    if (cnt_xunhuan==cnt_tsp):
                        break

                    cnt_xunhuan = cnt_xunhuan + 1

                    tsp_usestoreold1 = np.array(tsp_usestoreold2[::-1])

                    print('tsp_usestoreold1')
                    print(tsp_usestoreold1)

                    tsp_usestoreold = remove_duplicates(tsp_usestoreold1, prev_alreatsp)

                    print('tsp_usestoreold')
                    print(tsp_usestoreold)

                    if len(tsp_usestoreold) >= 3:
                        base_point = tsp_usestoreold[1]
                        min_distance = float('inf')
                        closest_index = -1

                        for i in range(len(tsp_usestoreold)):
                            if i == 1:
                                continue
                            point = tsp_usestoreold[i]
                            dx = point[0] - base_point[0]
                            dy = point[1] - base_point[1]
                            distance = (dx ** 2 + dy ** 2) ** 0.5
                            if distance < min_distance:
                                min_distance = distance
                                closest_index = i

                        if closest_index != -1 and closest_index != 2:
                            tsp_usestoreold[2], tsp_usestoreold[closest_index] = tsp_usestoreold[closest_index], \
                            tsp_usestoreold[2]


                    print(tsp_usestoreold)


                    if len(tsp_usestoreold) > 1:
                        arr_list = tsp_usestoreold[1].tolist()
                        prev_alreatsp.append(arr_list)
                    else:
                        print("Error: tsp_usestoreold does not contain enough elements.")
                        exit()

                    print('prev_alreatsp')
                    print(prev_alreatsp)

                    regp = [[x * scale_factor, y * scale_factor, r * scale_factor] for x, y, r in regpold]
                    regnz = [[x * scale_factor, y * scale_factor, r * scale_factor] for x, y, r in regnzold]

                    cirobstacle = [[x * scale_factor, y * scale_factor, r * scale_factor] for x, y, r in cirobstacle1]
                    recobstacle = [[x * scale_factor, y * scale_factor, r1 * scale_factor, r2 * scale_factor] for
                                   x, y, r1, r2 in recobstacle1]

                    newpath = []


                    if len(tsp_usestoreold) >= 3:
                        base_point = tsp_usestoreold[1]
                        min_distance = float('inf')
                        closest_index = -1

                        for i in range(len(tsp_usestoreold)):
                            if i == 1:
                                continue
                            point = tsp_usestoreold[i]
                            dx = point[0] - base_point[0]
                            dy = point[1] - base_point[1]
                            distance = (dx ** 2 + dy ** 2) ** 0.5
                            if distance < min_distance:
                                min_distance = distance
                                closest_index = i

                        if closest_index != -1 and closest_index != 2:
                            tsp_usestoreold[2], tsp_usestoreold[closest_index] = tsp_usestoreold[closest_index],tsp_usestoreold[2]

                    new_usepoint1 = [[x * scale_factor, y * scale_factor, r * scale_factor] for x, y, r in
                                     tsp_usestoreold]

                    problem = get_problem_input(env_config_list[env_config_index])
                    problem['env_dict']['env_dims'] = [224, 224]
                    problem['env_dict']['rectangle_obstacles'] = [[2, 2, 2, 1]]
                    problem['env_dict']['circle_obstacles'] = [[2, 2, 1]]
                    problem['env_dict']['circle_obstacles'].extend(regp)
                    problem['env_dict']['circle_obstacles'].extend(regnz)
                    problem['env_dict']['circle_obstacles'].extend(cirobstacle)
                    problem['env_dict']['rectangle_obstacles'].extend(recobstacle)
                    new_usepoint1 = [[x * scale_factor, y * scale_factor, r * scale_factor] for x, y, r in
                                     tsp_usestoreold]


                    t = time.time()

                    problem = get_problem_input(env_config_list[env_config_index])
                    t1 = time.time()
                    problem['env_dict']['env_dims'] = [224, 224]
                    problem['env_dict']['rectangle_obstacles'] = [[2, 2, 2, 1]]
                    problem['env_dict']['circle_obstacles'] = [[2, 2, 1]]
                    problem['env_dict']['circle_obstacles'].extend(regp)
                    problem['env_dict']['circle_obstacles'].extend(regnz)
                    problem['env_dict']['circle_obstacles'].extend(cirobstacle)
                    problem['env_dict']['rectangle_obstacles'].extend(recobstacle)
                    problem['x_start'] = (new_usepoint1[0][0], new_usepoint1[0][1])
                    problem['x_goal'] = (new_usepoint1[1][0], new_usepoint1[1][1])

                    path_planner = get_path_planner(
                        args,
                        problem,
                        neural_wrapper,
                    )

                    # path = path_planner.planning(visualize=True)/25
                    path = path_planner.planning(visualize=True) / 35
                    interpolated_path = interpolate_directions(path)

                    interpolated_path = np.vstack([interpolated_path, tsp_usestoreold[1]])

                    newpath.append(interpolated_path)

                    flag = 0

                    print(time.time() - t)
                    print(newpath)

                    path1 = np.vstack(newpath)

                    path1[:, :2] -= 3.0

                    path = np.array(path1)
                    path.reshape(1, len(path) * 3)
                    path_list = path.tolist()
                    for i in range(len(path)):
                        path_list[i][0] = round(path_list[i][0], 4)
                        path_list[i][1] = round(path_list[i][1], 4)
                        path_list[i][2] = round(path_list[i][2], 4)
                    path_list2 = str(path_list)
                    print(path_list2)

                socket.setsockopt(zmq.RCVTIMEO, 1000)

                try:
                    flag1 = socket.recv_string()
                    if flag1 == "12":
                        print("Reactivating path planning")
                        flag = 1
                    else:
                        print(f"Received unexpected flag: {flag1}")
                        flag = 0
                except zmq.Again:
                    print("Timeout waiting for flag1")
                    continue

                socket.send_string(path_list2)

            except ViconDataStream.DataStreamException as e:
                print('Handled data stream error', e)

    except ViconDataStream.DataStreamException as e:
        print('Handled data stream error', e)

if __name__ == '__main__':
    global flag,t1,pi,prvrad, prvnz,cirobstacle1,recobstacle1,scale_factor,robotCon,static_person,prev_points ,max_history_length,cnt_xunhuan

    flag = 1

    prev_alreatsp = []

    cnt_xunhuan = 0

    t1 = time.time()
    args = arg_parse()
    # * sanity check
    if args.path_planner == 'rrt_star' or args.path_planner == 'irrt_star':
        assert args.neural_net == 'none'
    else:
        assert args.neural_net != 'none'
    #  * set get_path_planner
    if args.neural_net == 'none':
        path_planner_name = args.path_planner
    elif args.neural_net == 'pointnet2' or args.neural_net == 'pointnet':
        path_planner_name = args.path_planner + '_png'
    elif args.neural_net == 'unet':
        path_planner_name = args.path_planner + '_gng'
    else:
        raise NotImplementedError
    if args.connect != 'none':
        path_planner_name = path_planner_name + '_c'
    path_planner_name = path_planner_name + '_2d'
    get_path_planner = getattr(import_module('path_planning_classes.' + path_planner_name), 'get_path_planner')
    #  * set NeuralWrapper
    if args.neural_net == 'none':
        NeuralWrapper = None
    elif args.neural_net == 'pointnet2' or args.neural_net == 'pointnet':
        neural_wrapper_name = args.neural_net + '_wrapper'
        if args.connect != 'none':
            neural_wrapper_name = neural_wrapper_name + '_connect_' + args.connect
        NeuralWrapper = getattr(import_module('wrapper.pointnet_pointnet2.' + neural_wrapper_name), 'PNGWrapper')
    elif args.neural_net == 'unet':
        neural_wrapper_name = args.neural_net + '_wrapper'
        if args.connect != 'none':
            raise NotImplementedError
        NeuralWrapper = getattr(import_module('wrapper.unet.' + neural_wrapper_name), 'GNGWrapper')
    else:
        raise NotImplementedError
    #  * set planning problem
    get_env_configs = getattr(import_module('datasets.planning_problem_utils_2d'),
                              'get_' + args.problem + '_env_configs')
    get_problem_input = getattr(import_module('datasets.planning_problem_utils_2d'),
                                'get_' + args.problem + '_problem_input')

    # * main
    if NeuralWrapper is None:
        neural_wrapper = None
    else:
        neural_wrapper = NeuralWrapper(
            device=args.device,
        )
    if args.problem == 'random_2d':
        args.clearance = 3

    env_config_list = get_env_configs()
    env_config_index = np.random.randint(len(env_config_list))

    pi = np.pi
    prvrad = 0.36
    prvnz = 0.6


    '''
    cirobstacle = [[-1.8, 0.0, 0.3]]  
    recobstacle = [[-0.1, -2.1, 0.40, 0.40],[1.15,-0.05,0.35,0.28],[1.7,-0.5,0.45,0.35]]  

    cirobstacle1 = [[x + 3, y + 3, r] for [x, y, r] in cirobstacle]
    recobstacle1 = [[x + 3, y + 3, w, h] for [x, y, w, h] in recobstacle]
    '''

    cirobstacle = [[-0.9, -0.3, 0.3]]
    recobstacle = [[1.00, -0.50, 0.4, 0.4],[-0.25, -1.55, 0.3, 0.5],[-0.05,1.15, 0.35, 0.35]]
    cirobstacle1 = [[x + 3, y + 3, r] for [x, y, r] in cirobstacle]
    recobstacle1 = [[x + 3, y + 3, w, h] for [x, y, w, h] in recobstacle]

    print(cirobstacle1)
    print(recobstacle1)
    print()

    scale_factor = 35

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5558")

    main()