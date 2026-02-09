import argparse
from importlib import import_module
import numpy as np
import time
import networkx as nx
from itertools import combinations
from scipy.optimize import fsolve
from itertools import permutations
from scipy.spatial.distance import cdist
import pandas as pd

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
        Beta = 0.75
    elif n == 2:
        Beta = 20
    elif n == 3:
        Beta = 30
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
        Beta = 1
    elif n == 2:
        Beta = 50
    elif n == 3:
        Beta = 30
    elif n == 4:
        Beta = 8

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

        if CosA < 0:
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

                    solution = np.array(fsolve(f1, x0[ll, 0:2],maxfev=1000, xtol=1e-2))
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

#rrt_star setting
'''def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_planner', default='rrt_star',
                        help='rrt_star, irrt_star, nrrt_star, dsgrrt_star')
    parser.add_argument('-n', '--neural_net', default='none', help='none, pointnet2, unet, pointnet')
    parser.add_argument('-c', '--connect', default='none', help='none, bfs, astar')
    parser.add_argument('--device', default='cuda', help='cuda, cpu')

    parser.add_argument('--step_len', type=float, default=20)
    parser.add_argument('--iter_max', type=int, default=2000)
    parser.add_argument('--clearance', type=float, default=0, help='0 for block and gap, 3 for random_2d.')
    parser.add_argument('--pc_n_points', type=int, default=2048)
    parser.add_argument('--pc_over_sample_scale', type=int, default=5)
    parser.add_argument('--pc_sample_rate', type=float, default=0.5)
    parser.add_argument('--pc_update_cost_ratio', type=float, default=0.9)
    parser.add_argument('--connect_max_trial_attempts', type=int, default=5)

    parser.add_argument('--problem', default='random_2d', help='block, gap, random_2d')
    parser.add_argument('--result_folderpath', default='results')
    parser.add_argument('--path_len_threshold_percentage', type=float, default=0.02, help='block use only.')
    parser.add_argument('--iter_after_initial', type=int, default=1000, help='random_2d use only.')

    return parser.parse_args()'''

#irrt setting
'''def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_planner', default='irrt_star',
                        help='rrt_star, irrt_star, nrrt_star, dsgrrt_star')
    parser.add_argument('-n', '--neural_net', default='none', help='none, pointnet2, unet, pointnet')
    parser.add_argument('-c', '--connect', default='none', help='none, bfs, astar')
    parser.add_argument('--device', default='cuda',  help='cuda, cpu')

    parser.add_argument('--step_len', type=float, default= 20 ) 
    parser.add_argument('--iter_max', type=int, default=2000)
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

    return parser.parse_args()'''

#nrrt setting
'''def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_planner', default='nrrt_star',
                        help='rrt_star, irrt_star, nrrt_star, dsgrrt_star')
    parser.add_argument('-n', '--neural_net', default='pointnet2', help='none, pointnet2, unet, pointnet')
    parser.add_argument('-c', '--connect', default='bfs', help='none, bfs, astar')
    parser.add_argument('--device', default='cuda',  help='cuda, cpu')

    parser.add_argument('--step_len', type=float, default= 20 ) 
    parser.add_argument('--iter_max', type=int, default=2000)
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

    return parser.parse_args()'''

#dsgrrt setting
def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_planner', default='dsgrrt_star',
                        help='rrt_star, irrt_star, nrrt_star, dsgrrt_star')
    parser.add_argument('-n', '--neural_net', default='pointnet2', help='none, pointnet2, unet, pointnet')
    parser.add_argument('-c', '--connect', default='bfs', help='none, bfs, astar')
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
    path_planner_name = args.path_planner+'_png'
elif args.neural_net == 'unet':
    path_planner_name = args.path_planner+'_gng'
else:
    raise NotImplementedError
if args.connect != 'none':
    path_planner_name = path_planner_name+'_c'
path_planner_name = path_planner_name+'_2d'
get_path_planner = getattr(import_module('path_planning_classes.'+path_planner_name), 'get_path_planner')
#  * set NeuralWrapper
if args.neural_net == 'none':
    NeuralWrapper = None
elif args.neural_net == 'pointnet2' or args.neural_net == 'pointnet':
    neural_wrapper_name = args.neural_net+'_wrapper'
    if args.connect != 'none':
        neural_wrapper_name = neural_wrapper_name+'_connect_'+args.connect
    NeuralWrapper = getattr(import_module('wrapper.pointnet_pointnet2.'+neural_wrapper_name), 'PNGWrapper')
elif args.neural_net == 'unet':
    neural_wrapper_name = args.neural_net+'_wrapper'
    if args.connect != 'none':
        raise NotImplementedError
    NeuralWrapper = getattr(import_module('wrapper.unet.'+neural_wrapper_name), 'GNGWrapper')
else:
    raise NotImplementedError
#  * set planning problem
get_env_configs = getattr(import_module('datasets.planning_problem_utils_2d'), 'get_'+args.problem+'_env_configs')
get_problem_input = getattr(import_module('datasets.planning_problem_utils_2d'), 'get_'+args.problem+'_problem_input')

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
prvrad = 0.3
prvnz = 0.5


#scenario 1 setting
robotCon = np.array([1.0,1.0,0.0])

static_person1 = pd.read_csv(r"D:\DSGRRT\simulation\dsgrrt\scenario_1\dsgrrt_static_person.csv", header=0)
static_person = np.array(static_person1.values)

print(static_person)

cirobstacle = pd.read_csv(r"D:\DSGRRT\simulation\dsgrrt\scenario_1\dsgrrt_cirobstacle.csv", header=0)
cirobstacle1 = np.array(cirobstacle.values)
print(cirobstacle1)

recobstacle = pd.read_csv(r"D:\DSGRRT\simulation\dsgrrt\scenario_1\dsgrrt_recobstacle.csv", header=0)
recobstacle1 = np.array(recobstacle.values)

t1 = time.time()
tsp_usestoreold1 , regpold,regnzold = GroupsPerceptionPart3(robotCon,static_person,prvrad,prvnz)
tsp_usestoreold = np.array(tsp_usestoreold1[::-1])

scale_factor = 20


regp = [[x * scale_factor, y * scale_factor, r * scale_factor] for x, y, r in regpold]
regnz = [[x * scale_factor, y * scale_factor, r * scale_factor] for x, y, r in regnzold]





cirobstacle = [[x * scale_factor, y * scale_factor, r * scale_factor] for x, y, r in cirobstacle1]
recobstacle = [[x * scale_factor, y * scale_factor, r1 * scale_factor,r2 * scale_factor] for x, y, r1,r2 in recobstacle1]

problem = get_problem_input(env_config_list[env_config_index])
newpath = []
problem['env_dict']['env_dims'] = [240,240]
problem['env_dict']['rectangle_obstacles'] = [[2, 2, 2, 1]]
problem['env_dict']['circle_obstacles'] = [[2, 2, 1]]
problem['env_dict']['circle_obstacles'].extend(regp)
problem['env_dict']['circle_obstacles'].extend(regnz)
problem['env_dict']['circle_obstacles'].extend(cirobstacle)
problem['env_dict']['rectangle_obstacles'].extend(recobstacle)
new_usepoint1 = [[x * scale_factor, y * scale_factor, r * scale_factor] for x, y, r in tsp_usestoreold]

s1 = np.size(new_usepoint1,0)


t = time.time()
for i in range(0,s1-1):

    problem = get_problem_input(env_config_list[env_config_index])

    t1 = time.time()
    problem['env_dict']['env_dims'] = [240, 240]
    problem['env_dict']['rectangle_obstacles'] = [[2, 2, 2, 1]]
    problem['env_dict']['circle_obstacles'] = [[2, 2, 1]]
    problem['env_dict']['circle_obstacles'].extend(regp)
    problem['env_dict']['circle_obstacles'].extend(regnz)
    problem['env_dict']['circle_obstacles'].extend(cirobstacle)
    problem['env_dict']['rectangle_obstacles'].extend(recobstacle)
    problem['x_start'] = (new_usepoint1[i][0], new_usepoint1[i][1])
    problem['x_goal'] = (new_usepoint1[i + 1][0], new_usepoint1[i + 1][1])

    path_planner = get_path_planner(
        args,
        problem,
        neural_wrapper,
    )


    #path = path_planner.planning(visualize=True)/25
    path = path_planner.planning(visualize=True) / 20
    interpolated_path = interpolate_directions(path)

    interpolated_path = np.vstack([interpolated_path, tsp_usestoreold[i + 1]])
    newpath.append(interpolated_path)

    print(time.time() - t1)

print(time.time() - t)

combined_data = np.vstack(newpath)

print(combined_data)

data1 = pd.DataFrame(combined_data, columns=['x', 'y', 'theta'])

data1.to_csv("D:/DSGRRT/simulation/dsgrrt/scenario_1/dsgrrt_newpath_1.csv", index=False)

