
import numpy as np
import random
import matplotlib.pyplot as plt
from math import pi, cos, sin

import pandas as pd

def is_valid_circle(circle, static_persons, min_distance):
    x, y, r = circle
    for person in static_persons:
        px, py, _ = person
        distance = np.sqrt((px - x) ** 2 + (py - y) ** 2)
        if distance < (r + min_distance):
            return False
    return True


def generate_circles(static_persons, max_count=2, max_radius=0.3, min_radius = 0.2, min_distance=0.4):
    circles = []
    attempts = 0
    while len(circles) < max_count and attempts < 1000:
        attempts += 1
        r = np.random.uniform(min_radius, max_radius)
        x = np.random.uniform(1, 11)
        y = np.random.uniform(1, 11)
        circle = [x, y, r]
        if is_valid_circle(circle, static_persons, min_distance):
            circles.append(circle)
            attempts = 0
    return circles


def is_valid_rect(rect, static_persons, min_distance):
    x, y, w, h = rect
    for person in static_persons:
        px, py, _ = person
        dx = max(x - px, px - (x + w), 0)
        dy = max(y - py, py - (y + h), 0)
        distance = np.sqrt(dx ** 2 + dy ** 2)
        if distance < min_distance:
            return False
    return True

def generate_rects(static_persons, max_count=2, max_size=0.5,min_size = 0.25, min_distance=0.4):
    rects = []
    attempts = 0
    while len(rects) < max_count and attempts < 1000:
        attempts += 1
        w = np.random.uniform(min_size, max_size)
        h = np.random.uniform(min_size, max_size)
        x = np.random.uniform(1, 11 - w)
        y = np.random.uniform(1, 11 - h)
        rect = [x, y, w, h]
        if is_valid_rect(rect, static_persons, min_distance):
            rects.append(rect)
            attempts = 0
    return rects

def plot_positions(static_person, cirobstacle1, recobstacle1):
    plt.figure(figsize=(12, 12))

    if static_person.size > 0:
        x, y, angle = static_person[:, 0], static_person[:, 1], static_person[:, 2]
        plt.scatter(x, y, color="blue", label="Person")
        for xi, yi, ang in zip(x, y, angle):
            plt.arrow(xi, yi, 0.5 * np.cos(ang), 0.5 * np.sin(ang), head_width=0.2, head_length=0.2, fc='red', ec='red')

    for circle in cirobstacle1:
        x, y, r = circle
        plt.gca().add_patch(plt.Circle((x, y), r, color='gray', alpha=0.5))

    for rect in recobstacle1:
        x, y, w, h = rect
        plt.gca().add_patch(plt.Rectangle((x, y), w, h, color='green', alpha=0.5))

    plt.title("Generated Person Positions with Obstacles")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(0, 12)
    plt.ylim(0, 12)
    plt.show()



def generate_group_positions(num_groups):
    groups = []
    center_positions = []

    for group_size, count in num_groups.items():
        for _ in range(count):
            while True:
                center = np.random.uniform(low=1, high=11, size=2)
                if all(np.linalg.norm(center - existing_center) >= 3.5 for existing_center in center_positions):
                    center_positions.append(center)
                    break

            if group_size == 1:
                angle = random.uniform(0, 2 * pi)
                group = np.array([np.hstack((center, angle))])
            elif group_size == 2:

                angle = random.uniform(0, 2 * pi)
                distance = 1.0
                person1 = np.hstack((center + np.array([cos(angle), sin(angle)]) * distance / 2, angle + pi))
                person2 = np.hstack((center - np.array([cos(angle), sin(angle)]) * distance / 2, angle + pi + pi))
                group = np.array([person1, person2])
            elif group_size == 3:

                angle = random.uniform(0, 2 * pi)
                distance = 0.75
                group = np.array([
                    np.hstack((center + np.array([cos(angle), sin(angle)]) * distance, angle + pi)),
                    np.hstack((center + np.array([cos(angle + 2 * pi / 3), sin(angle + 2 * pi / 3)]) * distance,
                               angle + pi + 2 * pi / 3)),
                    np.hstack((center + np.array([cos(angle - 2 * pi / 3), sin(angle - 2 * pi / 3)]) * distance,
                               angle + pi - 2 * pi / 3))
                ])

            elif group_size == 4:

                distance = 0.8

                rotation_angle = random.uniform(0, 2 * pi)

                base_angles = [pi / 4, 3 * pi / 4, 5 * pi / 4, 7 * pi / 4]
                angles = [angle + rotation_angle for angle in base_angles]

                group = []
                for angle in angles:

                    person_position = center + np.array([cos(angle) * distance, sin(angle) * distance])

                    person_angle = np.arctan2(center[1] - person_position[1], center[0] - person_position[0])
                    group.append(np.hstack((person_position, person_angle)))

                group = np.array(group)

            groups.append(group)

    return np.vstack(groups) if groups else np.array([])

num_groups = {
    4: 1,
    3: 1,
    2: 3,
    1: 2
}

static_person = generate_group_positions(num_groups)

cirobstacle1 = generate_circles(static_person)
recobstacle1 = generate_rects(static_person)

print("Circular_obstacles:", cirobstacle1)
print("Rectangular_obstacles:", recobstacle1)


if static_person.size > 0 or cirobstacle1 or recobstacle1:
    plot_positions(static_person, cirobstacle1, recobstacle1)
else:
    print("No data to plot.")

data2 = pd.DataFrame(static_person,columns=['x', 'y', 'theta'])

data2.to_csv("D:/DSGRRT/simulation/dsgrrt/scenario_5/dsgrrt_static_person.csv", index=False)

data3 = pd.DataFrame(cirobstacle1,columns=['x', 'y', 'radius'])

data3.to_csv("D:/DSGRRT/simulation/dsgrrt/scenario_5/dsgrrt_cirobstacle.csv", index=False)

data4 = pd.DataFrame(recobstacle1,columns=['x', 'y', 'width','height'])

data4.to_csv("D:/DSGRRT/simulation/dsgrrt/scenario_5/dsgrrt_recobstacle.csv", index=False)

'''
data2 = pd.DataFrame(static_person)

data2.to_csv("D:/DSGRRT/simulation/dsgrrt/scenario_1/dsgrrt_static_person.csv", index=False)

data3 = pd.DataFrame(cirobstacle1)

data3.to_csv("D:/DSGRRT/simulation/dsgrrt/scenario_1/dsgrrt_cirobstacle.csv", index=False)

data4 = pd.DataFrame(recobstacle1)

data4.to_csv("D:/DSGRRT/simulation/dsgrrt/scenario_1/dsgrrt_recobstacle.csv", index=False)
'''