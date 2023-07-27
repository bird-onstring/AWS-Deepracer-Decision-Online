# AWS-Deepracer-Decision-Online
亚马逊云科技deepracer线上决策
这个线上决策参照了22年世界冠军队，23年峰会经验

总共训练时长大约在12个小时上下
1.先学2-3个小时的寻中&Zig-zag算法

算法如下：
def reward_function(params):
    '''
    Example of penalize steering, which helps mitigate zig-zag behaviors
    '''

    # Read input parameters
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    abs_steering = abs(params['steering_angle']) # Only need the absolute steering angle

    # Calculate 3 marks that are farther and father away from the center line
    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    marker_3 = 0.5 * track_width

    # Give higher reward if the car is closer to center line and vice versa
    if distance_from_center <= marker_1:
        reward = 1.0
    elif distance_from_center <= marker_2:
        reward = 0.5
    elif distance_from_center <= marker_3:
        reward = 0.1
    else:
        reward = 1e-3  # likely crashed/ close to off track

    # Steering penality threshold, change the number based on your action space setting
    ABS_STEERING_THRESHOLD = 15 
    
    # Penalize reward if the car is steering too much
    if abs_steering > ABS_STEERING_THRESHOLD:
        reward *= 0.8
    return float(reward)

让小车能够完赛的同时，防止为了分数而左右抖动
2.将赛道的.npy文件下载，用bast.py脚本计算最佳路径，通常迭代1000次-1200次
将到处的最佳路径.npy文件放在./racelines/下
3.使用speed.py脚本计算每个路径点的转弯角度，速度以及时间
4.使用如下的奖励函数，将其关键点替换为speed.py生成的txt文件中的关键点（注意：需要设置最低速度，最高速度，最低速度最能决定完赛时常）

import math


class Reward:
    def __init__(self, verbose=False):
        self.first_racingpoint_index = None
        self.verbose = verbose

    def reward_function(self, params):

        ################## HELPER FUNCTIONS ###################

        def dist_2_points(x1, x2, y1, y2):
            return abs(abs(x1 - x2) ** 2 + abs(y1 - y2) ** 2) ** 0.5

        def closest_2_racing_points_index(racing_coords, car_coords):

            # Calculate all distances to racing points
            distances = []
            for i in range(len(racing_coords)):
                distance = dist_2_points(x1=racing_coords[i][0], x2=car_coords[0],
                                         y1=racing_coords[i][1], y2=car_coords[1])
                distances.append(distance)

            # Get index of the closest racing point
            closest_index = distances.index(min(distances))

            # Get index of the second closest racing point
            distances_no_closest = distances.copy()
            distances_no_closest[closest_index] = 999
            second_closest_index = distances_no_closest.index(
                min(distances_no_closest))

            return [closest_index, second_closest_index]

        def dist_to_racing_line(closest_coords, second_closest_coords, car_coords):

            # Calculate the distances between 2 closest racing points
            a = abs(dist_2_points(x1=closest_coords[0],
                                  x2=second_closest_coords[0],
                                  y1=closest_coords[1],
                                  y2=second_closest_coords[1]))

            # Distances between car and closest and second closest racing point
            b = abs(dist_2_points(x1=car_coords[0],
                                  x2=closest_coords[0],
                                  y1=car_coords[1],
                                  y2=closest_coords[1]))
            c = abs(dist_2_points(x1=car_coords[0],
                                  x2=second_closest_coords[0],
                                  y1=car_coords[1],
                                  y2=second_closest_coords[1]))

            # Calculate distance between car and racing line (goes through 2 closest racing points)
            # try-except in case a=0 (rare bug in DeepRacer)
            try:
                distance = abs(-(a ** 4) + 2 * (a ** 2) * (b ** 2) + 2 * (a ** 2) * (c ** 2) -
                               (b ** 4) + 2 * (b ** 2) * (c ** 2) - (c ** 4)) ** 0.5 / (2 * a)
            except:
                distance = b

            return distance

        # Calculate which one of the closest racing points is the next one and which one the previous one
        def next_prev_racing_point(closest_coords, second_closest_coords, car_coords, heading):

            # Virtually set the car more into the heading direction
            heading_vector = [math.cos(math.radians(
                heading)), math.sin(math.radians(heading))]
            new_car_coords = [car_coords[0] + heading_vector[0],
                              car_coords[1] + heading_vector[1]]

            # Calculate distance from new car coords to 2 closest racing points
            distance_closest_coords_new = dist_2_points(x1=new_car_coords[0],
                                                        x2=closest_coords[0],
                                                        y1=new_car_coords[1],
                                                        y2=closest_coords[1])
            distance_second_closest_coords_new = dist_2_points(x1=new_car_coords[0],
                                                               x2=second_closest_coords[0],
                                                               y1=new_car_coords[1],
                                                               y2=second_closest_coords[1])

            if distance_closest_coords_new <= distance_second_closest_coords_new:
                next_point_coords = closest_coords
                prev_point_coords = second_closest_coords
            else:
                next_point_coords = second_closest_coords
                prev_point_coords = closest_coords

            return [next_point_coords, prev_point_coords]

        def racing_direction_diff(closest_coords, second_closest_coords, car_coords, heading):

            # Calculate the direction of the center line based on the closest waypoints
            next_point, prev_point = next_prev_racing_point(closest_coords,
                                                            second_closest_coords,
                                                            car_coords,
                                                            heading)

            # Calculate the direction in radius, arctan2(dy, dx), the result is (-pi, pi) in radians
            track_direction = math.atan2(
                next_point[1] - prev_point[1], next_point[0] - prev_point[0])

            # Convert to degree
            track_direction = math.degrees(track_direction)

            # Calculate the difference between the track direction and the heading direction of the car
            direction_diff = abs(track_direction - heading)
            if direction_diff > 180:
                direction_diff = 360 - direction_diff

            return direction_diff

        # Gives back indexes that lie between start and end index of a cyclical list 
        # (start index is included, end index is not)
        def indexes_cyclical(start, end, array_len):
            if start is None:
                start = 0
            if end is None:
                end = 0
            if end < start:
                end += array_len

            return [index % array_len for index in range(start, end)]

        # Calculate how long car would take for entire lap, if it continued like it did until now
        def projected_time(first_index, closest_index, step_count, times_list):

            # Calculate how much time has passed since start
            current_actual_time = (step_count - 1) / 15

            # Calculate which indexes were already passed
            indexes_traveled = indexes_cyclical(first_index, closest_index, len(times_list))

            # Calculate how much time should have passed if car would have followed optimals
            current_expected_time = sum([times_list[i] for i in indexes_traveled])

            # Calculate how long one entire lap takes if car follows optimals
            total_expected_time = sum(times_list)

            # Calculate how long car would take for entire lap, if it continued like it did until now
            try:
                projected_time = (current_actual_time / current_expected_time) * total_expected_time
            except:
                projected_time = 9999

            return projected_time

        #################### RACING LINE ######################

        # Optimal racing line for the Spain track
        # Each row: [x,y,speed,timeFromPreviousPoint]
        racing_track = [[3.14604, 0.93435, 3.7, 0.07998],
[2.86653, 0.83008, 3.7, 0.08063],
[2.58681, 0.72186, 3.7, 0.08106],
[2.30709, 0.61131, 3.7, 0.08129],
[2.02779, 0.49938, 3.7, 0.08132],
[1.74888, 0.38653, 3.7, 0.08132],
[1.47024, 0.27308, 3.7, 0.08131],
[1.19171, 0.15934, 3.7, 0.08131],
[0.9133, 0.04534, 3.7, 0.08131],
[0.635, -0.06894, 3.7, 0.08131],
[0.3568, -0.18344, 3.7, 0.08131],
[0.07867, -0.29809, 3.7, 0.08131],
[-0.19936, -0.41298, 3.7, 0.08131],
[-0.47625, -0.52754, 3.7, 0.08099],
[-0.75232, -0.63897, 3.7, 0.08046],
[-1.02738, -0.74451, 3.7, 0.07963],
[-1.30124, -0.84157, 3.60486, 0.0806],
[-1.57341, -0.92755, 3.17157, 0.09],
[-1.84315, -0.99977, 2.77807, 0.10052],
[-2.10942, -1.05528, 2.47235, 0.11001],
[-2.37066, -1.09066, 2.20843, 0.11937],
[-2.62497, -1.10285, 1.97564, 0.12887],
[-2.86999, -1.08889, 1.75262, 0.14003],
[-3.10276, -1.04574, 1.55538, 0.15221],
[-3.31914, -0.96955, 1.3885, 0.16521],
[-3.51315, -0.85566, 1.3885, 0.16203],
[-3.67512, -0.69849, 1.65308, 0.13653],
[-3.81062, -0.51257, 1.76054, 0.13067],
[-3.91902, -0.30215, 1.86921, 0.12663],
[-3.99934, -0.07129, 1.99166, 0.12273],
[-4.05109, 0.17535, 2.07883, 0.12122],
[-4.07329, 0.43282, 2.1501, 0.12019],
[-4.06561, 0.69531, 2.15815, 0.12168],
[-4.02804, 0.95646, 2.09804, 0.12576],
[-3.96117, 1.20958, 2.00833, 0.13036],
[-3.8663, 1.44798, 1.89941, 0.13508],
[-3.74553, 1.66548, 1.77487, 0.14017],
[-3.60149, 1.8568, 1.65742, 0.14449],
[-3.43693, 2.01733, 1.44563, 0.15903],
[-3.25461, 2.14335, 1.29739, 0.17082],
[-3.05526, 2.22542, 1.29739, 0.16617],
[-2.84147, 2.25293, 1.65085, 0.13057],
[-2.62155, 2.2451, 1.76915, 0.12439],
[-2.39767, 2.20392, 1.79275, 0.12698],
[-2.17161, 2.12637, 1.77451, 0.13468],
[-1.94719, 2.00815, 1.63784, 0.15487],
[-1.73827, 1.93644, 1.50515, 0.14675],
[-1.53619, 1.90214, 1.31617, 0.15573],
[-1.34289, 1.90441, 1.0, 0.19332],
[-1.16345, 1.9474, 1.0, 0.18452],
[-1.01687, 2.04954, 1.10886, 0.16111],
[-0.90565, 2.19218, 1.21763, 0.14855],
[-0.83123, 2.36467, 1.36089, 0.13804],
[-0.79314, 2.55813, 1.49741, 0.13168],
[-0.79114, 2.76576, 1.63971, 0.12663],
[-0.82426, 2.98135, 1.78335, 0.12231],
[-0.89066, 3.19919, 1.91926, 0.11866],
[-0.98783, 3.41413, 2.04062, 0.11559],
[-1.11282, 3.62169, 2.14772, 0.11281],
[-1.26239, 3.81824, 2.24645, 0.10995],
[-1.43317, 4.0011, 2.34529, 0.10668],
[-1.6218, 4.16859, 2.43496, 0.1036],
[-1.82535, 4.3198, 2.4717, 0.10259],
[-2.04139, 4.45421, 2.41922, 0.10517],
[-2.26867, 4.57003, 2.29476, 0.11116],
[-2.50582, 4.66532, 1.84934, 0.1382],
[-2.75163, 4.73673, 1.84934, 0.13841],
[-3.00493, 4.76978, 2.02228, 0.12631],
[-3.25912, 4.77036, 2.17068, 0.11711],
[-3.51087, 4.7431, 2.33576, 0.10841],
[-3.75836, 4.69224, 2.4719, 0.10222],
[-4.00057, 4.62069, 2.62372, 0.09626],
[-4.23702, 4.53108, 2.74873, 0.09199],
[-4.46739, 4.4252, 2.80863, 0.09027],
[-4.69136, 4.30426, 2.76927, 0.09192],
[-4.90806, 4.16801, 2.59083, 0.0988],
[-5.1165, 4.0161, 2.40641, 0.10718],
[-5.31439, 3.84642, 2.40641, 0.10833],
[-5.4984, 3.65638, 2.48506, 0.10645],
[-5.66759, 3.44756, 2.7662, 0.09716],
[-5.8235, 3.22384, 3.05153, 0.08936],
[-5.96757, 2.98799, 3.37419, 0.08191],
[-6.10128, 2.74231, 3.64063, 0.07683],
[-6.22554, 2.48834, 3.7, 0.07642],
[-6.34102, 2.22734, 3.7, 0.07714],
[-6.44816, 1.96047, 3.7, 0.07772],
[-6.5473, 1.68882, 3.7, 0.07815],
[-6.63868, 1.41345, 3.7, 0.07842],
[-6.72241, 1.13529, 3.7, 0.07851],
[-6.79852, 0.85515, 3.7, 0.07846],
[-6.86713, 0.57374, 3.7, 0.07828],
[-6.92825, 0.29164, 3.7, 0.07801],
[-6.98184, 0.00935, 3.7, 0.07766],
[-7.02731, -0.27268, 3.7, 0.07721],
[-7.06393, -0.55397, 3.7, 0.07666],
[-7.09087, -0.834, 3.451, 0.08152],
[-7.1064, -1.11211, 3.04626, 0.09144],
[-7.10867, -1.38749, 2.70698, 0.10173],
[-7.09452, -1.65895, 2.41232, 0.11268],
[-7.06045, -1.92493, 2.15492, 0.12444],
[-7.00238, -2.18334, 2.13065, 0.12431],
[-6.91558, -2.43093, 2.13065, 0.12314],
[-6.80089, -2.66532, 2.31202, 0.11286],
[-6.66389, -2.88641, 2.48345, 0.10473],
[-6.50882, -3.09431, 2.67524, 0.09695],
[-6.33934, -3.28957, 2.80293, 0.09224],
[-6.15779, -3.47235, 2.89248, 0.08907],
[-5.96589, -3.64269, 2.99016, 0.08581],
[-5.76521, -3.80083, 3.02277, 0.08452],
[-5.5568, -3.9466, 3.01787, 0.08428],
[-5.34153, -4.07971, 2.98856, 0.08469],
[-5.12023, -4.1998, 2.80689, 0.0897],
[-4.89363, -4.30639, 2.62722, 0.09532],
[-4.66203, -4.39752, 2.39926, 0.10373],
[-4.42593, -4.47091, 2.1826, 0.11328],
[-4.18581, -4.52305, 1.93324, 0.1271],
[-3.94235, -4.54958, 1.93324, 0.12668],
[-3.69585, -4.54316, 2.45246, 0.10055],
[-3.44757, -4.51565, 2.69416, 0.09272],
[-3.19775, -4.46984, 2.95028, 0.08609],
[-2.94655, -4.40795, 3.16642, 0.0817],
[-2.69408, -4.33127, 3.42509, 0.07704],
[-2.44048, -4.24119, 3.7, 0.07274],
[-2.18589, -4.13918, 3.7, 0.07413],
[-1.93049, -4.02696, 3.7, 0.0754],
[-1.67444, -3.90602, 3.7, 0.07653],
[-1.41785, -3.77734, 3.7, 0.07758],
[-1.16085, -3.64193, 3.7, 0.07851],
[-0.90353, -3.50053, 3.7, 0.07935],
[-0.64467, -3.36374, 3.7, 0.07913],
[-0.38593, -3.23442, 3.7, 0.07818],
[-0.12754, -3.11446, 3.66205, 0.07779],
[0.13019, -3.00565, 3.35418, 0.08341],
[0.38685, -2.90984, 3.11074, 0.08807],
[0.64193, -2.82868, 2.75958, 0.097],
[0.89485, -2.76351, 2.4675, 0.10585],
[1.14464, -2.71728, 2.20678, 0.11511],
[1.38998, -2.69292, 1.96762, 0.1253],
[1.62913, -2.69365, 1.75029, 0.13663],
[1.85964, -2.72323, 1.51997, 0.1529],
[2.07802, -2.7863, 1.46843, 0.15479],
[2.27776, -2.89074, 1.46843, 0.1535],
[2.45319, -3.03777, 1.91334, 0.11963],
[2.61155, -3.20982, 2.09213, 0.11177],
[2.75361, -3.40335, 2.31867, 0.10354],
[2.88063, -3.61495, 2.53772, 0.09725],
[2.99355, -3.8422, 2.53246, 0.1002],
[3.11985, -4.05878, 2.28322, 0.10981],
[3.26031, -4.26096, 2.03491, 0.12098],
[3.4169, -4.44576, 1.59139, 0.15221],
[3.59217, -4.60868, 1.59139, 0.15037],
[3.794, -4.73412, 1.7126, 0.13876],
[4.01441, -4.82473, 1.83963, 0.12954],
[4.24718, -4.88304, 1.98913, 0.12063],
[4.4871, -4.91221, 2.09331, 0.11546],
[4.73034, -4.9141, 2.18232, 0.11146],
[4.97368, -4.89043, 2.26641, 0.10787],
[5.21447, -4.843, 2.32283, 0.10566],
[5.45051, -4.77317, 2.37422, 0.10368],
[5.67995, -4.68231, 2.4205, 0.10195],
[5.90132, -4.57172, 2.45198, 0.10092],
[6.11328, -4.44246, 2.33708, 0.10623],
[6.31461, -4.29545, 2.21523, 0.11253],
[6.50241, -4.12929, 1.84345, 0.13603],
[6.67317, -3.94285, 1.84345, 0.13715],
[6.81522, -3.72995, 2.05091, 0.12479],
[6.93077, -3.49861, 2.25337, 0.11476],
[7.0224, -3.25409, 2.47638, 0.10545],
[7.09292, -3.0004, 2.65092, 0.09933],
[7.14409, -2.74027, 2.81965, 0.09402],
[7.17755, -2.4759, 2.98664, 0.08923],
[7.19484, -2.20902, 3.12913, 0.08547],
[7.19719, -1.94102, 3.23822, 0.08277],
[7.18553, -1.673, 3.35355, 0.08],
[7.16081, -1.40588, 3.46496, 0.07742],
[7.12396, -1.14039, 3.48367, 0.07694],
[7.07532, -0.87725, 3.40982, 0.07848],
[7.01504, -0.61725, 3.28344, 0.08129],
[6.94291, -0.3613, 3.10283, 0.0857],
[6.85844, -0.11058, 2.85067, 0.09281],
[6.76073, 0.13337, 2.6083, 0.10075],
[6.64823, 0.36819, 2.35617, 0.11051],
[6.51935, 0.59091, 2.14401, 0.12001],
[6.37229, 0.79748, 1.85915, 0.13639],
[6.20571, 0.98338, 1.65333, 0.15097],
[6.0167, 1.1405, 1.46709, 0.16754],
[5.80383, 1.25961, 1.46709, 0.16627],
[5.56646, 1.32699, 1.9737, 0.12502],
[5.31776, 1.36351, 2.21289, 0.11359],
[5.0607, 1.37369, 2.41369, 0.10658],
[4.79701, 1.35998, 2.67009, 0.09889],
[4.52824, 1.32521, 2.97563, 0.09108],
[4.25569, 1.27223, 3.36027, 0.08263],
[3.98046, 1.20406, 3.7, 0.07663],
[3.70339, 1.12339, 3.7, 0.07799],
[3.4251, 1.03283, 3.7, 0.07909]]

        ################## INPUT PARAMETERS ###################

        # Read all input parameters
        all_wheels_on_track = params['all_wheels_on_track']
        x = params['x']
        y = params['y']
        distance_from_center = params['distance_from_center']
        is_left_of_center = params['is_left_of_center']
        heading = params['heading']
        progress = params['progress']
        steps = params['steps']
        speed = params['speed']
        steering_angle = params['steering_angle']
        track_width = params['track_width']
        waypoints = params['waypoints']
        closest_waypoints = params['closest_waypoints']
        is_offtrack = params['is_offtrack']

        ############### OPTIMAL X,Y,SPEED,TIME ################

        # Get closest indexes for racing line (and distances to all points on racing line)
        closest_index, second_closest_index = closest_2_racing_points_index(
            racing_track, [x, y])

        # Get optimal [x, y, speed, time] for closest and second closest index
        optimals = racing_track[closest_index]
        optimals_second = racing_track[second_closest_index]

        # Save first racingpoint of episode for later
        if self.verbose == True:
            self.first_racingpoint_index = 0  # this is just for testing purposes
        if steps == 1:
            self.first_racingpoint_index = closest_index

        ################ REWARD AND PUNISHMENT ################

        ## Define the default reward ##
        reward = 1

        ## Reward if car goes close to optimal racing line ##
        DISTANCE_MULTIPLE = 1
        dist = dist_to_racing_line(optimals[0:2], optimals_second[0:2], [x, y])
        distance_reward = max(1e-3, 1 - (dist / (track_width * 0.5)))
        reward += distance_reward * DISTANCE_MULTIPLE

        ## Reward if speed is close to optimal speed ##
        SPEED_DIFF_NO_REWARD = 1
        SPEED_MULTIPLE = 2
        speed_diff = abs(optimals[2] - speed)
        if speed_diff <= SPEED_DIFF_NO_REWARD:
            # we use quadratic punishment (not linear) bc we're not as confident with the optimal speed
            # so, we do not punish small deviations from optimal speed
            speed_reward = (1 - (speed_diff / (SPEED_DIFF_NO_REWARD)) ** 2) ** 2
        else:
            speed_reward = 0
        reward += speed_reward * SPEED_MULTIPLE

        # Reward if less steps
        REWARD_PER_STEP_FOR_FASTEST_TIME = 1
        STANDARD_TIME = 37
        FASTEST_TIME = 27
        times_list = [row[3] for row in racing_track]
        projected_time = projected_time(self.first_racingpoint_index, closest_index, steps, times_list)
        try:
            steps_prediction = projected_time * 15 + 1
            reward_prediction = max(1e-3, (-REWARD_PER_STEP_FOR_FASTEST_TIME * (FASTEST_TIME) /
                                           (STANDARD_TIME - FASTEST_TIME)) * (
                                            steps_prediction - (STANDARD_TIME * 15 + 1)))
            steps_reward = min(REWARD_PER_STEP_FOR_FASTEST_TIME, reward_prediction / steps_prediction)
        except:
            steps_reward = 0
        reward += steps_reward

        # Zero reward if obviously wrong direction (e.g. spin)
        direction_diff = racing_direction_diff(
            optimals[0:2], optimals_second[0:2], [x, y], heading)
        if direction_diff > 30:
            reward = 1e-3

        # Zero reward of obviously too slow
        speed_diff_zero = optimals[2] - speed
        if speed_diff_zero > 0.5:
            reward = 1e-3

        ## Incentive for finishing the lap in less steps ##
        REWARD_FOR_FASTEST_TIME = 1500  # should be adapted to track length and other rewards
        STANDARD_TIME = 37  # seconds (time that is easily done by model)
        FASTEST_TIME = 27  # seconds (best time of 1st place on the track)
        if progress == 100:
            finish_reward = max(1e-3, (-REWARD_FOR_FASTEST_TIME /
                                       (15 * (STANDARD_TIME - FASTEST_TIME))) * (steps - STANDARD_TIME * 15))
        else:
            finish_reward = 0
        reward += finish_reward

        ## Zero reward if off track ##
        if all_wheels_on_track == False:
            reward = 1e-3

        ####################### VERBOSE #######################

        if self.verbose == True:
            print("Closest index: %i" % closest_index)
            print("Distance to racing line: %f" % dist)
            print("=== Distance reward (w/out multiple): %f ===" % (distance_reward))
            print("Optimal speed: %f" % optimals[2])
            print("Speed difference: %f" % speed_diff)
            print("=== Speed reward (w/out multiple): %f ===" % speed_reward)
            print("Direction difference: %f" % direction_diff)
            print("Predicted time: %f" % projected_time)
            print("=== Steps reward: %f ===" % steps_reward)
            print("=== Finish reward: %f ===" % finish_reward)

        #################### RETURN REWARD ####################

        # Always return a float value
        return float(reward)


reward_object = Reward()  # add parameter verbose=True to get noisy output for testing


def reward_function(params):
    return reward_object.reward_function(params)

