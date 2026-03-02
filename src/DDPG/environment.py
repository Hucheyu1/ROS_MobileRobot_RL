#!/usr/bin/env python
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert #
# Added: Risk Perception of the Moving Crowd (by Hafiq Anas) #

import rospy
import numpy as np
import math
import time
from math import pi
from geometry_msgs.msg import Twist, Pose, Point, PointStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion
from visualization_msgs.msg import Marker

from collections import deque
from uuid import uuid4
from itertools import chain
import utils
import random



class Env:
    def __init__(self, action_dim=2, max_step=200):
        self.position = Pose()
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.get_odometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.action_dim = action_dim
        # Keys CTRL + c will stop script
        rospy.on_shutdown(self.shutdown)

        # Added
        self.vel_cmd = 0.0
        self.orientation = 0.0
        self.previous_yaw = 3.14
        self.robot_yaw = 3.14   # 指机器人当前的偏航角（yaw angle）
        self.linear_twist = 0.0
        self.angular_twist = 0.0
        self.previous_heading = 0.0
        self.previous_way_distance = 0.0
        self.previous_distance_actual = 0.0
        self.previous_obs_distance_min = 0.0
        self.previous_v = 0.0
        self.previous_w = 0.0
        self.episode_success = False
        self.episode_failure = False
       
        self.test = True
        self.starting_point = Point()
        self.starting_point.x = rospy.get_param("/turtlebot3/starting_pose/x")
        self.starting_point.y = rospy.get_param("/turtlebot3/starting_pose/y")
        self.starting_point.z = rospy.get_param("/turtlebot3/starting_pose/z")

        self.original_desired_point = Point()
        self.original_desired_point.x = rospy.get_param("/turtlebot3/desired_pose/x")
        self.original_desired_point.y = rospy.get_param("/turtlebot3/desired_pose/y")
        self.original_desired_point.z = rospy.get_param("/turtlebot3/desired_pose/z")

        self.waypoint_desired_point = Point()
        self.waypoint_desired_point.x = self.original_desired_point.x
        self.waypoint_desired_point.y = self.original_desired_point.y
        self.waypoint_desired_point.z = self.original_desired_point.z

        self.linear_forward_speed = rospy.get_param('/turtlebot3/linear_forward_speed')
        self.linear_turn_speed = rospy.get_param('/turtlebot3/linear_turn_speed')
        self.angular_speed = rospy.get_param('/turtlebot3/angular_speed')


        self.scan_ranges = rospy.get_param('/turtlebot3/scan_ranges')
        self.max_scan_range = rospy.get_param('/turtlebot3/max_scan_range')
        self.min_scan_range = rospy.get_param('/turtlebot3/min_scan_range')
        self.max_steps = max_step
        self.done = False

        self.actual_distance_to_goal_origin = 0
        self.actual_distance_to_goal_current = 0
        self.actual_heading_to_goal = 0
        self.count = 0
  

        # Temporary (delete)  用于记录不同类型的奖励计数和机器人的行为状态
        self.step_reward_count = 0
        self.waypoint_reward_count = 0
        self.rd_reward_count = 0
        self.ra_reward_count = 0
        self.rd_penalty_count = 0
        self.ra_penalty_count = 0

        self.forward_action_reward_count = 0
        self.left_turn_action_reward_count = 0
        self.right_turn_action_reward_count = 0
        self.stop_action_reward_count = 0
    
        self.last_action = "FORWARD"



    def shutdown(self):
        rospy.loginfo("Stopping TurtleBot")
        self.pub_cmd_vel.publish(Twist())
        time.sleep(1)

    def get_robot_obs_xy_diff(self, robot_pose_x, robot_pose_y, obs_pose_x, obs_pose_y):
        """
        Args:
            robot_pose_x: robot's x position
            robot_pose_y: robot's y position
            obs_pose_x: obstacle's x position
            obs_pose_y: obstacle's y position

        Returns: returns distance in x and y axis between robot and obstacle

        """
        # 计算机器人和障碍物在 x 和 y 轴上的距离差
        robot_obs_x = abs(robot_pose_x - obs_pose_x)
        robot_obs_y = abs(robot_pose_y - obs_pose_y)

        return [robot_obs_x, robot_obs_y]

    def get_distance_from_point(self, pstart, p_end):
        a = np.array((pstart.x, pstart.y, pstart.z))
        b = np.array((p_end.x, p_end.y, p_end.z))

        distance = np.linalg.norm(a - b)

        return distance

    def get_distance_to_goal(self, current_position):
        distance = self.get_distance_from_point(current_position,
                                                self.waypoint_desired_point)

        return distance

    def get_actual_distance_to_goal(self, current_position):
        distance = self.get_distance_from_point(current_position,
                                                self.original_desired_point)

        return distance

    def get_angle_from_point(self, current_orientation):
        current_ori_x = current_orientation.x
        current_ori_y = current_orientation.y
        current_ori_z = current_orientation.z
        current_ori_w = current_orientation.w

        orientation_list = [current_ori_x, current_ori_y, current_ori_z, current_ori_w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        return yaw

    def get_heading_to_goal(self, current_position, current_orientation):
        # current_pos_x = current_position.x + self.starting_point.x
        # current_pos_y = current_position.y + self.starting_point.y
        current_pos_x =  current_position.x
        current_pos_y =  current_position.y

        yaw = self.get_angle_from_point(current_orientation)
        goal_angle = math.atan2(self.waypoint_desired_point.y - current_pos_y,
                                self.waypoint_desired_point.x - current_pos_x)

        heading = goal_angle - yaw
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        return heading
    
    def get_heading_to_actual_goal(self, current_position, current_orientation):
        # current_pos_x = current_position.x + self.starting_point.x
        # current_pos_y = current_position.y + self.starting_point.y
        current_pos_x =  current_position.x
        current_pos_y =  current_position.y

        yaw = self.get_angle_from_point(current_orientation)
        goal_angle = math.atan2(self.original_desired_point.y - current_pos_y,
                                self.original_desired_point.x - current_pos_x)

        heading = goal_angle - yaw
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        return heading

    def get_odometry(self, odom):
        self.position = odom.pose.pose.position
        if self.test:
            self.position.x = self.position.x * 0.6
            self.position.y = self.position.y * 0.6
        self.orientation = odom.pose.pose.orientation
        self.linear_twist = odom.twist.twist.linear
        self.angular_twist = odom.twist.twist.angular

    def get_state(self, scan, step_counter=0, action=[0, 0]):
        # 这段代码用于更新目标点。在步数为1时，它首先根据机器人的当前位置和原始目标点，计算机器人在视野范围内的新目标点。
        # 然后，更新目标点的 x 和 y 坐标。在每个步数的倍数为5或者当前距离小于之前的距离时，它也会更新目标点，确保机器人朝着目标前进。
        if step_counter == 1:
            # Get updated waypoints according to the Point Of Intersection at circle (Robot FOV)
            # 这个函数的作用是根据机器人的当前位置、目标位置以及一个边界半径来确定机器人下一个目标位置。
            # 实现的原理是通过从机器人位置到目标位置的一条直线和以机器人位置为圆心、最大激光扫描距离为半径的圆的边界之间的交点来确定新的目标位置。
            # 如果找不到交点，则使用原始目标点作为下一个路径点
            goal_waypoints = utils.get_local_goal_waypoints([self.position.x, self.position.y],
                                                            [self.original_desired_point.x,
                                                             self.original_desired_point.y], 0.3)

            self.waypoint_desired_point.x = round(goal_waypoints[0],2)
            self.waypoint_desired_point.y = round(goal_waypoints[1],2)

        distance_to_goal = round(self.get_distance_to_goal(self.position), 2)
        heading_to_goal = round(self.get_heading_to_goal(self.position, self.orientation), 2)
        
        self.actual_heading_to_goal = round(self.get_heading_to_actual_goal(self.position, self.orientation), 2)
        self.actual_distance_to_goal_origin = round(self.get_actual_distance_to_goal(self.starting_point), 2)
        self.actual_distance_to_goal_current = round(self.get_actual_distance_to_goal(self.position), 2)

        # if step_counter % 5 == 0 or distance_to_goal < self.previous_distance:
        if self.count % 10 == 0 :
            goal_waypoints = utils.get_local_goal_waypoints([self.position.x, self.position.y],
                                                            [self.original_desired_point.x,
                                                             self.original_desired_point.y], 0.3)

            self.waypoint_desired_point.x = round(goal_waypoints[0],2)
            self.waypoint_desired_point.y = round(goal_waypoints[1],2)
            
            rospy.set_param('/turtlebot3/way_pose/x', float(self.waypoint_desired_point.x))
            rospy.set_param('/turtlebot3/way_pose/y', float(self.waypoint_desired_point.y))
        self.count = self.count + 1
        
        # 分别表示机器人在局部 x 轴和 y 轴上的线速度    
        agent_vel_x = -1.0 * (self.linear_twist.x * math.cos(self.angular_twist.z))
        agent_vel_y = self.linear_twist.x * math.sin(self.angular_twist.z)

        # Get scan ranges from sensor, reverse the scans and remove the final scan because the scan reads in an
        # anti-clockwise manner and the final scan is the same as the first scan, so it is omitted
        _scan_range = utils.get_scan_ranges(scan, self.scan_ranges, self.max_scan_range)
        scan_range = _scan_range[:]

        # 机器人当前朝向的角度
        yaw = self.get_angle_from_point(self.orientation)
        self.robot_yaw = yaw
        # 将激光雷达的扫描数据转换为障碍物的坐标
        obstacle_poses = utils.convert_laserscan_to_coordinate(scan_range, self.scan_ranges, self.position, yaw, 360)


        current_scans = scan_range

        if not self.done:
            if min(current_scans) <= self.min_scan_range:
                print("DONE: MINIMUM RANGE")
                print("MINIMUM: ", str(min(current_scans)))
                self.done = True

            if self.is_in_true_desired_position(self.position):
                print("DONE: IN DESIRED POSITION")
                self.done = True

            if step_counter >= self.max_steps:
                print("DONE: STEP COUNTER > MAX STEP")
                self.done = True

        agent_position = [round(self.position.x, 2), round(self.position.y, 2)]
        agent_orientation = [round(self.robot_yaw, 3)]
        agent_velocity = [round(agent_vel_x, 3), round(agent_vel_y, 3)]
        desired_point = [round(self.original_desired_point.x, 2), round(self.original_desired_point.y, 2)]
        way_point =[self.waypoint_desired_point.x , self.waypoint_desired_point.y]

        # goal_heading_distance = [heading_to_goal, distance_to_goal]
        goal_heading_distance = [self.actual_heading_to_goal, self.actual_distance_to_goal_current]
        general_obs_distance = current_scans
   
        state = (general_obs_distance + goal_heading_distance  + agent_position + desired_point )
        # state = (general_obs_distance + goal_heading_distance  + [distance_to_goal] + agent_position + desired_point )
        
        # Round items in state to 2 decimal places
        state = list(np.around(np.array(state), 3))

        return state, self.done

    def compute_reward(self, state, step_counter, done):
        obs_distance  = state[:37]
        current_heading = state[37]
        important_obs_distance = obs_distance[5:32]
        current_actual_distance = self.actual_distance_to_goal_current
        v = round(math.sqrt(self.linear_twist.x ** 2 + self.linear_twist.y ** 2),2)
        w = round(self.angular_twist.z , 2)

        current_way_distance = state[39]
        distance_difference_way = current_way_distance - self.previous_way_distance
        
        if step_counter == 1:
            distance_difference_actual = 0
            obs_distance_difference = 0
            v_difference = 0
            w_difference = 0
        if step_counter > 1:
            distance_difference_actual = current_actual_distance - self.previous_distance_actual
            obs_distance_difference = min(obs_distance[14:23]) - self.previous_obs_distance_min
            v_difference = abs(v - self.previous_v)
            w_difference = abs(w - self.previous_w)
        r_d = round( - 180 * distance_difference_actual)
        r_a = round (4 * math.cos(current_heading))
        r_s = -4
        waypoint_reward = 0
        r_v = 0
        # r_v = round (35 * v - 3 * abs(w)- 40 * v_difference- 5 * w_difference)
        print(v,w)
        # print(important_obs_distance)
        if min(obs_distance) > 0.45:
            if v < 0.1:
                r_v = -5
            else:
                r_v = round(15 * v) - round(abs(w))
        # else:
        #     if v < 0.1 :
        #         r_v = - 5
        #     else:
        #         r_v = round(15 * (0.22 - v))

        if min(obs_distance) > 0.55:
            r_h = 0
        else:
            if min(important_obs_distance) <= min(obs_distance):
                if min(state[15:21]) <= min(important_obs_distance):
                    r_h = round(25*(min(state[15:21])-0.55))
                else:
                    r_h = round(20*(min(important_obs_distance)-0.55))
            else:
                r_h = round(10*(min(obs_distance)-0.55))
        # if distance_difference_actual <= 0 :
        #     if min(obs_distance) > 0.55:
        #         r_h = 3
        #     else:
        #         # r_h = round(15*(min(obs_distance)-0.45))
        #         if min(important_obs_distance) <= min(obs_distance):
        #             if min(state[15:21]) <= min(important_obs_distance):
        #                 r_h = round(15*(min(state[15:21])-0.55))
        #             else:
        #                 r_h = round(10*(min(important_obs_distance)-0.55))
        #         else:
        #             r_h = round(5*(min(obs_distance)-0.55))
        # else:
        #     if min(obs_distance) > 0.55:
        #         r_h = -2
        #     else:
        #         r_h = -5

        # Action reward
        if self.last_action == "FORWARD":
            self.forward_action_reward_count += 1
            action_reward = 5
        if self.last_action == "TURN_LEFT":
            self.left_turn_action_reward_count += 1
            action_reward = 1
        if self.last_action == "TURN_RIGHT":
            self.right_turn_action_reward_count += 1
            action_reward = 1
        if self.last_action == "STOP":
            self.stop_action_reward_count += 1
            action_reward = 1
        if r_d < 0:
            self.rd_penalty_count = self.rd_penalty_count + 1
        else:
            self.rd_reward_count = self.rd_reward_count + 1
        if r_a < 0:
            self.ra_penalty_count = self.ra_penalty_count + 1
        else:
            self.ra_reward_count = self.ra_reward_count + 1
        # print(state)
        # print(self.position)
        # print(self.original_desired_point)
        # print(self.waypoint_desired_point)
        # Waypoint reward
        if self.is_in_desired_position(self.position):
            rospy.loginfo("Reached waypoint position!!")
            goal_waypoints = utils.get_local_goal_waypoints([self.position.x, self.position.y],
                                                            [self.original_desired_point.x,
                                                             self.original_desired_point.y], 0.3)
            self.waypoint_desired_point.x = round(goal_waypoints[0],2)
            self.waypoint_desired_point.y = round(goal_waypoints[1],2)
            waypoint_reward = 50
            self.count = 0
            print("Change desired point")
            print(self.waypoint_desired_point)

            # Check if waypoint is within the goal point
            if self.is_in_true_desired_position(self.waypoint_desired_point):
                self.waypoint_desired_point.x = self.original_desired_point.x
                self.waypoint_desired_point.y = self.original_desired_point.y
                print("Change desired point to actual goal point since it is near")
                print(self.waypoint_desired_point)

        non_terminating_reward = r_d + r_a + r_s + r_h + r_v
        # non_terminating_reward = r_d + r_a + r_s 
        print(r_d , r_a, r_s, r_h ,r_v, non_terminating_reward)
        # non_terminating_reward = step_reward + dtg_reward + htg_reward + waypoint_reward + action_reward
        self.step_reward_count += 1

        if self.last_action is not None:
            reward = non_terminating_reward

        self.previous_distance_actual = current_actual_distance
        self.previous_way_distance    = current_way_distance
        self.previous_obs_distance_min  = min(obs_distance[14:23])
        self.previous_v = v
        self.previous_w = w

        if done:
            print(" step_reward_count : ", str(self.step_reward_count))
            print(" ra_reward_count : ", str(self.ra_reward_count))
            print(" ra_penalty_count : ", str(self.ra_penalty_count))
            print(" rd_reward_count : ", str(self.rd_reward_count))
            print(" rd_penalty_count : ", str(self.rd_penalty_count))
            print("forward action reward count: ", str(self.forward_action_reward_count))
            print("left action reward count: ", str(self.left_turn_action_reward_count))
            print("right action reward count: ", str(self.right_turn_action_reward_count))
            print("stop action reward count: ", str(self.stop_action_reward_count))
            print("----------------------------")
            if self.is_in_true_desired_position(self.position):
                rospy.loginfo("Reached goal position!!")
                self.episode_failure = False
                self.episode_success = True
                goal_reward = 500
                reward = goal_reward + non_terminating_reward
            else:
                rospy.loginfo("Collision!!")
                self.episode_failure = True
                self.episode_success = False
                collision_reward = -200
                reward = collision_reward + non_terminating_reward
            self.pub_cmd_vel.publish(Twist())

        return reward, done

    def step(self, action, step_counter, mode="discrete"):
        if mode == "discrete":
            if action == 0:  # FORWARD
                linear_speed = self.linear_forward_speed
                angular_speed = 0.0
                self.last_action = "FORWARD"
            elif action == 1:  # LEFT
                linear_speed = self.linear_turn_speed
                angular_speed = self.angular_speed
                self.last_action = "TURN_LEFT"
            elif action == 2:  # RIGHT
                linear_speed = self.linear_turn_speed
                angular_speed = -1 * self.angular_speed
                self.last_action = "TURN_RIGHT"
        else:
            linear_speed = action[0]
            angular_speed = action[1]
            if linear_speed >= 0 and (((1.0 / 16.0) * -2.0) <= angular_speed <= (1.0 / 16.0) * 2.0):
                self.last_action = "FORWARD"
            elif linear_speed >= 0 and angular_speed > (1.0 / 8.0):
                self.last_action = "TURN_LEFT"
            elif linear_speed >= 0 and angular_speed < (-1.0 / 8.0):
                self.last_action = "TURN_RIGHT"
            elif linear_speed == 0 and angular_speed == 0:
                self.last_action = "STOP"

        vel_cmd = Twist()
        vel_cmd.linear.x = linear_speed
        vel_cmd.angular.z = angular_speed
        self.vel_cmd = vel_cmd

        

        # Execute the actions to move the robot for 1 timestep
        start_timestep = time.time()
        self.pub_cmd_vel.publish(vel_cmd)
        time.sleep(0.15)
        end_timestep = time.time() - start_timestep
        if end_timestep < 0.05:
            time.sleep(0.05 - end_timestep)
            end_timestep += 0.05 - end_timestep + 0.1  # Without 0.1, the velocity is doubled

        # Update previous robot yaw, to check for heading changes, for RVIZ tracking visualization
        self.previous_yaw = self.robot_yaw

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        state, done = self.get_state(data, step_counter, action)
        reward, done = self.compute_reward(state, step_counter, done)

        return np.asarray(state), reward, done

    def reset(self):
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            print("RESET PROXY")
            self.reset_proxy()
        except rospy.ServiceException as e:
            print("gazebo/reset_simulation service call failed")

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass


        if self.test:
            # random_number_x = round(random.uniform(-1.6,1.6),2)
            # if random_number_x > 0:
            #     random_number_y = round(random.uniform(0, 1.6),2)
            # else:
            #     random_number_y = round(random.uniform(-1.6, 1.6),2)
            random_number_x = round(random.uniform(-1.6, 0),2) 
            random_number_y = round(random.uniform(-1.6,1.6),2)
        else:
            random_number_x = round(random.uniform(-1, 1),2)
            if random_number_x > 0:
                random_number_y = round(random.uniform(0, 1),2)
            else:
                random_number_y = round(random.uniform(-1, 1),2)           
            
        rospy.set_param('/turtlebot3/desired_pose/x', random_number_x)
        rospy.set_param('/turtlebot3/desired_pose/y', random_number_y)
        self.original_desired_point.x = rospy.get_param("/turtlebot3/desired_pose/x")
        self.original_desired_point.y = rospy.get_param("/turtlebot3/desired_pose/y")
        if self.test:
            self.original_desired_point.x = round(self.original_desired_point.x * 0.6 , 2)
            self.original_desired_point.y = round(self.original_desired_point.y * 0.6 , 2)
            self.starting_point.x = round(self.original_desired_point.x * 0.6 , 2)
            self.starting_point.y = round(self.original_desired_point.y * 0.6 , 2)
        self.original_desired_point.z = rospy.get_param("/turtlebot3/desired_pose/z")
        # print(self.original_desired_point)
        self.waypoint_desired_point.x = self.original_desired_point.x
        self.waypoint_desired_point.y = self.original_desired_point.y
        self.waypoint_desired_point.z = self.original_desired_point.z
        self.count = 0
        
        # Get initial heading and distance to goal
        self.previous_distance = self.get_distance_to_goal(self.position)
        self.previous_distance_actual = self.get_actual_distance_to_goal(self.position)
        self.previous_way_distance = self.get_distance_to_goal(self.position)
        self.previous_heading = self.get_heading_to_goal(self.position, self.orientation)
        self.previous_yaw = 3.14
        self.previous_obs_distance_min = 0.0
        # print(self.previous_distance_actual)
        state, _ = self.get_state(data)

        # Temporary (delete)
        self.step_reward_count = 0
        self.waypoint_reward_count = 0
        self.ra_reward_count = 0
        self.rd_reward_count = 0
        self.ra_penalty_count = 0
        self.rd_penalty_count = 0

        self.forward_action_reward_count = 0
        self.left_turn_action_reward_count = 0
        self.right_turn_action_reward_count = 0
        self.stop_action_reward_count = 0
        return np.asarray(state)

    def get_episode_status(self):

        return self.episode_success, self.episode_failure


    def is_in_desired_position(self, current_position, epsilon=0.10):  # originally 0.05, changed to 0.20
        is_in_desired_pos = False
        if self.test:
            epsilon = epsilon * 0.6
        x_pos_plus = self.waypoint_desired_point.x + epsilon
        x_pos_minus = self.waypoint_desired_point.x - epsilon
        y_pos_plus = self.waypoint_desired_point.y + epsilon
        y_pos_minus = self.waypoint_desired_point.y - epsilon

        x_current = current_position.x
        y_current = current_position.y

        x_pos_are_close = (x_current <= x_pos_plus) and (x_current > x_pos_minus)
        y_pos_are_close = (y_current <= y_pos_plus) and (y_current > y_pos_minus)

        is_in_desired_pos = x_pos_are_close and y_pos_are_close

        return is_in_desired_pos

    def is_in_true_desired_position(self, current_position, epsilon=0.20):  # originally 0.05, changed to 0.20
        is_in_desired_pos = False
        if self.test:
            epsilon = epsilon * 0.6
        x_pos_plus = self.original_desired_point.x + epsilon
        x_pos_minus = self.original_desired_point.x - epsilon
        y_pos_plus = self.original_desired_point.y + epsilon
        y_pos_minus = self.original_desired_point.y - epsilon

        x_current = current_position.x
        y_current = current_position.y

        x_pos_are_close = (x_current <= x_pos_plus) and (x_current > x_pos_minus)
        y_pos_are_close = (y_current <= y_pos_plus) and (y_current > y_pos_minus)

        is_in_desired_pos = x_pos_are_close and y_pos_are_close

        return is_in_desired_pos
