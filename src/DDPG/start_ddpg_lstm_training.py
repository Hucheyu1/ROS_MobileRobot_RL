#!/usr/bin/env python

"""
Based on:
https://github.com/dranaju/project
"""
import ddpg
import ddpg_per
import ddpg_lstm
import rospy
import numpy as np
import rospkg
import utils
import time
import os
from environment import Env  # <-- used in latest work

# from environment_stage_1_original import Env  # For thesis

if __name__ == '__main__':
    rospy.init_node('ddpg_training', anonymous=True, log_level=rospy.WARN)

    max_step = rospy.get_param("/turtlebot3/nsteps") 
    env = Env(action_dim=2, max_step=max_step)

    stage_name = "stage_sparse_lstm"
    pkg_path = '/home/hu/catkin_ws/src/DDPG'

    result_outdir = pkg_path + '/results/ddpg' + '/' + stage_name
    model_outdir = pkg_path + '/models/ddpg' + '/' + stage_name
    if not os.path.exists(result_outdir):
        os.makedirs(result_outdir)
    if not os.path.exists(model_outdir):
        os.makedirs(model_outdir)
    actor_model_param_path = model_outdir + '/ddpg_actor_model_ep'
    critic_model_param_path = model_outdir + '/ddpg_critic_model_ep'

    resume_epoch = 1000
    continue_execution = True
    learning = False
    prioritized = False 
    lstm = True
    k_obstacle_count = 8
    max_hist_len = 10

    if not continue_execution:
        nepisodes = rospy.get_param("/turtlebot3/nepisodes")     
        nsteps = rospy.get_param("/turtlebot3/nsteps")           
        actor_learning_rate = rospy.get_param("/turtlebot3/actor_alpha")
        critic_learning_rate = rospy.get_param("/turtlebot3/critic_alpha")
        discount_factor = rospy.get_param("/turtlebot3/gamma")
        softupdate_coefficient = rospy.get_param("/turtlebot3/tau")
        batch_size = 64
        memory_size = 150000
        # network_inputs = 370 + (4 * k_obstacle_count - 4)
        network_inputs = 43
        hidden_layers = 256  
        network_outputs = 2  
        action_v_max = 0.22  # m/s
        action_w_max = 2.0  # rad/s
        resume_epoch = 0
        if not prioritized:
            if lstm:
                ddpg_trainer = ddpg_lstm.Agent(network_inputs, network_outputs, hidden_layers, actor_learning_rate,
                                               critic_learning_rate, batch_size, memory_size, discount_factor,
                                               softupdate_coefficient, action_v_max, action_w_max)
            else:
                ddpg_trainer = ddpg.Agent(network_inputs, network_outputs, hidden_layers, actor_learning_rate,
                                          critic_learning_rate, batch_size, memory_size, discount_factor,
                                          softupdate_coefficient, action_v_max, action_w_max)

        else:
            ddpg_trainer = ddpg_per.Agent(network_inputs, network_outputs, hidden_layers, actor_learning_rate,
                            critic_learning_rate, batch_size, memory_size, discount_factor,
                            softupdate_coefficient, action_v_max, action_w_max, prioritized)

    else:
        nepisodes = rospy.get_param("/turtlebot3/nepisodes")
        nsteps = rospy.get_param("/turtlebot3/nsteps")
        actor_learning_rate = rospy.get_param("/turtlebot3/actor_alpha")
        critic_learning_rate = rospy.get_param("/turtlebot3/critic_alpha")
        discount_factor = rospy.get_param("/turtlebot3/gamma")
        softupdate_coefficient = rospy.get_param("/turtlebot3/tau")
        batch_size = 64  
        memory_size = 150000
        network_inputs = 43
        hidden_layers = 256  
        network_outputs = 2  
        action_v_max = 0.22  
        action_w_max = 2.0 

        actor_resume_path = actor_model_param_path + str(resume_epoch)
        critic_resume_path = critic_model_param_path + str(resume_epoch)
        actor_path = actor_resume_path + '.pt'
        critic_path = critic_resume_path + '.pt'

        if not prioritized:
            if lstm:
                ddpg_trainer = ddpg_lstm.Agent(network_inputs, network_outputs, hidden_layers, actor_learning_rate,
                                               critic_learning_rate, batch_size, memory_size, discount_factor,
                                               softupdate_coefficient, action_v_max, action_w_max)
            else:
                ddpg_trainer = ddpg.Agent(network_inputs, network_outputs, hidden_layers, actor_learning_rate,
                                          critic_learning_rate, batch_size, memory_size, discount_factor,
                                          softupdate_coefficient, action_v_max, action_w_max)

        else:
            ddpg_trainer = ddpg_per.Agent(network_inputs, network_outputs, hidden_layers, actor_learning_rate,
                            critic_learning_rate, batch_size, memory_size, discount_factor,
                            softupdate_coefficient, action_v_max, action_w_max, prioritized)
            
        ddpg_trainer.load_models(actor_path, critic_path)

        resume_epoch = 0
        nepisodes = 350

    step_counter = 0

    for ep in range(resume_epoch, nepisodes):
        cumulated_reward = 0
        observation = env.reset()
        time.sleep(0.1) 
        start_time = time.time()
        env.done = False
        state = observation

        if max_hist_len > 0:
            o_buff = np.zeros([max_hist_len, network_inputs])
            a_buff = np.zeros([max_hist_len, network_outputs])
            o_buff[0, :] = state
            o_buff_len = 0
        else:
            o_buff = np.zeros([1, network_inputs])
            a_buff = np.zeros([1, network_outputs])
            o_buff_len = 0

        for step in range(nsteps):
            rospy.logwarn("EPISODE: " + str(ep + 1) + " | STEP: " + str(step + 1))
            step_counter += 1
            state = np.float32(state)
            
            if max_hist_len != 0:
                if o_buff_len == max_hist_len:
                    o_buff[:max_hist_len - 1] = o_buff[1:]
                    o_buff[max_hist_len - 1] = list(state)
                else:
                    o_buff[o_buff_len + 1 - 1] = list(state)
                    o_buff_len += 1

            if learning:
                action = ddpg_trainer.act(state, o_buff, a_buff, o_buff_len, step_counter, ep, add_noise=True)
            else:
                action = ddpg_trainer.act(state, o_buff, a_buff, o_buff_len, step_counter, ep, add_noise=False)
            _action = action.flatten().tolist()

            observation, reward, done = env.step(_action, step + 1, mode="continuous")
            success_episode, failure_episode = env.get_episode_status()
            cumulated_reward += reward

            next_state = observation
            next_state = np.float32(next_state)

            # Learning
            if learning:
                ddpg_trainer.memory.add(state, action, reward, next_state, done)
                if len(ddpg_trainer.memory) > batch_size and step_counter > 2500:
                        ddpg_trainer.learn(step_counter)       
            if not done:
                rospy.logwarn("NOT DONE")
                state = next_state

            if done:
                time_lapse = time.time() - start_time
                if (step + 1) <= 2:
                    env.shutdown()
                if learning:
                    if ep < 600:
                        if (ep + 1) % 100 == 0:
                            ddpg_trainer.save_actor_model(model_outdir, "ddpg_actor_model_ep" + str(ep + 1) + '.pt')
                            ddpg_trainer.save_critic_model(model_outdir, "ddpg_critic_model_ep" + str(ep + 1) + '.pt')
                    else:
                        if (ep + 1) % 50 == 0 :
                            ddpg_trainer.save_actor_model(model_outdir, "ddpg_actor_model_ep" + str(ep + 1) + '.pt')
                            ddpg_trainer.save_critic_model(model_outdir, "ddpg_critic_model_ep" + str(ep + 1) + '.pt')

                rospy.logwarn("DONE")
   
                if learning:
                    data = [ep + 1, success_episode, failure_episode, cumulated_reward, step + 1]
                else:
                    data = [ep + 1, success_episode, failure_episode, cumulated_reward, step + 1, time_lapse]
                if learning:
                    utils.record_data(data, result_outdir,"ddpg_training_trajectory_9")
                else:
                    utils.record_data(data, result_outdir,"ddpg_test_no_move_1000_9")
                print("EPISODE REWARD: ", cumulated_reward)
                print("EPISODE STEP: ", step + 1)
                print("EPISODE SUCCESS: ", success_episode)
                print("EPISODE FAILURE: ", failure_episode)
                
                break

   