import gym
import numpy as np
from DQN.DQN import Agent
from DQN.utils import plotLearning,plot_estmation,plot_z
from quad_gym_descrete import Quadrotor_Env_discrete
from display.GUI_quadcopter import plot_quad_3d  
import global_var
import json
from os import path ,remove
def main():
    save_data = True
    train = True
    env = Quadrotor_Env_discrete()
    if train:
        agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=3, eps_end=0.01,
                    input_dims=env.observation_space.shape, lr=0.001)
        n_games = 200
    else:
        agent = Agent(gamma=0.99, epsilon= 0, batch_size=64, n_actions=3, eps_end=0.01,
                    input_dims=env.observation_space.shape, lr=0.001)
        n_games = 10 
        agent.load_models("./DQN/check_point/dqn_model_number_55")
    score_history, eps_history = [], []
    ave_reward,reward_error,z_height = [],[],[]
    reward_list = []
    filename = f'DQN_{n_games}games_'

    

    ###for plot_estmation(x, mass, estimated_mass, filename, lines=None)
    mass,estimated_mass = [],[]
    best_score = env.reward_range[0]
    if save_data:
        if path.exists('./DQN/info_file/info.json'):
            remove('./DQN/info_file/info.json')
    
    
        
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
         
        print(f"... doing {i} run ...")
        ctr = 0
        global_var._init()
        global_var.set_value("done",done)
        def ani_loop(i_frame):
            nonlocal estimated_mass
            nonlocal z_height
            nonlocal mass
            nonlocal reward_list
            nonlocal ave_reward
            nonlocal reward_error
            nonlocal env
            nonlocal agent
            nonlocal n_games
            nonlocal filename           
            nonlocal best_score
            nonlocal score_history
            nonlocal eps_history
            nonlocal observation
            nonlocal done
            nonlocal score
            nonlocal ctr
                
            ctr+= 1 
            action = agent.choose_action(observation)  
            observation_, reward, done, info = env.step(action)
            reward_list.append(reward)
            global_var.set_value("done",done)
            if train:
                agent.store_transition(observation, action, reward, observation_, done)
                agent.learn()
            score += reward
            observation = observation_
            mass.append(env.real_mass)
            z_height.append(env.sensor.position_noisy()[2])
            estimated_mass.append(env.mass_estimate)
            if done:
                ave_reward.append(np.mean(reward_list, axis=0) )
                reward_error.append(np.std(reward_list, axis=0) )
                if (save_data):  # save inputs and states graphs
                    with open('./DQN/info_file/info.json', 'a+') as f:
                        print("Saving simulation data...")
                        info_string = json.dumps(info)
                        json.dump(info_string,f)
                reward_list = []        
            return env.quadcopter.world_frame()
        try:
            plot_quad_3d(np.vstack((env.startpoint , env.targetpoint)), ani_loop)
        except :
            pass
        score_history.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(score_history[-100:])
        if train:
            #if avg_score > best_score:
            save_path_ = "./DQN/check_point/dqn_model_"
            best_score = avg_score
            save_path = save_path_ + "number_" + str(i+1)
            agent.save_models(save_path)

        print('episode ', i, 'score %.1f' % score,
                'average score %.1f' % avg_score)
    x = [i+1 for i in range(n_games)]
    plotLearning(x, ave_reward,reward_error, eps_history, filename+"reward")
    x = [i+1 for i in range(len(mass))] 
    plot_estmation(x, mass, estimated_mass, filename+"mass" ) 
    z_target = [5 for _ in range(len(z_height))]
    x = [i+1 for i in range(len(z_height))]
    plot_z(x, z_target, z_height,filename +"drone_altitude")


if __name__ == "__main__":

    main()
    
    


