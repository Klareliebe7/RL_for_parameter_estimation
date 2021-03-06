import gym
import numpy as np
from DQN.DQN import Agent
from DQN.utils import plotLearning
from quad_gym_descrete import Quadrotor_Env_discrete
from display.GUI_quadcopter import plot_quad_3d , init_ani
import global_var

def main():
    env = Quadrotor_Env_discrete()

    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=3, eps_end=0.01, input_dims=env.observation_space.shape, lr=0.001,save_path = "./DQN/check_point/dqn_model")
    score_history, eps_history = [], []
 
    n_games = 1
    filename = 'Quadrotor_alpha_discrete'


    best_score = env.reward_range[0]

    
    
        
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
         
        print(f"... doing {i} run ...")
        ctr = 0
        global_var._init()
        global_var.set_value("done",done)
        def ani_loop(i_frame):
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
            global_var.set_value("done",done)
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            score += reward
            observation = observation_

            if False:
                print(f"=====  Step:   {ctr}========")
                print(info)
                print(f"Reward: {reward}")
                print(f"Score: {score}")
                print(f"Real mass:{env.quadcopter.mass}")
                print(f"action :{action }") 
                print(f"Obssservation is :\n{observation}")
                
            return env.quadcopter.world_frame()
        try:
            plot_quad_3d(np.vstack((env.startpoint , env.targetpoint)), ani_loop)
        except ValueError:
            init_ani()
        score_history.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(score_history[-100:])
        
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode ', i, 'score %.1f' % score,
                'average score %.1f' % avg_score)
    x = [i+1 for i in range(n_games)]
    plotLearning(x, score_history, eps_history, filename)



if __name__ == "__main__":

    main()
    
    


