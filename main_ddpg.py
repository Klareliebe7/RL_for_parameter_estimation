import gym
import numpy as np
from DDPG.ddpg_torch import Agent
from DDPG.utils import plot_learning_curve
from my_gym_integrated import Quadrotor_Env
from display.GUI_quadcopter import plot_quad_3d
 

if __name__ == '__main__':
    env = Quadrotor_Env()
    #env = gym.make('LunarLanderContinuous-v2')

    agent = Agent(alpha=0.0001, beta=0.001, 
                    input_dims=env.observation_space.shape, tau=0.001,
                    batch_size=64, fc1_dims=400, fc2_dims=300, 
                    n_actions=env.action_space.shape[0])
    n_games = 1000
    filename = 'Quadrotor_alpha_' + str(agent.alpha) + '_beta_' + str(agent.beta) + '_' + str(n_games) + 'simulations'
    figure_file = 'plots/' + filename + '.png'

    best_score = env.reward_range[0]
    score_history = []
    for i in range(n_games):
        #plot_quad_3d(env.waypoints, env.quadcopter.world_frame())
        observation = env.reset()
        done = False
        score = 0
        agent.noise.reset()
        print(f"... doing {i} run ...")
        ctr = 0
        while not done:
            ctr+= 1 
            action = agent.choose_action(observation)
            action = action.T*(np.array([1,0.001,0.001,0.001]))
            
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            score += reward
            observation = observation_
            if ctr%1== 0:
                print(f"=====  step:   {ctr}========")
                print(f"m :{action[0]}\na0:{action[1]} \na1:{action[2]} \na2:{action[3]}  ")
                print(f"observation is :\n{observation}")
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode ', i, 'score %.1f' % score,
                'average score %.1f' % avg_score)
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)




