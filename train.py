import os
import copy
import torch
import argparse

import numpy as np
import matplotlib.pyplot as plt

from agents.agent_pg import Agent_PG
from agents.agent_dqn import Agent_DQN
from agents.agent_ddqn import Agent_DDQN
from reversi.reversi_env import Reversi
from alfazero.config import CFG
from alfazero.train_net import Train
from alfazero.neural_net import NeuralNetworkWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Code to read command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model",
                    help="Pick a training model (0 - PG (default); 1 - DQN; 2 - DDQN; 3 - AlfaZero;).",
                    dest="model",
                    type=int,
                    default=0)
def modelPG():
    blue_check_point = "Brain_PG_Blue10000"
    red_check_point = "Brain_PG_Red10000"
    agent_Blue = Agent_PG("Blue", device=device).to(device)
    agent_Red = Agent_PG("Red", device=device).to(device)
    if blue_check_point:
        agent_Blue.load_model(blue_check_point)
    if red_check_point:
        agent_Red.load_model(red_check_point)

    env = Reversi(human_VS_machine=False)
    reward_history, winning_rate = [], []
    is_Blue = []
    max_epoch = 10000
    RENDER = False
    for ep in range(1, max_epoch+1):
        ep_reward = []
        obs, info = env.reset()
        done = False
        if RENDER: env.render()
        while True:
            next_palyer = info["next_player"]
            next_possible_actions = info["next_possible_actions"]
            
            if next_palyer == "Blue":  # We train the blue
                action = agent_Blue.choose_action(obs, next_possible_actions)
                obs_, reward, done, info = env.step(action)
                ep_reward.append(reward)
                agent_Blue.store_transition(obs, action, reward, next_possible_actions)
            else:
                action = agent_Red.choose_action(obs, next_possible_actions)
                # action = env.get_random_action()
                obs_, reward, done, info = env.step(action)
                if done:
                    if info["winner"] == "Red":  # when red take the last turn and game over, rewards of blue player should be updated
                        agent_Blue.ep_rs[-1] -= 10
                        ep_reward[-1] -= 10
                    elif info["winner"] == "Blue":
                        agent_Blue.ep_rs[-1] += 10
                        ep_reward[-1] += 10
                    else:  # "Tie"
                        agent_Blue.ep_rs[-1] += 2
                        ep_reward[-1] += 2
            obs = copy.deepcopy(obs_)

            if RENDER: env.render()
            if done:  # Game Over
                loss = agent_Blue.learn()
                print("ep: {:d}/{:d}, blue player taining loss value: {:.4f}".format(ep, max_epoch, loss))
                is_Blue.append(True if info["winner"] == "Blue" else False)
                break
        reward_history.append(np.sum(ep_reward))

        if ep % 20 == 0:  # update the weights of the red player
            winning_rate.append(np.mean(is_Blue))
            is_Blue = []
            print("ep: {:d}/{:d}, blue player winning rate in latest 20 rounds: {:.2%}.".format(ep, max_epoch, winning_rate[-1]))

        if len(winning_rate) >= 3 and all([w >= 0.65 for w in winning_rate[-3:]]):
            agent_Red.weights_assign(agent_Blue.brain)
            print("ep: {:d}/{:d}, red player updated.".format(ep, max_epoch))

    # end of training
    agent_Blue.save_model("Brain_PG_Blue10000")
    agent_Red.save_model("Brain_PG_Red10000")
    # plot
    plt.figure("Blue winning rate")
    plt.plot(range(0, max_epoch, 20), winning_rate)
    plt.show()

def modelDQN():
    blue_check_point = None
    red_check_point = None
    agent_Blue = Agent_DQN("Blue", device=device).to(device)
    agent_Red = Agent_DQN("Red", device=device).to(device)
    if blue_check_point:
        agent_Blue.load_model(blue_check_point)
    if red_check_point:
        agent_Red.load_model(red_check_point)

    env = Reversi(human_VS_machine=False)
    reward_history, winning_rate = [], []
    best_model, best_winning_rate = None, 0.  # the one obtained the highest winning rate, regardless of opponent
    is_Blue = []
    max_epoch = 20000
    dominant_counter_blue = 0
    RENDER = False
    for ep in range(1, max_epoch + 1):
        ep_reward = []
        obs, info = env.reset()
        done = False
        if RENDER: env.render()
        while True:
            next_palyer = info["next_player"]
            next_possible_actions = info["next_possible_actions"]

            if next_palyer == "Blue":  # We train the blue
                action = agent_Blue.choose_action(obs, next_possible_actions)
                obs_, reward, done, info = env.step(action)
                ep_reward.append(reward)
                agent_Blue.store_transition(obs, action, reward, done, obs_, next_possible_actions)
            else:
                action = agent_Red.choose_action(obs, next_possible_actions)
                # action = env.get_random_action()
                obs_, reward, done, info = env.step(action)
                if done:
                    if info["winner"] == "Red":  # when red take the last turn and game over, rewards of blue player should be updated
                        agent_Blue.reward_transition_update(-10.)
                    elif info["winner"] == "Blue":
                        agent_Blue.reward_transition_update(10.)
                    else:  # "Tie"
                        agent_Blue.reward_transition_update(2.)
            obs = copy.deepcopy(obs_)

            if RENDER: env.render()
            if done:  # Game Over
                loss = agent_Blue.learn()
                print("ep: {:d}/{:d}, blue player taining loss value: {:.4f}".format(ep, max_epoch, loss))
                is_Blue.append(True if info["winner"] == "Blue" else False)
                break
        reward_history.append(np.sum(ep_reward))

        if ep % 20 == 0:  # log winning rate in every 20 eps
            winning_rate.append(np.mean(is_Blue))
            is_Blue = []
            print("ep: {:d}/{:d}, blue player winning rate in latest 20 rounds: {:.2%}.".format(ep, max_epoch, winning_rate[-1]))
            if best_winning_rate <= winning_rate[-1]:
                best_model = copy.deepcopy(agent_Blue)
                best_winning_rate = winning_rate[-1]
            if winning_rate[-1] >= 0.60:
                dominant_counter_blue += 1
            else:
                dominant_counter_blue = 0
            if dominant_counter_blue >= 3:
                dominant_counter_blue = 0
                agent_Red.weights_assign(agent_Blue.brain_evl)
                print("ep: {:d}/{:d}, red player updated.".format(ep, max_epoch))

    # end of training
    agent_Blue.save_model("Brain_DQN_prioritized_Blue20000")
    agent_Red.save_model("Brain_DQN_prioritized_Red20000")
    best_model.save_model("Brain_DQN_prioritized_Best20000")
    # plot
    plt.figure("Blue winning rate")
    plt.plot(range(0, max_epoch, 20), winning_rate)
    plt.show()
    
def modelDDQN():
    blue_check_point = None
    red_check_point = None
    agent_Blue = Agent_DDQN("Blue", device=device).to(device)
    agent_Red = Agent_DDQN("Red", device=device).to(device)
    if blue_check_point:
        agent_Blue.load_model(blue_check_point)
    if red_check_point:
        agent_Red.load_model(red_check_point)

    env = Reversi(human_VS_machine=False)
    reward_history, winning_rate = [], []
    best_model, best_winning_rate = None, 0.  # the one obtained the highest winning rate, regardless of opponent
    is_Blue = []
    max_epoch = 20000
    dominant_counter_blue = 0
    RENDER = False
    for ep in range(1, max_epoch + 1):
        ep_reward = []
        obs, info = env.reset()
        done = False
        if RENDER: env.render()
        while True:
            next_palyer = info["next_player"]
            next_possible_actions = info["next_possible_actions"]

            if next_palyer == "Blue":  # We train the blue
                action = agent_Blue.choose_action(obs, next_possible_actions)
                obs_, reward, done, info = env.step(action)
                ep_reward.append(reward)
                agent_Blue.store_transition(obs, action, reward, done, obs_, next_possible_actions)
            else:
                action = agent_Red.choose_action(obs, next_possible_actions)
                obs_, reward, done, info = env.step(action)
                if done:
                    if info["winner"] == "Red":  # when red take the last turn and game over, rewards of blue player should be updated
                        agent_Blue.reward_transition_update(-10.)
                    elif info["winner"] == "Blue":
                        agent_Blue.reward_transition_update(10.)
                    else:  # "Tie"
                        agent_Blue.reward_transition_update(2.)
            obs = copy.deepcopy(obs_)

            if RENDER: env.render()
            if done:  # Game Over
                loss = agent_Blue.learn()
                print("ep: {:d}/{:d}, blue player taining loss value: {:.4f}".format(ep, max_epoch, loss))
                is_Blue.append(True if info["winner"] == "Blue" else False)
                break
        reward_history.append(np.sum(ep_reward))

        if ep % 20 == 0:  # log winning rate in every 20 eps
            winning_rate.append(np.mean(is_Blue))
            is_Blue = []
            print("ep: {:d}/{:d}, blue player winning rate in latest 20 rounds: {:.2%}.".format(ep, max_epoch, winning_rate[-1]))
            if best_winning_rate <= winning_rate[-1]:
                best_model = copy.deepcopy(agent_Blue)
                best_winning_rate = winning_rate[-1]
            if winning_rate[-1] >= 0.60:
                dominant_counter_blue += 1
            else:
                dominant_counter_blue = 0
            if dominant_counter_blue >= 3:
                dominant_counter_blue = 0
                agent_Red.weights_assign(agent_Blue.brain_evl)
                print("ep: {:d}/{:d}, red player updated.".format(ep, max_epoch))

    # end of training
    agent_Blue.save_model("Brain_DDQN_prioritized_Blue20000")
    agent_Red.save_model("Brain_DDQN_prioritized_Red20000")
    best_model.save_model("Brain_DDQN_prioritized_Best20000")
    # plot
    plt.figure("Blue winning rate")
    plt.plot(range(0, max_epoch, 20), winning_rate)
    plt.show()

def modelAlfaZero():
    game = Reversi()
    net = NeuralNetworkWrapper(game)

    # Initialize the network with the best model.
    if CFG.load_model:
        file_path = CFG.model_directory + "best_model.meta"
        if os.path.exists(file_path):
            net.load_model("best_model")
        else:
            print("Trained model doesn't exist. Starting from scratch.")
    else:
        print("Trained model not loaded. Starting from scratch.")

    # Play vs the AI as a human instead of training.
    train = Train(game, net)
    train.start()

if __name__=="__main__":
    arguments = parser.parse_args()

    if arguments.model == 0:
        modelPG()
    elif arguments.model == 1:
       modelDQN()
    elif arguments.model == 2:
        modelDDQN()
    elif arguments.model == 3:
        modelAlfaZero()
    else:
        print("Invalid input")