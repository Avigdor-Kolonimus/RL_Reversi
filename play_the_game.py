import cv2
import torch
import argparse

from agents.agent_pg import Agent_PG
from agents.agent_dqn import Agent_DQN
from agents.agent_ddqn import Agent_DDQN
from reversi.reversi_env import Reversi
from alfazero.mcts import MonteCarloTreeSearch, TreeNode
from alfazero.neural_net import NeuralNetworkWrapper
from alfazero.config import CFG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Code to read command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model",
                    help="Pick a model to play with (0 - PG (default); 1 - DQN; 2 - DDQN; 3 - AlphaZero;).",
                    dest="model",
                    type=int,
                    default=0)


def playPG(HUMAN_vs_MACHINE=True):
    machine = Agent_PG("Blue", device=device).to(device)
    machine.load_model(name="Brain_PG_Blue10000")
    play(HUMAN_vs_MACHINE, machine)

def playDGN(HUMAN_vs_MACHINE=True):
    machine = Agent_DQN("Blue", device=device).to(device)
    machine.load_model("Brain_DQN_prioritized_Blue20000")
    play(HUMAN_vs_MACHINE, machine)

def playDDGN(HUMAN_vs_MACHINE=True):
    machine = Agent_DDQN("Blue", device=device).to(device)
    machine.load_model("Brain_DDQN_prioritized_Blue20000")
    play(HUMAN_vs_MACHINE, machine)

def play(HUMAN_vs_MACHINE=True, machine=object):
    env = Reversi(human_VS_machine=HUMAN_vs_MACHINE)
    
    if HUMAN_vs_MACHINE:  # human vs machine
        for ep in range(1):
            obs, info = env.reset()  # {"next_player": self.next_player, "next_possible_actions":
            env.render()  # show the initialization
            while True:
                if info["next_player"] == "Blue":  # machine's turn
                    action = machine.choose_action(obs, info["next_possible_actions"])
                else:  # human's turn
                    action = env.get_human_action()
                obs, _, done, info = env.step(action)
                env.render()
                if done:
                    break
            cv2.waitKey(3000)  # wait for 3 seconds after the end of each ep
        cv2.waitKey()
        env.close()
    else:
        win_counter_blue = 0
        round_max = 200
        RENDER = False
        for ep in range(round_max):
            obs, info = env.reset()  # {"next_player": self.next_player, "next_possible_actions":
            if RENDER: env.render()  # show the initialization
            while True:
                if info["next_player"] == "Blue":  # machine's turn
                    action = machine.choose_action(obs, info["next_possible_actions"])
                else:
                    action = env.get_random_action()
                obs, _, done, info = env.step(action)
                if RENDER: env.render()
                if done:
                    win_counter_blue += 1 if info["winner"] == "Blue" else 0
                    print("Round: {:d}/{:d}, winner is ".format(ep, round_max), info["winner"])
                    break
            cv2.waitKey(3000)  # wait for 3 seconds after the end of each ep
        print("Winning rate of the blue player is {:.2%}".format(win_counter_blue/round_max))
        cv2.waitKey()
        env.close()    

def playAlfaZero():
    env = Reversi()
    net = NeuralNetworkWrapper(env)

    mcts = MonteCarloTreeSearch(net)
    game = env.clone()  # Create a fresh clone for each game.
    node = TreeNode()

    if HUMAN_vs_MACHINE:  # human vs machine
        for ep in range(1):
            obs, info = game.reset()  # {"next_player": self.next_player, "next_possible_actions":
            game.render()  # show the initialization
            while True:
                if info["next_player"] == "Blue":  # machine's turn
                    best_child = mcts.search(game, node, CFG.temp_final)
                else:  # human's turn
                    action = game.get_human_action()
                    best_child = TreeNode()
                    best_child.action = (1, action[0], action[1])
            
                action = (best_child.action[1], best_child.action[2])
                obs, _, done, info = game.step(action)
                # game.play_action(action)  # Play the child node's action.

                # done, _ = game.check_game_over(game.current_player)

                best_child.parent = None
                node = best_child  # Make the child node the root node.

                game.render()
                if done:
                    break
            cv2.waitKey(3000)  # wait for 3 seconds after the end of each ep
        cv2.waitKey()
        game.close()
        env.close()

if __name__ == "__main__":
    arguments = parser.parse_args()
    HUMAN_vs_MACHINE = True  # Human are always Red players

    if arguments.model == 0:
        playPG(HUMAN_vs_MACHINE)
    elif arguments.model == 1:
        playDGN(HUMAN_vs_MACHINE)
    elif arguments.model == 2:
        playDDGN(HUMAN_vs_MACHINE)
    elif arguments.model == 3:
        playAlfaZero()
    else:
        print("Invalid input")

