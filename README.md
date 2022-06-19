# RL_Reversi
[Reversi](https://en.wikipedia.org/wiki/Reversi) is a strategy board game for two players, played on an 8×8 uncheckered board. 
It was invented in 1883. Othello, a variant with a fixed initial setup of the board, was patented in 1971. 
You can play with a PC or another human player. The rule is very simple. You can have a try [here](https://www.mathsisfun.com/games/reversi.html).
### Python with Gym
#### reversi/reversi_env.py
It includes essential classes and methods for realizing the interactive Reversi game.
The class Reversi may be the most important part:
```python
class Reversi:
    def __init__(self, human_VS_machine=False):
        ...
    def reset(self):
        ...
    def __refresh_canvas(self):
        ...
    def render(self, mode='human'):
        ...
    def step(self, action: tuple):
        ...
    def __put(self, i:int, j:int, color:str):
        ...
    def __get_possible_actions(self, color):
        ...
    def get_random_action(self):
        ...
    def close(self):
        ...
    def __piece_count(self):
        ...
    def __check_termination(self):
        ...
    def get_human_action(self):
        ...
    def clone(self):
        ...
    def play_action(self, action):
        ...
    def get_valid_moves(self, current_player):
        ...
    def check_game_over(self, current_player):
        ...
    def print_board(self):
        ...
```
Actually, it just weakly relies on Gym since only the data type is used such as `self.observation_space = spaces.Discrete(8 * 8)`, 
all the other realizations are indepandent from Gym (but the entire framework was referred to Gym realizations).


#### agent/...
The algorithms PG, DQN, and DDQN are in this folder.


#### alfazero/...
AlfaZero's implementation can be found in this folder. AlphaZero implementation based on ["Mastering the game of Go without human knowledge"](https://deepmind.com/documents/119/agz_unformatted_nature.pdf) 
and ["Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"](https://arxiv.org/abs/1712.01815) by DeepMind. The algorithm learns to play games like Chess and Go without any human knowledge. 
It uses Monte Carlo Tree Search and a Deep Residual Network to evaluate the board state and play the most promising move.

#### train.py
The process of training your models. Pick a training model (0 - PG (default); 1 - DQN; 2 - DDQN; 3 - AlfaZero;)
```
python train.py --model 0
``` 

#### play_the_game.py
Play the game with a trained AI. Pick a model to play with (0 - PG (default); 1 - DQN; 2 - DDQN; 3 - AlphaZero;).
```
python play_the_game.py --model 0
``` 
