import os
import gym
import cv2
import random

import numpy as np

from gym import Env, spaces
from copy import deepcopy
from collections import Counter

# global variable
font = cv2.FONT_HERSHEY_TRIPLEX
mouse_X = -1
mouse_Y = -1
click_count = 0

EMP, RED, BLU = 0, -1, 1
GRID_STATE = {"Red":RED, "Blue":BLU, "Empty":EMP}

class Disk:
    def __init__(self, color):
        self.color = color
        assert color in ["Red", "Blue"], "invalid color"
        self.shape = (50, 50)  # red and blue disks are of 50 pi in height by 50 pi in width, these are fixed values

class Red(Disk):
    def __init__(self):
        super(Red, self).__init__(color="Red")
        self.icon = cv2.imread(os.path.join("images", "red_disk.png"))

class Blue(Disk):
    def __init__(self):
        super(Blue, self).__init__(color="Blue")
        self.icon = cv2.imread(os.path.join("images", "blue_disk.png"))

class Reversi:
    def __init__(self, human_VS_machine=False):
        super(Reversi, self).__init__()
        self.canvas_shape = (620, 520, 3)  # the chess board, 620 pi in height by 520 pi in width
        self.observation_shape = (8, 8)  # it is a 8 row by 8 col grid
        
        self.row = 8
        self.column = 8
        self.action_size = self.row * self.column

        self.observation_space = spaces.Discrete(8 * 8)  # the observation is a very large discrete space, and I do not want to use it

        self.action_space = spaces.Discrete(8 * 8)  # 64 locations
        self.board = cv2.imread(os.path.join("images", "board.png"))  # shape = (height: num of rows, width:num of cols, channel in (b, g, r))

        # below is the coordinates (y:row, x:col) of the upper left corner of each grid where to put the disk
        self.grid_coordinates = (
            ((122, 17), (122, 78), (122, 140), (122, 202), (122, 264), (122, 324), (122, 386), (122, 446)),
            ((184, 17), (184, 78), (184, 140), (184, 202), (184, 264), (184, 324), (184, 386), (184, 446)),
            ((246, 17), (246, 78), (246, 140), (246, 202), (246, 264), (246, 324), (246, 386), (246, 446)),
            ((304, 17), (304, 78), (304, 140), (304, 202), (304, 264), (304, 324), (304, 386), (304, 446)),
            ((366, 17), (366, 78), (366, 140), (366, 202), (366, 264), (366, 324), (366, 386), (366, 446)),
            ((428, 17), (428, 78), (428, 140), (428, 202), (428, 264), (428, 324), (428, 386), (428, 446)),
            ((490, 17), (490, 78), (490, 140), (490, 202), (490, 264), (490, 324), (490, 386), (490, 446)),
            ((550, 17), (550, 78), (550, 140), (550, 202), (550, 264), (550, 324), (550, 386), (550, 446)))
        
        self.grids = [[EMP, EMP, EMP, EMP, EMP, EMP, EMP, EMP],
                      [EMP, EMP, EMP, EMP, EMP, EMP, EMP, EMP],
                      [EMP, EMP, EMP, EMP, EMP, EMP, EMP, EMP],
                      [EMP, EMP, EMP, RED, BLU, EMP, EMP, EMP],
                      [EMP, EMP, EMP, BLU, RED, EMP, EMP, EMP],
                      [EMP, EMP, EMP, EMP, EMP, EMP, EMP, EMP],
                      [EMP, EMP, EMP, EMP, EMP, EMP, EMP, EMP],
                      [EMP, EMP, EMP, EMP, EMP, EMP, EMP, EMP],
                      ]  # 0=empty, -1=red, 1=blue

        self.red = Red()  # red disk
        self.blue = Blue() # blue disk
        self.red_counter_coor = (205, 83)  # (x, y) of the bottom-left of the text, x: horizontal dist, y: vertical dist, where to display the number of red disks
        self.blue_counter_coor = (285, 83)  # where to display the number of blue disks
        self.winner_display_coor = (235, 40)  # where to display the winner
        self.prompt_display_coor = (195, 40)  # where to display hints, e.g., next player, game over, winner
        
        self.next_player = "Red"  # red alwayst take the first turn
        self.current_player = RED  # red alwayst take the first turn
        
        self.red_count = 2  # initial count
        self.blue_count = 2  # initial count
        self.done = False
        self.next_possible_actions = set()  # a set of the possible coordinates (row, col) for the next player
        self.show_next_possible_actions_hint = True  # if True, show a red plus sign in the grid where the player is allowed to put a disk
        self.dirs = ((0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1))  # eight directions
        self.directions = {
            0: (-1, -1),
            1: (-1, 0),
            2: (-1, 1),
            3: (0, -1),
            4: (0, 1),
            5: (1, -1),
            6: (1, 0),
            7: (1, 1)
        }
        self.human_VS_machine = human_VS_machine
        self.next_possible_actions = self.__get_possible_actions(color="Red")

    def reset(self):
        """
        initialize the environment
        :return: the initialized grid with four disks (2 red and 2 blue at board center),
        info gives who is the next player and what are the possible grids to put the disk
        """
        self.grids = [[EMP, EMP, EMP, EMP, EMP, EMP, EMP, EMP],
                      [EMP, EMP, EMP, EMP, EMP, EMP, EMP, EMP],
                      [EMP, EMP, EMP, EMP, EMP, EMP, EMP, EMP],
                      [EMP, EMP, EMP, RED, BLU, EMP, EMP, EMP],
                      [EMP, EMP, EMP, BLU, RED, EMP, EMP, EMP],
                      [EMP, EMP, EMP, EMP, EMP, EMP, EMP, EMP],
                      [EMP, EMP, EMP, EMP, EMP, EMP, EMP, EMP],
                      [EMP, EMP, EMP, EMP, EMP, EMP, EMP, EMP],
                      ]  # 0=empty, -1=red, 1=blue
        self.red_count = 2  # initial count
        self.blue_count = 2  # initial count
        
        self.next_player = "Red"  # in each ep, red takes the first turn
        self.current_player = RED
        
        self.done = False
        self.next_possible_actions = self.__get_possible_actions(color="Red")
        self.__refresh_canvas()
        self.__img_counter = 0  # for debug purpose
        info = {"next_player": self.next_player, "next_possible_actions": self.next_possible_actions}
        return self.grids, info

    def __refresh_canvas(self):
        self.canvas = self.board.copy()
        for i in range(8):
            for j in range(8):
                if self.grids[i][j] == GRID_STATE["Empty"]: continue
                r, c = self.grid_coordinates[i][j]
                if self.grids[i][j] == GRID_STATE["Red"]: self.canvas[r:r + self.red.shape[0], c:c + self.red.shape[1]] = self.red.icon
                else: self.canvas[r:r + self.blue.shape[0], c:c + self.blue.shape[1]] = self.blue.icon
        self.canvas = cv2.putText(self.canvas, str(self.red_count), self.red_counter_coor, font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        self.canvas = cv2.putText(self.canvas, str(self.blue_count), self.blue_counter_coor, font, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

        if self.show_next_possible_actions_hint:
            for i, j in self.next_possible_actions:
                coor = (self.grid_coordinates[i][j][1]+15, self.grid_coordinates[i][j][0]+32)
                self.canvas = cv2.putText(self.canvas, "+", coor, font, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

        if self.done:
            if self.red_count > self.blue_count:
                self.canvas = cv2.putText(self.canvas, "Red", self.winner_display_coor, font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            elif self.red_count < self.blue_count:
                self.canvas = cv2.putText(self.canvas, "Blue", self.winner_display_coor, font, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
            else:
                self.canvas = cv2.putText(self.canvas, "Tie", self.winner_display_coor, font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        else:
            self.canvas = cv2.putText(self.canvas, self.next_player+"'s turn", self.prompt_display_coor, font, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

    def render(self, mode='human'):
        assert mode in ["human"], "Invalid mode, must be \"human\""
        if mode == "human":
            cv2.imshow("Reversi", self.canvas)
            # cv2.imwrite(".//images//reversi_process//board in step {:0>2d}.png".format(self.__img_counter), self.canvas)  # save images to examine each step if needed
            # self.__img_counter += 1
            cv2.waitKey(200)

    def step(self, action: tuple):
        """
        :param action: a tuple,  (row, col), where to put the disk
        :return: obs, reward, done, info
        info is a dict, telling plays, whos is the next, and what are the possible actions
        """
        assert len(action) == 2, "Invalid Action"
        assert self.action_space.contains(action[0]*8+action[1]), "Invalid Action"
        assert action in self.next_possible_actions, "Invalid Action"
        done = False
        color_self = GRID_STATE[self.next_player]
        color_opponent = GRID_STATE["Blue"] if self.next_player=="Red" else GRID_STATE["Red"]
        reward = 0.

        self.__put(action[0], action[1], self.next_player)

        next_player = "Red" if self.next_player == "Blue" else "Blue"  # opponent's turn
        next_possible_actions = self.__get_possible_actions(next_player)
        if not next_possible_actions:  # if there is no way to put the disk, the opponent skips
            next_player = "Red" if self.next_player == "Red" else "Blue"  # my turn again
            next_possible_actions = self.__get_possible_actions(next_player)
            if not next_possible_actions:
                self.next_possible_actions = set()
                self.next_player = None
                done = True  # there is no way for both players, Game Over
            else:
                reward += 0  # get rewards when force opponent to skip but I can continue to play
                self.next_player = next_player
                self.next_possible_actions = next_possible_actions
        else:
            self.next_player = next_player  # it is opponent's turn
            self.next_possible_actions = next_possible_actions

        self.done = done
        info = {"next_player": self.next_player, "next_possible_actions": self.next_possible_actions}
        if done:
            conclusion = "Gave Over! "
            if self.red_count == self.blue_count:  # Tie
                reward += 2
                info["winner"] = "Tie"
                conclusion += "No winner, ends up a Tie"
            elif self.red_count > self.blue_count:
                info["winner"] = "Red"
                reward += 10 if color_self == GRID_STATE["Red"] else -10
                conclusion += "Winner is Red."
            else:
                info["winner"] = "Blue"
                reward += 10 if color_self == GRID_STATE["Blue"] else -10
                conclusion += "Winner is Blue."
            if self.human_VS_machine:
                print(conclusion)
        self.__refresh_canvas()
        self.current_player = -self.current_player
        return self.grids, reward, done, info

    def __put(self, i:int, j:int, color:str):
        """
        put a disk at (i, j)
        :param i: row index
        :param j: col index
        :param color: the player, it is equivalent to self.next_player
        :return: None
        """
        assert self.grids[i][j] == GRID_STATE["Empty"], "Cannot put a disk in a occupied grid"
        assert color in ("Red", "Blue"), "illegal color input"
        color_self = GRID_STATE[color]
        color_opponent = GRID_STATE["Blue"] if color == "Red" else GRID_STATE["Red"]

        self.grids[i][j] = color_self
        flips = []

        def check_flip(dir, dist, opponent_cnt, i, j, candidates, flips):
            """
            :param dir: direction
            :param dist: number of neighboring disk that has been checked in this direction
            :param opponent_cnt: number of opponent disk that has been checked
            :param i: current row
            :param j: current col
            :param candidates: a list of (row, col) that could be possibly flipped in the direction dir
            :param flips: a list of (row, col) where opponents sits and will be flipped
            :return: None
            """
            if i < 0 or i >= 8 or j < 0 or j >= 8: return  # out of boundary
            if self.grids[i][j] == GRID_STATE["Empty"]: return  # empty, could be an empty neighbor or an empty cell in the direction
            if self.grids[i][j] == color_self:  # find my color in the direction
                if opponent_cnt == 0: return  # it is the neighbor, nothing to flip
                if opponent_cnt > 0:  # yes, can flip component's disks
                    flips += candidates
                    return
            if self.grids[i][j] == color_opponent:  # it is the opponent in the direction, continue the search
                check_flip(dir, dist+1, opponent_cnt+1, i+dir[0], j+dir[1], candidates+[(i, j)], flips)

        for dir in self.dirs:
            check_flip(dir=dir, dist=1, opponent_cnt=0, i=i+dir[0], j=j+dir[1], candidates=[], flips=flips)

        flips = set(flips)
        for f_i, f_j in flips:  # remove duplicated flips
            self.grids[f_i][f_j] = color_self
        if color == "Red":
            self.blue_count, self.red_count = self.blue_count - len(flips), self.red_count + len(flips) + 1
        else:
            self.blue_count, self.red_count = self.blue_count + len(flips) + 1, self.red_count - len(flips)

    def __get_possible_actions(self, color):
        """
        :param color: player
        :return: a set of possible coordinates (in tuples) where the player can put the disk
        """
        assert color in ("Red", "Blue"), "illegal color input"
        actions = []
        color_self = GRID_STATE[color]
        color_opponent = GRID_STATE["Blue"] if color == "Red" else GRID_STATE["Red"]

        def check_possible(dir, dist, opponent_cnt, i, j, actions):
            """
            :param dir: direction
            :param dist: number of neighboring disk that has been checked in this direction
            :param opponent_cnt: number of opponent disk that has been checked
            :param i: current row
            :param j: current col
            :param actions: a list of (row, col) where the next play can put a disk
            :return: None
            """
            if i < 0 or i >= 8 or j < 0 or j >= 8: return  # out of boundary
            if dist == 1 and self.grids[i][j] == GRID_STATE["Empty"]: return  # empty neighbor
            if self.grids[i][j] == color_self: return  # same color in the direction
            if self.grids[i][j] == GRID_STATE["Empty"] and opponent_cnt>0:  # empty and opponents exist in this direction
                actions.append((i, j))
                return
            if self.grids[i][j] == color_opponent: # opponent color in the direction, continue to search in this direction
                check_possible(dir, dist+1, opponent_cnt+1, i+dir[0], j+dir[1], actions)

        for r in range(8):
            for c in range(8):
                if self.grids[r][c] in (GRID_STATE["Empty"], color_opponent): continue
                for dir in self.dirs:
                    check_possible(dir=dir, dist=1, opponent_cnt=0, i=r+dir[0], j=c+dir[1], actions=actions)
        return set(actions)  # duplicates must be removed

    def get_random_action(self):
        if self.next_possible_actions:
            return random.choice(list(self.next_possible_actions))
        return ()

    def close(self):
        cv2.destroyAllWindows()

    def __disk_count(self):
        """
        deprecated function,
        the numbers of red and blue are updated whenever put a new piece
        :return:
        """
        self.red_count, self.blue_count = 0, 0
        for row in self.grids:
            for p in row:
                if p < 0: continue
                self.red_count += 1 if p==0 else 0
                self.blue_count += 1 if p==1 else 0

    def __check_termination(self):
        """
        deprecated function
        check if game over, when both sides do not have further possible actions, game over!
        :return: bool, whether the game is over
        """
        if self.next_player is None:
            return True
        if sum([sum([p < 0 for p in row]) for row in self.grids]) == 0:  # there is no empty grid
            return True
        return False

    def get_human_action(self):
        global mouse_X, mouse_Y, click_count
        mouse_X, mouse_Y, click_count = -1, -1, 0

        def get_mouse_location(event, x, y, flags, param):
            global mouse_X, mouse_Y, click_count
            if event == cv2.EVENT_LBUTTONDOWN:
                mouse_X, mouse_Y = x, y
                click_count += 1

        while click_count < 3:  # after 3 times of illegal put, a random action is generated
            cv2.waitKey(20)
            cv2.setMouseCallback("Reversi", get_mouse_location)
            if mouse_X < 0 or mouse_Y < 0: continue  # no mouse detected
            for i in range(8):
                for j in range(8):
                    y1, x1 = self.grid_coordinates[i][j]
                    y2, x2 = y1 + 49, x1 + 49
                    if y1 <= mouse_Y <= y2 and x1 <= mouse_X <= x2 and (i, j) in self.next_possible_actions:
                        return i, j
            print("illegal action")
            mouse_X, mouse_Y = -1, -1
        return self.get_random_action()

    def clone(self):
        """
        Creates a deep clone of the game object.
        Returns:
            the cloned game object.
        """
        game_clone = Reversi()
        game_clone.grids = deepcopy(self.grids)
        game_clone.current_player = self.current_player
        return game_clone
    
    def play_action(self, action):
        """
        Plays an action on the game board.
        Args:
            action: A tuple in the form of (row, column).
        """
        self.next_player = "Red" if self.current_player == RED else "Blue"
        self.next_possible_actions = self.__get_possible_actions(self.next_player)
        assert len(action) == 3, "Invalid Action"
        assert self.action_space.contains(action[1]*8+action[2]), "Invalid Action"
        done = False

        self.__put(action[1], action[2], self.next_player)

        next_player = "Red" if self.next_player == "Blue" else "Blue"  # opponent's turn
        next_possible_actions = self.__get_possible_actions(next_player)
        if not next_possible_actions:  # if there is no way to put the disk, the opponent skips
            next_player = "Red" if self.next_player == "Red" else "Blue"  # my turn again
            next_possible_actions = self.__get_possible_actions(next_player)
            if not next_possible_actions:
                self.next_possible_actions = set()
                self.next_player = None
                done = True  # there is no way for both players, Game Over
            else:
                self.next_player = next_player
                self.next_possible_actions = next_possible_actions
        else:
            self.next_player = next_player  # it is opponent's turn
            self.next_possible_actions = next_possible_actions

        self.done = done
        self.current_player = -self.current_player
    
    def get_valid_moves(self, current_player):
        """
        Returns a list of moves along with their validity.
        Searches the board for valid sandwich moves.
        Returns:
            A list containing moves as (validity, row, column).
        """
        next_player = "Red" if current_player == RED else "Blue"
        possible_actions = self.__get_possible_actions(next_player)

        valid_moves = []

        for x in range(self.row):
            for y in range(self.column):
                if ((x,y) in possible_actions):
                    valid_moves.append((1, x, y))
                else:
                    valid_moves.append((0, None, None)) 
        return np.array(valid_moves)

    def check_game_over(self, current_player):
        """
        Checks if the game is over and return a possible winner.
        There are 3 possible scenarios.
            a) The game is over and we have a winner.
            b) The game is over but it is a draw.
            c) The game is not over.
        Args:
            current_player: An integer representing the current player.
        Returns:
            A bool representing the game over state.
            An integer action value. (win: 1, loss: -1, draw: 0)
        """

        player_a = current_player
        player_b = -current_player

        player_a_moves = self.get_valid_moves(player_a)
        player_b_moves = self.get_valid_moves(player_b)

        player_a_valid_count = Counter(x[0] == 1 for x in player_a_moves)
        player_b_valid_count = Counter(x[0] == 1 for x in player_b_moves)

        # Check if both players can't play any more moves.
        if player_a_valid_count[True] == 0 or player_b_valid_count[True] == 0:
            _, piece_count = np.unique(self.grids, return_counts=True)

            # Check for the player with the most number of pieces.
            if piece_count[player_a] > piece_count[player_b]:
                return True, 1
            elif piece_count[player_a] == piece_count[player_b]:
                return True, 0
            else:
                return True, -1
        else:
            return False, 0
    
    def print_board(self):
        """Prints the board state."""
        print("   0    1    2    3    4    5    6    7")
        for x in range(self.row):
            print(x, end='')
            for y in range(self.column):
                if self.grids[x][y] == 0:
                    print('  -  ', end='')
                elif self.grids[x][y] == 1:
                    print('  X  ', end='')
                elif self.grids[x][y] == -1:
                    print('  O  ', end='')
            print('\n')
        print('\n')