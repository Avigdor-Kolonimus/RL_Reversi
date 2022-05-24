from alfazero.config import CFG
from alfazero.mcts import TreeNode

# Class to evaluate network.
class Evaluate(object):
    """
    Represents the Policy and Value Resnet.
    Attributes:
        current_mcts: An object for the current network's MCTS.
        eval_mcts: An object for the evaluation network's MCTS.
        game: An object containing the game state.
    """

    def __init__(self, current_mcts, eval_mcts, game):
        """Initializes Evaluate with the both network's MCTS and game state."""
        self.current_mcts = current_mcts
        self.eval_mcts = eval_mcts
        self.game = game

    def evaluate(self):
        """
        Play self-play games between the two networks and record game stats.
        Returns:
            Wins and losses count from the perspective of the current network.
        """
        wins = 0
        losses = 0

        # Self-play loop
        for i in range(CFG.num_eval_games):
            print("Start Evaluation Self-Play Game:", i+1)

            game = self.game.clone()  # Create a fresh clone for each game.
            game_over = False
            value = 0
            node = TreeNode()

            player = game.current_player

            # Keep playing until the game is in a terminal state.
            while not game_over:
                # MCTS simulations to get the best child node.
                # If player_to_eval is 1 play using the current network
                # Else play using the evaluation network.
                if game.current_player == 1:
                    best_child = self.current_mcts.search(game, node, CFG.temp_final)
                else:
                    best_child = self.eval_mcts.search(game, node, CFG.temp_final)

                action = best_child.action
                game.play_action(action)  # Play the child node's action.

                game_over, value = game.check_game_over(player)

                best_child.parent = None
                node = best_child  # Make the child node the root node.

            if value == 1:
                wins += 1
            elif value == -1:
                losses += 1
        return wins, losses
