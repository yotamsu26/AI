import numpy as np
import abc
import ex3.util as util
from enum import Enum
from ex3.game import Agent, Action

class AgentEnum(Enum):
    OUR_AGENT = 0
    OPPONENT = 1
class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        get_action takes a game_state and returns some Action.X for some X in the set {UP, DOWN, LEFT, RIGHT, STOP}
        """

        # Collect legal moves and successor states
        legal_moves = game_state.get_agent_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (GameState.py) and returns a number, where higher numbers are better.

        """

        # Useful information you can extract from a GameState (game_state.py)

        successor_game_state = current_game_state.generate_successor(action=action)
        board = successor_game_state.board
        max_tile = successor_game_state.max_tile
        score = 0
        
        # give weight to the max_tile, to the number of empty tiles and to the position of the tiles
        # the heuristics is based on the fact that the max_tile should be in the corner of the board
        for row in range(len(board)):
            for col in range(len(board[row])):
                score += board[row][col] * (row + len(board) * col) + max_tile*successor_game_state.get_empty_tiles()[0].size
        
        return score
        


def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.score


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        self.evaluation_function = util.lookup(evaluation_function, globals())
        self.depth = depth

    @abc.abstractmethod
    def get_action(self, game_state):
        return


class MinmaxAgent(MultiAgentSearchAgent):
    def get_action(self, game_state):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1

        Action.STOP:
            The stop direction, which is always legal

        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action
        """
        legal_moves = game_state.get_agent_legal_actions()
        
        scores = [self.min_value(game_state.generate_successor(AgentEnum.OUR_AGENT.value, action)) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        return legal_moves[chosen_index]
    
    def min_value(self, game_state, depth=0):
        if depth == self.depth:
            return self.evaluation_function(game_state)
        
        legal_moves = game_state.get_opponent_legal_actions()
        
        if not legal_moves:
            return self.evaluation_function(game_state)
        # Min player increases depth - one ply is the Agent turn and then the board turn
        # so after the board turn update depth
        return min([self.max_value(game_state.generate_successor(AgentEnum.OPPONENT.value, action), depth+1) for action in legal_moves])
    
    def max_value(self, game_state, depth=0):
        if depth == self.depth:
            return self.evaluation_function(game_state)
        
        legal_moves = game_state.get_agent_legal_actions()
        
        if not legal_moves:
            return self.evaluation_function(game_state)

        # Max player doesnt increases depth - one ply is the Agent turn and then the board turn
        # so after the agent turn dont update depth
        return max([self.min_value(game_state.generate_successor(AgentEnum.OUR_AGENT.value, action), depth) for action in legal_moves])


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        legal_moves = game_state.get_agent_legal_actions()
        
        scores = [self.min_value(game_state.generate_successor(AgentEnum.OUR_AGENT.value, action)) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        return legal_moves[chosen_index]
    
    def min_value(self, game_state, alpha=float("-inf"), beta=float("inf"),  depth=0):
        if depth == self.depth:
            return self.evaluation_function(game_state)
        
        legal_moves = game_state.get_opponent_legal_actions()
        
        if not legal_moves:
            return self.evaluation_function(game_state)
        
        v = float('inf')
        for action in legal_moves:
            # Min player increases depth - one ply is the Agent turn and then the board turn
            # so after the board turn update depth
            v = min(v, self.max_value(game_state.generate_successor(AgentEnum.OPPONENT.value, action), alpha, beta, depth+1))
            if v <= alpha:
                return v
            beta = min(beta, v)
            
        return v
    
    def max_value(self, game_state, alpha=float("-inf"), beta=float("inf"), depth=0):
        if depth == self.depth:
            return self.evaluation_function(game_state)
        
        legal_moves = game_state.get_agent_legal_actions()
        
        if not legal_moves:
            return self.evaluation_function(game_state)
        
        v = float('-inf')
        for action in legal_moves:
            # Max player doesnt increases depth - one ply is the Agent turn and then the board turn
            # so after the agent turn dont update depth
            v = max(v, self.min_value(game_state.generate_successor(AgentEnum.OUR_AGENT.value, action), alpha, beta, depth))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        
        return v


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        legal_moves = game_state.get_agent_legal_actions()
        
        scores = [self.exp_value(game_state.generate_successor(AgentEnum.OUR_AGENT.value, action)) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        return legal_moves[chosen_index]
    
    def exp_value(self, game_state, depth=0):
        res_expectation = 0
        
        if depth == self.depth:
            return self.evaluation_function(game_state)
        
        legal_moves = game_state.get_opponent_legal_actions()
        
        if not legal_moves:
            return self.evaluation_function(game_state)

        # Uniform distribution
        probability = 1 / len(legal_moves)
        # Min player increases depth - one ply is the Agent turn and then the board turn
        # so after the board turn update depth
        results = [self.max_value(game_state.generate_successor(AgentEnum.OPPONENT.value, action), depth+1) for action in legal_moves]
        
        for res in results:
            res_expectation += res * probability
        
        return res_expectation
    
    def max_value(self, game_state, depth=0):
        if depth == self.depth:
            return self.evaluation_function(game_state)
        
        legal_moves = game_state.get_agent_legal_actions()
        
        if not legal_moves:
            return self.evaluation_function(game_state)
        # Max player doesnt increases depth - one ply is the Agent turn and then the board turn
        # so after the agent turn dont update depth
        return max([self.exp_value(game_state.generate_successor(AgentEnum.OUR_AGENT.value, action), depth) for action in legal_moves])


def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION: This heuristic function is a linear combination of a few aspect, all of them
    are fulfilling the idea of the strategy that keeps the board as much closer to a "snake" like shape, where the
    highest number is at the corner and then slowly decreasing in a shape that seems like snake:
    - max tile score
    - number of empty tiles
    - merging potential - how much tiles can be merged
    - smoothness of the neighbors values
    - monotonicity in the rows and cols
    - tile alignment to a shape of "snake" matrix
    - corner bias (max tile on the corner)

    we also tried to compute the heuristic on all 4 rotations of the board as it doesnt matter that we
    chose the top left corner as our reference corner, but it really slowed down and didnt improved a lot the results

    """
    board = current_game_state.board
    if not current_game_state.get_agent_legal_actions():
        return float("-inf")

    max_tile_score = current_game_state.max_tile
    empty_tiles_score = np.sum(board == 0)
    merging_potential_score = merging_potential(board)
    smoothness_score = smoothness(board)
    monotonicity_score = monotonicity(board)
    tile_alignment_score = tile_alignment(board)
    corner_bias_score = corner_bias(board)
    heuristic_score = (monotonicity_score * 1.0 +
                       empty_tiles_score * (max_tile_score / 2) +
                       merging_potential_score * 1.5 +
                       max_tile_score * 1.0 +
                       smoothness_score * 0.5 +
                       tile_alignment_score * 1.0 +
                       corner_bias_score * 2.0)
    return heuristic_score


def monotonicity(board):
    """
    Check Monotonicity in the cols and rows, for each row and each cols, if it is monotonic (increasing or decreasing)
    than add the sum of the tiles in that row/col - we want to enforce the "snake" positioning so
    the first and third row should be decreasing and the second and forth rows should be increasing.
    all the cols need to be decreasing
    return the sum of all the monotonic rows and cols.
    """
    scores = []
    # Check rows
    for index, row in enumerate(board):
        if index in [0, 2]:
            cond = all(row[i] >= row[i + 1] for i in range(len(row) - 1))
        else:
            cond = all(row[i] <= row[i + 1] for i in range(len(row) - 1))
        if cond:
            scores.append(np.sum(row))
        else:
            scores.append(0)

    # Check columns
    for col in board.T:
        decreasing = all(col[i] >= col[i + 1] for i in range(len(col) - 1))
        if decreasing:
            scores.append(np.sum(col))
        else:
            scores.append(0)

    return np.sum(scores)


def merging_potential(board):
    """
    For each 2 blocks that could be merged together add the sum of those 2 tiles. (for example if there is 2 128 tiles
    that are neighbors then add 2*128 to the potential.
    """
    potential = 0
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] != 0:
                if i < len(board)-1 and board[i][j] == board[i+1][j]:
                    potential += board[i][j] * 2
                if j < len(board[0])-1 and board[i][j] == board[i][j+1]:
                    potential += board[i][j] * 2
    return potential


def smoothness(board):
    """
    calculate the diff between each tiles and its 2 neighbors - we need that diff to be as minimal as possible.
    (similar reason as monotonicity)
    this score decrease the value of the node as higher diffs are not good (near 512 should be 256 and not 2)
    """
    smoothness_score = 0
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] != 0:
                if i < len(board)-1:
                    smoothness_score -= abs(board[i][j] - board[i+1][j])
                if j < len(board[0])-1:
                    smoothness_score -= abs(board[i][j] - board[i][j+1])
    return smoothness_score


def tile_alignment(board):
    """
    enforce "snake" positioning on the board
    """
    weights = np.array([
        [16, 15, 14, 13],
        [9, 10, 11, 12],
        [8, 7, 6, 5],
        [1, 2, 3, 4]
    ])
    return np.sum(board * weights)


def corner_bias(board):
    """
    Add a bias that the max tile would be in the 0,0 corner
    """
    max_tile = np.max(board)
    if board[0, 0] == max_tile:
        return 1000
    else:
        return 0


# def get_rotations(mat):
#     """
#     Get all 4 rotations of the boards - it doesnt matter which is the corner you are playing relative to
#     so try on all 4 variations of the board.
#     """
#     rotations = [mat]
#     rotations.append(np.rot90(mat, k=3))  # 90 degrees clockwise
#     rotations.append(np.rot90(mat, k=2))  # 180 degrees clockwise
#     rotations.append(np.rot90(mat, k=1))  # 270 degrees clockwise
#     return rotations


# Abbreviation
better = better_evaluation_function
