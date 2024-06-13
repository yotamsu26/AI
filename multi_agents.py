import numpy as np
import abc
import util
from enum import Enum
from game import Agent, Action

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
        """*** YOUR CODE HERE ***"""
        legal_moves = game_state.get_agent_legal_actions()
        
        scores = [self.min_value(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]
    
    def min_value(self, game_state, action, depth=0):
        if depth == self.depth:
            return self.evaluation_function(game_state)
        
        successor = game_state.generate_successor(AgentEnum.OUR_AGENT.value, action)
        legal_moves = successor.get_opponent_legal_actions()
        
        if not legal_moves:
            return self.evaluation_function(successor)
        
        return min([self.max_value(successor, action, depth+1) for action in legal_moves])
    
    def max_value(self, game_state, action, depth=0):
        if depth == self.depth:
            return self.evaluation_function(game_state)
        
        successor = game_state.generate_successor(AgentEnum.OPPONENT.value, action)
        legal_moves = successor.get_agent_legal_actions()
        
        if not legal_moves:
            return self.evaluation_function(successor)
        
        return max([self.min_value(successor, action, depth+1) for action in legal_moves])
    
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        """*** YOUR CODE HERE ***"""
        legal_moves = game_state.get_agent_legal_actions()
        
        scores = [self.min_value(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]
    
    def min_value(self, game_state, action, alpha=float("-inf"), beta=float("inf"),  depth=0):
        if depth == self.depth:
            return self.evaluation_function(game_state)
        
        successor = game_state.generate_successor(AgentEnum.OUR_AGENT.value, action)
        legal_moves = successor.get_opponent_legal_actions()
        
        if not legal_moves:
            return self.evaluation_function(successor)
        
        v = float('inf')
        for action in legal_moves:
            v = min(v, self.max_value(successor, action, alpha, beta, depth+1))
            if v <= alpha:
                return v
            beta = min(beta, v)
            
        return v
    
    def max_value(self, game_state, action, alpha=float("-inf"), beta=float("inf"), depth=0):
        if depth == self.depth:
            return self.evaluation_function(game_state)
        
        successor = game_state.generate_successor(AgentEnum.OPPONENT.value, action)
        legal_moves = successor.get_agent_legal_actions()
        
        if not legal_moves:
            return self.evaluation_function(successor)
        
        v = float('-inf')
        for action in legal_moves:
            v = max(v, self.min_value(successor, action, alpha, beta, depth+1))
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
        """*** YOUR CODE HERE ***"""
        util.raiseNotDefined()





def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = better_evaluation_function
