"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import math

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    my_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    weight, height = game.width / 2., game.height / 2.
    coord_y, coord_x = game.get_player_location(player)
    coord_y2, coord_x2 = game.get_player_location(game.get_opponent(player))

    distance_to_center_my = float((height - coord_y)**2 + (weight - coord_x)**2)
    distance_to_center_opp = float((height - coord_y2)**2 + (weight - coord_x2)**2)
    return float((my_moves - opp_moves) + (distance_to_center_opp - distance_to_center_my)*0.6/(game.move_count))


def custom_score_2(game, player):
    my_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(my_moves - opp_moves)

def custom_score_3(game, player):
    
    def percent_game_completed(low, high, game):
        game_completion_percentage = game.move_count/(game.height * game.width)*100
        if low <= game_completion_percentage and high > game_completion_percentage:
            return True
        return False

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    my_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    coord_y, coord_x = game.get_player_location(player)
    boxes_to_center = abs(coord_x - math.ceil(game.width/2)) + \
        abs(coord_y - math.ceil(game.height/2)) - 1

    if percent_game_completed(0, 20, game):
        return 2*my_moves - 0.5*boxes_to_center
    elif percent_game_completed(20, 50, game):
        return float(3*my_moves - opp_moves - 0.5*boxes_to_center)
    else:
        return 2*my_moves - opp_moves

class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):

        self.time_left = time_left

        best_move = (-1, -1)

        try:
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  

        return best_move

    def __min_value(self, game, depth):
        self.__check_time()
        if self.__is_terminal(game, depth):
            return self.score(game, self)
        min_val = float("inf")
        legal_moves = game.get_legal_moves()
        for move in legal_moves:
            forecast = game.forecast_move(move)
            min_val = min(min_val, self.__max_value(forecast, depth - 1))
        return min_val

    def __max_value(self, game, depth):
        self.__check_time()
        if self.__is_terminal(game, depth):
            return self.score(game, self)
        max_val = float("-inf")
        legal_moves = game.get_legal_moves()
        for move in legal_moves:
            forecast = game.forecast_move(move)
            max_val = max(max_val, self.__min_value(forecast, depth - 1))
        return max_val

    def __is_terminal(self, game, depth):
        """Helper method to check if we've reached the end of the game tree or
        if the maximum depth has been reached.
        """
        self.__check_time()
        if len(game.get_legal_moves()) != 0 and depth > 0:
            return False
        return True

    def __check_time(self):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

    def minimax(self, game, depth):

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return (-1, -1)
        vals = [(self.__min_value(game.forecast_move(m), depth - 1), m) for m in legal_moves]
        _, move = max(vals)
        return move



class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def __max_value(self, game, depth, alpha, beta):
        self.__check_time()
        best_move = (-1, -1)
        if self.__is_terminal(game, depth):
            return (self.score(game, self), best_move)
        value = float("-inf")
        legal_moves = game.get_legal_moves()
        for move in legal_moves:
            result = self.__min_value(game.forecast_move(move), depth - 1, alpha, beta)
            if result[0] > value:
                value, _ = result
                best_move = move
            if value >= beta:
                return (value, best_move)
            alpha = max(alpha, value)
        return (value, best_move)

    def __min_value(self, game, depth, alpha, beta):
        self.__check_time()
        best_move = (-1, -1)
        if self.__is_terminal(game, depth):
            return (self.score(game, self), best_move)
        value = float("inf")
        legal_moves = game.get_legal_moves()
        for move in legal_moves:
            result = self.__max_value(game.forecast_move(move), depth - 1, alpha, beta)
            if result[0] < value:
                value, _ = result
                best_move = move
            if value <= alpha:
                return (value, best_move)
            beta = min(beta, value)
        return (value, best_move)

    def __is_terminal(self, game, depth):
        self.__check_time()
        if len(game.get_legal_moves()) != 0 and depth > 0:
            return False
        return True

    def __check_time(self):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

    def get_move(self, game, time_left):
        self.time_left = time_left

        legal_moves = game.get_legal_moves(self)
        
        if len(legal_moves) > 0:
            best_move = legal_moves[random.randint(0, len(legal_moves)-1)]
        else:
            best_move = (-1, -1)
        
        try:
            depth = 1
            while True:
                current_move = self.alphabeta(game, depth)
                if current_move == (-1, -1):
                    return best_move
                else:
                    best_move = current_move
                depth += 1
        except SearchTimeout:
            return best_move

        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        self.__check_time()
        _, move = self.__max_value(game, depth, alpha, beta)
        return move
