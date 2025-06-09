import copy
from collections import deque
import numpy as np
from Former.Cpp_code.former_class_cpp import FormerGame, heuristic_min_groups_1_look_ahead, heuristic_min_groups_2_look_ahead, heuristic_min_groups_3_look_ahead


def board_to_tuple(board):
    """
    Converts a NumPy board (2D array) into a tuple of tuples so that it can be used as a key in a set.
    """
    return tuple(tuple(int(cell) for cell in row) for row in board)

def get_optimal_list_of_actions(initial_board):
    """
    Finds the optimal (fewest-turn) sequence of valid moves to clear the board.
    Each move is a (row, col) tuple.
    This function uses a breadth-first search (BFS) approach.
    """
    start_board = initial_board.copy() if isinstance(initial_board, np.ndarray) else np.array(initial_board)
    start_state = board_to_tuple(start_board)
    queue = deque([(start_board, [])])
    visited = {start_state}
    
    while queue:
        current_board, moves = queue.popleft()
        
        if FormerGame.is_game_over_static(current_board):
            return moves
        
        game = FormerGame(custom_board=current_board.copy())
        valid_turns = game.get_valid_turns()
        for turn in valid_turns:
            new_game = FormerGame(custom_board=current_board.copy())
            new_game.make_turn(turn)
            new_board = new_game.get_board()
            new_state = board_to_tuple(new_board)
            
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_board, moves + [turn]))
    
    return None

def _normalize_point(pt):
    """
    Ensure the point is a flat (int, int) tuple,
    even if it comes in as numpy scalars or nested sequences.
    """
    try:
        r, c = pt
    except Exception:
        raise ValueError(f"Invalid point format: {pt!r}")
    def to_int(x):
        if hasattr(x, "item"):
            x = x.item()
        if isinstance(x, (list, tuple)) and len(x) == 1:
            x = x[0]
        return int(x)
    return (to_int(r), to_int(c))


def find_point_minimizing_groups_static(board, T=1):
    """
    Recursively finds the move whose look-ahead number of groups
    (at depth T) are lexicographically minimal, using the static C++ bindings.
    """
    if hasattr(board, "tolist"):
        board_list = board.tolist()
    else:
        board_list = board

    raw_actions = FormerGame.get_valid_turns_static(board_list)
    valid_actions = [_normalize_point(a) for a in raw_actions]
    current_groups = len(FormerGame.get_groups_static(board_list))

    if not valid_actions:
        return ((current_groups,) * T, None)

    best_score = None
    best_move  = None

    # For each action
    for action in valid_actions:
        next_board = FormerGame.apply_turn_static(board_list, action)
        g1 = len(FormerGame.get_groups_static(next_board))

        # Find the action that minimizes the number of groups T-1 moves ahead (recursively)
        if T > 1:
            future_score, _ = find_point_minimizing_groups_static(next_board, T-1)
            score = future_score + (g1,)
        else:
            score = (g1,)

        # If this is the first score or a better one, update best_score and best_move
        if best_score is None or score < best_score:
            best_score = score
            best_move  = action

    return best_score, best_move

def find_point_minimizing_groups_2(board, T=1):
    """
    Does the same as the previous, but with CPP code. Did not really make it any faster, but are more
    difficult to work with, so we do not use this one.
    """
    game = FormerGame(custom_board=board.copy())
    valid_turns = game.get_valid_turns()
    current_groups = len(game.get_groups())
    
    # Base case: If no moves are available, return a score tuple with the current groups repeated.
    if not valid_turns:
        return ((current_groups,) * T, None)
    
    if T == 1:
        best_point_index = int(np.argmax(heuristic_min_groups_1_look_ahead(board, valid_turns)))
        best_point = valid_turns[best_point_index]
    
    elif T == 2:
        best_point_index = int(np.argmax(heuristic_min_groups_2_look_ahead(board, valid_turns)))
        best_point = valid_turns[best_point_index]
    elif T == 3:
        best_point_index = int(np.argmax(heuristic_min_groups_3_look_ahead(board, valid_turns)))
        best_point = valid_turns[best_point_index]
    return best_point

def play_game_with_minimizing_groups(board_init, T=1):
    """
    Plays the game greedily with T-look-ahead heuristic.
    """
    game = FormerGame(custom_board=copy.deepcopy(board_init))
    step_counter = 0
    points = []
    
    while not game.is_game_over():
        groups_remaining = len(game.get_groups())
        best_point = find_point_minimizing_groups_static(copy.deepcopy(game.board), T)[1]
        if best_point is None:
            break
        
        points.append(best_point)
        game.make_turn(best_point)
        step_counter += 1
    
    return step_counter, points

def play_game_choosing_largest_group(board_init):
    """
    Play game always choosing the largest group
    """
    move_counter = 0
    while not FormerGame.is_game_over_static(board_init):
        groups = FormerGame.get_groups_static(board_init)
        length = 0
        for group in groups:
            if len(group) > length:
                length = len(group)
                best_action = group.pop()
        board_init = np.array(FormerGame.apply_turn_static(board_init, best_action))
        move_counter += 1
    return move_counter

def play_game_randomly(board_init):
    """
    Play game choosing random actions
    """
    move_counter = 0
    while not FormerGame.is_game_over_static(board_init):
        legal_actions = FormerGame.get_valid_turns_static(board_init)
        chosen_index = np.random.randint(0, len(legal_actions))
        chosen_action = legal_actions[chosen_index]
        board_init = np.array(FormerGame.apply_turn_static(board_init, chosen_action))
        move_counter += 1
    return move_counter

def play_game_leave_one_color(board_init, frac = 0.5, heur = 'random'):
    """
    A heuristic we tried that did not really work, so we left it out of the paper. Idea: do not select a color until frac of the board
    is occupied by that color. Then, always select the color with the most shapes. Inspired by Schadd et al. (see thesis pdf)
    """
    move_counter = 0
    max_count = 0
    max_col = None
    for i in range(4):
        shape_count = np.sum(board_init == i)
        if shape_count > max_count:
            max_col = i
            max_count = shape_count
    
    while not FormerGame.is_game_over_static(board_init):
        legal_actions = FormerGame.get_valid_turns_static(board_init)
        frac_max_col = np.sum(board_init == max_col) / np.sum(board_init != -1)
        if frac_max_col < frac:    
            actually_legal = []
            for action in legal_actions:
                if board_init[action] == max_col:
                    pass
                else:
                    actually_legal.append(action)
            legal_actions = actually_legal
        
        if heur == 'random':
            chosen_index = np.random.randint(0, len(legal_actions))
            chosen_action = legal_actions[chosen_index]
        
        if heur == '1lookahead':
            heur_prob = heuristic_min_groups_1_look_ahead(board_init, legal_actions)
            if len(legal_actions) == 1:
                move_counter += 1
                return move_counter
            best_heur_prob = 0
            best_action = None
            for i, action in enumerate(legal_actions):
                if heur_prob[i] > best_heur_prob:
                    best_heur_prob = heur_prob[i]
                    best_action = action
            chosen_action = best_action
        
        if heur == '2lookahead':
            heur_prob = heuristic_min_groups_2_look_ahead(board_init, legal_actions)
            if len(legal_actions) == 1:
                move_counter += 1
                return move_counter
            best_heur_prob = 0
            best_action = None
            for i, action in enumerate(legal_actions):
                if heur_prob[i] > best_heur_prob:
                    best_heur_prob = heur_prob[i]
                    best_action = action
            chosen_action = best_action

            
        
        board_init = np.array(FormerGame.apply_turn_static(board_init, chosen_action))
        move_counter += 1
    return move_counter