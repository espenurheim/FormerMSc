# Move root one step out
import os
import sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pickle
from Former.Cpp_code.former_class_cpp import FormerGame, heuristic_min_groups_1_look_ahead, heuristic_min_groups_2_look_ahead

def board_to_tuple(board):
    """
    Converts a NumPy board (2D array) into a tuple of tuples for hashing.
    """
    return tuple(tuple(int(cell) for cell in row) for row in board)

# Function to generate dataset of random boards
def generate_boards(num_boards, N, M, S):
    """
    Generates a dataset of random boards for the FormerGame.
    
    Parameters:
        num_boards (int): Number of boards to generate.
        N (int): Number of rows in the board.
        M (int): Number of columns in the board.
        S (int): Number of different shapes on the board.
        
    Returns:
        list: A list of random boards, each represented as a 2D NumPy array.
    """
    boards = []
    for _ in range(num_boards):
        board = np.random.randint(0, S, size=(N, M), dtype=np.int32)
        boards.append(board)
    return boards

def generate_beam_search_game(initial_board, beam_width=2, max_depth=50, network=None, T=2):
    """
    Uses beam search to find a sequence of moves that clears the board
    for FormerGame. The algorithm uses a heuristic function
    to compute a probability distribution over valid actions. These probabilities are multiplied along the branch
    to form a cumulative score, and nodes with the highest scores are prioritized.
    
    This version works for any custom board. It deduces the board dimensions and number of shapes from the input board.
    
    Parameters:
        initial_board (np.array): The starting board.
        beam_width (int): Maximum number of nodes to keep at each depth.
        max_depth (int): Maximum search depth (number of moves).
        
    Returns:
        (board_sequence, move_sequence, n_moves): A tuple where board_sequence is a list of boards from the same game,
        move_sequence is a list of moves (each a (row, col) tuple), and n_moves is a vector (list) of the number of moves
        required to clear the board from each board in the sequence. For example, if the board is cleared in L moves,
        then the board at index i requires (L - i) moves to be cleared.
    """
    rows, cols = initial_board.shape
    unique_vals = np.unique(initial_board)
    unique_vals = unique_vals[unique_vals != -1]
    S = len(unique_vals)
    
    initial_game = FormerGame(M=rows, N=cols, S=S, custom_board=initial_board.copy())
    
    # Each beam node is a tuple: (game_instance, board_sequence, move_sequence, cumulative_score)
    beam = [(initial_game, [initial_game.get_board().copy()], [], 1.0)]
    visited = set()
    
    for depth in range(max_depth):
        new_beam = []
        for game, board_seq, move_seq, cum_score in beam:
            # If the board is cleared, compute and return the data.
            if game.is_game_over():
                L = len(move_seq)
                n_moves = [L - i for i in range(L + 1)]
                return board_seq, move_seq, n_moves
            
            valid_actions = game.get_valid_turns()
            if not valid_actions:
                continue
            
            # Get the heuristic values
            if T == 1:
                probs = heuristic_min_groups_1_look_ahead(game.get_board(), valid_actions)
            elif T == 2:
                probs = heuristic_min_groups_2_look_ahead(game.get_board(), valid_actions)
                
            
            # Check all valid actions and expand the beam
            for action, prob in zip(valid_actions, probs):
                new_game = FormerGame(M=rows, N=cols, S=S, custom_board=game.get_board())
                new_game.make_turn(action)
                new_board_seq = board_seq + [new_game.get_board().copy()]
                new_move_seq = move_seq + [action]
                new_cum_score = cum_score * prob
                board_key = board_to_tuple(new_game.get_board())
                if board_key in visited:
                    continue
                visited.add(board_key)
                new_beam.append((new_game, new_board_seq, new_move_seq, new_cum_score))
        
        if not new_beam:
            break
        
        new_beam.sort(key=lambda node: node[3], reverse=True)
        beam = new_beam[:beam_width]
    
    best_node = max(beam, key=lambda node: node[3])
    L = len(best_node[2])
    n_moves = [L - i for i in range(L + 1)]
    return best_node[1], best_node[2], n_moves


def generate_beam_search_data(num_boards, N, M, S, beam_width=2, max_depth=50, save_path=None, T=2):
    """
    Generates training data using beam search for the FormerGame.
    
    Parameters:
        num_boards (int): Number of boards to generate.
        N (int): Number of rows in the board.
        M (int): Number of columns in the board.
        S (int): Number of different shapes on the board.
        beam_width (int): Maximum number of nodes to keep at each depth.
        max_depth (int): Maximum search depth (number of moves).
        
    Returns:
        list: A list of tuples where each tuple contains all boards from the same game and corresponding move sequence, as well as number of moves.
    """
    print("Generating boards...")
    boards = generate_boards(num_boards, N, M, S)
    training_data = []
    print("Generating training data...")
    

    for i, board in enumerate(boards):
        if i % 100 == 0:
            print(f"# {i}/{num_boards}...")
        board_seq, move_seq, n_moves = generate_beam_search_game(board, beam_width, max_depth,T=T)
        training_data.append((board_seq, move_seq, n_moves))
        
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(training_data, f)
    return training_data
    
def load_training_data(file_path):
    """
    Loads training data from a pickle file.
    
    Parameters:
        file_path (str): Path to the pickle file.
        
    Returns:
        list: Loaded training data.
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data