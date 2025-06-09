from Cpp_code.former_class_cpp import FormerGame, heuristic_min_groups_1_look_ahead, heuristic_min_groups_2_look_ahead
import numpy as np
import pickle
import torch
import time
import daily_board as db
import matplotlib.pyplot as plt
from heuristics import find_point_minimizing_groups_static
from PPO_classes import evaluate_state_critic, load_ppo_models

############################################################################################################
# BEAM SEARCH WITH NEURAL NETWORKS
############################################################################################################        

N_ROWS, N_COLUMNS, N_SHAPES = 9, 7, 4
def one_hot_encode(board, S):
    """
    Turns a "regular" board into a one-hot encoded representation (like psi in the paper).
    """
    H, W = board.shape
    onehot = np.zeros((S+1, H, W), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            v = board[i, j]
            if v < 0:
                onehot[S, i, j] = 1.0
            else:
                onehot[int(v), i, j] = 1.0
    return onehot

def beam_search_neural(initial_state, model, beam_width = 2, max_depth=50, value_type = 'network', T = 1):
    """
    Perform a beam search where at each level the beam_width best states 
    (according to the neural network's evaluation) are kept.
    
    Input:
        initial_state:  The initial board state as a numpy array.
        model:          The neural network model with an evaluate_state() method. If value_type is not network, this one is ignored.
        beam_width:     The number of states to keep at each depth.
        max_depth:      The maximum depth of the search (number of moves).
        value_type:     'network' for neural network evaluation, 'heuristic' for heuristic evaluation,
                        or 'ppo' for PPO critic evaluation.
        T:              Look-ahead depth for heuristics (1 or 2), and PPO critic choice: 1 is (32, 5) critic, 2 is (64, 10) critic.
    """
    # Each beam entry is a tuple: (score, current_state, path) where score is provided by heuristics, value networks or PPO critic
    if value_type == 'network':
        beam = [(model.evaluate_state(initial_state), initial_state, [])]
    elif value_type == 'heuristic':
        beam = [(find_point_minimizing_groups_static(initial_state, T=T)[0][0], initial_state, [])]
    elif value_type == 'ppo':
        beam = [(evaluate_state_critic(initial_state, model), initial_state, [])]
    for depth in range(max_depth):
        next_beam = []  # Store all possible next actions in this list
        
        for score, state, path in beam:
            # If we have reached a goal state, return the path and state
            if FormerGame.is_game_over_static(state):
                return state, path
            
            valid_turns = FormerGame.get_valid_turns_static(state)
            for move in valid_turns:
                new_state = np.array(FormerGame.apply_turn_static(state, move))
                if value_type == 'network':
                    new_score = model.evaluate_state(new_state)
                elif value_type == 'heuristic':
                    if T == 0:
                        new_score = len(FormerGame.get_groups_static(new_state))
                    else:
                        new_score = find_point_minimizing_groups_static(new_state, T=T)[0][0]
                elif value_type == 'ppo':
                    new_score = evaluate_state_critic(new_state, model)
                next_beam.append((new_score, new_state, path + [move]))
        
        if not next_beam:
            # No more states to expand, search ends.
            print("No further states to expand.")
            break

        next_beam.sort(key=lambda x: x[0])
        beam = next_beam[:beam_width]
    
    print("Goal state not found within max_depth.")
    return None

def test_value_net_with_beam_search(model, w=32, d=5, beam_widths = [1, 2, 4, 8, 16, 32, 64, 128, 256], save=True, is_value_net=True):
    daily_boards = db.get_daily_board()
    beam_sol_array = np.zeros((len(daily_boards), len(beam_widths)))
    real_sol_array = np.zeros((len(daily_boards), len(beam_widths)))
    time_array = np.zeros((len(daily_boards), len(beam_widths)))
    if model is not None:
        model.to(torch.device("cpu"))
    # For each daily board: use beam search with each beam width. Store the results.
    i = 0
    for key, board_tuple in daily_boards.items():
        print(f"Board {i+1} / {len(daily_boards)}: {key}")
        board = board_tuple[0]
        for j, beam_width in enumerate(beam_widths):
            # Perform beam search with the given board and beam width.
            
            
            if model is None:
                start_time = time.time()
                move_sequence, _ = beam_search_with_heuristic(board, beam_width)
                end_time = time.time()
            else:
                start_time = time.time()
                _, move_sequence = beam_search_neural(board, model, beam_width, is_value_net=is_value_net)
                end_time = time.time()
            
            
            # Store the results.
            beam_sol_array[i, j] = len(move_sequence)
            real_sol_array[i, j] = board_tuple[1]
            time_array[i, j] = end_time - start_time
        i += 1
    
    if save:
        if model is not None:
            if is_value_net:
                np.savez(f'/Users/espen/Desktop/masteroppgave_uten_chat/results/daily_board_comparison/beam_search/neural_search_{w}_{d}.npz',
                        beam_sol_array=beam_sol_array,
                        real_sol_array=real_sol_array,
                        time_array=time_array,
                        beam_widths=beam_widths)
            else:
                np.savez(f'/Users/espen/Desktop/masteroppgave_uten_chat/results/daily_board_comparison/beam_search/neural_search_dual_{w}_{d}.npz',
                        beam_sol_array=beam_sol_array,
                        real_sol_array=real_sol_array,
                        time_array=time_array,
                        beam_widths=beam_widths)
        else:
            np.savez(f'/Users/espen/Desktop/masteroppgave_uten_chat/results/daily_board_comparison/beam_search/heuristic_search.npz',
                    beam_sol_array=beam_sol_array,
                    real_sol_array=real_sol_array,
                    time_array=time_array,
                    beam_widths=beam_widths)
    
    return {'neural_solution': beam_sol_array, 'real_solution': real_sol_array, 'times': time_array, 'beam_widths': beam_widths}

def beam_search_neural_anytime(board, best_sol, network, return_best = True, max_depth=50, max_beam_width=None, value_type='network', t=2):
    """
    An anytime wrapper around beam_search_neural.
    It tries beam widths 2, 4, 8, … until it finds a solution of length == best_sol.
    
    Args:
        board:         initial board state
        best_sol:      known optimal number of moves
        network:       your neural-network model, with evaluate_state()
        max_depth:     forwarded to beam_search_neural
        max_beam_width: upper cap on beam width (optional)
    
    Returns:
        (final_state, moves) for the first solution of length == best_sol,
        or None if no such solution is found by max_beam_width.
    """
    beam_width = 1
    best_sol_found = 63
    print(f"Best actual solution: {best_sol}")
    while True:
        print(f"Beam width = {beam_width}")
        a = time.time()
        result = beam_search_neural(board, network, beam_width, max_depth, value_type, T=t)
        b = time.time()
        if result is not None:
            state, moves = result
            if len(moves) < best_sol_found:
                print(f"Found solution length = {len(moves)} (runtime = {round(b-a,2)} seconds): {moves}")
                best_sol_found = len(moves)
            if len(moves) == best_sol and return_best:
                # optimal solution found
                return len(moves), moves
        
        beam_width *= 2
        
def plot_value_beam_performance(files):
    """
    Plot metrics to compare beam search with heuristic and beam search with value network.
    Input format: dictionary, 'name' -> 'saved_file_path')
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    plt.suptitle('Beam Search Performance Metrics with Different Beam Widths')
    legend = []
    most_widths = None
    most_widths_length = 0
    for name, path in files.items():
        data = np.load(path)
        beam_widths = data["beam_widths"]
        if len(beam_widths) > most_widths_length:
            most_widths = beam_widths
            most_widths_length = len(beam_widths)
        beam_sol = data["beam_sol_array"]
        real_sol = data["real_sol_array"]
        times = data['time_array']
        ax1.plot(beam_widths, np.mean(beam_sol == real_sol, axis=0), marker='o')
        ax2.errorbar(beam_widths, np.mean(beam_sol - real_sol, axis=0), 
             yerr=np.std(beam_sol - real_sol, axis=0), marker='o', capsize=5)
        ax3.plot(beam_widths, np.mean(times, axis=0), marker='o')
        legend.append(name)
    
    ax1.set_xscale('log')
    ax1.set_xticks(most_widths)
    ax1.set_xticklabels(most_widths)  # Explicitly set the tick labels
    ax1.set_xlabel('Beam Width')
    ax1.set_ylabel('Proportion of Correct Solutions')
    ax1.set_title('Correct rate')
    ax1.grid()
    ax1.legend(legend)
    
    ax2.set_xscale('log')
    ax2.set_xticks(most_widths)
    ax2.set_xticklabels(most_widths)  # Explicitly set the tick labels
    ax2.set_xlabel('Beam Width')
    ax2.set_ylabel('Mean Difference')
    ax2.set_title('Mean Difference with Std Dev')
    ax2.grid()
    ax2.legend(legend)
    
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xticks(most_widths)
    ax3.set_xticklabels(most_widths)  # Explicitly set the tick labels
    ax3.set_xlabel('Beam Width')
    ax3.set_ylabel('Mean Time (s)')
    ax3.set_title('Mean Time')
    ax3.grid()
    ax3.legend(legend)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(top=0.85)
    plt.show()
    
    
# These are older implementations, that I keep in case I need them later.
# Find moves according to beam_search
def beam_search_with_heuristic(initial_board, beam_width=2, max_depth=50, T=1):
    """
    Uses beam search to find a sequence of moves that clears the board, based on
    1-look-ahead (T=1) and 2-look-ahead (T=2) heuristics.
    
    Parameters:
        initial_board (np.array): The starting board.
        beam_width (int): Maximum number of nodes to keep at each depth.
        max_depth (int): Maximum search depth (number of moves).
        
    Returns:
        (move_sequence, cumulative_score): A tuple where move_sequence is a list of moves (each a (row, col) tuple)
        that leads to a cleared board (if found) or the best path found within max_depth, and cumulative_score is the
        cumulative heuristic probability for that branch.
    """
    rows, cols = initial_board.shape
    unique_vals = np.unique(initial_board)
    unique_vals = unique_vals[unique_vals != -1]
    S = len(unique_vals)
    
    initial_game = FormerGame(M=rows, N=cols, S=S, custom_board=initial_board.copy())
    beam = [(initial_game, [], 1.0)]
    visited = set()
    
    for depth in range(max_depth):
        new_beam = []
        for game, moves, cum_score in beam:
            # If the board is cleared, return this move sequence.
            if game.is_game_over():
                return moves, cum_score
            valid_actions = game.get_valid_turns()
            if not valid_actions:
                continue
            if T == 1:
                probs = heuristic_min_groups_1_look_ahead(game.get_board(), valid_actions)
            elif T == 2:
                probs = heuristic_min_groups_2_look_ahead(game.get_board(), valid_actions)
            
            # Expand each valid action.
            for action, prob in zip(valid_actions, probs):
                # Create a new game instance using the current board.
                new_game = FormerGame(M=rows, N=cols, S=S, custom_board=game.get_board())
                new_game.make_turn(action)
                new_moves = moves + [action]
                new_cum_score = cum_score * prob
                board_key = board_to_tuple(new_game.get_board())
                if board_key in visited:
                    continue
                visited.add(board_key)
                new_beam.append((new_game, new_moves, new_cum_score))
        if not new_beam:
            break
        # Sort new nodes by cumulative score (higher is better) and keep the top beam_width.
        new_beam.sort(key=lambda node: node[2], reverse=True)
        beam = new_beam[:beam_width]
    
    # If no solution is found within max_depth, return the best branch found.
    best_node = max(beam, key=lambda node: node[2])
    return best_node[1], best_node[2]

def board_to_tuple(board):
    """
    Converts a NumPy board (2D array) into a tuple of tuples for hashing.
    """
    return tuple(tuple(int(cell) for cell in row) for row in board)

def randomize_beam_search_data(data):
    """
    Randomizes the game states and solutions in the data set.
    
    Args:
        data (list): List of tuples. Each tuple corresponds to a single game, and contains the consecutive game states, actions and remaining moves.
        
    Returns:
        list: Randomized data set, on the form [(game_state, action, remaining_moves), ...], no longer ordered by game or game state.
    """
    randomized_data = []
    for game in data:
        game_states, actions, remaining_moves = game
        for i in range(len(game_states)-1): # Do not include the last game state, as it is an array of only -1s
            randomized_data.append((game_states[i], actions[i], remaining_moves[i]))
    np.random.shuffle(randomized_data)
    return randomized_data

def beam_search_anytime(board, target=14, initial_beam_width=2, max_beam_width=15000):
    """
    Runs beam search until a solution with fewer than 'target' moves is found.
    The beam width doubles after each iteration. Each iteration runs a beam search from scratch.
    
    Parameters:
        board (np.array): The initial board.
        target (int): The target (optimal) number of moves.
        initial_beam_width (int): The initial beam width.
        
    Returns:
        None (prints solutions as they are found).
    """
    curr_best = np.inf
    beam_width = initial_beam_width
    start_time = time.time()
    while curr_best > target and beam_width <= max_beam_width:
        print(f"Beam width: {beam_width}")
        moves, score = beam_search_with_heuristic(board, beam_width=beam_width)
        if len(moves) < curr_best:
            curr_best = len(moves)
            elapsed = time.time() - start_time
            print(f"New best found: {curr_best} moves, beam width: {beam_width}, time: {elapsed:.2f}s")
            print("Moves:", moves)
        beam_width *= 2