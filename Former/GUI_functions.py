"""
This file contains the functions for the GUI. It also had a separate MCTS implementation which only
relies on the one policy network, so that the user does not have to download Stable Baselines3, etc.
"""
# External libraries
import pygame
import sys
import threading
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local code
from Former.Cpp_code.former_class_cpp import FormerGame
import Former.daily_board as db

# LOAD NETWORK AND ALL ASSOCIATED HELPER FUNTCIONS
def accumulate_probabilities(board, full_p, valid_turns):
    """
    Helper function to mask probabilities so that only valid turns are given a probability.
    Args:
        board:  2D array-like of shape (R, C), with -1 for empty cells and non-negative ints for shapes.
        full_p: 1D torch.Tensor of length R*C containing the network’s softmax output
                (before any masking). full_p[i] is the prob for flat index i = r*C + c.

    Returns:
        A dict mapping each valid turn (r, c) to the total probability mass
        assigned to its entire shape-group, normalized so the values sum to 1.
    """
    valid_set   = set(valid_turns)

    R, C = board.shape
    probs = {}

    # Iterate over valid turns and accumulate probabilities from each group
    for turn in valid_turns:
        group_pts = FormerGame.find_group_static(board, turn)
        idxs = [
            rr * C + cc
            for (rr, cc) in group_pts
            if (rr, cc) in valid_set
        ]
        if idxs:
            mass = full_p[idxs].sum().item()
        else:
            mass = 0.0
        probs[turn] = mass

    # Normalize sum to 1
    total = sum(probs.values())
    if total > 0:
        for turn in probs:
            probs[turn] /= total
    else:
        # If no valid turns are assigned probabilities, assign uniform distribution
        uni = 1.0 / len(valid_turns)
        for turn in probs:
            probs[turn] = uni

    return probs

class PolicyNet(nn.Module):
    """
    Class for policy networks.
    """
    def __init__(self, input_channels, board_shape, d=5, w=64):
        super().__init__()
        self.board_shape = board_shape
        self.d = d
        self.w = w

        # First convolution + batch norm
        self.conv1 = nn.Conv2d(input_channels, w, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm2d(w)

        # Middle convolutions + batch norms
        for i in range(2, d+1):
            setattr(self, f'conv{i}', nn.Conv2d(w, w, kernel_size=3, padding=1))
            setattr(self, f'bn{i}', nn.BatchNorm2d(w))

        # 1x1 convolution head for per-cell policy logits
        self.policy_head = nn.Conv2d(w, 1, kernel_size=1)

    def forward(self, x):
        # Initial conv -> BN -> ReLU
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # d-1 middle blocks: conv -> BN -> ReLU
        for i in range(2, self.d+1):
            conv = getattr(self, f'conv{i}')
            bn = getattr(self, f'bn{i}')
            x = conv(x)
            x = bn(x)
            x = F.relu(x)

        # 1x1 conv head -> flatten (SoftMax done in evaluate_state)
        p = self.policy_head(x)            # (B, 1, R, C)
        p = p.view(p.size(0), -1)          # (B, R*C)
        return p
    
    def evaluate_state(self, board):
        """
        Single-board inference: returns
          policy (dict mapping legal (r,c)->prob)
        """
        R, C = board.shape
        S = 4
        # one‐hot encode channels = S+1
        onehot = np.zeros((S+1, R, C), dtype=np.float32)
        for r in range(R):
            for c in range(C):
                v = int(board[r,c])
                onehot[S if v<0 else v, r, c] = 1.0

        # Evaluate the board state. Apply softmax to get probabilities.
        x = torch.from_numpy(onehot)[None].to(next(self.parameters()).device)
        self.eval()
        with torch.no_grad():
            full_p = F.softmax(self(x), dim=1)
        full_p = full_p.squeeze(0)
        valid = FormerGame.get_valid_turns_static(board)
        policy = accumulate_probabilities(board, full_p, valid)
        return policy

policy_net_w64d10 = PolicyNet(5,(9,7), 10, 64)
policy_net_w64d10.load_state_dict(torch.load("models/trained_models/supervised/policy/w64d10/w64d10.pth", torch.device("cpu")))
policy_net_w64d10.eval()

print("Loaded network successfully.")

import sys
import pygame
from pathlib import Path
import Former.daily_board as db

def select_board(window_width=400, window_height=600, font_size=24, padding=10):
    """
    Open a Pygame window listing all daily‐board dates (and a "Custom board" entry).
    Returns (board_array or None, display_date, best_display_str).
    """
    # 1) Build and sort the list of entries
    daily = db.get_daily_board()  # {'jan27': (board, best, _), …}
    month_map = {'jan':'January','feb':'February','mar':'March',
                 'apr':'April','may':'May'}
    entries = []
    for key, (board, best, _) in daily.items():
        # Format “best” for display
        best_str = f"{best:.2f}" if isinstance(best, float) else str(best)
        # Turn "jan27" → "January 27"
        abbr, day = key[:3].lower(), int(key[3:])
        month = month_map.get(abbr, abbr.capitalize())
        display_date = f"{month} {day}"
        entries.append((key, board, best_str, display_date))

    # Sort by month then day
    month_order = {m: i for i, m in enumerate(month_map, start=1)}
    entries.sort(key=lambda e: (month_order.get(e[0][:3].lower(), 99), int(e[0][3:])))
    # Finally, allow a “Custom board” option
    entries.append(('custom', None, '', 'Custom board'))

    # 2) Set up Pygame UI
    pygame.init()
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Select a Daily Board")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, font_size)

    line_h = font_size + 6
    top_y = padding*2 + font_size
    avail_h = window_height - top_y - padding
    max_visible = max(1, avail_h // line_h)
    scroll = 0

    def clamp(idx):
        return max(0, min(idx, len(entries) - max_visible))

    selected = None
    while selected is None:
        clock.tick(30)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif ev.type == pygame.MOUSEWHEEL:
                scroll = clamp(scroll - ev.y)
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                # Mouse wheel up/down
                if ev.button in (4, 5):
                    scroll = clamp(scroll + (1 if ev.button == 5 else -1))
                else:
                    mx, my = ev.pos
                    # Check clicks inside the list area
                    if top_y <= my < top_y + max_visible*line_h:
                        idx = scroll + (my - top_y) // line_h
                        if 0 <= idx < len(entries):
                            selected = idx
                            break

        # Draw background and header
        screen.fill((240, 240, 240))
        header = font.render("Choose a date:", True, (0, 0, 0))
        screen.blit(header, (padding, padding))

        # Draw each visible entry
        for i in range(max_visible):
            gi = scroll + i
            if gi >= len(entries):
                break
            _, _, best_str, date_str = entries[gi]
            y = top_y + i*line_h
            rect = pygame.Rect(padding, y, window_width-2*padding, line_h)
            # Highlight on hover
            if rect.collidepoint(pygame.mouse.get_pos()):
                pygame.draw.rect(screen, (200, 200, 255), rect)
            screen.blit(font.render(date_str, True, (0, 0, 0)), (padding+4, y+2))
            if best_str:
                suf = font.render(f"(best: {best_str} moves)", True, (50, 50, 50))
                screen.blit(suf, (window_width-padding - suf.get_width(), y+2))

        # If list is longer than the view, draw a scrollbar
        if len(entries) > max_visible:
            bar_h = int(avail_h * (max_visible/len(entries)))
            bar_y = top_y + int((avail_h - bar_h) * (scroll/(len(entries)-max_visible)))
            bar = pygame.Rect(window_width - padding//2, bar_y, padding//2, bar_h)
            pygame.draw.rect(screen, (150, 150, 150), bar)

        pygame.display.flip()

    pygame.quit()
    # Unpack and return the chosen entry
    _, board_arr, best_str, date_str = entries[selected]
    return board_arr, date_str, best_str


def play_game(M=9, N=7, S=4, board=None, date_str="", best_known=""):
    """
    Launches a Pygame interface for Former with MCTS suggestions under a time limit.

    Displays:
      - Moves used
      - Best known solution
      - MCTS suggestion text: “Group X (max. Y moves)”
      - Time‐limit input box + Search button
      - Reset button to restart the board
    """
    # initialize Pygame and compute layout
    pygame.init()
    cell_size = 60
    board_w = N * cell_size
    board_h = M * cell_size
    stats_w = 300
    screen = pygame.display.set_mode((board_w + stats_w, board_h))
    pygame.display.set_caption("FormerGame")
    clock = pygame.time.Clock()

    # color mapping for board values and grid lines
    colors = {
        -1: (0, 0, 0),
         0: (255, 20, 147),
         1: (135, 206, 250),
         2: (0, 128, 0),
         3: (255, 165, 0),
    }
    grid_c = (50, 50, 50)

    # helper to reset the game to the initial board
    def new_game():
        return FormerGame(M, N, S, custom_board=board)
    game = new_game()

    # fonts for various text elements
    stats_font = pygame.font.SysFont(None, 18)
    title_font = pygame.font.SysFont(None, 28)
    small_font = pygame.font.SysFont(None, 18)

    name_x = board_w + 10
    padding = 10

    # set up input and button rectangles
    label_text = "Time limit for MCTS:"
    label_w = stats_font.size(label_text)[0]
    input_box = pygame.Rect(
        name_x + label_w + padding,
        board_h - 90 - padding,
        40, 30
    )
    search_button = pygame.Rect(
        input_box.right + padding,
        input_box.y,
        80, 30
    )
    reset_button = pygame.Rect(
        name_x,
        board_h - 40 - padding,
        80, 30
    )

    time_text = ""
    active_input = False

    # state for MCTS suggestions
    suggestion_actions = None
    suggestion_idx = 0
    suggestion_rem = None
    search_in_progress = False

    # MCTS configuration parameters
    mcts_params = {
        "num_simulations": 1000,
        "c_puct": 10,
        "min_exploration_limit": 10,
        "N": N, "M": M, "S": S,
        "rollout": True,
        "Q_type": "avg",
        "use_prior": True
    }

    # try to parse best_known as a float
    try:
        best_sol_num = float(best_known)
    except ValueError:
        best_sol_num = None

    # runs MCTS in a background thread when Search is clicked
    def run_mcts_search(board_state, t_lim):
        nonlocal suggestion_actions, suggestion_idx, suggestion_rem, search_in_progress
        search_in_progress = True

        mcts = MCTS(
            board_state, None,
            policy_net_w64d10, mcts_params,
            True, best_sol=0,  # avoid immediate stop on repeated calls
            policy_type='network'
        )
        sol_lens, times, actions, n_sims = mcts.search_with_time_limit(t_lim)

        if actions:
            suggestion_actions = actions
            suggestion_idx = 0
            suggestion_rem = sol_lens[-1]
        else:
            suggestion_actions = None
            suggestion_rem = None

        search_in_progress = False

    running = True
    action_count = 0

    while running:
        clock.tick(30)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

            elif ev.type == pygame.MOUSEBUTTONDOWN:
                mx, my = ev.pos
                active_input = input_box.collidepoint((mx, my))

                # Search button logic
                if search_button.collidepoint((mx, my)):
                    try:
                        t_lim = float(time_text)
                        suggestion_actions = None
                        if not search_in_progress:
                            threading.Thread(
                                target=run_mcts_search,
                                args=(game.get_board(), t_lim),
                                daemon=True
                            ).start()
                    except ValueError:
                        pass  # ignore invalid input
                    time_text = ""
                    active_input = False

                # Reset button logic
                if reset_button.collidepoint((mx, my)):
                    game = new_game()
                    action_count = 0
                    suggestion_actions = None
                    suggestion_idx = 0
                    suggestion_rem = None

                # Board click logic
                if mx < board_w:
                    r, c = my // cell_size, mx // cell_size
                    if (0 <= r < M and 0 <= c < N and
                        game.board[r][c] != -1):
                        old_board = game.get_board()
                        clicked_group = FormerGame.find_group_static(old_board, (r, c))

                        action_count += 1
                        game.make_turn((r, c))

                        # advance or drop suggestion based on click
                        if suggestion_actions and suggestion_idx < len(suggestion_actions):
                            target_rep = suggestion_actions[suggestion_idx]
                            if target_rep in clicked_group:
                                suggestion_idx += 1
                                suggestion_rem -= 1
                                if suggestion_idx >= len(suggestion_actions):
                                    suggestion_actions = None
                                    suggestion_rem = None
                            else:
                                suggestion_actions = None
                                suggestion_rem = None

            # Time input typing
            elif ev.type == pygame.KEYDOWN and active_input:
                if ev.key == pygame.K_BACKSPACE:
                    time_text = time_text[:-1]
                elif ev.unicode.isdigit() or ev.unicode == '.':
                    time_text += ev.unicode

        # ─── Draw the board ─────────────────────────────────────
        screen.fill((255, 255, 255))
        for r in range(M):
            for c in range(N):
                val = game.board[r][c]
                col = colors.get(val, (200, 200, 200))
                rect = pygame.Rect(
                    c * cell_size, r * cell_size,
                    cell_size, cell_size
                )
                pygame.draw.rect(screen, col, rect)
                pygame.draw.rect(screen, grid_c, rect, 1)

        # ─── Draw group labels ───────────────────────────────────
        groups = game.get_groups()
        reps = [min(grp) for grp in groups]
        rep2id = {rep: idx for idx, rep in enumerate(reps)}
        cell2rep = {}
        for grp in groups:
            rep = min(grp)
            for cell in grp:
                cell2rep[cell] = rep

        for (r, c), rep in cell2rep.items():
            if game.board[r][c] != -1:
                gid = rep2id.get(rep, "?")
                txt = small_font.render(str(gid), True, (0, 0, 0))
                tr = txt.get_rect(center=(
                    c * cell_size + cell_size/2,
                    r * cell_size + cell_size/2
                ))
                screen.blit(txt, tr)

        # ─── Draw stats panel ────────────────────────────────────
        pygame.draw.rect(
            screen, (230, 230, 230),
            (board_w, 0, stats_w, board_h)
        )
        x0 = board_w + padding

        screen.blit(
            title_font.render(f"Former – {date_str}", True, (0, 0, 0)),
            (x0, padding)
        )
        screen.blit(
            stats_font.render(f"Best known solution: {best_known} moves", True, (0, 0, 0)),
            (x0, padding + 30)
        )
        screen.blit(
            stats_font.render(f"Moves used: {action_count}", True, (0, 0, 0)),
            (x0, padding + 50)
        )

        # ─── Draw MCTS suggestion ────────────────────────────────
        if suggestion_actions and suggestion_idx < len(suggestion_actions):
            nr, nc = suggestion_actions[suggestion_idx]
            group_cells = FormerGame.find_group_static(game.get_board(), (nr, nc))
            rep = min(group_cells)
            gid = rep2id.get(rep, "?")
            sug_text = (
                f"MCTS suggests: Group {gid} "
                f"(max. {suggestion_rem:.0f} moves)"
            )
            screen.blit(
                stats_font.render(sug_text, True, (0, 0, 0)),
                (x0, padding + 100)
            )

        # ─── Draw time‐limit UI ──────────────────────────────────
        screen.blit(
            stats_font.render(label_text, True, (0, 0, 0)),
            (name_x, input_box.y + 5)
        )
        box_col = (200, 200, 255) if active_input else (220, 220, 220)
        pygame.draw.rect(screen, box_col, input_box)
        screen.blit(
            stats_font.render(time_text, True, (0, 0, 0)),
            (input_box.x + 5, input_box.y + 7)
        )
        pygame.draw.rect(screen, (180, 180, 180), search_button)
        sb_txt = stats_font.render("Search", True, (0, 0, 0))
        sb_rect = sb_txt.get_rect(center=search_button.center)
        screen.blit(sb_txt, sb_rect)

        # ─── Draw Reset button ──────────────────────────────────
        pygame.draw.rect(screen, (180, 180, 180), reset_button)
        rb_txt = stats_font.render("Reset", True, (0, 0, 0))
        rb_rect = rb_txt.get_rect(center=reset_button.center)
        screen.blit(rb_txt, rb_rect)

        pygame.display.flip()

        if game.is_game_over():
            print("Game Over! Moves used:", action_count)
            pygame.time.wait(800)
            running = False

    pygame.quit()
    sys.exit()

    
    
# MCTS IMPLEMENTATION
"""
MCTS implementation.
"""

# MCTS
class MCTSNode:
    """
    One node in a MCTS tree. The "expected_best_solution" is the actual best solution obtained - it is called expected
    since the implementation of MCTS allows for using value networks to predict the value instead of full simulations,
    but when using full simulations, it is the exact best solution found.
    """
    def __init__(self, state, prior, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action_made = action
        self.children = {}
        self.visit_count = 0
        self.prior = prior
        self.depth = parent.depth + 1 if parent is not None else 0
        self.expected_best_solution = state.shape[0]*state.shape[1] # Initially, expect the best solution of a node to be the worst possible
        self.move_sum = 0 # The total sum of number of moves used from this state
        
class MCTS:
    def __init__(self, root_state, net1, net2=None, params = {"num_simulations": 1000,
                                                              "c_puct": 10.0,
                                                              "min_exploration_limit": 10,
                                                              "N": 9,
                                                              "M": 7,
                                                              "S": 4,
                                                              "rollout": False,
                                                              "Q_type": "max",
                                                              "use_prior": True}, anytime = False, best_sol = None, policy_type = 'network', t = 0):
        self.root_node = MCTSNode(root_state, 1)
        
        self.params = params
        self.net1 = net1 # Value network (we do not really use this anymore, but it is an option if rollout is False)
        self.net2 = net2 # Policy network
            
        self.num_simulations = params["num_simulations"]
        self.c_puct = params["c_puct"]
        self.N, self.M, self.S = params["N"], params["M"], params["S"]
        self.min_exploration_limit = params["min_exploration_limit"]
        self.rollout = params["rollout"]
        if net1 is None and net2 is None:
            self.rollout = True # If no networks are given, we have to perform rollouts with heuristic
        
        self.Q_type = params["Q_type"]
        self.use_prior = params["use_prior"]
        self.start_time = time.time()
        self.anytime = anytime
        
        # Always keep track of the best found solution
        self.best_found_solution = float('inf') 
        self.best_action_sequence = None
        self.action_list_from_previous_node = []
        self.best_sol = best_sol
        
        self.policy_type = policy_type
        self.t = t # For heuristic
    
        
    def _puct(self, node):
        """
        Calculate the PUCT value for a node.
        """
        parent_visit_count = node.parent.visit_count if node.parent else 1
        if self.Q_type == 'max':
            Q = -node.expected_best_solution # Value is set to be the (negative) best solution, i.e. the fewer moves, the higher the Q.
        elif self.Q_type == 'avg':
            Q = -node.move_sum / node.visit_count # Negative average, to encourage less moves
        U = self.c_puct * node.prior * np.sqrt(parent_visit_count) / (node.visit_count + 1)
        return Q + U
        
    def select(self, node):
        """
        Select a leaf node. 
        """
        while node.children:
            max_puct = float('-inf')
            best_child = None
            for action, child in node.children.items():
                if child.visit_count == 0:     # If child has not been visited, choose it immediately 
                    return child
                if node.parent is None:       # At root node, force a minimum amount of visits
                    if child.visit_count < self.min_exploration_limit:
                        return child
                puct = self._puct(child)
                if puct > max_puct:
                    max_puct = puct
                    best_child = child
            node = best_child
        return node
    
    def expand(self, node, use_prior = True):
        """ 
        Expand the node by generating its children. Returns the last child.
        """
        legal_actions = FormerGame.get_valid_turns_static(node.state)
        priors = self.net2.evaluate_state(node.state)
        
        # Add all children
        for action in legal_actions:
            prior = priors[action]
            new_state = np.array(FormerGame.apply_turn_static(node.state, action))
            child_node = MCTSNode(new_state, prior=prior, parent=node, action=action)
            node.children[action] = child_node
            
    def simulate(self, node):
        """
        Simulate a random game from the current node. Return the expected number of moves from the current node.
        If self.rollout = params['rollout'] is set to False, we replace rollout by an estimation from the value net. If it is
        set to True, we perform a full rollout using the policy network as heuristic.
        """
        state = node.state.copy()
        if FormerGame.is_game_over_static(state):
            return 0
        
        # Rollout is false -> use value net instead
        if not self.rollout:
            if self.net2 is None: # Dual-head structure
                return self.net1.evaluate_state(state)[0]
            return self.net1.evaluate_state(state)
        
        # Rollout is true -> perform rollout with policy net / heuristic as policy function
        action_count = 0
        action_list = []
        while not FormerGame.is_game_over_static(state):
            policy = self.net2.evaluate_state(state) # Policy net
            
            actions = list(policy.keys())
            probs = list(policy.values())
            chosen_action = random.choices(actions, weights=probs, k=1)[0] # Sample from policy

            state = np.array(FormerGame.apply_turn_static(state, chosen_action))
            action_list = action_list + [chosen_action]
            action_count += 1
        
        return action_count, action_list
        
    def backpropagate(self, node, value):
        """
        Backtrack the tree and update the expected best solution and visit counts of all nodes.
        """
        estimated_solution = node.depth + value # The estimated solution found is the value of the leaf node + the depth of the leaf node
        action_list = []
        while node.parent is not None:
            action_list = [node.action_made] + action_list
            if estimated_solution < node.expected_best_solution: # Update the expected best solution of the node if the estimation is lower
                node.expected_best_solution = estimated_solution
            node.move_sum += estimated_solution
            node.visit_count += 1
            node = node.parent
        
        # Update for root node as well
        if estimated_solution < node.expected_best_solution:
            node.expected_best_solution = estimated_solution
        node.move_sum += estimated_solution
        node.visit_count += 1
        return action_list

    
    def search_with_time_limit(self, time_limit, swap_freq = float('inf')):
        """
        Perform a full iteration of MCTS, based on a maximum time limit.
        """
        root = self.root_node
        i = 0
        solution_moves = []
        solution_times = []
        start_time = time.time()
        last_meta = time.time()
        while time.time() - start_time < time_limit:
            if time.time() > last_meta + swap_freq: # Reset
                best_action = self.get_best_action()
                self.root_node.children.pop(best_action)
                print("swap")
                last_meta = time.time()
            
            selected_node = self.select(root) # Select leaf node. Can be the root itself.
            if FormerGame.is_game_over_static(selected_node.state): # If leaf node represent a finished game state, backpropagate
                self.backpropagate(selected_node, 0)
                continue
            self.expand(selected_node) # Expand child nodes of the leaf node
            value, simulation_actions = self.simulate(selected_node)
            initial_actions = self.backpropagate(selected_node, value)
            
            sim_value = value + selected_node.depth
            if self.anytime and sim_value < self.best_found_solution: # Print the solution if anytime
                solution_times.append(time.time()-start_time)
                solution_moves.append(sim_value)
                self.best_found_solution = sim_value
                self.best_action_sequence = self.action_list_from_previous_node + initial_actions + simulation_actions
                #print(f"New best: {sim_value}: {self.best_action_sequence} Time passed: {round(time.time()-self.start_time,2)}.")
                if self.best_found_solution == self.best_sol:
                    return solution_moves, solution_times, self.best_action_sequence, i
            i += 1
                
        
        visit_counts = {}
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count / (root.visit_count - 1)
        
        return solution_moves, solution_times, self.best_action_sequence, i
    
    def get_best_action(self):
        """
        Return the best action to take from the root node, based on the visit counts of its children.
        """
        best_action = None
        max_visit = 0
        for action, child_node in self.root_node.children.items():
            if child_node.visit_count > max_visit:
                best_action = action
                max_visit = child_node.visit_count
                
        return best_action
        
