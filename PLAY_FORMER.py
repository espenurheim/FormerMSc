"""
This file contains the code to run our Former GUI. For it to run, the following points must be ensured:

- All external libraries are downloaded (pygame, numpy)
- Necessary files from the GitHub repository are downloaded (Cpp_code folder, daily_board.py, MCTS.py, )
- The CPP code is compiled (instructions on this in the Cpp_code folder and the readme-file)

The game can be played on any of the 100 boards in the daily_boards.pkl file, or on a custom board.
"""

from GUI_functions import select_board, play_game
import numpy as np

if __name__ == "__main__":
    board_arr, display_date, best_disp = select_board()
    if board_arr is None: # Add your own custom board here! Or generate randomly using the code below. Then choose "Custom board" after running the code.
        
        # Change values to fit your own board here. We typically use 0 = pink shape, 1 = blue shape, 2 = green shape and 3 = orange shape, but it does not make a difference.
        board_arr = np.array([
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0],
        ])
        
        # Or generate board randomly
        #board_arr = np.random.randint(0,4,(9,7))
        
    play_game(M=9, N=7, S=4,
              board=board_arr,
              date_str=display_date,
              best_known=best_disp)