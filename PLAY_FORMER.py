"""
This file contains the code to run our Former GUI. For it to run, the following points must be ensured:

- All external libraries are downloaded (either requirements_GUI.txt or requirements.txt)
- GitHub repository is downloaded
- The CPP code is compiled (instructions on this in the readme-file)

The game can be played on any of the 100 boards in the daily_boards.pkl file, or on a custom board.
"""
import numpy as np
from Former.GUI_functions import select_board, play_game

if __name__ == "__main__":
    board_arr, display_date, best_disp = select_board()
    if board_arr is None: # Add your own custom board here! Or generate randomly using the code below. Then choose "Custom board" after running the code.
        
        # Change values to fit your own board here. We typically use 0 = pink shape, 1 = blue shape, 2 = green shape and 3 = orange shape, but it does not make a difference.
        board_arr = np.array([
            [1, 3, 1, 0, 2, 0, 2],
            [0, 2, 2, 2, 0, 3, 3],
            [2, 2, 2, 1, 3, 2, 2],
            [2, 3, 1, 1, 3, 2, 2],
            [3, 1, 3, 2, 0, 0, 1],
            [0, 2, 3, 0, 1, 2, 3],
            [3, 3, 1, 1, 2, 0, 0],
            [1, 1, 0, 3, 0, 3, 2],
            [2, 0, 3, 2, 2, 2, 2]])
        
        # Or generate board randomly to use a custom board
        #board_arr = np.random.randint(0,4,(9,7))
        
    play_game(M=9, N=7, S=4,
              board=board_arr,
              date_str=display_date,
              best_known=best_disp)