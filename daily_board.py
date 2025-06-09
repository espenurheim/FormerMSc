import numpy as np
import pickle
        
def get_daily_board(date = None):
 
    # Load the daily boards from a file
    with open('/Users/espen/Desktop/masteroppgave/daily_boards/daily_boards.pkl', 'rb') as f:
        daily_boards = pickle.load(f)
        
    if date is None:
        return daily_boards
    
    # Get the daily board for the specified date
    return daily_boards.get(date, None)