import pickle
from pathlib import Path

def get_daily_board(date=None):
    """
    Load the dict of daily boards from daily_boards.pkl (sitting
    next to this file), and optionally return just one date.
    """
    pkl_path = Path(__file__).resolve().parent / 'daily_boards.pkl'
    with pkl_path.open('rb') as f:
        daily_boards = pickle.load(f)

    if date is None:
        return daily_boards

    return daily_boards.get(date)
