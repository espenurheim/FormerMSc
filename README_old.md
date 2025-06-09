# Solving *Former* with machine learning techniques

Author: Espen Bjørge Urheim
Supervisor: Jo Eidsvik
Master's thesis in Applied Physics and Mathematics at the Norwegian University of Science and Technology

This repository contains code relevant to solving the Former game, as a part my Master's thesis in Industrial Mathematics
of spring 2025. The contents of the repository are listed below.

We used Python version 3.12.7. After creating a virtual environment, we provide two options for downloading packages:
- Run 'pip install -r requirements_GUI.txt' - this file contains the minimum required packages to run the GUI.
- Run 'pip install -r requirements.txt'     - this file contains all packages used.

The repository contains the following main components.
- 4 folders:
    1) Cpp_code - contains necessary C++ code with bindings and setup files to compile and bind with Python (using pybind11). How-to on compiling is included below.
    2) figures - contains all figures in the thesis.
    3) models - contains supervised learning-based value and policy networks obtained from training, and PPO-based actor and critic networks.
    4) results - Results stored in CSV files and pickle-files, used in the plotting functions in figures.py.
- Code for *Former* GUI: GUI_functions.py holds the necessary functions, PLAY_FORMER.py is the file to run to play the game (see how-to below)
- Code for each of the three model types developed (used to predict based on game states):
    1) Heuristics - analysis in heuristics.ipynb, code implementations in heuristics.py.
    2) Supervised learning - analysis in supervised.ipynb, related code for network implementation, hyperparameter tuning and helper functions in the three 'supervised_<...>.py' files.
    3) PPO - analysis in the PPO.ipynb file, Gymnasium environments etc. in the PPO_classes.py file, and code for training in PPO_train.py.
- Code for each search technique:
    1) MCTS implementation in MCTS.py
    2) Beam search implementation in beam_search.py
- Code for plotting - plotting functions in figure_functions.py, obtain each plot from figures.py.
- Code for comparing C++ to Python code in compare_cpp_py.ipynb
- Two requirements-files - requirements_GUI.txt contains the minimum packages required to run the GUI, requirements.txt contains all packages used. They are run using 'pip install -r <filename>'

####################################################################################################################################
Step-by-step for running the Former GUI
####################################################################################################################################
STEP 1: Install necessary packages - run 'pip install -r requirements_GUI.txt'

STEP 2: Compile the C++ code (game implementation):
1) Run 'cd Cpp_code' in terminal (switch to correct directory)
2) Run 'python setup.py build_ext --inplace' in terminal (compile code)
3) Run 'cd -' in terminal (exit directory).

STEP 3: Run the 'PLAY_FORMER.py' file. The game should now work, although the first time loading may take a little longer than usual.