# Solving *Former* with Machine Learning Techniques

**Author:** Espen Bjørge Urheim

**Supervisor:** Jo Eidsvik

**Program:** Master's Thesis in Applied Physics and Mathematics, Norwegian University of Science and Technology

**Semester:** Spring 2025

## Overview

This repository contains code and resources for solving the *Former* game (NRK, 2024) as part of my Master's thesis in Industrial Mathematics. This includes a GUI that anyone can use to play the game, with recommendations from our best-performing Monte Carlo Tree Search solver. Instructions on how to set up a matching environment, install dependencies, compile the C++ code and running the GUI are included below.

## Requirements

* Python 3.12.7 (other versions may work, but this is the one we used)
* Compiler (g++, clang, or MSVC) for building C++ extensions
* pybind11

## Setup
If not already done, clone the repository by opening up a terminal and running 
```bash
git clone https://github.com/espenurheim/FormerMSc.git
cd FormerMSc
```

Then do the following steps. Steps 1-3 are required for running any code, and Step 4 is the GUI - if the user is looking to play our implementation.

### 1. Create and activate a matching virtual environment

```bash
python3.12 -m venv venv
source venv/bin/activate     # macOS/Linux
.\venv\Scripts\Activate.ps1  # Windows PowerShell
```

### 2. Install dependencies

* **For GUI only:**

  ```bash
  pip install -r requirements_GUI.txt
  ```
* **For full functionality:**

  ```bash
  pip install -r requirements.txt
  ```

### 3. Building the C++ game implementation extension

```bash
cd Former/Cpp_code
python setup.py build_ext --inplace
cd ..
cd ..
```

### 4. Running the GUI

```bash
python PLAY_FORMER.py
```

## Repository structure

```
FormerMSc/
├── Former/
├── models/
├── results/
├── search/
├── PLAY_FORMER.py
└── README.md
```


## Contents

### Former
Game implementation and other helper functions. The `Cpp_code` folder holds all C++ code, bindings and setup. `GUI_functions.py` holds functions for the GUI, `compare_cpp_py.ipynb` has code to compare Python and C++ implementations, `daily_boards.py` holds function to load daily boards from `daily_boards.pkl`, and `descriptive_statistics.ipynb` is used for analysis of the game itself.

### Models
The `models` folder holds all code for the three models we use to predict based on boards: heuristics, supervised learning, and PPO. It also holds the trained models in the `trained_models` directory.

* **Heuristics:** `heuristics.py`, `heuristics.ipynb`
* **Supervised Learning:** `supervised_functions.py`, `supervised_hyper.py`, `supervised_networks.py`, `supervised.ipynb`
* **PPO:** `PPO_classes.py`, `PPO_train.py`, `PPO.ipynb`

### Results
The `results` folder has all CSV results, figures, and plotting functions.

### Search
The `search` folder has all code relevant to the implementation and use of MCTS and beam search.


### *Former* GUI
The `PLAY_FORMER.py` is run to play our GUI implementation, after following the steps given earlier.
