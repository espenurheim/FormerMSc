# Solving *Former* with Machine Learning Techniques

**Author:** Espen Bjørge Urheim

**Supervisor:** Jo Eidsvik

**Program:** Master's Thesis in Applied Physics and Mathematics, Norwegian University of Science and Technology

**Semester:** Spring 2025

## Overview

This repository contains code and resources for solving the *Former* game (NRK, 2024) as part of my Master's thesis in Industrial Mathematics. This includes a GUI that anyone can use to play the game, with recommendations from our best-performing Monte Carlo Tree Search solver. Instructions on how to setup a matching environment, install dependencies, compile the C++ code and running the GUI are included below.

## Requirements

* Python 3.12.7 (other versions may work, but this is the one we used)
* Compiler (g++, clang, or MSVC) for building C++ extensions
* pybind11

## Setup

### 1. Create and activate a virtual environment

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

## Repository structure

```
├── Cpp_code/                # C++ source, bindings, and setup files
├── figures/                 # Thesis figures
├── models/                  # Trained ML models (both from supervised learning and PPO)
├── results/                 # CSV and pickle results for plotting
├── GUI_functions.py         # GUI-relevant functions and classes
├── PLAY_FORMER.py           # Run Former GUI
├── heuristics.py            # Heuristic implementations
├── heuristics.ipynb         # Heuristics analysis notebook
├── supervised.ipynb         # Supervised learning (SL) analysis notebook
├── supervised_functions.py  # SL helper functions
├── supervised_hyper.py      # SL hyperparameter tuning implementation
├── supervised_networks.py   # SL network implementation
├── PPO_classes.py           # Gymnasium environments and classes for PPO
├── PPO_train.py             # PPO training script (ran on Idun, so may need fixing)
├── PPO.ipynb                # PPO analysis notebook
├── MCTS.py                  # Monte Carlo Tree Search implementation
├── beam_search.py           # Beam search implementation
├── figure_functions.py      # Plotting helper functions
├── figures.py               # Script to generate thesis figures
├── compare_cpp_py.ipynb     # Compare C++ vs Python implementations
├── requirements_GUI.txt     # GUI dependencies
└── requirements.txt         # All project dependencies
```

## Building the C++ game implementation extension

```bash
cd Cpp_code
python setup.py build_ext --inplace
cd -
```

## Running the GUI

```bash
python PLAY_FORMER.py
```

## Models

* **Heuristics:** `heuristics.py`, `heuristics.ipynb`
* **Supervised Learning:** `supervised_functions.py`, `supervised_hyper.py`, `supervised_networks.py`, `supervised.ipynb`
* **PPO:** `PPO_classes.py`, `PPO_train.py`, `PPO.ipynb`

## Search Techniques

* **MCTS:** `MCTS.py`
* **Beam Search:** `beam_search.py`

## Plotting

* **Helper functions:** `figure_functions.py`
* **Generate figures:** `figures.py`

## Comparing Implementations

* **C++ vs Python:** `compare_cpp_py.ipynb`
