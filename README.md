# Solving Former with Machine Learning Techniques

**Author:** Espen Bjørge Urheim
**Supervisor:** Jo Eidsvik
**Program:** Master's Thesis in Applied Physics and Mathematics, Norwegian University of Science and Technology
**Semester:** Spring 2025

## Overview

This repository contains code and resources for solving the Former game as part of my Master's thesis in Industrial Mathematics.

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
├── Cpp_code/              # C++ source, bindings, and setup files
├── figures/               # Thesis figures
├── models/                # Trained ML models
├── results/               # CSV and pickle results for plotting
├── GUI_functions.py       # Core GUI logic
├── PLAY_FORMER.py         # Entry point for the Former game GUI
├── heuristics.py          # Heuristic-based strategies
├── heuristics.ipynb       # Heuristics analysis notebook
├── supervised_*.py        # Supervised learning implementations
├── supervised.ipynb       # Supervised learning analysis notebook
├── PPO_classes.py         # Gymnasium environments and classes for PPO
├── PPO_train.py           # PPO training script
├── PPO.ipynb              # PPO analysis notebook
├── MCTS.py                # Monte Carlo Tree Search implementation
├── beam_search.py         # Beam search implementation
├── figure_functions.py    # Plotting helper functions
├── figures.py             # Script to generate thesis figures
├── compare_cpp_py.ipynb   # Compare C++ vs Python implementations
├── requirements_GUI.txt   # GUI dependencies
└── requirements.txt       # All project dependencies
```

## Building the C++ Extension

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
* **Supervised Learning:** `supervised_*.py`, `supervised.ipynb`
* **PPO:** `PPO_classes.py`, `PPO_train.py`, `PPO.ipynb`

## Search Techniques

* **MCTS:** `MCTS.py`
* **Beam Search:** `beam_search.py`

## Plotting

* **Helper functions:** `figure_functions.py`
* **Generate figures:** `figures.py`

## Comparing Implementations

* **C++ vs Python:** `compare_cpp_py.ipynb`

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
