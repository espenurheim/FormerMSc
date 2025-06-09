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

### Models

* **Heuristics:** `heuristics.py`, `heuristics.ipynb`
* **Supervised Learning:** `supervised_functions.py`, `supervised_hyper.py`, `supervised_networks.py`, `supervised.ipynb`
* **PPO:** `PPO_classes.py`, `PPO_train.py`, `PPO.ipynb`

### Results

### Search

* **MCTS:** `MCTS.py`
* **Beam Search:** `beam_search.py`

### Plotting

* **Helper functions:** `figure_functions.py`
* **Generate figures:** `figures.py`

### Comparing Implementations

* **C++ vs Python:** `compare_cpp_py.ipynb`

