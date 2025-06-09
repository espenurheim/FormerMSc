from Cpp_code.former_class_cpp import FormerGame
import itertools
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.ticker import LogLocator, LogFormatter, LogFormatterMathtext
import daily_board as db
import pandas as pd
from matplotlib.patches import Patch
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from tensorboard.backend.event_processing import event_accumulator
from matplotlib.lines import Line2D


###################################################################################################
# CHAPTER 2 - The Former game
# 2.2.3 - The average number of shapes
# 2.2.4 - The branching factor

def plot_avg_number_of_shapes(save_path = None, show = False):
    """
    Plot average number of shapes in the 100 daily boards.
    """

    daily_board_dict = db.get_daily_board()  # Dictionary, {date: (board, solution, actions)}
    n_boards = len(daily_board_dict)

    # Count shapes in each board
    shape_counts = []
    for _, (board, _, _) in daily_board_dict.items():
        unique, freq = np.unique(board, return_counts=True)
        counts = np.zeros(4, dtype=int)
        for u, f in zip(unique, freq):
            counts[int(u)] = f
        shape_counts.append(counts)

    shape_counts = np.array(shape_counts)

    # Use same colors as in the actual game
    raw_colors = {
        0: [249, 87, 171],    # Pink
        1: [ 52, 177, 232],   # Blue
        2: [ 93, 196, 121],   # Green
        3: [255, 156,  84],   # Orange
    }
    
    colors = [tuple(c/255 for c in raw_colors[i]) for i in range(4)]
    shape_labels = ['Pink', 'Blue', 'Green', 'Orange']

    fig, ax = plt.subplots(figsize=(7.5, 4))
    parts = ax.violinplot(shape_counts, showmeans=True, showmedians=False)
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(1)
        pc.set_edgecolor('black')
        pc.set_linewidth(0.8)
        
    for key in ['cbars', 'cmins', 'cmaxes', 'cmeans']:
        artist = parts[key]
        artist.set_edgecolor('black')
        artist.set_linewidth(1)    
    
    ax.set_xticks(np.arange(1,5))
    ax.set_xticklabels(shape_labels)
    ax.set_xlabel('Shape')
    ax.set_ylabel('Count per board')
    ax.set_title(f'Distribution of shape counts over {n_boards} daily boards')
    ax.grid(axis='y', linestyle=':', linewidth=0.5)
    ax.set_ylim(5, 28)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, format='pdf', bbox_inches='tight')
        
    if show:
        plt.show()

def plot_branching_factor(save_path = None, show = False, N=1000, K=16, P=20):
    """
    Plot empirical branching factor of the Former game.
    N: number of boards to sample
    K: number of moves to explore + 1
    P: number of random trajectories per board
    """

    boards = []
    for n in range(N):
        boards.append(np.random.randint(0,4,size=(9,7), dtype=np.int8))

    BF = {d: [] for d in range(K)}

    for b in boards:
        for _ in range(P):
            state = b.copy()
            for d in range(K):
                actions = FormerGame.get_valid_turns_static(state)  
                BF[d].append(len(actions))
                if len(actions) == 0:
                    for r in range(d, K):
                        BF[r].append(1)
                    break
                a = random.choice(actions)
                state = FormerGame.apply_turn_static(state, a)

    depths = np.arange(K)
    lower = [np.percentile(BF[d], 10) for d in depths]
    median = [np.percentile(BF[d], 50) for d in depths]
    upper = [np.percentile(BF[d], 90) for d in depths]

    cum_lower  = list(itertools.accumulate(lower, func=lambda x,y: x*y))
    cum_median = list(itertools.accumulate(median, func=lambda x,y: x*y))
    cum_upper  = list(itertools.accumulate(upper, func=lambda x,y: x*y))

    depths = np.arange(K, dtype=int)
    fig, ax = plt.subplots(figsize=(7.5, 4))

    ax.plot(depths, cum_median, label='Median combinations')
    ax.plot(depths, cum_lower, linestyle='--', label='10th percentile')
    ax.plot(depths, cum_upper, linestyle='--', label='90th percentile')

    ax.set_yscale('log')

    ax.set_xticks(depths)
    ax.set_xlabel(r'Move depth ($d$)')
    ax.set_ylabel('Number of possible action sequences')
    ax.set_title(r"Exponential growth in action combinations in $\emph{Former}$")

    # Highlight B_0
    initial_d = 0
    initial_val = cum_median[initial_d]
    ax.scatter(initial_d, initial_val, color='red', s=50, zorder=10)
    initial_val = median[0]
    ax.annotate(
        r'$B_0=36$',
        xy=(0, initial_val),
        xytext=(1.7, 1.1*initial_val),
        textcoords='data',
        ha='left', va='center',
        fontsize=12,
        color='red',
        bbox=dict(
            facecolor='white',
            edgecolor='red',
            boxstyle='round,pad=0.3'
        ),
        arrowprops=dict(
            arrowstyle='-',
            color='red',
            linewidth=1.5,
            shrinkA=0, 
            shrinkB=0  
        )
    )

    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.grid(True, which='major', linestyle=':', linewidth=0.5)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.2)
    ax.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, format='pdf', bbox_inches='tight')
    if show:
        plt.show()

###################################################################################################
# Chapter 5 - Methodology
# 5.1: Game implementation
def plot_time_comparison_cpp_vs_python(show = True, save_path = None):
    df = pd.read_csv('results/5_methodology/game_simulation_results.csv')
    times_py  = np.sort(df['python_time'].values)
    times_cpp = np.sort(df['cpp_time'].values)
    n = len(times_py)

    median_py  = np.median(times_py)
    median_cpp = np.median(times_cpp)

    # Create a fine grid of x-values between 1e-4 and 1e-2
    x_vals = np.logspace(-4, -2, 1000)

    # Compute empirical CDFs
    y_py  = np.searchsorted(times_py,  x_vals, side='right') / n
    y_cpp = np.searchsorted(times_cpp, x_vals, side='right') / n

    fig, ax = plt.subplots(figsize=(6.5, 3))
    ax.plot(x_vals, y_py,  label='Python', color='C0', linewidth=1.8)
    ax.plot(x_vals, y_cpp, label='C++',    color='C1', linewidth=1.8)

    # Draw medians
    ax.axvline(median_py, ymax=0.5,  color='C0', linestyle='--', linewidth=1.2)
    ax.axvline(median_cpp, ymax=0.5, color='C1', linestyle='--', linewidth=1.2)
    #ax.axhline(0.5, color='k', linestyle='--', linewidth=1.2)

    # Highlight median values
    ax.text(
        median_py, 
        -0.05, 
        f"${median_py:.5f}$ s", 
        transform=ax.get_xaxis_transform(), 
        ha='center', 
        va='top', 
        color='C0'
    )
    ax.text(
        median_cpp, 
        -0.05, 
        f"${median_cpp:.5f}$ s", 
        transform=ax.get_xaxis_transform(), 
        ha='center', 
        va='top', 
        color='C1'
    )
    ax.set_xscale('log')
    ax.set_xlim(1e-4, 1e-2)
    ax.set_ylim(-0.05, 1.05)

    ax.set_xlabel("Time per game (s)")
    ax.set_ylabel("Fraction of runs finished")
    ax.set_title("Empirical CDF of runtimes for Python and C++ implementations")
    ax.legend(loc='lower right')
    plt.subplots_adjust(bottom=0.18)

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, format='pdf', bbox_inches='tight')
    if show:
        plt.show()

# 5.3: Supervised learning
# 5.3.1: Data generation
def plot_move_counts(show = True, save_path = None):
    """
    Reads a CSV with columns ["remaining_moves","count"] and
    plots a bar chart of sample counts per remaining-move bin.
    """
    csv_path = "data/remaining_moves_counts.csv"
    df = pd.read_csv(csv_path)
    x = np.array(df["remaining_moves"].tolist())
    y = np.array(df["count"].tolist())

    x = x[np.where(x < 21)]
    y = y[np.where(x < 21)]
    
    plt.figure(figsize=(6.5, 3))
    plt.bar(x, y, align='center')
    plt.xlabel("Remaining moves")
    plt.ylabel("Number of samples")
    plt.title("Number of samples per number of remaining moves")
    plt.xticks(x)
    plt.xlim(min(x) - 1, max(x) + 1)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()


###################################################################################################
# CHAPTER 6 - Results
# 6.1: Self-made heuristics
def plot_heuristic_distributions(save_path = None, show = False):
    """
    Plot move distributions for self-made heuristics.
    """
    
    df = pd.read_csv('results/6.1_selfmade/selfmade.csv')
    methods = df['Method'].unique()
    moves_data = [df.loc[df['Method'] == method, 'Moves'] for method in methods]
    
    cmap_heur = plt.cm.Blues # Blue colors for heuristics
    t = np.linspace(0.2, 0.9, 5)
    colors = [cmap_heur(tt) for tt in t]
    
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.grid(axis='y', linewidth=0.5)
    parts = ax.violinplot(moves_data, showmeans=True)

    j = 0
    for pc in parts['bodies']:
        
        pc.set_alpha(1)
        pc.set_edgecolor('black')
        pc.set_facecolor(colors[j])
        j += 1
        pc.set_linewidth(0.8)
        
    for key in ['cbars', 'cmins', 'cmaxes', 'cmeans']:
        artist = parts[key]
        artist.set_edgecolor('black')
        artist.set_linewidth(1)  
        
    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels(['Random', 'Largest group', '1 look-ahead', '2 look-ahead', '3 look-ahead'])
    ax.set_ylabel("Number of moves to clear board")
    ax.set_title(f"Move distribution by heuristic")
    ax.set_axisbelow(True)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, format='pdf', bbox_inches='tight')
    if show:
        plt.show()

# 6.2: Supervised learning-based heuristics
# 6.2.1: Hyperparameter tuning
def plot_hyperparameter_tuning(show=False, save=False):
    """
    Plot results from hyperparameter tuning of selectec networks.
    The networks can be changed by changing the 'net_type' parameter.
    """

    # Plot convergence of hyperparameter tuning trials
    base = 'results/6.2_supervised/hyper_tuning'
    net_type = 'w64d5'  # Change this to 'w32d5', 'w32d10' or 'w64d10' to plot for other networks
    paths = {
        'policy_csv': f'{base}/optuna_policy_{net_type}.csv',
        'value_csv':  f'{base}/optuna_value_{net_type}.csv',
        'policy_db':  f'{base}/optuna_policy_{net_type}.db',
        'value_db':   f'{base}/optuna_value_{net_type}.db',
    }

    df_policy = pd.read_csv(paths['policy_csv'], parse_dates=['datetime_start','datetime_complete'])
    df_value  = pd.read_csv(paths['value_csv'],   parse_dates=['datetime_start','datetime_complete'])
    for df in (df_policy, df_value):
        df['duration'] = pd.to_timedelta(df['duration'])

    cmap_pol = plt.cm.Greens # Green palette for policy network
    cmap_val = plt.cm.Reds  # Red palette for value network

    t = np.linspace(0.4, 0.9, 4)
    policy_colors = [cmap_pol(tt) for tt in t]
    value_colors  = [cmap_val(tt) for tt in t]

    fig1, axes1 = plt.subplots(1, 2, figsize=(7.5, 4))
    for ax, df, label in zip(axes1, (df_policy, df_value), ('Policy', 'Value')):
        if label == 'Policy':
            ax.plot(df['number'], df['value'],       'o', markersize=4, label='Loss per trial', color=policy_colors[-2])
            ax.plot(df['number'], df['value'].cummin(), '--', linewidth=2, label='Best so far', color='k')
        elif label == 'Value':
            ax.plot(df['number'], df['value'],       'o', markersize=4, label='Loss per trial', color=value_colors[-2])
            ax.plot(df['number'], df['value'].cummin(), '--', linewidth=2, label='Best so far', color='k')
        ax.set_xlabel('Trial')
        ax.set_ylabel('MSE')
        ax.grid()
        ax.set_title(f'{label} network convergence')
        ax.legend()
    plt.tight_layout()
    if save:
        fig1.savefig(f'figures/ch6/06hyper_convergence.pdf', dpi=300, bbox_inches='tight')
    if show:
        plt.show()

    # Compute permutation importances for hyperparameters
    base = 'results/6.2_supervised/hyper_tuning'
    specs = {
        'Value':  {'study': 'value_hpo',  'db': f'{base}/optuna_value_{net_type}.db'},
        'Policy': {'study': 'policy_hpo', 'db': f'{base}/optuna_policy_{net_type}.db'}
    }

    imps = {}
    for label, spec in specs.items():
        st = optuna.load_study(study_name=spec['study'], storage=f"sqlite:///{spec['db']}")
        df = st.trials_dataframe(attrs=('params', 'value'))
        param_cols = [c for c in df.columns if c.startswith('params_')]
        X = df[param_cols].apply(pd.to_numeric)
        X.columns = [c.replace('params_', '') for c in X.columns]
        y = df['value']

        rf = RandomForestRegressor(n_estimators=100, random_state=0)
        rf.fit(X, y)
        perm = permutation_importance(rf, X, y, n_repeats=10, random_state=0)
        imps[label] = pd.Series(perm.importances_mean, index=X.columns)

    df_imps = pd.DataFrame(imps).fillna(0)
    desired = ['bs', 'wd', 'lr']
    df_imps = df_imps.reindex(desired)

    fig, ax = plt.subplots(figsize=(6.5, 3))
    df_imps.plot.barh(ax=ax, color=[value_colors[-2], policy_colors[-2]], edgecolor='k')
    ax.set_xlabel('Permutation importance')
    ax.set_yticklabels(['Batch size', 'Weight decay', 'Learning rate'])
    ax.set_title('Hyperparameter importance for policy and value networks')
    proxies = [Patch(facecolor=policy_colors[-2], edgecolor="k"),Patch(facecolor=value_colors[-2], edgecolor="k")]
    ax.legend(proxies, [r'$f_{\pi,(64,5)}$', r'$f_{v,(64,5)}$'], 
        frameon=True, loc="lower right")
    plt.tight_layout()

    if save:
        plt.savefig(f'figures/ch6/06hyper_importance.pdf', dpi=300, bbox_inches='tight')
    if show:
        plt.show()

# 6.2.2: Training and validation
# Policy networks
def plot_train_and_validation_for_policy_net(show=False, save=False):
    """
    Plot training and validation loss for policy networks.
    """
    paths = {
        r"$w=32$, $d=5$":  "results/6.2_supervised/training_validation/policy/w32d5_policy.csv",
        r"$w=32$, $d=10$": "results/6.2_supervised/training_validation/policy/w32d10_policy.csv",
        r"$w=64$, $d=5$":  "results/6.2_supervised/training_validation/policy/w64d5_policy.csv",
        r"$w=64$, $d=10$": "results/6.2_supervised/training_validation/policy/w64d10_policy.csv",
    }

    logs = {}
    for label, fp in paths.items():
        df = pd.read_csv(fp)
        df = df.rename(columns={
            "epoch": "Epoch",
            "train_loss": "Training loss",
            "val_loss": "Validation loss",
            "val_top1_acc": "Top-1 accuracy (val)",
            "val_top3_acc": "Top-3 accuracy (val)"
        })
        logs[label] = df[["Epoch", "Training loss", "Validation loss", "Top-1 accuracy (val)", "Top-3 accuracy (val)"]]

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(7.5, 4), sharex=True)

    cmap_pol = plt.cm.Greens # Green colors for policy networks
    t = np.linspace(0.4, 0.9, 4)
    colors = [cmap_pol(tt) for tt in t]
    
    # Plot each network's training and validation loss
    j = 0
    for label, df in logs.items():
        ax1.plot(df["Epoch"], df["Training loss"], label=label, marker="o",linewidth=2, markersize=5, color = colors[j])
        ax1.plot(df["Epoch"], df["Validation loss"], label=label, linestyle="--", marker="o",linewidth=2, markersize=5, color = colors[j])
        j += 1
    ax1.set_xlabel("Epoch")
    ax1.set_title("Training (full lines) and validation loss (stapled lines) for policy networks")
    ax1.set_ylabel("Loss")
    ax1.grid(True)
    
    # Make nice legend
    labels = [r"$f_{\pi,(32,5)}$", r"$f_{\pi,(32,10)}$", r"$f_{\pi,(64,5)}$", r"$f_{\pi,(64,10)}$"]
    proxies = [Patch(facecolor=c, edgecolor="k") for c in colors]
    ax1.legend(proxies, labels, title=r"Policy network", ncol=2, 
           frameon=True, loc="upper right")
    
    plt.tight_layout()
    if save:
        plt.savefig("figures/ch6/06policy_net_train_and_validation.pdf", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    
    # Evaluation metrics for policy networks
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(7.5,4.5))
    j = 0
    for label, df in logs.items():
        ax1.plot(df["Epoch"], df["Top-1 accuracy (val)"], label=label, linestyle="-", marker="o",linewidth=2, markersize=5, color = colors[j])
        ax2.plot(df["Epoch"], df["Top-3 accuracy (val)"], label=label, linestyle="-", marker="o",linewidth=2, markersize=5, color = colors[j])
        j += 1
    
    ax1.set_xlabel("Epoch")
    ax1.set_title("Top-1 validation accuracy")
    ax1.set_ylabel("Top-1 accuracy")
    ax1.grid(True)
    
    ax2.set_xlabel("Epoch")
    ax2.set_title("Top-3 validation accuracy")
    ax2.set_ylabel("Top-3 accuracy")
    ax2.grid(True)
    
    # Make nice legend, underneath the axes since it is common for both plots
    labels = [r"$f_{\pi,(32,5)}$", r"$f_{\pi,(32,10)}$", r"$f_{\pi,(64,5)}$", r"$f_{\pi,(64,10)}$"]
    proxies = [Patch(facecolor=c, edgecolor="k") for c in colors]    
    fig.legend(
        proxies, labels,
        ncol=4,
        frameon=True,
        loc="lower center",
        title=r"Policy network",
        bbox_to_anchor=(0.54, 0.02)
    )
    fig.tight_layout(rect=[0, 0.15, 1, 1])#
    if save:
        plt.savefig("figures/ch6/06policy_validation_accuracy.pdf", dpi=300, bbox_inches="tight")
    if show:
        plt.show()

def plot_violin_confidence_accuracy(show = True, save_path = None):
    """
    Plots a violin plot of the top-1 probability distribution for each remaining-move bin,
    overlaid with the top-1 accuracy curve.
    
    We have already calculated these parameters: 
    - p_max_list: list of top-1 probabilities.
    - correct_list: list of 0/1 correctness indicators.
    - moves_list: list of remaining moves for each state.
    
    These parameters are stored in the csv file. For calculations, check
    the 'results_supervised.ipynb' notebook.
    """
    df = pd.read_csv('results/6.2_supervised/training_validation/policy_confidence.csv')
    p = np.asarray(df.top1_prob)
    c = np.asarray(df.correct)
    m = np.asarray(df.remaining_moves)
    unique_moves = np.unique(m)
    
    # Filter out moves greater than 20 (mostly just noise)
    data = [p[m == mm] for mm in unique_moves if mm <= 20]
    # Compute accuracy per difficulty
    accuracy = [c[m == mm].mean() for mm in unique_moves if mm <= 20]
    
    fig, ax = plt.subplots(figsize=(6.5, 4))
    parts = ax.violinplot(data, positions=np.arange(1,21), widths=0.7,
                          showmeans=True, showmedians=False)
    
    cmap_pol = plt.cm.Greens
    t = np.linspace(0.4, 0.9, 4)
    colors = [cmap_pol(tt) for tt in t]
    for pc in parts['bodies']:
        pc.set_alpha(0.4)
        pc.set_edgecolor(colors[-1])
        pc.set_facecolor(colors[-1])
        
    for key in ['cbars', 'cmins', 'cmaxes', 'cmeans']:
        artist = parts[key]
        artist.set_edgecolor(colors[-1])
        artist.set_linewidth(1)  
        artist.set_alpha(0.6)
    
    ax.plot(np.arange(1,21), accuracy, 'o-', color=colors[-1], linewidth=2,
            label=r'Top-1 accuracy')
    
    ax.set_xlabel('Remaining moves')
    ax.set_ylabel('Top-1 probability')
    ax.set_title('Distribution of Top-1 probabilities and accuracy by difficulty')
    ax.set_xticks(unique_moves)
    ax.legend(loc='lower left')
    ax.set_xlim(0.5,20.5)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()

# Value networks
def plot_train_and_validation_for_value_net(save_path = None, show=False):
    """
    Plot training and validation loss for value networks.
    """
    paths = {
        r"$w=32$, $d=5$":  "results/6.2_supervised/training_validation/value/w32d5_logs.csv",
        r"$w=32$, $d=10$": "results/6.2_supervised/training_validation/value/w32d10_logs.csv",
        r"$w=64$, $d=5$":  "results/6.2_supervised/training_validation/value/w64d5_logs.csv",
        r"$w=64$, $d=10$": "results/6.2_supervised/training_validation/value/w64d10_logs.csv",
    }

    logs = {}
    for label, fp in paths.items():
        df = pd.read_csv(fp)
        df = df.rename(columns={
            "epoch": "Epoch",
            "train_loss": "Train Loss",
            "val_mse": "Validation MSE",
            "val_mae": "Validation MAE" # We ended up not using this one
        })
        logs[label] = df[["Epoch", "Train Loss", "Validation MSE"]]

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(7.5, 4.5), sharex=True)

    cmap_val = plt.cm.Reds
    t = np.linspace(0.4, 0.9, 4)
    colors = [cmap_val(tt) for tt in t]
    
    j = 0
    for label, df in logs.items():
        ax1.plot(df["Epoch"], df["Train Loss"], label=label, marker="o",linewidth=2, markersize=5, color = colors[j])
        ax2.plot(df["Epoch"], df["Validation MSE"], label=label, marker="o",linestyle = "--",linewidth=2, markersize=5, color = colors[j])
        j += 1
    ax1.set_xlabel("Epoch")
    ax1.set_title(r"Training loss over epochs")
    ax1.set_ylabel(r"Training loss")
    ax1.grid(True)

    ax2.set_title("Validation loss over epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE")
    ax2.grid(True)
    
    # Common, nice legend
    labels = [r"$f_{v,(32,5)}$", r"$f_{v,(32,10)}$", r"$f_{v,(64,5)}$", r"$f_{v,(64,10)}$"]
    proxies = [Patch(facecolor=c, edgecolor="k") for c in colors]
    fig.legend(
        proxies, labels,
        ncol=4,
        frameon=True,
        loc="lower center",
        title=r"Value network",
        bbox_to_anchor=(0.54, 0.02)
    )
    fig.tight_layout(rect=[0, 0.15, 1, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

def plot_violin_predicted_moves(show = False, save_path = None):
    """
    Plots a violin plot of the value network's predicted remaining moves distribution
    for each true remaining moves bin (0 - 20), overlaid with the mean predicted move.
    """
    # Get predictions and true values from CSV
    csv_path = "results/6.2_supervised/training_validation/value_predictions.csv"
    df = pd.read_csv(csv_path)
    preds = np.asarray(df.predicted_moves, dtype=float)
    trues = np.asarray(df.actual_moves, dtype=int)
    
    # Distribute data over number of remaining moves (only worth between 0 and 20)
    all_bins = np.arange(0, 21)
    data = []
    positions = []
    mean_preds = []
    for b in all_bins:
        arr = preds[trues == b]
        if arr.size > 0:
            data.append(arr)
            positions.append(b)
            mean_preds.append(arr.mean())
            
    fig, ax = plt.subplots(figsize=(6.5, 4))
    parts = ax.violinplot(
        data,
        positions=positions,
        widths=0.7,
        showmeans=True,
        showmedians=False,
        showextrema=True
    )
    
    cmap_pol = plt.cm.Reds
    t = np.linspace(0.4, 0.9, 4)
    colors = [cmap_pol(tt) for tt in t]
    for pc in parts['bodies']:
        pc.set_alpha(0.7)
        pc.set_edgecolor(colors[-1])
        pc.set_facecolor(colors[-1])


    for key in ['cbars', 'cmins', 'cmaxes', 'cmeans']:
        artist = parts[key]
        artist.set_edgecolor(colors[-1])
        artist.set_linewidth(1)  
        
    # Perfect prediction line
    ax.plot(positions, positions, '.:', linewidth=2, color='orange', label='Perfect prediction',  markersize=7)

    ax.set_xlabel("Actual remaining moves")
    ax.set_ylabel("Predicted remaining moves")
    ax.set_title(r"Distribution of predicted remaining moves by actual moves")
    ax.set_xticks(all_bins)           # show all ticks 0–20
    ax.set_xlim(-0.5, 20.5)
    ax.set_ylim(0, 20.5)
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()

# 6.2.3: Evaluation
def plot_value_vs_policy_distributions(save_path=None, show=False):
    """
    Plot move distributions for value and policy networks side by side.
    """
    value_csv  = "results/6.2_supervised/evaluation/value_evaluation.csv"
    policy_csv = "results/6.2_supervised/evaluation/policy_evaluation.csv"
    
    # Load and make dataframe
    df_val = pd.read_csv(value_csv); df_val['Type']='Value'
    df_pol = pd.read_csv(policy_csv); df_pol['Type']='Policy'
    df = pd.concat([df_pol, df_val], ignore_index=True)

    # Sort by network
    methods   = df['Method'].unique()
    n         = len(methods)
    positions = np.arange(1, n+1)

    # Set violin width and offset from center
    w      = 0.4
    offset = 0.02

    pol_data = [df.loc[(df.Method==m)&(df.Type=='Policy'),'Moves'] for m in methods]
    val_data = [df.loc[(df.Method==m)&(df.Type=='Value'),'Moves']  for m in methods]
    cmap_pol = plt.cm.Greens
    cmap_val = plt.cm.Reds
    t = np.linspace(0.4, 0.9, n)
    policy_colors = [cmap_pol(tt) for tt in t]
    value_colors  = [cmap_val(tt) for tt in t]

    # Plot violins
    fig, ax = plt.subplots(figsize=(7.5,4))
    ax.grid(axis='y', linewidth=0.5)

    # Policy halves
    parts_pol = ax.violinplot(
        pol_data,
        positions=positions-offset,
        widths=w,
        showmeans=False,
        showextrema=False
    )
    for i, pc in enumerate(parts_pol['bodies']):
        verts = pc.get_paths()[0].vertices
        center = verts[0,0]
        verts[:,0] = np.minimum(verts[:,0], center)
        pc.set_facecolor(policy_colors[i])
        pc.set_edgecolor('black')
        pc.set_linewidth(1)
        pc.set_alpha(1)

    # Value halves
    parts_val = ax.violinplot(
        val_data,
        positions=positions+offset,
        widths=w,
        showmeans=False,
        showextrema=False
    )
    for i, pc in enumerate(parts_val['bodies']):
        verts = pc.get_paths()[0].vertices
        center = verts[0,0]
        verts[:,0] = np.maximum(verts[:,0], center)
        pc.set_facecolor(value_colors[i])
        pc.set_edgecolor('black')
        pc.set_linewidth(1)
        pc.set_alpha(1)

    # Draw custom means, maxes and mins to make positions nice
    for i, pos in enumerate(positions):
        m_pol = np.mean(pol_data[i])
        max_pol = np.max(pol_data[i])
        min_pol = np.min(pol_data[i])
        x0_pol, x1_pol = pos-offset-w/4, pos-offset
        ax.hlines(m_pol, x0_pol, x1_pol, color='black', linewidth=1)
        ax.hlines(max_pol, x0_pol+w/8, x1_pol, color='black', linewidth=1)
        ax.hlines(min_pol, x0_pol+w/8, x1_pol, color='black', linewidth=1)

        m_val = np.mean(val_data[i])
        max_val = np.max(val_data[i])
        min_val = np.min(val_data[i])
        x0_val, x1_val = pos+offset, pos+offset+w/4
        ax.hlines(m_val, x0_val, x1_val, color='black', linewidth=1)
        ax.hlines(max_val, x0_val, x1_val-w/8, color='black', linewidth=1)
        ax.hlines(min_val, x0_val, x1_val-w/8, color='black', linewidth=1)

    ax.set_xticks(positions)
    ax.set_xticklabels([r"$f_{\pi,(32,5)}\mid f_{v,(32,5)}$", r"$f_{\pi,(32,10)}\mid f_{v,(32,10)}$", 
                        r"$f_{\pi,(64,5)}\mid f_{v,(64,5)}$", r"$f_{\pi,(64,10)}\mid f_{v,(64,10)}$"])
    ax.set_ylabel("Number of moves to clear board")
    ax.set_title("Move distribution by policy (green) and value networks (red)")
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    if save_path: 
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:      
        plt.show()

# 6.3: PPO
# 6.3.1: Training
def plot_ppo_training(show=False, save=False):
    """
    Plot training results from PPO. This includes reward over time, and
    a 3-way-split of each component of the loss function. All data is gathered
    from a TensorBoard event file.
    """
    event_file_big   = 'results/6.3_reinforcement/training/tensorboards/w64d10/events.out.tfevents.1748025998.idun-09-17.3020223-9.0'
    event_file_small = 'results/6.3_reinforcement/training/tensorboards/w32d5/events.out.tfevents.1748607018.idun-06-01.2601453-5.0'

    ea_big = event_accumulator.EventAccumulator(
        event_file_big,
        size_guidance={event_accumulator.SCALARS: 0}
    )
    ea_big.Reload()

    ea_small = event_accumulator.EventAccumulator(
        event_file_small,
        size_guidance={event_accumulator.SCALARS: 0}
    )
    ea_small.Reload()

    tag = 'rollout/ep_rew_mean'
    events_big   = ea_big.Scalars(tag)
    events_small = ea_small.Scalars(tag)

    # Big model
    wall_times_big     = [e.wall_time for e in events_big]
    t0_big            = wall_times_big[0]
    elapsed_days_big  = [(wt - t0_big) / (3600 * 24) for wt in wall_times_big]
    rewards_big       = [e.value for e in events_big]

    # Small model
    wall_times_small    = [e.wall_time for e in events_small]
    t0_small           = wall_times_small[0]
    elapsed_days_small = [(wt - t0_small) / (3600 * 24) for wt in wall_times_small]
    rewards_small      = [e.value for e in events_small]

    window = 1000
    ser_big   = pd.Series(rewards_big)
    ser_small = pd.Series(rewards_small)

    # Calculate the moving averages
    ma_big    = ser_big.rolling(window=window, center=True, min_periods=1).mean().values
    ma_small  = ser_small.rolling(window=window, center=True, min_periods=1).mean().values

    plt.figure(figsize=(6.93, 3.5))

    plt.plot(
        elapsed_days_big[::5], # Only plot one in five, since it is so dense
        rewards_big[::5], 
        color=plt.cm.YlOrBr(0.4), 
        alpha=0.6, 
        label='Loss $(64,10)$'
    )
    
    
    plt.plot(
        elapsed_days_small[::5], 
        rewards_small[::5], 
        color=plt.cm.YlOrBr(0.2), 
        alpha=0.6, 
        label='Loss $(32,5)$'
    )

    plt.plot(
        elapsed_days_big[::5], 
        ma_big[::5], 
        color='darkgrey', 
        linewidth=1.5,
        linestyle='--', 
        label=f'MA $(64,10)$'
    )
    
    plt.plot(
        elapsed_days_small[::5], 
        ma_small[::5], 
        color='lightgrey', 
        linewidth=1.5, 
        linestyle='--',
        label=f'MA $(32,5)$'
    )

    plt.xlabel('Training time (days)')
    plt.ylabel(r'Mean episode reward, $\bar{R}$')
    plt.title('Mean reward per episode over time')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    
    color_big   = plt.cm.YlOrBr(0.4)
    color_small = plt.cm.YlOrBr(0.2)
    
    proxy_big_raw   = Line2D([0], [0], color=color_big,   alpha=0.6, linestyle='-', label='Big raw')
    proxy_big_ma    = Line2D([0], [0], color='darkgrey',   linewidth=1.5, linestyle='--', label='Big MA (1000)')
    proxy_small_raw = Line2D([0], [0], color=color_small, alpha=0.6, linestyle='-', label='Small raw')
    proxy_small_ma  = Line2D([0], [0], color='lightgray', linewidth=1.5, linestyle='--', label='Small MA (1000)')

    proxies = [proxy_big_raw, proxy_big_ma, proxy_small_raw, proxy_small_ma]
    labels  = ["Loss, $(64,10)$", "Moving average, $(64,10)$", "Loss, $(32,5)$", "Moving average, $(32,5)$"]
    
    plt.legend(proxies, labels, loc='lower right')
    plt.tight_layout()
    
    if save:
        plt.savefig("figures/ch6/06ppo_rewards.pdf", dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    
    # Find the exact tags for policy (actor), value (critic), and entropy losses
    tags_big   = ea_big.Tags()['scalars']
    policy_tag_big  = next(t for t in tags_big if 'policy' in t.lower() and 'loss' in t.lower())
    value_tag_big   = next(t for t in tags_big if 'value'  in t.lower() and 'loss' in t.lower())
    entropy_tag_big = next(t for t in tags_big if 'entropy' in t.lower())

    tags_small   = ea_small.Tags()['scalars']
    policy_tag_small  = next(t for t in tags_small if 'policy' in t.lower() and 'loss' in t.lower())
    value_tag_small   = next(t for t in tags_small if 'value'  in t.lower() and 'loss' in t.lower())
    entropy_tag_small = next(t for t in tags_small if 'entropy' in t.lower())

    # Pull out the events
    events_pol_big = ea_big.Scalars(policy_tag_big)
    events_val_big = ea_big.Scalars(value_tag_big)
    events_ent_big = ea_big.Scalars(entropy_tag_big)

    events_pol_small = ea_small.Scalars(policy_tag_small)
    events_val_small = ea_small.Scalars(value_tag_small)
    events_ent_small = ea_small.Scalars(entropy_tag_small)

    # ──────────────────────────────────────────────────────────────────────────────
    # 2) Convert wall_time to “days since start” and collect loss values

    # For Big model
    t0_big         = events_pol_big[0].wall_time
    time_pol_big   = np.array([(e.wall_time - t0_big) / (3600*24) for e in events_pol_big])
    time_val_big   = np.array([(e.wall_time - t0_big) / (3600*24) for e in events_val_big])
    time_ent_big   = np.array([(e.wall_time - t0_big) / (3600*24) for e in events_ent_big])

    loss_pol_big   = np.array([e.value for e in events_pol_big])
    loss_val_big   = np.array([e.value for e in events_val_big])
    loss_ent_big   = np.array([e.value for e in events_ent_big])

    # For Small model
    t0_small         = events_pol_small[0].wall_time
    time_pol_small   = np.array([(e.wall_time - t0_small) / (3600*24) for e in events_pol_small])
    time_val_small   = np.array([(e.wall_time - t0_small) / (3600*24) for e in events_val_small])
    time_ent_small   = np.array([(e.wall_time - t0_small) / (3600*24) for e in events_ent_small])

    loss_pol_small   = np.array([e.value for e in events_pol_small])
    loss_val_small   = np.array([e.value for e in events_val_small])
    loss_ent_small   = np.array([e.value for e in events_ent_small])

    window = 1000

    ma_pol_big   = pd.Series(loss_pol_big).rolling(window=window, center=True, min_periods=1).mean().values
    ma_pol_small = pd.Series(loss_pol_small).rolling(window=window, center=True, min_periods=1).mean().values
    ma_val_big   = pd.Series(loss_val_big).rolling(window=window, center=True, min_periods=1).mean().values
    ma_val_small = pd.Series(loss_val_small).rolling(window=window, center=True, min_periods=1).mean().values
    ma_ent_big   = pd.Series(loss_ent_big).rolling(window=window, center=True, min_periods=1).mean().values
    ma_ent_small = pd.Series(loss_ent_small).rolling(window=window, center=True, min_periods=1).mean().values

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4), sharey=False)

    color_big   = plt.cm.YlOrBr(0.4)
    color_small = plt.cm.YlOrBr(0.2)
    
    ax1.plot(
        time_pol_big[::5], 
        loss_pol_big[::5], 
        color=color_big, 
        alpha=0.6, 
        linestyle='-', 
        label='Loss, $(64,10)$'
    )
    ax1.plot(
        time_pol_small[::5], 
        loss_pol_small[::5], 
        color=color_small, 
        alpha=0.6, 
        linestyle='-', 
        label='Loss, $(32,5)$'
    )
    ax1.plot(
        time_pol_big[::5], 
        ma_pol_big[::5], 
        color='darkgrey', 
        linewidth=1.5, 
        linestyle='--', 
        label='Moving average, $(64,10)$'
    )
    ax1.plot(
        time_pol_small[::5], 
        ma_pol_small[::5], 
        color='lightgrey', 
        linewidth=1.5, 
        linestyle='--', 
        label='Moving average, $(32,5)$'
    )
    ax1.set_xlabel('Training time (days)', fontsize=14)
    ax1.set_title('Actor loss over time', fontsize=16)
    ax1.set_ylabel(r'Actor loss, $\mathcal{L}_{\mathrm{CLIP}}(\boldsymbol{\theta})$', fontsize=14)
    ax1.set_ylim(ymax=0.5)
    ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    ax2.plot(
        time_val_big[::5], 
        loss_val_big[::5], 
        color=color_big, 
        alpha=0.6, 
        linestyle='-', 
        label='Big raw'
    )
    ax2.plot(
        time_val_small[::5], 
        loss_val_small[::5], 
        color=color_small, 
        alpha=0.6, 
        linestyle='-', 
        label='Small raw'
    )
    ax2.plot(
        time_val_big[::5], 
        ma_val_big[::5], 
        color='darkgrey', 
        linewidth=1.5, 
        linestyle='--', 
        label='Big MA (1000)'
    )
    ax2.plot(
        time_val_small[::5], 
        ma_val_small[::5], 
        color='lightgrey', 
        linewidth=1.5, 
        linestyle='--', 
        label='Small MA (1000)'
    )
    ax2.set_xlabel('Training time (days)', fontsize=14)
    ax2.set_title('Critic loss over time', fontsize=16)
    ax2.set_ylabel(r'Critic loss, $\mathcal{L}_{\mathrm{V}}(\boldsymbol{\phi})$', fontsize=14)
    ax2.set_ylim(ymin=0)
    ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    # 4c) Entropy bonus (negative entropy) loss
    ax3.plot(
        time_ent_big[::5], 
        loss_ent_big[::5], 
        color=color_big, 
        alpha=0.6, 
        linestyle='-', 
        label='Big raw'
    )
    ax3.plot(
        time_ent_small[::5], 
        loss_ent_small[::5], 
        color=color_small, 
        alpha=0.6, 
        linestyle='-', 
        label='Small raw'
    )
    ax3.plot(
        time_ent_big[::5], 
        ma_ent_big[::5], 
        color='darkgrey', 
        linewidth=1.5, 
        linestyle='--', 
        label='Big MA (1000)'
    )
    ax3.plot(
        time_ent_small[::5], 
        ma_ent_small[::5], 
        color='lightgrey', 
        linewidth=1.5, 
        linestyle='--', 
        label='Small MA (1000)'
    )
    ax3.set_xlabel('Training time (days)')
    ax3.set_title('Entropy loss over time', fontsize=16)
    ax3.set_ylabel(r'Entropy loss, $-\mathcal{H}(\boldsymbol{\theta})$')
    ax3.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    # ──────────────────────────────────────────────────────────────────────────────
    # 5) Build a common legend underneath all subplots

    # Collect all handles & labels from ax1, ax2, ax3
    handles, labels = [], []
    for ax in (ax1, ax2, ax3):
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels  += l

    # Remove duplicate legend entries by using an ordered dict:
    from collections import OrderedDict
    by_label = OrderedDict(zip(labels, handles))
    unique_labels = list(by_label.keys())
    unique_handles = list(by_label.values())

    # 1) Define proxy artists matching your plotted lines
    # Define proxy artists matching your plotted lines
    proxy_big_raw   = Line2D([0], [0], color=color_big,   alpha=0.6, linestyle='-', label='Big raw')
    proxy_big_ma    = Line2D([0], [0], color='darkgrey',   linewidth=1.5, linestyle='--', label='Big MA (1000)')
    proxy_small_raw = Line2D([0], [0], color=color_small, alpha=0.6, linestyle='-', label='Small raw')
    proxy_small_ma  = Line2D([0], [0], color='lightgray', linewidth=1.5, linestyle='--', label='Small MA (1000)')

    proxies = [proxy_big_raw, proxy_big_ma, proxy_small_raw, proxy_small_ma]
    labels  = ["Loss, $(64,10)$", "Moving average, $(64,10)$", "Loss, $(32,5)$", "Moving average, $(32,5)$"]

    # Create a single legend for all subplots, placed underneath
    fig.legend(
        proxies, labels,
        ncol=4,
        frameon=True,
        loc='lower center',
        title='Loss curves',
        bbox_to_anchor=(0.54, 0.02)
    )

    # Adjust layout so there’s room for the legend
    #plt.subplots_adjust(bottom=0.2)
    fig.tight_layout(rect=[0, 0.15, 1, 1])
    if save:
        plt.savefig("figures/ch6/06ppo_losses.pdf", dpi=300, bbox_inches='tight')
    if show:
        plt.show()

def plot_ppo_critic_calibration(show=False, save_path=None):
    """
    Plot the actor vs. critic calibration for one PPO model.
    """
    arr = np.load("results/6.3_reinforcement/evaluation/ppo_critic_calib_data.npz")
    predictions = arr["predictions"]

    bins      = np.arange(1, 21)
    data      = []
    positions = []
    for i, b in enumerate(bins):
        col = predictions[:, i]
        col = col[~np.isnan(col)]
        if col.size > 0:
            data.append(col)
            positions.append(b)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    parts = ax.violinplot(
        data,
        positions=positions,
        widths=0.7,
        showmeans=True,
        showmedians=False,
        showextrema=True
    )

    col = plt.cm.YlOrBr(0.35)
    for pc in parts['bodies']:
        pc.set_facecolor(col)
        pc.set_edgecolor(col)
        pc.set_alpha(0.7)
    for key in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
        if key in parts:
            art = parts[key]
            try:    art.set_edgecolor(col)
            except: art.set_color(col)
            art.set_linewidth(1)
            
    ax.plot(
        positions, positions,
        '.:', linewidth=2, color='darkorange',
        label='Perfect calibration',
        markersize=7
    )

    ax.set_xlabel("Actual remaining moves")
    ax.set_ylabel("Predicted remaining moves")
    ax.set_title("Actor-critic calibration")
    ax.set_xticks(bins)
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(0,   20.5)
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()

# 6.3.2: Evaluation
def plot_ppo_moves(save_path=None,show=False):
    """
    Plot evaluation of actors and critics from the two PPO models.
    """ 
    # Large model
    df = pd.read_csv("results/6.3_reinforcement/evaluation/ppo_evaluation.csv")
    methods = list(df['Method'].unique())
    if len(methods) != 4:
        raise ValueError(f"Expected exactly 4 Method labels, got: {methods}")

    # 2) Identify big vs. small and actor vs. critic
    big_actor_label   = next(m for m in methods if '(64,10)' in m and m.strip().startswith('$\\pi'))
    big_critic_label  = next(m for m in methods if '(64,10)' in m and m.strip().startswith('$v'))
    small_actor_label = next(m for m in methods if '(32,5)' in m and m.strip().startswith('$\\pi'))
    small_critic_label= next(m for m in methods if '(32,5)' in m and m.strip().startswith('$v'))

    # 3) Extract move‐counts
    actor_big_moves   = df.loc[df.Method == big_actor_label,   'Moves'].values
    critic_big_moves  = df.loc[df.Method == big_critic_label,  'Moves'].values
    actor_small_moves = df.loc[df.Method == small_actor_label, 'Moves'].values
    critic_small_moves= df.loc[df.Method == small_critic_label,'Moves'].values

    # 4) Styling params
    positions = [1, 2]     # we'll put "small" at pos=1, "big" at pos=2
    w         = 0.4
    offset    = 0.02
    big_color = plt.cm.YlOrBr(0.4)  # π color
    small_color = plt.cm.YlOrBr(0.2)   # v color

    fig, ax = plt.subplots(figsize=(4.5,4))
    ax.grid(axis='y', linewidth=0.5)

    # ======================
    # 5) Draw the “small” split‐violin at x=1 (positions[0])
    pos = positions[0]

    # 5a) Actor‐small (left half)
    parts_pol_small = ax.violinplot(
        [actor_small_moves],
        positions=[pos - offset],
        widths=w,
        showmeans=False,
        showextrema=False
    )
    for pc in parts_pol_small['bodies']:
        verts  = pc.get_paths()[0].vertices
        center = pos - offset
        verts[:,0] = np.minimum(verts[:,0], center)
        pc.set_facecolor(small_color)
        pc.set_edgecolor('black')
        pc.set_linewidth(1)
        pc.set_alpha(1)

    # 5b) Critic‐small (right half)
    parts_val_small = ax.violinplot(
        [critic_small_moves],
        positions=[pos + offset],
        widths=w,
        showmeans=False,
        showextrema=False
    )
    for pc in parts_val_small['bodies']:
        verts  = pc.get_paths()[0].vertices
        center = pos + offset
        verts[:,0] = np.maximum(verts[:,0], center)
        pc.set_facecolor(small_color)
        pc.set_edgecolor('black')
        pc.set_linewidth(1)
        pc.set_alpha(1)

    # 5c) Custom mean/min/max bars for small
    m_pol_small   = actor_small_moves.mean()
    max_pol_small = actor_small_moves.max()
    min_pol_small = actor_small_moves.min()
    x0_pol_s, x1_pol_s = pos - offset - w/4, pos - offset
    ax.hlines(m_pol_small,   x0_pol_s,          x1_pol_s,          color='black', linewidth=1)
    ax.hlines(max_pol_small, x0_pol_s + w/8,    x1_pol_s,          color='black', linewidth=1)
    ax.hlines(min_pol_small, x0_pol_s + w/8,    x1_pol_s,          color='black', linewidth=1)

    m_val_small   = critic_small_moves.mean()
    max_val_small = critic_small_moves.max()
    min_val_small = critic_small_moves.min()
    x0_val_s, x1_val_s = pos + offset, pos + offset + w/4
    ax.hlines(m_val_small,   x0_val_s,          x1_val_s,          color='black', linewidth=1)
    ax.hlines(max_val_small, x0_val_s,          x1_val_s - w/8,    color='black', linewidth=1)
    ax.hlines(min_val_small, x0_val_s,          x1_val_s - w/8,    color='black', linewidth=1)

    # ======================
    # 6) Draw the “big” split‐violin at x=2 (positions[1])
    pos = positions[1]

    # 6a) Actor‐big (left half)
    parts_pol_big = ax.violinplot(
        [actor_big_moves],
        positions=[pos - offset],
        widths=w,
        showmeans=False,
        showextrema=False
    )
    for pc in parts_pol_big['bodies']:
        verts  = pc.get_paths()[0].vertices
        center = pos - offset
        verts[:,0] = np.minimum(verts[:,0], center)
        pc.set_facecolor(big_color)
        pc.set_edgecolor('black')
        pc.set_linewidth(1)
        pc.set_alpha(1)

    # 6b) Critic‐big (right half)
    parts_val_big = ax.violinplot(
        [critic_big_moves],
        positions=[pos + offset],
        widths=w,
        showmeans=False,
        showextrema=False
    )
    for pc in parts_val_big['bodies']:
        verts  = pc.get_paths()[0].vertices
        center = pos + offset
        verts[:,0] = np.maximum(verts[:,0], center)
        pc.set_facecolor(big_color)
        pc.set_edgecolor('black')
        pc.set_linewidth(1)
        pc.set_alpha(1)

    # 6c) Custom mean/min/max bars for big
    m_pol_big   = actor_big_moves.mean()
    max_pol_big = actor_big_moves.max()
    min_pol_big = actor_big_moves.min()
    x0_pol_b, x1_pol_b = pos - offset - w/4, pos - offset
    ax.hlines(m_pol_big,   x0_pol_b,          x1_pol_b,          color='black', linewidth=1)
    ax.hlines(max_pol_big, x0_pol_b + w/8,    x1_pol_b,          color='black', linewidth=1)
    ax.hlines(min_pol_big, x0_pol_b + w/8,    x1_pol_b,          color='black', linewidth=1)

    m_val_big   = critic_big_moves.mean()
    max_val_big = critic_big_moves.max()
    min_val_big = critic_big_moves.min()
    x0_val_b, x1_val_b = pos + offset, pos + offset + w/4
    ax.hlines(m_val_big,   x0_val_b,          x1_val_b,          color='black', linewidth=1)
    ax.hlines(max_val_big, x0_val_b,          x1_val_b - w/8,    color='black', linewidth=1)
    ax.hlines(min_val_big, x0_val_b,          x1_val_b - w/8,    color='black', linewidth=1)

    # ======================
    # 7) Final formatting
    ax.set_xticks(positions)
    ax.set_xticklabels([
        r"$\pi_{\boldsymbol{\theta},(32,5)}\mid v_{\boldsymbol{\phi},(32,5)}$",
        r"$\pi_{\boldsymbol{\theta},(64,10)}\mid v_{\boldsymbol{\phi},(64,10)}$"
    ], fontsize=14)
    ax.set_ylabel("Number of moves to clear board", fontsize=16)
    ax.set_title("Actor and critic move distributions", fontsize=18)
    ax.set_axisbelow(True)
    plt.tight_layout()

    # 8) Save/show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()

# 6.4: Search techniques
# 6.4.1: Testing on random boards
def plot_search_results_random_boards(show=False, save_path=None):
    """
    Plots best-so-far curves for MCTS and Beam on 1.000 random boards,
    """

    # 1) Time grid
    time_limit, n_points = 10.0, 200
    grid = np.linspace(0.5, time_limit, n_points)

    # 2) Method categories (PPO small → PPO large)
    net_keys  = ['w32d5', 'w32d10', 'w64d5', 'w64d10']
    heur_keys = ['1lookahead', '2lookahead']
    ppo_keys  = ['ppo_w32d5', 'ppo_w64d10']  # small first, then large

    # 3) Legend labels
    labels = {
        **{k: f'({k[1:].replace("d", ",")})' for k in net_keys},
        '1lookahead':  r'$\pi_1$',
        '2lookahead':  r'$\pi_2$',
        'ppo_w32d5':   r'$\pi_{\boldsymbol{\theta},(32,5)}$',
        'ppo_w64d10':  r'$\pi_{\boldsymbol{\theta},(64,10)}$',
    }

    # 4) Load & preprocess MCTS
    df_mcts_all = (
        pd.read_csv('results/6.4_search/mcts/mcts_random.csv')
          [['method', 'board', 'time', 'moves']]
          .rename(columns={'board': 'board_idx'})
    )
    df_mcts_all = df_mcts_all.sort_values(['method', 'board_idx', 'time']).copy()
    df_mcts_all['best_so_far'] = df_mcts_all.groupby(['method', 'board_idx'])['moves'].cummin()

    # 5) Load & preprocess Beam
    df_beam_all = (
        pd.read_csv('results/6.4_search/beam/beam_random.csv')
          [['method', 'board', 'time', 'moves']]
          .rename(columns={'board': 'board_idx'})
    )
    df_beam_all = df_beam_all.sort_values(['method', 'board_idx', 'time']).copy()
    df_beam_all['best_so_far'] = df_beam_all.groupby(['method', 'board_idx'])['moves'].cummin()

    # 6) Prepare figure & axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 4))

    # --- Plot MCTS on ax1 ---
    cmap_net   = plt.cm.Greens
    cmap_heur  = plt.cm.Blues
    cmap_ppo   = plt.cm.YlOrBr

    net_colors  = [cmap_net(x) for x in np.linspace(0.4, 0.9, len(net_keys))]
    heur_colors = [cmap_heur(x) for x in np.linspace(0.2, 0.9, len(heur_keys) + 2)][2:4]
    ppo_colors  = [cmap_ppo(x) for x in np.linspace(0.2, 0.4, len(ppo_keys))]

    color_map_mcts = {
        **{k: net_colors[i]   for i, k in enumerate(net_keys)},
        **{k: heur_colors[i]  for i, k in enumerate(heur_keys)},
        **{k: ppo_colors[i]   for i, k in enumerate(ppo_keys)}
    }

    for variant in net_keys + ppo_keys + heur_keys:
        if variant not in df_mcts_all['method'].unique():
            continue
        sub = df_mcts_all[df_mcts_all['method'] == variant]
        mean_curve = [
            sub[sub['time'] <= t]
               .sort_values(['board_idx', 'time'])
               .groupby('board_idx')['best_so_far']
               .last()
               .mean()
            for t in grid
        ]
        if variant in net_keys:
            label = f'MCTS, {labels[variant]}'
        else:
            label = labels[variant]
        ax1.plot(grid, mean_curve, label=label, color=color_map_mcts[variant])

    ax1.set(
        xlim=(-0.5, 10.5),
        ylim=(13.9, 17.1),
        xlabel='Time (s)',
        ylabel='Mean best solution',
        title='MCTS'
    )
    ax1.grid()

    # Legend for MCTS
    patches_mcts = []
    labels_mcts  = []
    # net keys
    for k in net_keys:
        patches_mcts.append(Patch(facecolor=color_map_mcts[k], edgecolor='k'))
        labels_mcts.append(f'$f_{{\\pi,{labels[k]}}}$')
    # PPO (small → large)
    for k in ppo_keys:
        patches_mcts.append(Patch(facecolor=color_map_mcts[k], edgecolor='k'))
        labels_mcts.append(labels[k])
    # heuristics
    for k in heur_keys:
        patches_mcts.append(Patch(facecolor=color_map_mcts[k], edgecolor='k'))
        labels_mcts.append(labels[k])
    ax1.legend(patches_mcts, labels_mcts, loc='upper right', frameon=True, ncol=2)

    # --- Plot Beam on ax2 ---
    cmap_net   = plt.cm.Reds
    cmap_heur  = plt.cm.Blues
    cmap_ppo   = plt.cm.YlOrBr

    net_colors  = [cmap_net(x) for x in np.linspace(0.4, 0.9, len(net_keys))]
    heur_colors = [cmap_heur(x) for x in np.linspace(0.2, 0.9, len(heur_keys) + 2)][2:4]
    ppo_colors  = [cmap_ppo(x) for x in np.linspace(0.2, 0.4, len(ppo_keys))]

    color_map_beam = {
        **{k: net_colors[i]   for i, k in enumerate(net_keys)},
        **{k: heur_colors[i]  for i, k in enumerate(heur_keys)},
        **{k: ppo_colors[i]   for i, k in enumerate(ppo_keys)}
    }

    for variant in net_keys + ppo_keys + heur_keys:
        if variant not in df_beam_all['method'].unique():
            continue
        sub = df_beam_all[df_beam_all['method'] == variant]
        mean_curve = [
            sub[sub['time'] <= t]
               .sort_values(['board_idx', 'time'])
               .groupby('board_idx')['best_so_far']
               .last()
               .mean()
            for t in grid
        ]
        if variant in net_keys:
            label = f'Beam, {labels[variant]}'
        elif variant in ppo_keys:
            # Replace \pi → v and \boldsymbol{\theta} → \boldsymbol{\phi}
            label = (
                labels[variant]
                  .replace(r'\pi', 'v')
                  .replace(r'\boldsymbol{\theta}', r'\boldsymbol{\phi}')
            )
        else:  # heuristic
            label = labels[variant].replace(r'\pi', 'v')
        ax2.plot(grid, mean_curve, label=label, color=color_map_beam[variant])

    ax2.set(
        xlim=(-0.5, 10.5),
        ylim=(13.9, 17.1),
        xlabel='Time (s)',
        ylabel='Mean best solution',
        title='Beam search'
    )
    ax2.grid()

    # Legend for Beam
    patches_beam = []
    labels_beam  = []
    # net keys
    for k in net_keys:
        patches_beam.append(Patch(facecolor=color_map_beam[k], edgecolor='k'))
        labels_beam.append(f'$f_{{v,{labels[k]}}}$')
    # PPO (small → large)
    for k in ppo_keys:
        patches_beam.append(Patch(facecolor=color_map_beam[k], edgecolor='k'))
        labels_beam.append(
            labels[k]
              .replace(r'\pi', 'v')
              .replace(r'\boldsymbol{\theta}', r'\boldsymbol{\phi}')
        )
    # heuristics
    for k in heur_keys:
        patches_beam.append(Patch(facecolor=color_map_beam[k], edgecolor='k'))
        labels_beam.append(labels[k].replace(r'\pi', 'v'))
    ax2.legend(patches_beam, labels_beam, loc='upper right', frameon=True, ncol=2)

    # Final layout
    fig.suptitle(r"Mean best-so-far solutions on 1.000 random boards")
    fig.tight_layout(rect=[0, 0, 1, 1.04])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()

# 6.4.2: Testing on daily boards
def plot_search_results_daily_boards(show=False, save_path=None):
    """
    Plots best-so-far solutions for MCTS and Beam search on 100 daily boards.
    """
    time_limit, n_points = 60.0, 200
    grid = np.linspace(0.5, time_limit, n_points)

    net_keys  = ['w32d5', 'w32d10', 'w64d5', 'w64d10']
    heur_keys = ['1lookahead', '2lookahead']
    ppo_keys  = ['ppo_w32d5', 'ppo_w64d10']  # small first, then large

    labels = {
        **{k: f'({k[1:].replace("d", ",")})' for k in net_keys},
        '1lookahead':  r'$\pi_1$',
        '2lookahead':  r'$\pi_2$',
        'ppo_w32d5':   r'$\pi_{\boldsymbol{\theta},(32,5)}$',
        'ppo_w64d10':  r'$\pi_{\boldsymbol{\theta},(64,10)}$',
    }

    df_mcts_all = (
        pd.read_csv('results/6.4_search/mcts/mcts_daily.csv')
          [['method', 'board', 'time', 'moves']]
          .rename(columns={'board': 'board_idx'})
    )
    df_mcts_all = df_mcts_all.sort_values(['method', 'board_idx', 'time']).copy()
    df_mcts_all['best_so_far'] = df_mcts_all.groupby(['method', 'board_idx'])['moves'].cummin()

    df_beam_all = (
        pd.read_csv('results/6.4_search/beam/beam_daily.csv')
          [['method', 'board', 'time', 'moves']]
          .rename(columns={'board': 'board_idx'})
    )
    df_beam_all = df_beam_all.sort_values(['method', 'board_idx', 'time']).copy()
    df_beam_all['best_so_far'] = df_beam_all.groupby(['method', 'board_idx'])['moves'].cummin()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 4))

    cmap_net   = plt.cm.Greens
    cmap_heur  = plt.cm.Blues
    cmap_ppo   = plt.cm.YlOrBr

    net_colors  = [cmap_net(x) for x in np.linspace(0.4, 0.9, len(net_keys))]
    heur_colors = [cmap_heur(x) for x in np.linspace(0.2, 0.9, len(heur_keys) + 2)][2:4]
    ppo_colors  = [cmap_ppo(x) for x in np.linspace(0.2, 0.4, len(ppo_keys))]

    color_map_mcts = {
        **{k: net_colors[i]   for i, k in enumerate(net_keys)},
        **{k: heur_colors[i]  for i, k in enumerate(heur_keys)},
        **{k: ppo_colors[i]   for i, k in enumerate(ppo_keys)}
    }

    for variant in net_keys + ppo_keys + heur_keys:
        if variant not in df_mcts_all['method'].unique():
            continue
        sub = df_mcts_all[df_mcts_all['method'] == variant]
        mean_curve = [
            sub[sub['time'] <= t]
               .sort_values(['board_idx', 'time'])
               .groupby('board_idx')['best_so_far']
               .last()
               .mean()
            for t in grid
        ]
        if variant in net_keys:
            label = f'MCTS, {labels[variant]}'
        else:
            label = labels[variant]
        ax1.plot(grid, mean_curve, label=label, color=color_map_mcts[variant])

    ax1.set(
        xlim=(-1, 61),
        ylim=(13.5, 15.3),
        xlabel='Time (s)',
        ylabel='Mean best solution',
        title='MCTS'
    )
    ax1.grid()

    patches_mcts = []
    labels_mcts  = []
    for k in net_keys:
        patches_mcts.append(Patch(facecolor=color_map_mcts[k], edgecolor='k'))
        labels_mcts.append(f'$f_{{\\pi,{labels[k]}}}$')
    for k in ppo_keys:
        patches_mcts.append(Patch(facecolor=color_map_mcts[k], edgecolor='k'))
        labels_mcts.append(labels[k])
    for k in heur_keys:
        patches_mcts.append(Patch(facecolor=color_map_mcts[k], edgecolor='k'))
        labels_mcts.append(labels[k])
    ax1.legend(patches_mcts, labels_mcts, loc='upper right', frameon=True, ncol=2)

    cmap_net   = plt.cm.Reds
    cmap_heur  = plt.cm.Blues
    cmap_ppo   = plt.cm.YlOrBr

    net_colors  = [cmap_net(x) for x in np.linspace(0.4, 0.9, len(net_keys))]
    heur_colors = [cmap_heur(x) for x in np.linspace(0.2, 0.9, len(heur_keys) + 2)][2:4]
    ppo_colors  = [cmap_ppo(x) for x in np.linspace(0.2, 0.4, len(ppo_keys))]

    color_map_beam = {
        **{k: net_colors[i]   for i, k in enumerate(net_keys)},
        **{k: heur_colors[i]  for i, k in enumerate(heur_keys)},
        **{k: ppo_colors[i]   for i, k in enumerate(ppo_keys)}
    }

    for variant in net_keys + ppo_keys + heur_keys:
        if variant not in df_beam_all['method'].unique():
            continue
        sub = df_beam_all[df_beam_all['method'] == variant]
        mean_curve = [
            sub[sub['time'] <= t]
               .sort_values(['board_idx', 'time'])
               .groupby('board_idx')['best_so_far']
               .last()
               .mean()
            for t in grid
        ]
        if variant in net_keys:
            label = f'Beam, {labels[variant]}'
        elif variant in ppo_keys:
            label = (
                labels[variant]
                  .replace(r'\pi', 'v')
                  .replace(r'\boldsymbol{\theta}', r'\boldsymbol{\phi}')
            )
        else:  # heuristic
            label = labels[variant].replace(r'\pi', 'v')
        ax2.plot(grid, mean_curve, label=label, color=color_map_beam[variant])

    ax2.set(
        xlim=(-1, 61),
        ylim=(13.5, 15.3),
        xlabel='Time (s)',
        ylabel='Mean best solution',
        title='Beam search'
    )
    ax2.grid()

    board_dictionary = db.get_daily_board()  
    avg_best = np.mean([info[1] for info in board_dictionary.values()])
    ax1.axhline(y=avg_best, color='k', linestyle='--')
    ax2.axhline(y=avg_best, color='k', linestyle='--')

    patches_beam = []
    labels_beam  = []
    for k in net_keys:
        patches_beam.append(Patch(facecolor=color_map_beam[k], edgecolor='k'))
        labels_beam.append(f'$f_{{v,{labels[k]}}}$')
    for k in ppo_keys:
        patches_beam.append(Patch(facecolor=color_map_beam[k], edgecolor='k'))
        labels_beam.append(
            labels[k]
              .replace(r'\pi', 'v')
              .replace(r'\boldsymbol{\theta}', r'\boldsymbol{\phi}')
        )
    for k in heur_keys:
        patches_beam.append(Patch(facecolor=color_map_beam[k], edgecolor='k'))
        labels_beam.append(labels[k].replace(r'\pi', 'v'))
    ax2.legend(patches_beam, labels_beam, loc='upper right', frameon=True, ncol=2)

    fig.suptitle(r"Mean best-so-far solutions on 100 daily boards")
    fig.tight_layout(rect=[0, 0, 1, 1.04])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
        
def plot_search_correct_proportion(show=True, save_path=None):
    """
    Plots the proportion of daily boards solved over time across all MCTS 
    and Beam search models, and across all models combined.
    """

    board_dictionary = db.get_daily_board()
    actual_sols = [ info[1] for _, info in board_dictionary.items() ]
    actual_series = pd.Series(actual_sols, index=range(len(actual_sols)))

    beam_csv = 'results/6.4_search/beam/beam_daily.csv'
    mcts_csv = 'results/6.4_search/mcts/mcts_daily.csv'
    time_limit = 60

    def _aggregate(df, actual_series, ensure_60=False, time_limit=None):
        df_sorted = df.sort_values(['board', 'time']).copy()
        df_sorted['best_moves_so_far'] = df_sorted.groupby('board')['moves'].cummin()
        df_pivot = (
            df_sorted
              .pivot_table(
                  index='time',
                  columns='board',
                  values='best_moves_so_far',
                  aggfunc='last'
              )
              .sort_index()
              .ffill()
        )

        mean_best = df_pivot.mean(axis=1)
        prop_solved = df_pivot.eq(actual_series, axis='columns').sum(axis=1) / df_pivot.shape[1]
        result = pd.DataFrame({
            'mean_best_moves': mean_best,
            'prop_solved':     prop_solved
        })

        if ensure_60 and 60 not in result.index:
            result = (
                result
                  .reindex(result.index.union([60]))
                  .sort_index()
                  .ffill()
            )

        if time_limit is not None:
            result = result[result.index <= time_limit]

        return result

    df_beam = pd.read_csv(beam_csv)
    df_beam = df_beam.rename(columns={'elapsed_s': 'time', 'n_moves': 'moves'}, errors='ignore')
    df_beam = df_beam[['board', 'time', 'moves']]

    df_mcts = pd.read_csv(mcts_csv)
    df_mcts = df_mcts.rename(columns={'elapsed_s': 'time', 'n_moves': 'moves'}, errors='ignore')
    df_mcts = df_mcts[['board', 'time', 'moves']]

    beam_df     = _aggregate(df_beam,     actual_series, ensure_60=False, time_limit=time_limit)
    mcts_df     = _aggregate(df_mcts,     actual_series, ensure_60=True,  time_limit=None)
    combined_df = _aggregate(
        pd.concat([df_beam, df_mcts], ignore_index=True),
        actual_series,
        ensure_60=False,
        time_limit=time_limit
    )

    plt.figure(figsize=(7.5, 4))
    plt.title("Proportion of Daily Boards Solved Over Time")
    plt.plot(combined_df.index, combined_df['prop_solved'], label='Total',       color='k')
    plt.plot(mcts_df.index,     mcts_df['prop_solved'],     label='MCTS only',   color='green')
    plt.plot(beam_df.index,     beam_df['prop_solved'],     label='Beam search', color='red')
    plt.xlabel('Time (s)')
    plt.ylabel('Proportion solved')
    plt.ylim(0, 1)
    plt.xlim(-1, 61)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
        
    def _prop_at(df, t):
        """
        If `t` is exactly in df.index, return df.loc[t, 'prop_solved'].
        Otherwise, return the last (earlier) prop_solved at index <= t.
        """
        # Make sure the index is sorted ascending (should already be from _aggregate)
        # Then pick the last index <= t
        idxs = df.index.values
        # if there's an exact match, just return it
        if t in idxs:
            return df.loc[t, 'prop_solved']
        # otherwise, filter to all indices <= t, then take the last one
        earlier = idxs[idxs <= t]
        if len(earlier) == 0:
            # no data point before t → assume 0 solved
            return 0.0
        last_idx = earlier.max()
        return df.loc[last_idx, 'prop_solved']

    # Specify the time‐points we care about:
    time_points = [1, 5, 10, 30, 60]

    print("\nProportion of daily boards solved at specific time checkpoints:")
    for τ in time_points:
        total_prop = _prop_at(combined_df, τ)
        mcts_prop  = _prop_at(mcts_df,      τ)
        beam_prop  = _prop_at(beam_df,      τ)
        print(f"  After {τ:>2d} second(s): Total: {total_prop:.3f},  MCTS: {mcts_prop:.3f},  Beam: {beam_prop:.3f}")
    
import pickle
def plot_number_of_optimal_solutions(show=False, save_path=None):

    with open("results/6.4_search/branching_may18.pkl", "rb") as f:
        data18 = pickle.load(f)
    
    with open("results/6.4_search/branching_may19.pkl", "rb") as f:
        data19 = pickle.load(f)

    runs = [
        ("May 18th", data18["branching"], data18["groups"]),
        ("May 19th", data19["branching"], data19["groups"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.5,4), sharey=False)

    for ax, (title, branching, groups) in zip(axes, runs):
        n_moves = len(branching)
        moves_remaining = np.arange(n_moves, 0, -1)

        ax.plot(moves_remaining, groups,    linestyle='-',  marker='o', label='Possible actions', color = 'cornflowerblue')
        ax.plot(moves_remaining, branching, linestyle='-',   marker='o', label='Optimal actions', color = 'darkorange')

        ax.set_title(title)
        ax.set_xlabel("Moves remaining")  
        #ax.set_xticks(moves_remaining)

        ax.set_ylabel("Number of actions")
        ax.set_yticks([0,5,10,15,20,25,30,35])
        ax.set_xticks(np.arange(2, len(moves_remaining)+1, 2))      
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(["Possible actions","Optimal actions"],frameon=True)#,edgecolor="black")
        ax.invert_xaxis()
        ax.set_ylim(-1, 38)

    fig.suptitle("Estimated optimal actions per number of remaining moves", y=0.97, fontsize=16)
    fig.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches = 'tight')
    if show:
        plt.show()

