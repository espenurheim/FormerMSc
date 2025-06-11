from figure_functions import *
import matplotlib as mpl

mpl.rcParams['font.size'] = 12
params = {
    "ytick.color": "black",
    "xtick.color": "black",
    "axes.labelcolor": "black",
    "axes.edgecolor": "black",
    "text.usetex": True,  # Enable LaTeX
    "font.family": "serif",
    "font.serif": ["Computer Modern Serif"],
}
plt.rcParams.update(params)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
# plt.style.use('seaborn-v0_8-whitegrid')

###################################################################################################
# CHAPTER 2 - The Former game
# 2.1 - The average number of shapes
# plot_avg_number_of_shapes(show=True)

# 2.2 - The branching factor
# plot_branching_factor(show=True, N=1000, K=16, P=20)

###################################################################################################
# CHAPTER 5 - Methodology
# plot_time_comparison_cpp_vs_python(show=True)
# plot_move_counts(show=True)

###################################################################################################
# CHAPTER 6 - Results
# 6.1: Self-made heuristics
# plot_heuristic_distributions(show=True)

# 6.2: Supervised learning-based heuristics
# 6.2.1: Hyperparameter tuning
# plot_hyperparameter_tuning(show=True)

# 6.2.2: Training and validation
# Policy networks:
# plot_train_and_validation_for_policy_net(show=True)
# plot_violin_confidence_accuracy(show=True)

# Value networks:
# plot_train_and_validation_for_value_net(show=True)
# plot_violin_predicted_moves(show=True)

# 6.2.3: Evaluation
# plot_value_vs_policy_distributions(show=True)

# 6.3: PPO
# 6.3.1: Training
# plot_ppo_training(show=True)

# plot_ppo_critic_calibration(show=True)

# 6.3.2: Evaluation
# plot_ppo_moves(show=True)

# 6.4: Search techniques with heuristics
# 6.4.1: Random boards
plot_search_results_random_boards(show=True)

# 6.4.2: Daily boards
# plot_search_results_daily_boards(show=True)

# plot_search_correct_proportion(show=True)
# plot_number_of_optimal_solutions(show=True)
