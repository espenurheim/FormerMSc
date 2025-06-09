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
#plt.style.use('seaborn-v0_8-whitegrid')

###################################################################################################
# CHAPTER 2 - The Former game
# 2.1 - The average number of shapes
#plot_avg_number_of_shapes(show=True, save_path='figures/ch2/02avg_shapes.pdf')

# 2.2 - The branching factor
#plot_branching_factor(show=True, save_path='02branching_factor.pdf', N=1000, K=16, P=20)

###################################################################################################
# CHAPTER 5 - Methodology
#plot_time_comparison_cpp_vs_python(show=True, save_path='figures/ch5/05cpp_vs_python.pdf')
#plot_move_counts(show=True, save_path = "figures/ch5/moves_in_dataset.pdf")

###################################################################################################
# CHAPTER 6 - Results
# 6.1: Self-made heuristics
#plot_heuristic_distributions(save_path='figures/ch6/06heuristic_distribution.pdf', show=True)

# 6.2: Supervised learning-based heuristics
# 6.2.1: Hyperparameter tuning
#plot_hyperparameter_tuning(show=True, save=True)

# 6.2.2: Training and validation
# Policy networks:
#plot_train_and_validation_for_policy_net(show=True, save=True)
#plot_violin_confidence_accuracy(show=True, save_path="figures/ch6/06policy_accuracy.pdf")

# Value networks:
#plot_train_and_validation_for_value_net(show = True, save_path = "figures/ch6/06value_net_train_and_validation.pdf")
#plot_violin_predicted_moves(show=True, save_path="figures/ch6/06value_accuracy.pdf")

# 6.2.3: Evaluation
#plot_value_vs_policy_distributions(show = True, save_path = "figures/ch6/06networks_distribution.pdf")

# 6.3: PPO
# 6.3.1: Training
plot_ppo_training(show=True, save=True)

#plot_ppo_critic_calibration(show=True, save_path="figures/ch6/06ppo_critic_calibration.pdf")

# 6.3.2: Evaluation
#plot_ppo_moves(show=True, save_path="figures/ch6/06ppo_move_distribution.pdf")

# 6.4: Search techniques with heuristics
# 6.4.1: Random boards
#plot_search_results_random_boards(save_path = "figures/ch6/06search2_random_boards.pdf", show=True)

# 6.4.2: Daily boards
#plot_search_results_daily_boards(show=True, save_path = "figures/ch6/06mcts_daily_movecounts.pdf")

#plot_search_correct_proportion(show=True, save_path = "figures/ch6/06mcts_daily_proportion.pdf")

#plot_number_of_optimal_solutions(show=True, save_path = "figures/ch6/06number_of_optimal_solutions.pdf")