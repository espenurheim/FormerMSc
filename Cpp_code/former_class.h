#ifndef FORMER_GAME_H
#define FORMER_GAME_H

#include <vector>
#include <set>
#include <utility>

using Board = std::vector<std::vector<int>>;
using Point = std::pair<int, int>;

class FormerGame {
public:
    int M, N, S;
    Board board;

    // Constructor: if a custom board is provided, use it; otherwise, create a random board.
    FormerGame(int M = 9, int N = 7, int S = 4, const Board* custom_board = nullptr);

    // Create a board with random integers in [0, S-1].
    Board create_board();

    // Return a copy of the current board.
    Board get_board() const;

    // Instance methods that call static implementations.
    std::set<Point> find_group(const Point & point) const;
    std::vector<std::set<Point>> get_groups() const;
    std::vector<Point> get_valid_turns() const;
    void remove_group(const std::set<Point> & group);
    void drop_board();
    void make_turn(const Point & point);
    bool is_game_over() const;

    // ---------- Static Methods ----------
    static std::set<Point> find_group_static(const Board & board, const Point & point);
    static std::vector<std::set<Point>> get_groups_static(const Board & board);
    static std::vector<Point> get_valid_turns_static(const Board & board);
    static Board remove_group_static(const Board & board, const std::set<Point> & group);
    static Board drop_board_static(const Board & board);
    static Board apply_turn_static(const Board & board, const Point & point);
    static std::vector<double> get_heuristic_probs_static(const Board & board, const std::vector<Point> & valid_turns);
    static bool is_game_over_static(const Board & board);
};

// Heuristic functions.
std::vector<double> heuristic_min_groups_1_look_ahead(const Board & board, const std::vector<Point> & valid_turns);
std::vector<double> heuristic_min_groups_2_look_ahead(const Board & board, const std::vector<Point> & valid_turns);
std::vector<double> heuristic_min_groups_3_look_ahead(const Board & board, const std::vector<Point> & valid_turns);
std::vector<double> heuristic_min_groups_2_look_ahead_softmax(const Board & board, const std::vector<Point> & valid_turns, double temperature = 1.0);

// Softmax function.
std::vector<double> softmax(const std::vector<double>& scores, double temperature = 1.0);

#endif // FORMER_GAME_H
