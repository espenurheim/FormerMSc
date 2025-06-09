#include "former_class.h"
#include <vector>
#include <set>
#include <queue>
#include <random>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <stdexcept>
#include <functional>
#include <cfloat>
using namespace std;

// ------------------- Constructor and Member Functions -------------------

FormerGame::FormerGame(int M, int N, int S, const Board* custom_board)
    : M(M), N(N), S(S)
{
    if (custom_board)
        board = *custom_board;
    else
        board = create_board();
}

Board FormerGame::create_board() {
    Board b(M, vector<int>(N, 0));
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, S - 1);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            b[i][j] = dis(gen);
        }
    }
    return b;
}

Board FormerGame::get_board() const {
    return board;
}

set<Point> FormerGame::find_group(const Point & point) const {
    return FormerGame::find_group_static(board, point);
}

vector<set<Point>> FormerGame::get_groups() const {
    return FormerGame::get_groups_static(board);
}

vector<Point> FormerGame::get_valid_turns() const {
    return FormerGame::get_valid_turns_static(board);
}

void FormerGame::remove_group(const set<Point> & group) {
    for (const auto &p : group) {
        int x = p.first, y = p.second;
        board[x][y] = -1;
    }
}

void FormerGame::drop_board() {
    for (int col = 0; col < N; col++) {
        vector<int> column_data;
        for (int row = 0; row < M; row++) {
            if (board[row][col] != -1)
                column_data.push_back(board[row][col]);
        }
        int num_empty = M - column_data.size();
        for (int row = 0; row < num_empty; row++)
            board[row][col] = -1;
        for (int row = num_empty; row < M; row++)
            board[row][col] = column_data[row - num_empty];
    }
}

void FormerGame::make_turn(const Point & point) {
    set<Point> group = find_group(point);
    remove_group(group);
    drop_board();
}

bool FormerGame::is_game_over() const {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            if (board[i][j] != -1)
                return false;
    return true;
}

// ------------------- Static Methods -------------------

set<Point> FormerGame::find_group_static(const Board & board, const Point & point) {
    set<Point> group;
    queue<Point> q;
    q.push(point);
    int x = point.first, y = point.second;
    int target_value = board[x][y];
    int rows = board.size();
    int cols = board[0].size();
    while (!q.empty()) {
        Point p = q.front();
        q.pop();
        if (group.find(p) != group.end())
            continue;
        group.insert(p);
        int cx = p.first, cy = p.second;
        vector<Point> neighbors = { {cx-1, cy}, {cx+1, cy}, {cx, cy-1}, {cx, cy+1} };
        for (const auto &np : neighbors) {
            int nx = np.first, ny = np.second;
            if (nx >= 0 && nx < rows && ny >= 0 && ny < cols) {
                if (board[nx][ny] == target_value && group.find({nx, ny}) == group.end()) {
                    q.push({nx, ny});
                }
            }
        }
    }
    return group;
}

vector<set<Point>> FormerGame::get_groups_static(const Board & board) {
    int rows = board.size();
    int cols = board[0].size();
    vector<set<Point>> groups;
    vector<vector<bool>> visited(rows, vector<bool>(cols, false));
    for (int x = 0; x < rows; x++) {
        for (int y = 0; y < cols; y++) {
            if (board[x][y] != -1 && !visited[x][y]) {
                set<Point> group = find_group_static(board, {x, y});
                groups.push_back(group);
                for (const auto &p : group)
                    visited[p.first][p.second] = true;
            }
        }
    }
    return groups;
}

vector<Point> FormerGame::get_valid_turns_static(const Board & board) {
    int rows = board.size();
    int cols = board[0].size();
    vector<Point> valid_turns;
    vector<vector<bool>> visited(rows, vector<bool>(cols, false));
    for (int x = 0; x < rows; x++) {
        for (int y = 0; y < cols; y++) {
            if (board[x][y] != -1 && !visited[x][y]) {
                set<Point> group = find_group_static(board, {x, y});
                valid_turns.push_back({x, y});
                for (const auto &p : group)
                    visited[p.first][p.second] = true;
            }
        }
    }
    return valid_turns;
}

Board FormerGame::remove_group_static(const Board & board, const set<Point> & group) {
    Board new_board = board; // copy board
    for (const auto &p : group)
        new_board[p.first][p.second] = -1;
    return new_board;
}

Board FormerGame::drop_board_static(const Board & board) {
    int rows = board.size();
    int cols = board[0].size();
    Board new_board = board; // copy board
    for (int col = 0; col < cols; col++) {
        vector<int> column_data;
        for (int row = 0; row < rows; row++) {
            if (new_board[row][col] != -1)
                column_data.push_back(new_board[row][col]);
        }
        int num_empty = rows - column_data.size();
        for (int row = 0; row < num_empty; row++)
            new_board[row][col] = -1;
        for (int row = num_empty; row < rows; row++)
            new_board[row][col] = column_data[row - num_empty];
    }
    return new_board;
}

Board FormerGame::apply_turn_static(const Board & board, const Point & point) {
    set<Point> group = find_group_static(board, point);
    Board board_after_removal = remove_group_static(board, group);
    Board new_board = drop_board_static(board_after_removal);
    return new_board;
}

vector<double> FormerGame::get_heuristic_probs_static(const Board & board, const vector<Point> & valid_turns) {
    return heuristic_min_groups_1_look_ahead(board, valid_turns);
    //return heuristic_min_groups_2_look_ahead_softmax(board, valid_turns);
}

bool FormerGame::is_game_over_static(const Board & board) {
    int rows = board.size();
    int cols = board[0].size();
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            if (board[i][j] != -1)
                return false;
    return true;
}

// ------------------- Heuristics -------------------

vector<double> heuristic_min_groups_1_look_ahead(
    const Board & board,
    const vector<Point> & valid_turns)
{
    size_t n = valid_turns.size();
    vector<double> heuristic_values(n);

    // 1) look one step ahead, take reciprocal of groups_count
    for (size_t i = 0; i < n; i++) {
        Board new_board = FormerGame::apply_turn_static(board, valid_turns[i]);
        int groups_count =
            (int)FormerGame::get_groups_static(new_board).size();
        if (groups_count > 0) {heuristic_values[i] = 1.0 / double(groups_count);
        } else {
              // no groups → pick a default (e.g. treat as “one group”)
          heuristic_values[i] = 1.0;
        }
    }

    // 2) normalize
    double sum = accumulate(heuristic_values.begin(),
                            heuristic_values.end(), 0.0);
    for (double &v : heuristic_values) v /= sum;

    return heuristic_values;
}

vector<double> heuristic_min_groups_2_look_ahead(
    const Board & board,
    const vector<Point> & valid_turns)
{
    size_t n = valid_turns.size();
    vector<double> rec1(n), rec2(n), score(n);

    // 1) look one step ahead → rec1
    for (size_t i = 0; i < n; ++i) {
        Board b1 = FormerGame::apply_turn_static(board, valid_turns[i]);
        int g1 = (int)FormerGame::get_groups_static(b1).size();
        // always safe since g1 >= 1
        rec1[i] = 1.0 / double(g1);

        // 2) look a second step → rec2
        if (g1 <= 1) {
            // already solved → same as rec1
            rec2[i] = rec1[i];
        } else {
            auto vt2 = FormerGame::get_valid_turns_static(b1);
            if (vt2.empty()) {
                // no further moves → fall back to rec1
                rec2[i] = rec1[i];
            } else {
                double min_g2 = DBL_MAX;
                for (const auto &pt2 : vt2) {
                    Board b2 = FormerGame::apply_turn_static(b1, pt2);
                    int g2 = (int)FormerGame::get_groups_static(b2).size();
                    min_g2 = std::min(min_g2, double(g2));
                }
                rec2[i] = 1.0 / min_g2;
            }
        }
    }

    // 3) combine with your weights
    for (size_t i = 0; i < n; ++i) {
        score[i] = 0.1 * rec1[i] + 0.9 * rec2[i];
    }

    // 4) normalize to get a probability distribution
    double sum = accumulate(score.begin(), score.end(), 0.0);
    if (sum > 0.0) {
        for (double &v : score) v /= sum;
    } else {
        // (should never happen, but just in case)
        for (double &v : score) v = 1.0 / n;
    }

    return score;
}



vector<double> heuristic_min_groups_3_look_ahead(
    const Board & board,
    const vector<Point> & valid_turns)
{
    size_t n = valid_turns.size();
    vector<double> rec1(n), rec2(n), rec3(n), score(n);

    // 1) First‐step reciprocal
    for (size_t i = 0; i < n; ++i) {
        Board b1 = FormerGame::apply_turn_static(board, valid_turns[i]);
        int g1 = (int)FormerGame::get_groups_static(b1).size();
        rec1[i] = (g1 > 0 ? 1.0 / double(g1) : 1.0);

        // get all second‐moves
        auto vt2 = FormerGame::get_valid_turns_static(b1);
        if (g1 <= 1 || vt2.empty()) {
            // already solved or no second moves
            rec2[i] = rec1[i];
            rec3[i] = rec1[i];
        } else {
            // 2) Two‐step lookahead: find min group count after step 2
            double min_g2 = DBL_MAX;
            // we'll also keep the boards that achieved that min
            vector<Board> best_b2;
            for (auto &pt2 : vt2) {
                Board b2 = FormerGame::apply_turn_static(b1, pt2);
                int g2 = (int)FormerGame::get_groups_static(b2).size();
                double dg2 = double(max(g2,1));
                if (dg2 < min_g2) {
                    min_g2 = dg2;
                    best_b2.clear();
                    best_b2.push_back(b2);
                } else if (dg2 == min_g2) {
                    best_b2.push_back(b2);
                }
            }
            rec2[i] = 1.0 / min_g2;

            // 3) Three‐step lookahead: from each best b2, try all third‐moves
            if (min_g2 <= 1) {
                // solved at depth 2 ⇒ no further refinement
                rec3[i] = rec2[i];
            } else {
                double min_g3 = DBL_MAX;
                for (auto &b2 : best_b2) {
                    auto vt3 = FormerGame::get_valid_turns_static(b2);
                    if (vt3.empty()) {
                        // no third‐moves ⇒ fallback
                        min_g3 = std::min(min_g3, min_g2);
                    } else {
                        for (auto &pt3 : vt3) {
                            Board b3 = FormerGame::apply_turn_static(b2, pt3);
                            int g3 = (int)FormerGame::get_groups_static(b3).size();
                            min_g3 = std::min(min_g3, double(max(g3,1)));
                        }
                    }
                }
                rec3[i] = 1.0 / min_g3;
            }
        }
    }

    // 4) Combine with weights
    constexpr double w1 = 0.05, w2 = 0.15, w3 = 0.80;
    for (size_t i = 0; i < n; ++i) {
        score[i] = w1*rec1[i] + w2*rec2[i] + w3*rec3[i];
    }

    // 5) Normalize to a probability distribution
    double sum = std::accumulate(score.begin(), score.end(), 0.0);
    if (sum > 0.0) {
        for (double &v : score) v /= sum;
    } else {
        // (shouldn't really happen) fall back to uniform
        for (double &v : score) v = 1.0 / n;
    }

    return score;
}



vector<double> softmax(const vector<double>& scores, double temperature) {
    vector<double> exp_values(scores.size());
    // We use the negative of the raw score so that lower raw scores (better moves) yield higher probabilities.
    for (size_t i = 0; i < scores.size(); i++) {
        exp_values[i] = exp(-scores[i] / temperature);
    }
    double sum_exp = accumulate(exp_values.begin(), exp_values.end(), 0.0);
    vector<double> probs(scores.size());
    for (size_t i = 0; i < scores.size(); i++) {
        probs[i] = exp_values[i] / sum_exp;
    }
    return probs;
}

vector<double> heuristic_min_groups_2_look_ahead_softmax(const Board & board, const vector<Point> & valid_turns, double temperature) {
    size_t n = valid_turns.size();
    vector<double> raw_scores(n, 0.0);
    
    // For each move, compute the raw score based on the number of groups two steps ahead.
    for (size_t i = 0; i < n; i++) {
        Board new_board = FormerGame::apply_turn_static(board, valid_turns[i]);
        // Evaluate groups after one move.
        vector<set<Point>> groups_after_one = FormerGame::get_groups_static(new_board);
        int groups1 = groups_after_one.size();
        
        if (groups1 == 1) {
            // If the move clears the board or leaves only one group, we set the score to 0.
            raw_scores[i] = 0.0;
        } else {
            // Evaluate two moves ahead.
            vector<Point> valid_turns_second = FormerGame::get_valid_turns_static(new_board);
            if (valid_turns_second.empty()) {
                raw_scores[i] = 0.0;
            } else {
                vector<double> second_move_scores;
                for (const auto &pt2 : valid_turns_second) {
                    Board new_board2 = FormerGame::apply_turn_static(new_board, pt2);
                    int groups2 = FormerGame::get_groups_static(new_board2).size();
                    second_move_scores.push_back(groups2);
                }
                double min_groups_after_two = *min_element(second_move_scores.begin(), second_move_scores.end());
                raw_scores[i] = min_groups_after_two;
            }
        }
    }
    
    // Directly apply softmax to the raw scores.
    vector<double> probabilities = softmax(raw_scores, temperature);
    return probabilities;
}