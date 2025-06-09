#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 
#include <pybind11/numpy.h> 
#include "former_class.h"

namespace py = pybind11;

// Helper function to convert a Board (vector<vector<int>>) to a numpy array.
py::array_t<int> get_board_numpy(const FormerGame & self) {
    Board b = self.get_board(); // Get a copy of the board from C++.
    size_t M = b.size();
    size_t N = (M > 0) ? b[0].size() : 0;
    // Create a NumPy array with shape (M, N)
    auto result = py::array_t<int>({M, N});
    py::buffer_info buf = result.request();
    int* ptr = static_cast<int*>(buf.ptr);
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            ptr[i * N + j] = b[i][j];
        }
    }
    return result;
}

PYBIND11_MODULE(former_class_cpp, m) {
    m.doc() = "Python wrapper for the FormerGame C++ class using pybind11";

    py::class_<FormerGame>(m, "FormerGame")
        .def(py::init([](int M, int N, int S, py::object custom_board) {
            // If custom_board is None, pass nullptr; otherwise, convert to C++ Board.
            if (custom_board.is_none()) {
                return new FormerGame(M, N, S, nullptr);
            } else {
                Board board = custom_board.cast<Board>();
                return new FormerGame(M, N, S, &board);
            }
        }),
        py::arg("M") = 9, py::arg("N") = 7, py::arg("S") = 4, py::arg("custom_board") = py::none())
        .def_readwrite("board", &FormerGame::board, "The game board")
        .def("create_board", &FormerGame::create_board, "Create a new game board")
        .def("get_board", &get_board_numpy, "Return a copy of the board as a numpy array")
        .def("make_turn", &FormerGame::make_turn, "Make a move on the board", py::arg("point"))
        .def("get_valid_turns", &FormerGame::get_valid_turns, "Get valid turns")
        .def("get_groups", &FormerGame::get_groups, "Return all groups on the board")
        .def("drop_board", &FormerGame::drop_board, "Apply gravity to the board")
        .def("is_game_over", &FormerGame::is_game_over, "Check if the game is over")
        
        // Expose static methods - we typically use these to avoid making instances of the game
        .def_static("find_group_static", &FormerGame::find_group_static)
        .def_static("get_groups_static", &FormerGame::get_groups_static)
        .def_static("get_valid_turns_static", &FormerGame::get_valid_turns_static)
        .def_static("remove_group_static", &FormerGame::remove_group_static)
        .def_static("drop_board_static", &FormerGame::drop_board_static)
        .def_static("apply_turn_static", &FormerGame::apply_turn_static)
        .def_static("get_heuristic_probs_static", &FormerGame::get_heuristic_probs_static)
        .def_static("is_game_over_static", &FormerGame::is_game_over_static)
        ;

    // Heuristics implemented in C++
    m.def(
      "heuristic_min_groups_1_look_ahead",
      &heuristic_min_groups_1_look_ahead,
      py::arg("board"),
      py::arg("valid_turns"),
      "Compute 1-step lookahead group-minimizing heuristic probabilities"
    );
  
    m.def(
      "heuristic_min_groups_2_look_ahead",
      &heuristic_min_groups_2_look_ahead,
      py::arg("board"),
      py::arg("valid_turns"),
      "Compute 2-step lookahead group-minimizing heuristic probabilities"
    );

    m.def(
      "heuristic_min_groups_3_look_ahead",
      &heuristic_min_groups_3_look_ahead,
      py::arg("board"),
      py::arg("valid_turns"),
      "Compute 3-step lookahead group-minimizing heuristic probabilities"
    );

    m.def(
      "softmax",
      &softmax,
      py::arg("scores"),
      py::arg("temperature") = 1.0,
      "Convert raw scores to probabilities with given temperature"
    );

    m.def(
      "heuristic_min_groups_2_look_ahead_softmax",
      &heuristic_min_groups_2_look_ahead_softmax,
      py::arg("board"),
      py::arg("valid_turns"),
      py::arg("temperature") = 1.0,
      "Compute 2-step lookahead heuristic probabilities via softmax"
    );
  
}
