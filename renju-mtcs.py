#!/usr/bin/env python3
"""Batch self-play generator for Renju/Gomoku training data.

The logger writes one CSV row per move:
    [board(225 cells as 0/1/2)] + [SEP token] + [move id 3..227]

Move ids map from board index `i` to `i + 3`.
The fixed SEP token is 228, which is outside the move-id range.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


BOARD_SIZE = 15
BOARD_CELLS = BOARD_SIZE * BOARD_SIZE
CENTER_INDEX = (BOARD_SIZE // 2) * BOARD_SIZE + (BOARD_SIZE // 2)
EMPTY = 0
BLACK = 1
WHITE = 2
SEP_TOKEN = 228
DRAW = 0
DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))


def idx_to_rc(index: int) -> tuple[int, int]:
    return divmod(index, BOARD_SIZE)


def rc_to_idx(row: int, col: int) -> int:
    return row * BOARD_SIZE + col


def inside(row: int, col: int) -> bool:
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


def board_with_move(board: list[int], index: int, player: int) -> list[int]:
    next_board = board.copy()
    next_board[index] = player
    return next_board


def move_to_id(index: int) -> int:
    return index + 3


def line_points_through(index: int, dr: int, dc: int) -> tuple[list[int], int]:
    row, col = idx_to_rc(index)
    move_row, move_col = row, col
    while inside(row - dr, col - dc):
        row -= dr
        col -= dc

    points: list[int] = []
    move_pos = 0
    while inside(row, col):
        if row == move_row and col == move_col:
            move_pos = len(points)
        points.append(rc_to_idx(row, col))
        row += dr
        col += dc
    return points, move_pos


def contiguous_count(board: list[int], index: int, player: int, dr: int, dc: int) -> int:
    total = 1
    row, col = idx_to_rc(index)

    step = 1
    while inside(row + dr * step, col + dc * step):
        if board[rc_to_idx(row + dr * step, col + dc * step)] != player:
            break
        total += 1
        step += 1

    step = 1
    while inside(row - dr * step, col - dc * step):
        if board[rc_to_idx(row - dr * step, col - dc * step)] != player:
            break
        total += 1
        step += 1

    return total


def has_five_or_more(board: list[int], index: int, player: int) -> bool:
    return any(contiguous_count(board, index, player, dr, dc) >= 5 for dr, dc in DIRECTIONS)


def is_overline(board: list[int], index: int, player: int) -> bool:
    return any(contiguous_count(board, index, player, dr, dc) >= 6 for dr, dc in DIRECTIONS)


def immediate_wins_in_direction(board: list[int], player: int, dr: int, dc: int, line_points: list[int]) -> set[int]:
    wins: set[int] = set()
    for candidate in line_points:
        if board[candidate] != EMPTY:
            continue
        next_board = board_with_move(board, candidate, player)
        if player == BLACK and is_overline(next_board, candidate, BLACK):
            continue
        if has_five_or_more(next_board, candidate, player):
            wins.add(candidate)
    return wins


def count_four_directions(board: list[int], move: int, player: int) -> int:
    count = 0
    for dr, dc in DIRECTIONS:
        line_points, _ = line_points_through(move, dr, dc)
        if immediate_wins_in_direction(board, player, dr, dc, line_points):
            count += 1
    return count


def count_open_three_directions(board: list[int], move: int, player: int) -> int:
    count = 0
    for dr, dc in DIRECTIONS:
        line_points, _ = line_points_through(move, dr, dc)
        found_open_three = False
        for candidate in line_points:
            if board[candidate] != EMPTY:
                continue
            next_board = board_with_move(board, candidate, player)
            if player == BLACK and is_overline(next_board, candidate, BLACK):
                continue
            winning_points = immediate_wins_in_direction(next_board, player, dr, dc, line_points)
            if len(winning_points) >= 2:
                found_open_three = True
                break
        if found_open_three:
            count += 1
    return count


def is_forbidden_for_black(board: list[int], index: int, move_number: int) -> bool:
    if board[index] != EMPTY:
        return True
    if move_number == 0:
        return index != CENTER_INDEX

    next_board = board_with_move(board, index, BLACK)
    if is_overline(next_board, index, BLACK):
        return True
    if count_four_directions(next_board, index, BLACK) >= 2:
        return True
    if count_open_three_directions(next_board, index, BLACK) >= 2:
        return True
    return False


def winner_after_move(board: list[int], index: int, player: int) -> int | None:
    if player == BLACK and is_overline(board, index, BLACK):
        return WHITE
    if has_five_or_more(board, index, player):
        return player
    return None


def occupied_indexes(board: list[int]) -> list[int]:
    return [i for i, value in enumerate(board) if value != EMPTY]


def neighbor_candidates(board: list[int], radius: int = 1) -> set[int]:
    stones = occupied_indexes(board)
    if not stones:
        return {CENTER_INDEX}

    candidates: set[int] = set()
    for index in stones:
        row, col = idx_to_rc(index)
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr = row + dr
                nc = col + dc
                if inside(nr, nc):
                    candidate = rc_to_idx(nr, nc)
                    if board[candidate] == EMPTY:
                        candidates.add(candidate)
    return candidates


def center_distance_sq(index: int) -> int:
    row, col = idx_to_rc(index)
    center = BOARD_SIZE // 2
    return (row - center) ** 2 + (col - center) ** 2


def local_density(board: list[int], index: int) -> int:
    row, col = idx_to_rc(index)
    score = 0
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            if dr == 0 and dc == 0:
                continue
            nr = row + dr
            nc = col + dc
            if inside(nr, nc) and board[rc_to_idx(nr, nc)] != EMPTY:
                score += 1
    return score


def ordered_candidates(board: list[int], limit: int = 12) -> list[int]:
    candidates = list(neighbor_candidates(board))
    candidates.sort(key=lambda move: (-local_density(board, move), center_distance_sq(move)))
    return candidates[:limit]


def generate_legal_moves(board: list[int], player: int, move_number: int) -> list[int]:
    candidates = ordered_candidates(board)
    legal_moves: list[int] = []

    for move in candidates:
        if player == BLACK:
            if not is_forbidden_for_black(board, move, move_number):
                legal_moves.append(move)
        elif board[move] == EMPTY:
            legal_moves.append(move)

    if legal_moves:
        return legal_moves

    if all(cell != EMPTY for cell in board):
        return []

    fallback = [i for i, cell in enumerate(board) if cell == EMPTY]
    if player == BLACK:
        fallback = [move for move in fallback if not is_forbidden_for_black(board, move, move_number)]
    return fallback


def immediate_winning_moves(board: list[int], player: int, legal_moves: Iterable[int]) -> list[int]:
    winning_moves: list[int] = []
    for move in legal_moves:
        next_board = board_with_move(board, move, player)
        if winner_after_move(next_board, move, player) == player:
            winning_moves.append(move)
    return winning_moves


def weighted_random_move(legal_moves: list[int], rng: random.Random) -> int:
    weights = [1.0 / (1.0 + center_distance_sq(move)) for move in legal_moves]
    return rng.choices(legal_moves, weights=weights, k=1)[0]


def choose_tactical_move(board: list[int], player: int, move_number: int, legal_moves: list[int], rng: random.Random) -> int:
    winning_moves = immediate_winning_moves(board, player, legal_moves)
    if winning_moves:
        return min(winning_moves, key=center_distance_sq)

    opponent = WHITE if player == BLACK else BLACK
    opponent_legal_moves = generate_legal_moves(board, opponent, move_number)
    opponent_wins = [move for move in immediate_winning_moves(board, opponent, opponent_legal_moves) if move in legal_moves]
    if opponent_wins:
        return min(opponent_wins, key=center_distance_sq)

    return weighted_random_move(legal_moves, rng)


@dataclass
class MCTSNode:
    board: list[int]
    player_to_move: int
    move_number: int
    move_played: int | None = None
    parent: MCTSNode | None = None
    root_player: int = BLACK
    wins: float = 0.0
    visits: int = 0
    children: list["MCTSNode"] = field(default_factory=list)
    untried_moves: list[int] = field(default_factory=list)
    terminal_winner: int | None = None

    def __post_init__(self) -> None:
        if self.terminal_winner is None:
            self.untried_moves = generate_legal_moves(self.board, self.player_to_move, self.move_number)
            if not self.untried_moves and all(cell != EMPTY for cell in self.board):
                self.terminal_winner = DRAW
            elif not self.untried_moves:
                self.terminal_winner = WHITE if self.player_to_move == BLACK else BLACK

    def is_terminal(self) -> bool:
        return self.terminal_winner is not None

    def fully_expanded(self) -> bool:
        return not self.untried_moves

    def best_child(self, c_param: float = 1.4) -> "MCTSNode":
        log_visits = math.log(self.visits)

        def score(child: MCTSNode) -> float:
            exploitation = child.wins / child.visits
            exploration = c_param * math.sqrt(log_visits / child.visits)
            return exploitation + exploration

        return max(self.children, key=score)

    def expand(self, rng: random.Random) -> "MCTSNode":
        move = self.untried_moves.pop(rng.randrange(len(self.untried_moves)))
        next_board = board_with_move(self.board, move, self.player_to_move)
        winner = winner_after_move(next_board, move, self.player_to_move)
        child = MCTSNode(
            board=next_board,
            player_to_move=WHITE if self.player_to_move == BLACK else BLACK,
            move_number=self.move_number + 1,
            move_played=move,
            parent=self,
            root_player=self.root_player,
            terminal_winner=winner,
        )
        self.children.append(child)
        return child

    def update(self, winner: int) -> None:
        self.visits += 1
        if winner == DRAW:
            self.wins += 0.5
        elif winner == self.root_player:
            self.wins += 1.0


def rollout(board: list[int], player: int, move_number: int, rng: random.Random, max_steps: int = 12) -> int:
    sim_board = board.copy()
    current_player = player
    current_move_number = move_number

    for _ in range(max_steps):
        legal_moves = generate_legal_moves(sim_board, current_player, current_move_number)
        if not legal_moves:
            if all(cell != EMPTY for cell in sim_board):
                return DRAW
            return WHITE if current_player == BLACK else BLACK

        move = choose_tactical_move(sim_board, current_player, current_move_number, legal_moves, rng)
        sim_board[move] = current_player
        winner = winner_after_move(sim_board, move, current_player)
        if winner is not None:
            return winner

        current_player = WHITE if current_player == BLACK else BLACK
        current_move_number += 1

    return DRAW


def run_mcts(board: list[int], player: int, move_number: int, simulations: int, rng: random.Random) -> int:
    legal_moves = generate_legal_moves(board, player, move_number)
    if not legal_moves:
        raise RuntimeError("No legal moves available.")
    if len(legal_moves) == 1:
        return legal_moves[0]

    root = MCTSNode(board=board.copy(), player_to_move=player, move_number=move_number, root_player=player)

    for _ in range(simulations):
        node = root

        while not node.is_terminal() and node.fully_expanded() and node.children:
            node = node.best_child()

        if not node.is_terminal() and node.untried_moves:
            node = node.expand(rng)

        if node.is_terminal():
            winner = node.terminal_winner if node.terminal_winner is not None else DRAW
        else:
            winner = rollout(node.board, node.player_to_move, node.move_number, rng)

        while node is not None:
            node.update(winner)
            node = node.parent

    best = max(
        root.children,
        key=lambda child: (
            child.visits,
            child.wins / child.visits if child.visits else -1.0,
            -center_distance_sq(child.move_played if child.move_played is not None else CENTER_INDEX),
        ),
    )
    if best.move_played is None:
        raise RuntimeError("MCTS failed to select a move.")
    return best.move_played


def write_training_row(handle, board: list[int], move: int) -> None:
    row = board + [SEP_TOKEN, move_to_id(move)]
    handle.write(",".join(map(str, row)))
    handle.write("\n")


def select_move(
    board: list[int],
    player: int,
    move_number: int,
    simulations: int,
    random_interval: int,
    rng: random.Random,
) -> int:
    legal_moves = generate_legal_moves(board, player, move_number)
    if not legal_moves:
        raise RuntimeError("No legal move to select.")

    if random_interval > 0 and (move_number + 1) % random_interval == 0:
        return rng.choice(legal_moves)

    return run_mcts(board, player, move_number, simulations, rng)


def play_game(
    simulations: int,
    random_interval: int,
    rng: random.Random,
    log_handle,
) -> tuple[int, int, bool]:
    board = [EMPTY] * BOARD_CELLS
    current_player = BLACK
    move_number = 0
    foul_loss = False

    while True:
        legal_moves = generate_legal_moves(board, current_player, move_number)
        if not legal_moves:
            winner = WHITE if current_player == BLACK else BLACK
            return winner, move_number, foul_loss

        move = select_move(board, current_player, move_number, simulations, random_interval, rng)
        write_training_row(log_handle, board, move)
        board[move] = current_player

        winner = winner_after_move(board, move, current_player)
        if winner is not None:
            if current_player == BLACK and winner == WHITE:
                foul_loss = True
            log_handle.write("\n")
            return winner, move_number + 1, foul_loss

        if all(cell != EMPTY for cell in board):
            log_handle.write("\n")
            return DRAW, move_number + 1, foul_loss

        current_player = WHITE if current_player == BLACK else BLACK
        move_number += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch self-play Renju generator with MCTS.")
    parser.add_argument("games", type=int, help="Number of games to generate.")
    parser.add_argument("logfile", type=Path, help="CSV log output path.")
    parser.add_argument(
        "--simulations",
        type=int,
        default=4,
        help="MCTS simulations per non-random move. Default: 4",
    )
    parser.add_argument(
        "--random-interval",
        type=int,
        default=7,
        help="Pick a random legal move every M plies. Set 0 to disable. Default: 7",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed.",
    )
    return parser.parse_args()


def result_label(winner: int, foul_loss: bool) -> str:
    if winner == DRAW:
        return "draw"
    if winner == BLACK:
        return "black"
    if foul_loss:
        return "white(foul)"
    return "white"


def main() -> None:
    args = parse_args()
    if args.games <= 0:
        raise SystemExit("games must be a positive integer.")
    if args.simulations <= 0:
        raise SystemExit("simulations must be a positive integer.")
    if args.random_interval < 0:
        raise SystemExit("random-interval must be zero or greater.")

    args.logfile.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    with args.logfile.open("w", encoding="utf-8", newline="") as handle:
        for game_index in range(1, args.games + 1):
            winner, plies, foul_loss = play_game(
                simulations=args.simulations,
                random_interval=args.random_interval,
                rng=rng,
                log_handle=handle,
            )
            print(f"game={game_index} winner={result_label(winner, foul_loss)} plies={plies}")


if __name__ == "__main__":
    main()
