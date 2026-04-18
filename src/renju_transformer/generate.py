"""Synthetic data generation pipeline for Renju next-move training."""

from __future__ import annotations

import multiprocessing
import math
import random
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import DictConfig

from .rules import (
    BLACK,
    BOARD_CELLS,
    BOARD_SIZE,
    CENTER_INDEX,
    DIRECTIONS,
    EMPTY,
    WHITE,
    board_winner,
    board_with_move,
    contiguous_count,
    idx_to_rc,
    infer_player,
    inside,
    is_forbidden_for_black,
    move_number as current_move_number,
    rc_to_idx,
    stone_counts,
    winner_after_move,
)

DRAW = 0
LIGHT_POLICY = "light"
MCTS_POLICY = "mcts"


@dataclass(slots=True)
class AugmentationConfig:
    enabled: bool = True
    move_ratio: float = 0.10
    add_stones_per_color: int = 1
    copies_per_sample: int = 1


@dataclass(slots=True)
class HighQualityConfig:
    enabled: bool = False
    games: int = 0
    simulations: int = 16
    random_interval: int = 0


@dataclass(slots=True)
class ParallelConfig:
    enabled: bool = False
    workers: int = 1
    keep_shards: bool = False


@dataclass(slots=True)
class GenerationConfig:
    games: int
    output_path: Path
    seed: int | None = None
    sep_token_id: int = 228
    move_id_offset: int = 3
    candidate_limit: int = 16
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    high_quality: HighQualityConfig = field(default_factory=HighQualityConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)


def move_to_id(index: int, move_id_offset: int) -> int:
    return index + move_id_offset


def other_player(player: int) -> int:
    return WHITE if player == BLACK else BLACK


def occupied_indexes(board: list[int]) -> list[int]:
    return [index for index, value in enumerate(board) if value != EMPTY]


def board_is_full(board: list[int]) -> bool:
    return all(cell != EMPTY for cell in board)


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


def ordered_candidates(board: list[int], limit: int) -> list[int]:
    candidates = list(neighbor_candidates(board))
    candidates.sort(key=lambda move: (-local_density(board, move), center_distance_sq(move), move))
    return candidates[:limit]


def generate_legal_moves(board: list[int], player: int, candidate_limit: int = 16) -> list[int]:
    candidates = ordered_candidates(board, candidate_limit)
    legal_moves: list[int] = []

    for move in candidates:
        if board[move] != EMPTY:
            continue
        if player == BLACK and is_forbidden_for_black(board, move):
            continue
        legal_moves.append(move)

    if legal_moves:
        return legal_moves

    if board_is_full(board):
        return []

    fallback = [index for index, cell in enumerate(board) if cell == EMPTY]
    if player == BLACK:
        fallback = [move for move in fallback if not is_forbidden_for_black(board, move)]
    return fallback


def immediate_winning_moves(
    board: list[int],
    player: int,
    legal_moves: list[int],
) -> list[int]:
    winning_moves: list[int] = []
    for move in legal_moves:
        next_board = board_with_move(board, move, player)
        if winner_after_move(next_board, move, player) == player:
            winning_moves.append(move)
    return winning_moves


def move_shape_score(board: list[int], move: int, player: int) -> float:
    next_board = board_with_move(board, move, player)
    lengths = [contiguous_count(next_board, move, player, dr, dc) for dr, dc in DIRECTIONS]
    longest = max(lengths)
    pressure = sum(length * length for length in lengths)
    score = float(longest * 100 + pressure * 10 + local_density(board, move) * 4)
    score -= center_distance_sq(move) * 0.25
    return score


def choose_tactical_move(
    board: list[int],
    player: int,
    legal_moves: list[int],
    rng: random.Random,
    candidate_limit: int,
) -> int:
    winning_moves = immediate_winning_moves(board, player, legal_moves)
    if winning_moves:
        return min(winning_moves, key=lambda move: (center_distance_sq(move), move))

    opponent = other_player(player)
    opponent_legal_moves = generate_legal_moves(board, opponent, candidate_limit)
    opponent_wins = [move for move in immediate_winning_moves(board, opponent, opponent_legal_moves) if move in legal_moves]
    if opponent_wins:
        return min(opponent_wins, key=lambda move: (center_distance_sq(move), move))

    scored_moves = [(move_shape_score(board, move, player), move) for move in legal_moves]
    scored_moves.sort(key=lambda item: (-item[0], center_distance_sq(item[1]), item[1]))

    top_k = min(4, len(scored_moves))
    top_moves = [move for _, move in scored_moves[:top_k]]
    weights = [top_k - index for index in range(top_k)]
    return rng.choices(top_moves, weights=weights, k=1)[0]


def non_terminal_legal_moves(board: list[int], player: int, candidate_limit: int = 16) -> list[int]:
    legal_moves = generate_legal_moves(board, player, candidate_limit)
    candidates: list[int] = []
    for move in legal_moves:
        next_board = board_with_move(board, move, player)
        if winner_after_move(next_board, move, player) is None:
            candidates.append(move)
    return candidates


def board_is_playable_snapshot(board: list[int], candidate_limit: int = 16) -> bool:
    black_count, white_count = stone_counts(board)
    if black_count < white_count or black_count > white_count + 1:
        return False
    if black_count > 0 and board[CENTER_INDEX] != BLACK:
        return False
    if board_winner(board) is not None:
        return False

    try:
        player = infer_player(board)
    except ValueError:
        return False

    return bool(generate_legal_moves(board, player, candidate_limit))


def move_stones(board: list[int], move_ratio: float, rng: random.Random, candidate_limit: int) -> list[int]:
    augmented = board.copy()
    stones = occupied_indexes(augmented)
    target_moves = int(len(stones) * move_ratio)
    if target_moves <= 0:
        return augmented

    rng.shuffle(stones)
    moved = 0
    for source in stones:
        color = augmented[source]
        if color == EMPTY:
            continue
        if color == BLACK and source == CENTER_INDEX:
            continue

        augmented[source] = EMPTY
        destinations = [index for index, cell in enumerate(augmented) if cell == EMPTY]
        rng.shuffle(destinations)
        placed = False

        for destination in destinations:
            candidate_board = board_with_move(augmented, destination, color)
            if board_is_playable_snapshot(candidate_board, candidate_limit):
                augmented = candidate_board
                moved += 1
                placed = True
                break

        if not placed:
            augmented[source] = color

        if moved >= target_moves:
            break

    return augmented


def add_balanced_stones(board: list[int], stones_per_color: int, rng: random.Random, candidate_limit: int) -> list[int]:
    augmented = board.copy()
    if stones_per_color <= 0:
        return augmented

    remaining = {BLACK: stones_per_color, WHITE: stones_per_color}
    player = infer_player(augmented)

    while remaining[BLACK] > 0 or remaining[WHITE] > 0:
        if remaining[player] <= 0:
            break

        candidates = non_terminal_legal_moves(augmented, player, candidate_limit)
        if not candidates:
            break

        move = rng.choice(candidates)
        augmented[move] = player
        remaining[player] -= 1
        player = other_player(player)

    return augmented


def augment_board(board: list[int], cfg: AugmentationConfig, rng: random.Random, candidate_limit: int) -> list[int] | None:
    if not cfg.enabled:
        return None

    augmented = move_stones(board, cfg.move_ratio, rng, candidate_limit)
    augmented = add_balanced_stones(augmented, cfg.add_stones_per_color, rng, candidate_limit)

    if augmented == board:
        return None
    if not board_is_playable_snapshot(augmented, candidate_limit):
        return None
    return augmented


@dataclass
class MCTSNode:
    board: list[int]
    player_to_move: int
    root_player: int
    candidate_limit: int
    move_played: int | None = None
    parent: MCTSNode | None = None
    terminal_winner: int | None = None
    wins: float = 0.0
    visits: int = 0
    children: list["MCTSNode"] = field(default_factory=list)
    untried_moves: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.terminal_winner is None:
            self.terminal_winner = board_winner(self.board)
        if self.terminal_winner is not None:
            return

        self.untried_moves = generate_legal_moves(self.board, self.player_to_move, self.candidate_limit)
        if self.untried_moves:
            return

        if board_is_full(self.board):
            self.terminal_winner = DRAW
        else:
            self.terminal_winner = other_player(self.player_to_move)

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
            player_to_move=other_player(self.player_to_move),
            root_player=self.root_player,
            candidate_limit=self.candidate_limit,
            move_played=move,
            parent=self,
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


def rollout(board: list[int], player: int, rng: random.Random, candidate_limit: int, max_steps: int = 12) -> int:
    sim_board = board.copy()
    current_player = player

    for _ in range(max_steps):
        legal_moves = generate_legal_moves(sim_board, current_player, candidate_limit)
        if not legal_moves:
            if board_is_full(sim_board):
                return DRAW
            return other_player(current_player)

        move = choose_tactical_move(sim_board, current_player, legal_moves, rng, candidate_limit)
        sim_board[move] = current_player
        winner = winner_after_move(sim_board, move, current_player)
        if winner is not None:
            return winner

        if board_is_full(sim_board):
            return DRAW

        current_player = other_player(current_player)

    return DRAW


def run_mcts(board: list[int], player: int, simulations: int, rng: random.Random, candidate_limit: int) -> int:
    legal_moves = generate_legal_moves(board, player, candidate_limit)
    if not legal_moves:
        raise RuntimeError("No legal moves available.")
    if len(legal_moves) == 1:
        return legal_moves[0]

    root = MCTSNode(
        board=board.copy(),
        player_to_move=player,
        root_player=player,
        candidate_limit=candidate_limit,
    )

    for _ in range(simulations):
        node = root

        while not node.is_terminal() and node.fully_expanded() and node.children:
            node = node.best_child()

        if not node.is_terminal() and node.untried_moves:
            node = node.expand(rng)

        if node.is_terminal():
            winner = node.terminal_winner if node.terminal_winner is not None else DRAW
        else:
            winner = rollout(node.board, node.player_to_move, rng, node.candidate_limit)

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


def select_move(
    board: list[int],
    player: int,
    policy: str,
    rng: random.Random,
    candidate_limit: int,
    mcts_simulations: int = 0,
    random_interval: int = 0,
) -> int:
    legal_moves = generate_legal_moves(board, player, candidate_limit)
    if not legal_moves:
        raise RuntimeError("No legal move to select.")

    if policy == LIGHT_POLICY:
        return choose_tactical_move(board, player, legal_moves, rng, candidate_limit)
    if policy != MCTS_POLICY:
        raise ValueError(f"Unsupported policy: {policy}")

    if random_interval > 0 and (current_move_number(board) + 1) % random_interval == 0:
        return rng.choice(legal_moves)
    return run_mcts(board, player, mcts_simulations, rng, candidate_limit)


def write_training_row(handle, board: list[int], move: int, sep_token_id: int, move_id_offset: int) -> None:
    row = board + [sep_token_id, move_to_id(move, move_id_offset)]
    handle.write(",".join(map(str, row)))
    handle.write("\n")


def emit_augmented_samples(
    handle,
    board: list[int],
    augmentation: AugmentationConfig,
    rng: random.Random,
    sep_token_id: int,
    move_id_offset: int,
    candidate_limit: int,
) -> None:
    if not augmentation.enabled:
        return

    for _ in range(augmentation.copies_per_sample):
        augmented = augment_board(board, augmentation, rng, candidate_limit)
        if augmented is None:
            continue
        player = infer_player(augmented)
        move = select_move(
            augmented,
            player,
            policy=LIGHT_POLICY,
            rng=rng,
            candidate_limit=candidate_limit,
        )
        write_training_row(handle, augmented, move, sep_token_id, move_id_offset)


def play_game(
    handle,
    policy: str,
    rng: random.Random,
    sep_token_id: int,
    move_id_offset: int,
    candidate_limit: int,
    augmentation: AugmentationConfig | None = None,
    mcts_simulations: int = 0,
    random_interval: int = 0,
) -> tuple[int, int, bool]:
    board = [EMPTY] * BOARD_CELLS
    current_player = BLACK
    plies = 0
    foul_loss = False

    while True:
        legal_moves = generate_legal_moves(board, current_player, candidate_limit)
        if not legal_moves:
            return other_player(current_player), plies, foul_loss

        move = select_move(
            board,
            current_player,
            policy=policy,
            rng=rng,
            candidate_limit=candidate_limit,
            mcts_simulations=mcts_simulations,
            random_interval=random_interval,
        )
        write_training_row(handle, board, move, sep_token_id, move_id_offset)

        if augmentation is not None and policy == LIGHT_POLICY:
            emit_augmented_samples(
                handle,
                board,
                augmentation,
                rng,
                sep_token_id,
                move_id_offset,
                candidate_limit,
            )

        board[move] = current_player
        plies += 1

        winner = winner_after_move(board, move, current_player)
        if winner is not None:
            if current_player == BLACK and winner == WHITE:
                foul_loss = True
            return winner, plies, foul_loss

        if board_is_full(board):
            return DRAW, plies, foul_loss

        current_player = other_player(current_player)


def result_label(winner: int, foul_loss: bool) -> str:
    if winner == DRAW:
        return "draw"
    if winner == BLACK:
        return "black"
    if foul_loss:
        return "white(foul)"
    return "white"


def run_game_batch(
    handle,
    games: int,
    policy: str,
    rng: random.Random,
    sep_token_id: int,
    move_id_offset: int,
    candidate_limit: int,
    augmentation: AugmentationConfig | None = None,
    mcts_simulations: int = 0,
    random_interval: int = 0,
    phase_label: str | None = None,
) -> None:
    for game_index in range(1, games + 1):
        winner, plies, foul_loss = play_game(
            handle=handle,
            policy=policy,
            rng=rng,
            sep_token_id=sep_token_id,
            move_id_offset=move_id_offset,
            candidate_limit=candidate_limit,
            augmentation=augmentation,
            mcts_simulations=mcts_simulations,
            random_interval=random_interval,
        )
        prefix = f"phase={phase_label} " if phase_label else ""
        print(f"{prefix}game={game_index} winner={result_label(winner, foul_loss)} plies={plies}")


def validate_generation_config(cfg: GenerationConfig) -> None:
    if cfg.games <= 0:
        raise ValueError("generate.games must be a positive integer.")
    if cfg.candidate_limit <= 0:
        raise ValueError("generate.candidate_limit must be a positive integer.")
    if not 0.0 <= cfg.augmentation.move_ratio <= 1.0:
        raise ValueError("generate.augmentation.move_ratio must be between 0.0 and 1.0.")
    if cfg.augmentation.add_stones_per_color < 0:
        raise ValueError("generate.augmentation.add_stones_per_color must be zero or greater.")
    if cfg.augmentation.copies_per_sample < 0:
        raise ValueError("generate.augmentation.copies_per_sample must be zero or greater.")
    if cfg.high_quality.games < 0:
        raise ValueError("generate.high_quality.games must be zero or greater.")
    if cfg.high_quality.simulations <= 0:
        raise ValueError("generate.high_quality.simulations must be a positive integer.")
    if cfg.high_quality.random_interval < 0:
        raise ValueError("generate.high_quality.random_interval must be zero or greater.")
    if cfg.parallel.workers <= 0:
        raise ValueError("generate.parallel.workers must be a positive integer.")


def run_generation_serial(cfg: GenerationConfig) -> None:
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg.seed)

    with cfg.output_path.open("w", encoding="utf-8", newline="") as handle:
        run_game_batch(
            handle=handle,
            games=cfg.games,
            policy=LIGHT_POLICY,
            rng=rng,
            sep_token_id=cfg.sep_token_id,
            move_id_offset=cfg.move_id_offset,
            candidate_limit=cfg.candidate_limit,
            augmentation=cfg.augmentation,
            phase_label="base",
        )

        if cfg.high_quality.enabled and cfg.high_quality.games > 0:
            run_game_batch(
                handle=handle,
                games=cfg.high_quality.games,
                policy=MCTS_POLICY,
                rng=rng,
                sep_token_id=cfg.sep_token_id,
                move_id_offset=cfg.move_id_offset,
                candidate_limit=cfg.candidate_limit,
                augmentation=None,
                mcts_simulations=cfg.high_quality.simulations,
                random_interval=cfg.high_quality.random_interval,
                phase_label="high_quality",
            )


def split_work(total_games: int, workers: int) -> list[int]:
    base, remainder = divmod(total_games, workers)
    return [base + (1 if worker_index < remainder else 0) for worker_index in range(workers)]


def shard_output_path(output_path: Path, worker_index: int) -> Path:
    shard_dir = output_path.parent / f"{output_path.stem}.shards"
    suffix = output_path.suffix or ".csv"
    return shard_dir / f"part-{worker_index:04d}{suffix}"


def shard_seed(seed: int | None, worker_index: int) -> int | None:
    if seed is None:
        return None
    return seed + worker_index


def build_shard_config(
    cfg: GenerationConfig,
    worker_index: int,
    base_games: int,
    high_quality_games: int,
) -> GenerationConfig:
    return GenerationConfig(
        games=base_games,
        output_path=shard_output_path(cfg.output_path, worker_index),
        seed=shard_seed(cfg.seed, worker_index),
        sep_token_id=cfg.sep_token_id,
        move_id_offset=cfg.move_id_offset,
        candidate_limit=cfg.candidate_limit,
        augmentation=cfg.augmentation,
        high_quality=HighQualityConfig(
            enabled=cfg.high_quality.enabled and high_quality_games > 0,
            games=high_quality_games,
            simulations=cfg.high_quality.simulations,
            random_interval=cfg.high_quality.random_interval,
        ),
        parallel=ParallelConfig(enabled=False, workers=1, keep_shards=cfg.parallel.keep_shards),
    )


def run_generation_shard(cfg: GenerationConfig) -> str:
    run_generation_serial(cfg)
    return str(cfg.output_path)


def merge_shards(output_path: Path, shard_paths: list[Path], keep_shards: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as merged_handle:
        for shard_path in shard_paths:
            with shard_path.open("r", encoding="utf-8", newline="") as shard_handle:
                shutil.copyfileobj(shard_handle, merged_handle)

    if keep_shards:
        return

    for shard_path in shard_paths:
        shard_path.unlink(missing_ok=True)

    shard_dir = shard_output_path(output_path, 0).parent
    try:
        shard_dir.rmdir()
    except OSError:
        pass


def run_generation_parallel(cfg: GenerationConfig) -> None:
    base_work = split_work(cfg.games, cfg.parallel.workers)
    high_quality_work = split_work(cfg.high_quality.games, cfg.parallel.workers)

    shard_configs = [
        build_shard_config(cfg, worker_index, base_games, high_quality_games)
        for worker_index, (base_games, high_quality_games) in enumerate(zip(base_work, high_quality_work))
        if base_games > 0 or high_quality_games > 0
    ]
    if not shard_configs:
        raise ValueError("No shard received any work. Increase generate.games or reduce generate.parallel.workers.")

    shard_dir = shard_output_path(cfg.output_path, 0).parent
    shard_dir.mkdir(parents=True, exist_ok=True)

    max_workers = min(cfg.parallel.workers, len(shard_configs), multiprocessing.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(run_generation_shard, shard_configs))

    shard_paths = [shard_cfg.output_path for shard_cfg in shard_configs]
    merge_shards(cfg.output_path, shard_paths, cfg.parallel.keep_shards)


def run_generation(cfg: GenerationConfig) -> None:
    validate_generation_config(cfg)
    if cfg.parallel.enabled and cfg.parallel.workers > 1:
        run_generation_parallel(cfg)
        return
    run_generation_serial(cfg)


def generation_config_from_hydra(cfg: DictConfig) -> GenerationConfig:
    return GenerationConfig(
        games=int(cfg.generate.games),
        output_path=Path(cfg.generate.output_path),
        seed=cfg.generate.seed if cfg.generate.seed is not None else cfg.seed,
        sep_token_id=int(cfg.data.sep_token_id),
        move_id_offset=int(cfg.data.move_id_offset),
        candidate_limit=int(cfg.generate.candidate_limit),
        augmentation=AugmentationConfig(
            enabled=bool(cfg.generate.augmentation.enabled),
            move_ratio=float(cfg.generate.augmentation.move_ratio),
            add_stones_per_color=int(cfg.generate.augmentation.add_stones_per_color),
            copies_per_sample=int(cfg.generate.augmentation.copies_per_sample),
        ),
        high_quality=HighQualityConfig(
            enabled=bool(cfg.generate.high_quality.enabled),
            games=int(cfg.generate.high_quality.games),
            simulations=int(cfg.generate.high_quality.simulations),
            random_interval=int(cfg.generate.high_quality.random_interval),
        ),
        parallel=ParallelConfig(
            enabled=bool(cfg.generate.parallel.enabled),
            workers=int(cfg.generate.parallel.workers),
            keep_shards=bool(cfg.generate.parallel.keep_shards),
        ),
    )


def generate_data(cfg: DictConfig) -> None:
    run_generation(generation_config_from_hydra(cfg))
