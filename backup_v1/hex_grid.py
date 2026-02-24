# -*- coding: utf-8 -*-
"""
交错六边形网格（8-9-8-9-8-9-8-9）最优步数搜索。
坐标系：Even-row Offset。偶数行 8 格，奇数行 9 格。

算法：Beam Search（束搜索）
- 每层保留最优的 BEAM_WIDTH 个状态
- 40 层深度在秒级完成
- 多轮搜索 + 不同评分策略取最优
"""

from __future__ import annotations

import os
import random
import sys
import time
from collections import deque
from typing import Iterator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── 棋盘常量（启动时预计算） ─────────────────────────────────
ROWS_LAYOUT = [8, 9, 8, 9, 8, 9, 8, 9]
NUM_ROWS = 8
TOTAL_CELLS = sum(ROWS_LAYOUT)  # 68

_rc_to_idx: dict[tuple[int, int], int] = {}
_idx_to_rc: list[tuple[int, int]] = []
_idx = 0
for _r in range(NUM_ROWS):
    for _c in range(ROWS_LAYOUT[_r]):
        _rc_to_idx[(_r, _c)] = _idx
        _idx_to_rc.append((_r, _c))
        _idx += 1

_NEIGHBORS: list[tuple[int, ...]] = [() for _ in range(TOTAL_CELLS)]
for _i in range(TOTAL_CELLS):
    _r, _c = _idx_to_rc[_i]
    if _r % 2 == 0:
        _cands = [(_r-1, _c), (_r-1, _c+1), (_r, _c-1), (_r, _c+1), (_r+1, _c), (_r+1, _c+1)]
    else:
        _cands = [(_r-1, _c-1), (_r-1, _c), (_r, _c-1), (_r, _c+1), (_r+1, _c-1), (_r+1, _c)]
    _nb = []
    for _nr, _nc in _cands:
        if (_nr, _nc) in _rc_to_idx:
            _nb.append(_rc_to_idx[(_nr, _nc)])
    _NEIGHBORS[_i] = tuple(_nb)

BOUNDARY_MASK = 0
_BOUNDARY_SET: frozenset[int]
_bs: set[int] = set()
for _i in range(TOTAL_CELLS):
    _r, _c = _idx_to_rc[_i]
    n = ROWS_LAYOUT[_r]
    if _r == 0 or _r == NUM_ROWS - 1 or _c == n - 1 or _c == 0:
        BOUNDARY_MASK |= (1 << _i)
        _bs.add(_i)
_BOUNDARY_SET = frozenset(_bs)

DEFAULT_BARRIERS: set[tuple[int, int]] = {
    (0, 1), (1, 7), (2, 5), (3, 0), (4, 7),
    (5, 2), (6, 0), (7, 3), (7, 6),
}


# ── 公共工具 ──────────────────────────────────────────────
def barriers_to_mask(cells: set[tuple[int, int]]) -> int:
    m = 0
    for r, c in cells:
        m |= 1 << _rc_to_idx[(r, c)]
    return m

def is_boundary_idx(i: int) -> bool:
    return i in _BOUNDARY_SET

def cell_count() -> int:
    return TOTAL_CELLS

def is_valid(r: int, c: int) -> bool:
    return (r, c) in _rc_to_idx

def is_boundary(r: int, c: int) -> bool:
    return (r, c) in _rc_to_idx and _rc_to_idx[(r, c)] in _BOUNDARY_SET

def get_neighbors(r: int, c: int) -> Iterator[tuple[int, int]]:
    if (r, c) not in _rc_to_idx:
        return
    for ni in _NEIGHBORS[_rc_to_idx[(r, c)]]:
        yield _idx_to_rc[ni]

def all_cells() -> Iterator[tuple[int, int]]:
    for r in range(NUM_ROWS):
        for c in range(ROWS_LAYOUT[r]):
            yield (r, c)


# ── BFS ──────────────────────────────────────────────────
def bfs_distance(start: int, bmask: int) -> int:
    if bmask & (1 << start):
        return -1
    if start in _BOUNDARY_SET:
        return 0
    visited = bmask | (1 << start)
    q = deque(((start, 0),))
    while q:
        u, d = q.popleft()
        for v in _NEIGHBORS[u]:
            if visited & (1 << v):
                continue
            if v in _BOUNDARY_SET:
                return d + 1
            visited |= (1 << v)
            q.append((v, d + 1))
    return -1

def bfs_path(start: int, bmask: int) -> list[int]:
    if bmask & (1 << start):
        return []
    if start in _BOUNDARY_SET:
        return [start]
    visited = bmask | (1 << start)
    prev: dict[int, int] = {}
    q = deque((start,))
    target = -1
    while q:
        u = q.popleft()
        for v in _NEIGHBORS[u]:
            if visited & (1 << v):
                continue
            prev[v] = u
            if v in _BOUNDARY_SET:
                target = v
                break
            visited |= (1 << v)
            q.append(v)
        if target >= 0:
            break
    if target < 0:
        return []
    path: list[int] = []
    cur = target
    while cur != start:
        path.append(cur)
        cur = prev[cur]
    path.append(start)
    path.reverse()
    return path

def get_monster_next_step(
    monster_pos: tuple[int, int],
    barriers: set[tuple[int, int]],
) -> tuple[int, int] | None:
    idx = _rc_to_idx.get(monster_pos)
    if idx is None:
        return None
    path = bfs_path(idx, barriers_to_mask(barriers))
    if len(path) <= 1:
        return None
    return _idx_to_rc[path[1]]


# ── Beam Search 求解器 ───────────────────────────────────
BEAM_WIDTH = 500
CANDIDATES_PER_STATE = 12

def _open_neighbors(idx: int, bmask: int) -> int:
    """小怪周围未被阻挡的邻居数。"""
    count = 0
    for nb in _NEIGHBORS[idx]:
        if not (bmask & (1 << nb)):
            count += 1
    return count

def _score_state(monster_idx: int, bmask: int) -> float:
    """评估状态：越高 = 小怪越难逃 = 越好。"""
    d = bfs_distance(monster_idx, bmask)
    if d < 0:
        return 99999  # 已困住
    open_nb = _open_neighbors(monster_idx, bmask)
    # 主要看 BFS 距离（越远越好），次要看开放邻居数（越少越好）
    return d * 100 + (6 - open_nb) * 10

def _get_candidates(monster_idx: int, bmask: int) -> list[int]:
    """生成候选放置位并按评分排序，返回 top-K。"""
    path = bfs_path(monster_idx, bmask)
    cands: set[int] = set()
    # 路径上及邻居
    for p in path[1:]:
        cands.add(p)
        for nb in _NEIGHBORS[p]:
            if not (bmask & (1 << nb)):
                cands.add(nb)
    # 小怪邻居
    for nb in _NEIGHBORS[monster_idx]:
        if not (bmask & (1 << nb)):
            cands.add(nb)
    cands.discard(monster_idx)

    # 按放砖后 BFS 距离降序
    scored: list[tuple[float, int]] = []
    for c in cands:
        new_bm = bmask | (1 << c)
        d = bfs_distance(monster_idx, new_bm)
        if d < 0:
            scored.append((99999.0, c))
        else:
            open_nb = _open_neighbors(monster_idx, new_bm)
            scored.append((d * 100.0 + (6 - open_nb) * 10.0, c))
    scored.sort(reverse=True)
    return [c for _, c in scored[:CANDIDATES_PER_STATE]]


def _beam_search_once(
    start_idx: int,
    bar_mask: int,
    max_turns: int,
    beam_width: int,
    noise: float = 0.0,
) -> tuple[int, list[int]]:
    """
    单轮 Beam Search。
    noise > 0 时给评分加随机扰动，用于多轮搜索的多样性。
    返回 (步数, 放置序列)。步数 = -1 表示未能困住。
    """
    # 状态: (monster_idx, bmask, placements_list)
    beam: list[tuple[int, int, list[int]]] = [(start_idx, bar_mask, [])]
    best_trapped: tuple[int, list[int]] = (-1, [])

    for turn in range(max_turns):
        next_states: list[tuple[float, int, int, list[int]]] = []

        for monster_idx, bmask, placements in beam:
            candidates = _get_candidates(monster_idx, bmask)

            for c in candidates:
                new_bm = bmask | (1 << c)
                new_pl = placements + [c]

                d = bfs_distance(monster_idx, new_bm)
                if d < 0:
                    # 困住了
                    steps = len(placements)  # 之前走的步数
                    if steps > best_trapped[0]:
                        best_trapped = (steps, new_pl)
                    continue

                # 小怪走一步
                path = bfs_path(monster_idx, new_bm)
                if len(path) <= 1:
                    steps = len(placements)
                    if steps > best_trapped[0]:
                        best_trapped = (steps, new_pl)
                    continue

                next_idx = path[1]
                if next_idx in _BOUNDARY_SET:
                    continue  # 逃脱了

                score = _score_state(next_idx, new_bm)
                if noise > 0:
                    score += random.uniform(-noise, noise)
                next_states.append((score, next_idx, new_bm, new_pl))

        if not next_states:
            break

        # 去重：相同 (monster_idx, bmask) 只保留最长路径的
        seen: dict[tuple[int, int], int] = {}
        unique: list[tuple[float, int, int, list[int]]] = []
        next_states.sort(reverse=True)
        for score, m, bm, pl in next_states:
            k = (m, bm)
            if k in seen:
                continue
            seen[k] = 1
            unique.append((score, m, bm, pl))
            if len(unique) >= beam_width:
                break

        beam = [(m, bm, pl) for _, m, bm, pl in unique]

    return best_trapped


def solve_max_steps(
    monster_start: tuple[int, int],
    initial_barriers: set[tuple[int, int]],
    *,
    max_placements: int = 40,
    on_progress: object | None = None,
) -> tuple[int, list[tuple[int, int]]]:
    """
    Beam Search 求解最大步数（困住小怪）。

    回合：玩家放一块砖 → 小怪走一步（BFS 最短路径）。
    目标：困住小怪，最大化放砖数。
    小怪策略：尽快逃出。

    返回 (步数, 放置序列)。步数 = -1 表示未能困住。
    """
    from typing import Callable
    progress_cb: Callable[[int, list[tuple[int, int]]], None] | None = (
        on_progress if callable(on_progress) else None
    )

    start_idx = _rc_to_idx[monster_start]
    bar_mask = barriers_to_mask(initial_barriers)

    best_steps = -1
    best_pl: list[int] = []
    last_report = [0.0]

    def try_update(steps: int, pl: list[int]) -> None:
        nonlocal best_steps, best_pl
        if steps <= best_steps:
            return
        best_steps = steps
        best_pl = list(pl)
        if progress_cb is None:
            return
        now = time.monotonic()
        if now - last_report[0] < 0.1:
            return
        last_report[0] = now
        progress_cb(best_steps, [_idx_to_rc[i] for i in best_pl])

    t0 = time.monotonic()

    # 第 1 轮：标准 Beam Search（无噪声，宽束）
    s, pl = _beam_search_once(start_idx, bar_mask, max_placements, BEAM_WIDTH, noise=0.0)
    if s >= 0:
        try_update(s, pl)

    # 第 2-N 轮：带噪声的多样性搜索，直到用完时间预算
    TIME_BUDGET = 8.0  # 秒
    round_num = 0
    while time.monotonic() - t0 < TIME_BUDGET:
        round_num += 1
        noise = 20.0 + round_num * 5.0
        bw = max(100, BEAM_WIDTH - round_num * 30)
        s, pl = _beam_search_once(start_idx, bar_mask, max_placements, bw, noise=noise)
        if s >= 0:
            try_update(s, pl)

    pl_rc = [_idx_to_rc[i] for i in best_pl]
    if progress_cb is not None:
        progress_cb(best_steps, pl_rc)

    return (best_steps, pl_rc)


# ── 测试与入口 ────────────────────────────────────────────
def run_quick_test() -> None:
    assert cell_count() == 68
    assert is_boundary(0, 0) and is_boundary(7, 0)
    d = bfs_distance(_rc_to_idx[(4, 4)], 0)
    assert d >= 0
    steps, pl = solve_max_steps((4, 4), set(), max_placements=10)
    print(f"Quick test: steps={steps}, bricks={len(pl)}")
    print("Quick test passed.")


def main() -> None:
    import sys as _sys
    if len(_sys.argv) > 1 and _sys.argv[1] == "test":
        run_quick_test()
        return

    initial_barriers = set(DEFAULT_BARRIERS)
    monster_start = (4, 4)
    max_placements = 40

    try:
        from hex_grid_gui import run_gui_with_solver
        run_gui_with_solver(
            monster_start=monster_start,
            initial_barriers=initial_barriers,
            max_placements=max_placements,
        )
    except ImportError:
        print("搜索最大步数...")
        steps, placements = solve_max_steps(
            monster_start, initial_barriers, max_placements=max_placements,
        )
        if steps >= 0:
            print(f"困住小怪！{steps} 步, 放置 {len(placements)} 块砖")
            for i, p in enumerate(placements):
                print(f"  {i+1}. {p}")
        else:
            print("未能困住小怪")


if __name__ == "__main__":
    main()
