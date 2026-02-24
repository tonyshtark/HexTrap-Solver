# -*- coding: utf-8 -*-
"""
交错六边形网格（8-9-8-9-8-9-8-9）最优步数搜索。
多进程并行 Beam Search + 可达区域评分。
"""

from __future__ import annotations

import multiprocessing as mp
import os
import random
import sys
import time
from collections import deque
from typing import Iterator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── 棋盘常量 ─────────────────────────────────────────────
ROWS_LAYOUT = [8, 9, 8, 9, 8, 9, 8, 9]
NUM_ROWS = 8
TOTAL_CELLS = sum(ROWS_LAYOUT)

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
    _nb = [_rc_to_idx[(_nr, _nc)] for _nr, _nc in _cands if (_nr, _nc) in _rc_to_idx]
    _NEIGHBORS[_i] = tuple(_nb)

_BOUNDARY_SET: frozenset[int]
_bs: set[int] = set()
for _i in range(TOTAL_CELLS):
    _r, _c = _idx_to_rc[_i]
    n = ROWS_LAYOUT[_r]
    if _r == 0 or _r == NUM_ROWS - 1 or _c == n - 1 or _c == 0:
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


# ── BFS（小怪寻路：离边缘最近的路径 = 步数最少的路径）────────────────
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
    """从 start 到边缘的步数最少路径（BFS 保证最短）。"""
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

def _count_reachable(idx: int, bmask: int) -> int:
    visited = bmask | (1 << idx)
    count = 1
    q = deque((idx,))
    while q:
        u = q.popleft()
        for v in _NEIGHBORS[u]:
            if not (visited & (1 << v)):
                visited |= (1 << v)
                count += 1
                q.append(v)
    return count

def get_monster_next_step(
    monster_pos: tuple[int, int],
    barriers: set[tuple[int, int]],
) -> tuple[int, int] | None:
    """小怪沿「离出口（边缘）最近」的路径走一步，即到边界步数最少的路径。"""
    idx = _rc_to_idx.get(monster_pos)
    if idx is None:
        return None
    path = bfs_path(idx, barriers_to_mask(barriers))
    if len(path) <= 1:
        return None
    return _idx_to_rc[path[1]]


# ── Beam Search 核心（可被子进程调用） ────────────────────
# 稍加深：每层保留更多状态、每状态探索更多候选，提高解质量
BEAM_WIDTH = 1200
CANDS_K = 22

def _beam_worker(args: tuple) -> tuple[int, list[int]]:
    """子进程入口：运行一轮 Beam Search。支持临时障碍(衰减2回合)与夹子(不挡寻路、踩到眩晕、消失后可放砖)。"""
    (
        start_idx, bar_mask, temp2_mask, trap_mask,
        max_turns, beam_width, weight_trap, noise, seed,
    ) = args
    rng = random.Random(seed)
    temp1_init = 0  # 初始只有耐久2的临时障碍

    def _open_nb(idx: int, bm: int) -> int:
        c = 0
        for nb in _NEIGHBORS[idx]:
            if not (bm & (1 << nb)):
                c += 1
        return c

    def _score(midx: int, bm: int, wt: float) -> float:
        d = bfs_distance(midx, bm)
        if d < 0:
            return 99999.0
        reach = _count_reachable(midx, bm)
        onb = _open_nb(midx, bm)
        trap_s = (TOTAL_CELLS - reach) * 15.0 + (6 - onb) * 10.0
        delay_s = d * 80.0
        return trap_s * wt + delay_s

    def _blocking(perm: int, t1: int, t2: int) -> int:
        """寻路用障碍：永久 + 临时（夹子不参与）。"""
        return perm | t1 | t2

    def _get_cands(midx: int, perm: int, t1: int, t2: int, trap: int) -> list[int]:
        bm = _blocking(perm, t1, t2)
        path = bfs_path(midx, bm)
        cs: set[int] = set()
        for p in path[1:]:
            cs.add(p)
            for nb in _NEIGHBORS[p]:
                if not (bm & (1 << nb)):
                    cs.add(nb)
        for nb in _NEIGHBORS[midx]:
            if not (bm & (1 << nb)):
                cs.add(nb)
            for nb2 in _NEIGHBORS[nb]:
                if not (bm & (1 << nb2)):
                    cs.add(nb2)
        cs.discard(midx)
        # 不能放在：永久、临时、夹子、怪物
        cs = {c for c in cs if not (perm & (1 << c)) and not (t1 & (1 << c)) and not (t2 & (1 << c)) and not (trap & (1 << c))}
        return list(cs)

    def _score_cands(midx: int, perm: int, t1: int, t2: int, trap: int, cs: list[int], wt: float) -> list[tuple[float, int]]:
        scored: list[tuple[float, int]] = []
        for c in cs:
            nperm = perm | (1 << c)
            bm = _blocking(nperm, t1, t2)
            d = bfs_distance(midx, bm)
            if d < 0:
                scored.append((99999.0, c))
            else:
                reach = _count_reachable(midx, bm)
                onb = _open_nb(midx, bm)
                trap_s = (TOTAL_CELLS - reach) * 15.0 + (6 - onb) * 10.0
                scored.append((trap_s * wt + d * 80.0, c))
        scored.sort(reverse=True)
        return scored[:CANDS_K]

    # 状态: (monster_idx, perm, temp1, temp2, trap_mask, stunned, placements)
    beam: list[tuple[int, int, int, int, int, int, list[int]]] = [
        (start_idx, bar_mask, temp1_init, temp2_mask, trap_mask, 0, [])
    ]
    best: tuple[int, list[int]] = (-1, [])

    for _ in range(max_turns):
        nxt: list[tuple[float, int, int, int, int, int, int, list[int]]] = []
        for midx, perm, t1, t2, trap, stunned, pls in beam:
            cs = _get_cands(midx, perm, t1, t2, trap)
            if not cs:
                continue
            scored = _score_cands(midx, perm, t1, t2, trap, cs, weight_trap)
            for _, c in scored:
                nperm = perm | (1 << c)
                npl = pls + [c]
                # 每回合临时障碍衰减：temp1 消失，temp2 -> temp1
                nt1, nt2 = t2, 0
                block = _blocking(nperm, nt1, nt2)

                if stunned:
                    # 眩晕回合：只放砖，怪物不移动，下一回合不眩晕
                    sc = _score(midx, block, weight_trap)
                    if noise > 0:
                        sc += rng.uniform(-noise, noise)
                    nxt.append((sc, midx, nperm, nt1, nt2, trap, 0, npl))
                    continue

                path = bfs_path(midx, block)
                if not path or len(path) <= 1:
                    steps = len(pls)
                    if steps > best[0]:
                        best = (steps, npl)
                    continue
                ni = path[1]
                if ni in _BOUNDARY_SET:
                    continue
                stepped_on_trap = (trap & (1 << ni)) != 0
                ntrap = trap & ~(1 << ni) if stepped_on_trap else trap
                nstun = 1 if stepped_on_trap else 0
                sc = _score(ni, block, weight_trap)
                if noise > 0:
                    sc += rng.uniform(-noise, noise)
                nxt.append((sc, ni, nperm, nt1, nt2, ntrap, nstun, npl))

        if not nxt:
            break
        seen: dict[tuple[int, int, int, int, int, int], int] = {}
        unique: list[tuple[float, int, int, int, int, int, int, list[int]]] = []
        nxt.sort(reverse=True)
        for sc, m, p, t1, t2, tr, st, pl in nxt:
            k = (m, p, t1, t2, tr, st)
            if k not in seen:
                seen[k] = 1
                unique.append((sc, m, p, t1, t2, tr, st, pl))
                if len(unique) >= beam_width:
                    break
        beam = [(m, p, t1, t2, tr, st, pl) for _, m, p, t1, t2, tr, st, pl in unique]

    return best


def solve_max_steps(
    monster_start: tuple[int, int],
    initial_barriers: set[tuple[int, int]],
    *,
    max_placements: int = 40,
    on_progress: object | None = None,
    exclude_cells: set[tuple[int, int]] | None = None,
    temp_barriers_cells: set[tuple[int, int]] | None = None,
) -> tuple[int, list[tuple[int, int]]]:
    """多进程并行 Beam Search。
    initial_barriers: 仅永久障碍。
    exclude_cells: 夹子位置，不可放砖（被踩后消失，求解器内会模拟）。
    temp_barriers_cells: 临时障碍位置（耐久2，每回合衰减，2回合后消失，影响寻路）。
    """
    from typing import Callable
    progress_cb: Callable[[int, list[tuple[int, int]]], None] | None = (
        on_progress if callable(on_progress) else None
    )

    start_idx = _rc_to_idx[monster_start]
    bar_mask = barriers_to_mask(initial_barriers)
    trap_mask = barriers_to_mask(exclude_cells) if exclude_cells else 0
    temp2_mask = barriers_to_mask(temp_barriers_cells) if temp_barriers_cells else 0

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

    # 构建多组参数：不同 weight_trap / noise / seed；worker 需要 (start_idx, bar_mask, temp2_mask, trap_mask, max_turns, beam_width, wt, noise, seed)
    tasks: list[tuple] = []
    base_seed = int(time.time() * 1000) & 0xFFFFFF

    def add_task(wt: float, nois: float, sid: int) -> None:
        tasks.append((start_idx, bar_mask, temp2_mask, trap_mask, max_placements, BEAM_WIDTH, wt, nois, sid))

    for i, wt in enumerate([3.0, 2.0, 1.5]):
        add_task(wt, 0.0, base_seed + i)
    for i in range(7):
        add_task(random.uniform(0.5, 1.5), random.uniform(5, 20), base_seed + 100 + i)
    for i in range(10):
        add_task(random.uniform(0.1, 0.5), random.uniform(5, 30), base_seed + 200 + i)
    for i in range(6):
        add_task(random.uniform(0.01, 0.1), random.uniform(10, 40), base_seed + 300 + i)

    n_workers = max(1, min(len(tasks), (os.cpu_count() or 4)))

    try:
        with mp.Pool(processes=n_workers) as pool:
            for result in pool.imap_unordered(_beam_worker, tasks):
                s, pl = result
                if s >= 0:
                    try_update(s, pl)
    except Exception:
        # 多进程失败时回退到单进程
        for task in tasks:
            s, pl = _beam_worker(task)
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
    max_placements = 50

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
            print(f"困住！{steps} 步, {len(placements)} 砖")
        else:
            print("未能困住")


if __name__ == "__main__":
    mp.freeze_support()
    main()
