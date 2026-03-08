# -*- coding: utf-8 -*-
"""
交错六边形网格（8-9-8-9-8-9-8-9）最优步数搜索。
Beam Search + 可达区域评分。
"""

from __future__ import annotations

import multiprocessing as _mp
import os
import random
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── C++ 引擎（可选加速）──────────────────────────────────
_CPP_ENGINE = None
try:
    import cpp_hex_engine as _CPP_ENGINE
except ImportError:
    pass

def is_using_cpp_engine() -> bool:
    """是否使用 C++ 引擎加速。True=C++，False=Python 实现。"""
    return _CPP_ENGINE is not None

def _to_cpp_barriers(bmask: int) -> list[int]:
    """将 int 位掩码转为 C++ 可接受的索引列表（支持 68 位）。"""
    return [i for i in range(TOTAL_CELLS) if (bmask >> i) & 1]

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

# 预计算：每个格子到最近边界的步数（无障碍时），用于优先边界附近放置
_DIST_TO_BOUNDARY: list[int] = [99] * TOTAL_CELLS
_dtb_q: deque = deque()
for _b in _BOUNDARY_SET:
    _DIST_TO_BOUNDARY[_b] = 0
    _dtb_q.append(_b)
while _dtb_q:
    _u = _dtb_q.popleft()
    for _v in _NEIGHBORS[_u]:
        if _DIST_TO_BOUNDARY[_v] > _DIST_TO_BOUNDARY[_u] + 1:
            _DIST_TO_BOUNDARY[_v] = _DIST_TO_BOUNDARY[_u] + 1
            _dtb_q.append(_v)

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
def _bfs_distance_py(start: int, bmask: int) -> int:
    """纯 Python 实现（回退用）。"""
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

def bfs_distance(start: int, bmask: int) -> int:
    if _CPP_ENGINE:
        return _CPP_ENGINE.bfs_distance(start, _to_cpp_barriers(bmask))
    return _bfs_distance_py(start, bmask)

def _bfs_path_py(start: int, bmask: int) -> list[int]:
    """纯 Python 实现（回退用）。"""
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

def bfs_path(start: int, bmask: int) -> list[int]:
    """从 start 到边缘的步数最少路径（BFS 保证最短）。"""
    if _CPP_ENGINE:
        return _CPP_ENGINE.bfs_path(start, _to_cpp_barriers(bmask))
    return _bfs_path_py(start, bmask)

def _count_reachable_py(idx: int, bmask: int) -> int:
    """纯 Python 实现（回退用）。"""
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

def _count_reachable(idx: int, bmask: int) -> int:
    if _CPP_ENGINE:
        return _CPP_ENGINE.count_reachable(idx, _to_cpp_barriers(bmask))
    return _count_reachable_py(idx, bmask)

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


# ── Beam Search 核心 ──────────────────────────────────────
BEAM_WIDTH = 2000
CANDS_K = 40
PATH_CAND_LIMIT = 15  # 路径候选延伸步数
BOUNDARY_PENALTY = 30.0  # 离边界越远惩罚越大
PROXIMITY_PENALTY = 800.0  # 贴脸放砖惩罚（社交距离）

def _hex_dist(idx1: int, idx2: int) -> int:
    """六边形网格上两格之间的步数距离（odd-r 布局）。"""
    r1, c1 = _idx_to_rc[idx1]
    r2, c2 = _idx_to_rc[idx2]
    x1 = c1 - (r1 - (r1 & 1)) // 2
    z1 = r1
    y1 = -x1 - z1
    x2 = c2 - (r2 - (r2 & 1)) // 2
    z2 = r2
    y2 = -x2 - z2
    return (abs(x1 - x2) + abs(y1 - y2) + abs(z1 - z2)) // 2

def _beam_worker(args: tuple) -> tuple[int, list[int]]:
    """子进程入口：支持临时障碍(耐久1/2/3回合)与夹子。"""
    (
        start_idx, bar_mask, temp1_mask, temp2_mask, temp3_mask, trap_mask,
        max_turns, beam_width, weight_trap, noise, seed,
        first_placement_idx,
    ) = args
    rng = random.Random(seed)

    def _open_nb(idx: int, bm: int) -> int:
        if _CPP_ENGINE:
            return _CPP_ENGINE.open_neighbor_count(idx, _to_cpp_barriers(bm))
        return sum(1 for nb in _NEIGHBORS[idx] if not (bm & (1 << nb)))

    def _score(midx: int, bm: int, wt: float) -> float:
        """高步数策略：距离优先，大笼子奖励。"""
        d = bfs_distance(midx, bm)
        reach = _count_reachable(midx, bm)
        onb = _open_nb(midx, bm)
        if d < 0:
            return 50000.0 + reach * 100.0
        score = d * 600.0
        if d > 3:
            score += reach * 30.0
        else:
            score -= reach * 2.0
        score += (6 - onb) * 20.0
        return score

    def _blocking(perm: int, t1: int, t2: int, t3: int) -> int:
        return perm | t1 | t2 | t3

    def _get_cands(midx: int, perm: int, t1: int, t2: int, t3: int, trap: int) -> list[int]:
        """候选格：路径前N步+贴身+边界出口+关键走廊+优先边界附近采样。"""
        bm = _blocking(perm, t1, t2, t3)
        path = bfs_path(midx, bm)
        cs: set[int] = set()
        limit = min(len(path), PATH_CAND_LIMIT)
        for p in path[1:limit]:
            cs.add(p)
            for nb in _NEIGHBORS[p]:
                if not (bm & (1 << nb)):
                    cs.add(nb)
        for nb in _NEIGHBORS[midx]:
            if not (bm & (1 << nb)):
                cs.add(nb)
        for b_idx in _BOUNDARY_SET:
            if not (bm & (1 << b_idx)):
                cs.add(b_idx)
                for nb in _NEIGHBORS[b_idx]:
                    if not (bm & (1 << nb)):
                        cs.add(nb)
        # 关键走廊：双向 BFS 找所有最短路径覆盖的格子（封住任意一个都能延长距离）
        if path and len(path) > 1:
            d_total = len(path) - 1
            d_m: dict[int, int] = {midx: 0}
            _q: deque = deque([midx])
            _vis = bm | (1 << midx)
            while _q:
                u = _q.popleft()
                for v in _NEIGHBORS[u]:
                    if not (_vis & (1 << v)):
                        _vis |= (1 << v)
                        d_m[v] = d_m[u] + 1
                        _q.append(v)
            d_b: dict[int, int] = {}
            _q2: deque = deque()
            for b in _BOUNDARY_SET:
                if b in d_m:
                    d_b[b] = 0
                    _q2.append(b)
            while _q2:
                u = _q2.popleft()
                for v in _NEIGHBORS[u]:
                    if not (bm & (1 << v)) and v in d_m and v not in d_b:
                        d_b[v] = d_b[u] + 1
                        _q2.append(v)
            for c, dc in d_m.items():
                if c != midx and not (bm & (1 << c)) and c in d_b:
                    if dc + d_b[c] <= d_total + 1:
                        cs.add(c)
        # 优先边界附近：dist<=2 的格子优先，内部格子仅少量补充
        near_b = [i for i in range(TOTAL_CELLS) if not (bm & (1 << i)) and i != midx and _DIST_TO_BOUNDARY[i] <= 2]
        far_b = [i for i in range(TOTAL_CELLS) if not (bm & (1 << i)) and i != midx and _DIST_TO_BOUNDARY[i] > 2]
        if near_b:
            cs.update(rng.sample(near_b, min(len(near_b), 12)))
        if far_b and len(cs) < 24:
            cs.update(rng.sample(far_b, min(len(far_b), 6)))
        # 强制加入：离怪物 3 步以外、靠近边界的空位（给 AI 往外面放的选项）
        outer_ring = [i for i in range(TOTAL_CELLS) if not (bm & (1 << i)) and i != midx
                      and _hex_dist(midx, i) >= 3 and _DIST_TO_BOUNDARY[i] <= 2]
        if outer_ring:
            cs.update(rng.sample(outer_ring, min(len(outer_ring), 8)))
        cs.discard(midx)
        cs = {c for c in cs if not (perm & (1 << c)) and not (t1 & (1 << c)) and not (t2 & (1 << c)) and not (t3 & (1 << c)) and not (trap & (1 << c))}
        return list(cs)

    def _simulate_greedy_survival(midx: int, bm: int) -> int:
        """贪心模拟：双方贪心策略下还能走多少步。"""
        curr_m, curr_bm = midx, bm
        for step in range(50):
            path = bfs_path(curr_m, curr_bm)
            if not path or len(path) <= 1:
                return step
            ni = path[1]
            if ni in _BOUNDARY_SET:
                return -1
            curr_bm |= (1 << ni)
            new_path = bfs_path(curr_m, curr_bm)
            if not new_path or len(new_path) <= 1:
                return step + 1
            curr_m = new_path[1]
        return 50

    def _score_cands(midx: int, perm: int, t1: int, t2: int, t3: int, trap: int, cs: list[int], wt: float) -> list[tuple[float, int]]:
        """评分候选：距离+缩减可达区域+大笼子+贴脸惩罚+边界惩罚，前N名贪心模拟。"""
        scored: list[tuple[float, int]] = []
        nt1, nt2, nt3 = t2, t3, 0
        bm_before = _blocking(perm, nt1, nt2, nt3)
        total_reach_before = _count_reachable(midx, bm_before)
        for c in cs:
            nperm = perm | (1 << c)
            bm = _blocking(nperm, nt1, nt2, nt3)
            d = bfs_distance(midx, bm)
            reach = _count_reachable(midx, bm)
            delta_reach = total_reach_before - reach  # 放砖后减少的可达格子数
            if d < 0:
                sc = 50000.0 + reach * 100.0
            else:
                sc = d * 600.0
                if d > 3:
                    sc += reach * 30.0
                    sc += delta_reach * 80.0  # 奖励大幅压缩可达区域
                else:
                    sc -= reach * 2.0
                    sc += delta_reach * 40.0  # 近身时也奖励压缩
            # 贴脸惩罚：离怪物 <=2 步施加惩罚
            dist_to_monster = _hex_dist(midx, c)
            if dist_to_monster <= 2:
                sc -= (3 - dist_to_monster) * PROXIMITY_PENALTY
            sc -= _DIST_TO_BOUNDARY[c] * BOUNDARY_PENALTY
            scored.append((sc, c))
        scored.sort(reverse=True)
        top_k = min(8, len(scored))
        for i in range(top_k):
            sc, c = scored[i]
            nperm = perm | (1 << c)
            bm = _blocking(nperm, nt1, nt2, nt3)
            sim = _simulate_greedy_survival(midx, bm)
            if sim >= 0:
                scored[i] = (sc + sim * 50.0, c)
        scored.sort(reverse=True)
        return scored[:CANDS_K]

    # 状态: (monster_idx, perm, t1, t2, t3, trap_mask, stunned, placements)  每回合衰减: t1消失, t2->t1, t3->t2
    beam: list[tuple[int, int, int, int, int, int, int, list[int]]] = [
        (start_idx, bar_mask, temp1_mask, temp2_mask, temp3_mask, trap_mask, 0, [])
    ]
    best: tuple[int, list[int]] = (-1, [])
    best_escape_pl: list[int] = []  # 未困住时，保存放置最多的逃脱路径供回放

    for turn in range(max_turns):
        nxt: list[tuple[float, int, int, int, int, int, int, int, list[int]]] = []
        for midx, perm, t1, t2, t3, trap, stunned, pls in beam:
            if midx in _BOUNDARY_SET:
                if len(pls) > len(best_escape_pl):
                    best_escape_pl = list(pls)
                continue
            cs = _get_cands(midx, perm, t1, t2, t3, trap)
            if not cs:
                continue
            if len(pls) == 0 and first_placement_idx >= 0:
                bm = _blocking(perm, t1, t2, t3)
                valid = not (perm & (1 << first_placement_idx)) and not (t1 & (1 << first_placement_idx)) and not (t2 & (1 << first_placement_idx)) and not (t3 & (1 << first_placement_idx)) and not (trap & (1 << first_placement_idx)) and first_placement_idx != midx and not (bm & (1 << first_placement_idx))
                if valid:
                    cs = [first_placement_idx]
            scored = _score_cands(midx, perm, t1, t2, t3, trap, cs, weight_trap)
            for _, c in scored:
                nperm = perm | (1 << c)
                npl = pls + [c]
                nt1, nt2, nt3 = t2, t3, 0
                block = _blocking(nperm, nt1, nt2, nt3)

                if stunned:
                    sc = _score(midx, block, weight_trap)
                    if noise > 0:
                        sc += rng.uniform(-noise, noise)
                    sc += turn * 10.0
                    nxt.append((sc, midx, nperm, nt1, nt2, nt3, trap, 0, npl))
                    continue

                path = bfs_path(midx, block)
                if not path or len(path) <= 1:
                    steps = len(pls)
                    if steps > best[0]:
                        best = (steps, npl)
                    continue
                ni = path[1]
                if ni in _BOUNDARY_SET:
                    if len(npl) > len(best_escape_pl):
                        best_escape_pl = list(npl)
                    continue
                stepped_on_trap = (trap & (1 << ni)) != 0
                ntrap = trap & ~(1 << ni) if stepped_on_trap else trap
                nstun = 1 if stepped_on_trap else 0
                sc = _score(ni, block, weight_trap)
                if noise > 0:
                    sc += rng.uniform(-noise, noise)
                sc += turn * 10.0
                nxt.append((sc, ni, nperm, nt1, nt2, nt3, ntrap, nstun, npl))

        if not nxt:
            break
        seen: dict[tuple[int, int, int, int, int, int, int], int] = {}
        unique: list[tuple[float, int, int, int, int, int, int, int, list[int]]] = []
        nxt.sort(reverse=True)
        for sc, m, p, t1, t2, t3, tr, st, pl in nxt:
            k = (m, p, t1, t2, t3, tr, st)
            if k not in seen:
                seen[k] = 1
                unique.append((sc, m, p, t1, t2, t3, tr, st, pl))
                if len(unique) >= beam_width:
                    break
        beam = [(m, p, t1, t2, t3, tr, st, pl) for _, m, p, t1, t2, t3, tr, st, pl in unique]

    if best[0] < 0 and best_escape_pl:
        return (-1, best_escape_pl)  # 未困住也返回最佳尝试，供回放
    return best


def _beam_worker_two(args: tuple) -> tuple[int, list[int]]:
    """双小怪 Beam Search：交替移动(先1后2循环)，夹子眩晕占该怪回合，目标围住两只。"""
    (
        m1_idx, m2_idx, bar_mask, temp1_mask, temp2_mask, temp3_mask, trap_mask,
        max_turns, beam_width, weight_trap, noise, seed,
        first_placement_idx,
    ) = args
    rng = random.Random(seed)

    def _open_nb(idx: int, bm: int) -> int:
        if _CPP_ENGINE:
            return _CPP_ENGINE.open_neighbor_count(idx, _to_cpp_barriers(bm))
        return sum(1 for nb in _NEIGHBORS[idx] if not (bm & (1 << nb)))

    def _score(midx: int, bm: int, wt: float) -> float:
        """高步数策略：距离优先，大笼子奖励。"""
        d = bfs_distance(midx, bm)
        reach = _count_reachable(midx, bm)
        onb = _open_nb(midx, bm)
        if d < 0:
            return 50000.0 + reach * 100.0
        score = d * 600.0
        if d > 3:
            score += reach * 30.0
        else:
            score -= reach * 2.0
        score += (6 - onb) * 20.0
        return score

    def _blocking(perm: int, t1: int, t2: int, t3: int) -> int:
        return perm | t1 | t2 | t3

    def _get_cands(midx: int, perm: int, t1: int, t2: int, t3: int, trap: int, m1: int, m2: int) -> list[int]:
        """候选格：路径前N步+贴身+边界+关键走廊+优先边界附近采样。"""
        bm = _blocking(perm, t1, t2, t3)
        path = bfs_path(midx, bm)
        cs: set[int] = set()
        limit = min(len(path), PATH_CAND_LIMIT)
        for p in path[1:limit]:
            cs.add(p)
            for nb in _NEIGHBORS[p]:
                if not (bm & (1 << nb)):
                    cs.add(nb)
        for nb in _NEIGHBORS[midx]:
            if not (bm & (1 << nb)):
                cs.add(nb)
        for b_idx in _BOUNDARY_SET:
            if not (bm & (1 << b_idx)):
                cs.add(b_idx)
                for nb in _NEIGHBORS[b_idx]:
                    if not (bm & (1 << nb)):
                        cs.add(nb)
        # 关键走廊：双向 BFS 找所有最短路径覆盖的格子
        if path and len(path) > 1:
            d_total = len(path) - 1
            d_m: dict[int, int] = {midx: 0}
            _q: deque = deque([midx])
            _vis = bm | (1 << midx)
            while _q:
                u = _q.popleft()
                for v in _NEIGHBORS[u]:
                    if not (_vis & (1 << v)):
                        _vis |= (1 << v)
                        d_m[v] = d_m[u] + 1
                        _q.append(v)
            d_b: dict[int, int] = {}
            _q2: deque = deque()
            for b in _BOUNDARY_SET:
                if b in d_m:
                    d_b[b] = 0
                    _q2.append(b)
            while _q2:
                u = _q2.popleft()
                for v in _NEIGHBORS[u]:
                    if not (bm & (1 << v)) and v in d_m and v not in d_b:
                        d_b[v] = d_b[u] + 1
                        _q2.append(v)
            for c, dc in d_m.items():
                if c != midx and not (bm & (1 << c)) and c in d_b:
                    if dc + d_b[c] <= d_total + 1:
                        cs.add(c)
        monster_mask = (1 << m1) | (1 << m2)
        near_b = [i for i in range(TOTAL_CELLS) if not (bm & (1 << i)) and i != midx and not (monster_mask & (1 << i)) and _DIST_TO_BOUNDARY[i] <= 2]
        far_b = [i for i in range(TOTAL_CELLS) if not (bm & (1 << i)) and i != midx and not (monster_mask & (1 << i)) and _DIST_TO_BOUNDARY[i] > 2]
        if near_b:
            cs.update(rng.sample(near_b, min(len(near_b), 12)))
        if far_b and len(cs) < 24:
            cs.update(rng.sample(far_b, min(len(far_b), 6)))
        outer_ring = [i for i in range(TOTAL_CELLS) if not (bm & (1 << i)) and i != midx and not (monster_mask & (1 << i))
                      and _hex_dist(midx, i) >= 3 and _DIST_TO_BOUNDARY[i] <= 2]
        if outer_ring:
            cs.update(rng.sample(outer_ring, min(len(outer_ring), 8)))
        cs.discard(midx)
        cs = {c for c in cs if not (perm & (1 << c)) and not (t1 & (1 << c)) and not (t2 & (1 << c)) and not (t3 & (1 << c)) and not (trap & (1 << c)) and not (monster_mask & (1 << c))}
        return list(cs)

    def _simulate_greedy_survival(midx: int, bm: int) -> int:
        curr_m, curr_bm = midx, bm
        for step in range(50):
            path = bfs_path(curr_m, curr_bm)
            if not path or len(path) <= 1:
                return step
            ni = path[1]
            if ni in _BOUNDARY_SET:
                return -1
            curr_bm |= (1 << ni)
            new_path = bfs_path(curr_m, curr_bm)
            if not new_path or len(new_path) <= 1:
                return step + 1
            curr_m = new_path[1]
        return 50

    def _score_cands(midx: int, perm: int, t1: int, t2: int, t3: int, trap: int, m1: int, m2: int, cs: list[int], wt: float) -> list[tuple[float, int]]:
        """评分候选：距离+缩减可达区域+大笼子+贴脸惩罚+边界惩罚，前N名贪心模拟。"""
        nt1, nt2, nt3 = t2, t3, 0
        bm_before = _blocking(perm, nt1, nt2, nt3)
        total_reach_before = _count_reachable(midx, bm_before)
        scored: list[tuple[float, int]] = []
        for c in cs:
            nperm = perm | (1 << c)
            bm = _blocking(nperm, nt1, nt2, nt3)
            d = bfs_distance(midx, bm)
            reach = _count_reachable(midx, bm)
            delta_reach = total_reach_before - reach
            if d < 0:
                sc = 50000.0 + reach * 100.0
            else:
                sc = d * 600.0
                if d > 3:
                    sc += reach * 30.0
                    sc += delta_reach * 80.0
                else:
                    sc -= reach * 2.0
                    sc += delta_reach * 40.0
            dist_to_monster = _hex_dist(midx, c)
            if dist_to_monster <= 2:
                sc -= (3 - dist_to_monster) * PROXIMITY_PENALTY
            sc -= _DIST_TO_BOUNDARY[c] * BOUNDARY_PENALTY
            scored.append((sc, c))
        scored.sort(reverse=True)
        top_k = min(8, len(scored))
        for i in range(top_k):
            sc, c = scored[i]
            nperm = perm | (1 << c)
            bm = _blocking(nperm, nt1, nt2, nt3)
            sim = _simulate_greedy_survival(midx, bm)
            if sim >= 0:
                scored[i] = (sc + sim * 50.0, c)
        scored.sort(reverse=True)
        return scored[:CANDS_K]

    # 状态: (m1, m2, turn, perm, t1, t2, t3, trap, stunned, pls)
    beam: list[tuple[int, int, int, int, int, int, int, int, int, list[int]]] = [
        (m1_idx, m2_idx, 0, bar_mask, temp1_mask, temp2_mask, temp3_mask, trap_mask, -1, [])
    ]
    best: tuple[int, list[int]] = (-1, [])
    best_escape_pl: list[int] = []

    for turn_idx in range(max_turns):
        nxt: list[tuple[float, int, int, int, int, int, int, int, int, int, list[int]]] = []
        for m1, m2, turn, perm, t1, t2, t3, trap, stunned, pls in beam:
            midx = m1 if turn == 0 else m2
            if midx in _BOUNDARY_SET:
                if len(pls) > len(best_escape_pl):
                    best_escape_pl = list(pls)
                continue
            cs = _get_cands(midx, perm, t1, t2, t3, trap, m1, m2)
            if not cs:
                continue
            if len(pls) == 0 and first_placement_idx >= 0:
                bm = _blocking(perm, t1, t2, t3)
                monster_mask = (1 << m1) | (1 << m2)
                valid = not (perm & (1 << first_placement_idx)) and not (t1 & (1 << first_placement_idx)) and not (t2 & (1 << first_placement_idx)) and not (t3 & (1 << first_placement_idx)) and not (trap & (1 << first_placement_idx)) and not (monster_mask & (1 << first_placement_idx)) and not (bm & (1 << first_placement_idx))
                if valid:
                    cs = [first_placement_idx]
            scored = _score_cands(midx, perm, t1, t2, t3, trap, m1, m2, cs, weight_trap)
            for _, c in scored:
                nperm = perm | (1 << c)
                npl = pls + [c]
                nt1, nt2, nt3 = t2, t3, 0
                block = _blocking(nperm, nt1, nt2, nt3)

                if stunned == turn:
                    sc = _score(m1, block, weight_trap) + _score(m2, block, weight_trap)
                    if noise > 0:
                        sc += rng.uniform(-noise, noise)
                    sc += turn_idx * 10.0
                    nxt.append((sc, m1, m2, (turn + 1) % 2, nperm, nt1, nt2, nt3, trap, -1, npl))
                    continue

                path = bfs_path(midx, block)
                if not path or len(path) <= 1:
                    d1 = bfs_distance(m1, block)
                    d2 = bfs_distance(m2, block)
                    if d1 < 0 and d2 < 0:
                        steps = len(pls)
                        if steps > best[0]:
                            best = (steps, npl)
                        continue
                    # 仅当前怪被困：不移动，轮到另一只，继续搜索
                    other = (turn + 1) % 2
                    sc = _score(m1, block, weight_trap) + _score(m2, block, weight_trap)
                    if noise > 0:
                        sc += rng.uniform(-noise, noise)
                    sc += turn_idx * 10.0
                    nxt.append((sc, m1, m2, other, nperm, nt1, nt2, nt3, trap, -1, npl))
                    continue
                ni = path[1]
                if ni in _BOUNDARY_SET:
                    if len(npl) > len(best_escape_pl):
                        best_escape_pl = list(npl)
                    continue
                stepped_on_trap = (trap & (1 << ni)) != 0
                ntrap = trap & ~(1 << ni) if stepped_on_trap else trap
                nstun = turn if stepped_on_trap else stunned
                nm1, nm2 = (ni, m2) if turn == 0 else (m1, ni)
                sc = _score(nm1, block, weight_trap) + _score(nm2, block, weight_trap)
                if noise > 0:
                    sc += rng.uniform(-noise, noise)
                sc += turn_idx * 10.0
                nxt.append((sc, nm1, nm2, (turn + 1) % 2, nperm, nt1, nt2, nt3, ntrap, nstun, npl))

        if not nxt:
            break
        seen: dict[tuple[int, int, int, int, int, int, int, int, int], int] = {}
        unique: list[tuple[float, int, int, int, int, int, int, int, int, int, list[int]]] = []
        nxt.sort(reverse=True)
        for sc, ma, mb, trn, p, t1, t2, t3, tr, st, pl in nxt:
            k = (ma, mb, trn, p, t1, t2, t3, tr, st)
            if k not in seen:
                seen[k] = 1
                unique.append((sc, ma, mb, trn, p, t1, t2, t3, tr, st, pl))
                if len(unique) >= beam_width:
                    break
        beam = [(ma, mb, trn, p, t1, t2, t3, tr, st, pl) for _, ma, mb, trn, p, t1, t2, t3, tr, st, pl in unique]

    if best[0] < 0 and best_escape_pl:
        return (-1, best_escape_pl)
    return best


def solve_max_steps_two(
    monster_starts: tuple[tuple[int, int], tuple[int, int]],
    initial_barriers: set[tuple[int, int]],
    *,
    max_placements: int = 200,  # 障碍无上限，仅作搜索深度
    on_progress: object | None = None,
    exclude_cells: set[tuple[int, int]] | None = None,
    temp_barriers: dict[tuple[int, int], int] | None = None,
    first_placement: tuple[int, int] | None = None,
) -> tuple[int, list[tuple[int, int]]]:
    """双小怪 Beam Search：交替移动(先1后2)，夹子眩晕占该怪回合，目标围住两只。"""
    from typing import Callable
    progress_cb: Callable[[int, list[tuple[int, int]]], None] | None = (
        on_progress if callable(on_progress) else None
    )

    m1_idx = _rc_to_idx[monster_starts[0]]
    m2_idx = _rc_to_idx[monster_starts[1]]
    bar_mask = barriers_to_mask(initial_barriers)
    trap_mask = barriers_to_mask(exclude_cells) if exclude_cells else 0
    tb = temp_barriers or {}
    temp1_mask = barriers_to_mask({c for c, d in tb.items() if d == 1})
    temp2_mask = barriers_to_mask({c for c, d in tb.items() if d == 2})
    temp3_mask = barriers_to_mask({c for c, d in tb.items() if d == 3})

    best_steps = -1
    best_pl: list[int] = []
    last_report = [0.0]

    def try_update(steps: int, pl: list[int]) -> None:
        nonlocal best_steps, best_pl
        if steps <= best_steps:
            return
        best_steps = steps
        best_pl = list(pl)
        if progress_cb:
            now = time.monotonic()
            if now - last_report[0] >= 0.1:
                last_report[0] = now
                progress_cb(best_steps, [_idx_to_rc[i] for i in best_pl])

    first_idx = _rc_to_idx[first_placement] if first_placement is not None and first_placement in _rc_to_idx else -1
    tasks: list[tuple] = []
    base_seed = int(time.time() * 1000) & 0xFFFFFF

    def add_task(wt: float, nois: float, sid: int) -> None:
        tasks.append((m1_idx, m2_idx, bar_mask, temp1_mask, temp2_mask, temp3_mask, trap_mask, max_placements, BEAM_WIDTH, wt, nois, sid, first_idx))

    has_temp = bool(tb)
    for i, wt in enumerate([3.0, 2.5, 2.0, 1.5, 1.0]):
        add_task(wt, 0.0, base_seed + i)
    if has_temp:
        for i, wt in enumerate([4.0, 3.5, 3.0]):
            add_task(wt, 0.0, base_seed + 50 + i)
    for i in range(5):
        add_task(random.uniform(0.5, 1.5), random.uniform(5, 20), base_seed + 100 + i)
    for i in range(5):
        add_task(random.uniform(0.1, 0.5), random.uniform(5, 30), base_seed + 200 + i)

    n_workers = min(len(tasks), max(1, _mp.cpu_count()))
    try:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_beam_worker_two, task): task for task in tasks}
            for future in as_completed(futures):
                try:
                    s, pl = future.result()
                except Exception:
                    continue  # 单个任务失败，跳过继续处理其他任务
                if s >= 0:
                    try_update(s, pl)
                elif pl and len(pl) > len(best_pl) and best_steps < 0:
                    best_pl = list(pl)
    except Exception:
        # 整个进程池不可用时才串行回退
        for task in tasks:
            s, pl = _beam_worker_two(task)
            if s >= 0:
                try_update(s, pl)
            elif pl and len(pl) > len(best_pl) and best_steps < 0:
                best_pl = list(pl)

    pl_rc = [_idx_to_rc[i] for i in best_pl]
    if progress_cb:
        progress_cb(best_steps, pl_rc)
    return (best_steps, pl_rc)


def solve_max_steps(
    monster_start: tuple[int, int],
    initial_barriers: set[tuple[int, int]],
    *,
    max_placements: int = 200,  # 障碍无上限，仅作搜索深度
    on_progress: object | None = None,
    exclude_cells: set[tuple[int, int]] | None = None,
    temp_barriers: dict[tuple[int, int], int] | None = None,
    first_placement: tuple[int, int] | None = None,
) -> tuple[int, list[tuple[int, int]]]:
    """Beam Search 求解。
    temp_barriers: 临时障碍 {cell: 耐久1/2/3}，每回合衰减，归零消失。
    """
    from typing import Callable
    progress_cb: Callable[[int, list[tuple[int, int]]], None] | None = (
        on_progress if callable(on_progress) else None
    )

    start_idx = _rc_to_idx[monster_start]
    bar_mask = barriers_to_mask(initial_barriers)
    trap_mask = barriers_to_mask(exclude_cells) if exclude_cells else 0
    tb = temp_barriers or {}
    temp1_mask = barriers_to_mask({c for c, d in tb.items() if d == 1})
    temp2_mask = barriers_to_mask({c for c, d in tb.items() if d == 2})
    temp3_mask = barriers_to_mask({c for c, d in tb.items() if d == 3})

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

    first_idx = _rc_to_idx[first_placement] if first_placement is not None and first_placement in _rc_to_idx else -1
    tasks: list[tuple] = []
    base_seed = int(time.time() * 1000) & 0xFFFFFF

    def add_task(wt: float, nois: float, sid: int) -> None:
        tasks.append((start_idx, bar_mask, temp1_mask, temp2_mask, temp3_mask, trap_mask, max_placements, BEAM_WIDTH, wt, nois, sid, first_idx))

    has_temp = bool(tb)
    for i, wt in enumerate([3.0, 2.5, 2.0, 1.5, 1.0]):
        add_task(wt, 0.0, base_seed + i)
    if has_temp:
        for i, wt in enumerate([4.0, 3.5, 3.0]):
            add_task(wt, 0.0, base_seed + 50 + i)
    for i in range(5):
        add_task(random.uniform(0.5, 1.5), random.uniform(5, 20), base_seed + 100 + i)
    for i in range(6):
        add_task(random.uniform(0.1, 0.5), random.uniform(5, 30), base_seed + 200 + i)
    for i in range(4):
        add_task(random.uniform(0.01, 0.1), random.uniform(10, 40), base_seed + 300 + i)

    n_workers = min(len(tasks), max(1, _mp.cpu_count()))
    try:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_beam_worker, task): task for task in tasks}
            for future in as_completed(futures):
                try:
                    s, pl = future.result()
                except Exception:
                    continue  # 单个任务失败，跳过继续处理其他任务
                if s >= 0:
                    try_update(s, pl)
                elif pl and len(pl) > len(best_pl) and best_steps < 0:
                    best_pl = list(pl)
    except Exception:
        # 整个进程池不可用时才串行回退
        for task in tasks:
            s, pl = _beam_worker(task)
            if s >= 0:
                try_update(s, pl)
            elif pl and len(pl) > len(best_pl) and best_steps < 0:
                best_pl = list(pl)

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

    initial_barriers: set[tuple[int, int]] = set()  # 无初始障碍，完全由用户手动添加
    monster_start = (4, 4)
    max_placements = 200

    try:
        from hex_grid_gui import run_gui_with_solver
        run_gui_with_solver(
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
    _mp.freeze_support()
    main()
