/**
 * hex_engine.cpp - 六边形网格寻路核心实现
 * 位运算 BFS、预计算、评分
 */

#include "hex_engine.hpp"
#include <algorithm>
#include <queue>

namespace hex_engine {

std::array<CellMask, TOTAL_CELLS> NEIGHBOR_MASKS;
CellMask BOUNDARY_MASK;
std::array<int, 8 * 10> RC_TO_IDX;

// idx -> (r,c) 反向映射
static std::array<int, TOTAL_CELLS> IDX_TO_R;
static std::array<int, TOTAL_CELLS> IDX_TO_C;

void init_backend() {
    // 初始化 RC_TO_IDX 为 -1
    RC_TO_IDX.fill(-1);

    int idx = 0;
    for (int r = 0; r < NUM_ROWS; ++r) {
        int n = ROWS_LAYOUT[r];
        for (int c = 0; c < n; ++c) {
            RC_TO_IDX[r * 10 + c] = idx;
            IDX_TO_R[idx] = r;
            IDX_TO_C[idx] = c;
            ++idx;
        }
    }

    // 预计算 NEIGHBOR_MASKS
    for (int i = 0; i < TOTAL_CELLS; ++i) {
        int r = IDX_TO_R[i], c = IDX_TO_C[i];
        CellMask nb_mask;
        auto add_neighbor = [&nb_mask](int nr, int nc) {
            int ni = rc_to_idx(nr, nc);
            if (ni >= 0) nb_mask.set(ni);
        };
        if (r % 2 == 0) {
            add_neighbor(r - 1, c);
            add_neighbor(r - 1, c + 1);
            add_neighbor(r, c - 1);
            add_neighbor(r, c + 1);
            add_neighbor(r + 1, c);
            add_neighbor(r + 1, c + 1);
        } else {
            add_neighbor(r - 1, c - 1);
            add_neighbor(r - 1, c);
            add_neighbor(r, c - 1);
            add_neighbor(r, c + 1);
            add_neighbor(r + 1, c - 1);
            add_neighbor(r + 1, c);
        }
        NEIGHBOR_MASKS[i] = nb_mask;
    }

    // 预计算 BOUNDARY_MASK
    BOUNDARY_MASK.reset();
    for (int i = 0; i < TOTAL_CELLS; ++i) {
        int r = IDX_TO_R[i], c = IDX_TO_C[i];
        int n = ROWS_LAYOUT[r];
        if (r == 0 || r == NUM_ROWS - 1 || c == 0 || c == n - 1)
            BOUNDARY_MASK.set(i);
    }
}

int rc_to_idx(int r, int c) {
    if (r < 0 || r >= NUM_ROWS) return -1;
    if (c < 0 || c >= ROWS_LAYOUT[r]) return -1;
    int v = RC_TO_IDX[r * 10 + c];
    return v;
}

void idx_to_rc(int idx, int& r, int& c) {
    if (idx < 0 || idx >= TOTAL_CELLS) {
        r = c = -1;
        return;
    }
    r = IDX_TO_R[idx];
    c = IDX_TO_C[idx];
}

/**
 * 位运算 BFS：每层通过掩码并行扩展
 */
int bfs_distance(int start_idx, const CellMask& barriers) {
    if (barriers.test(start_idx)) return -1;
    if (BOUNDARY_MASK.test(start_idx)) return 0;

    CellMask visited = barriers;
    visited.set(start_idx);

    CellMask current_frontier;
    current_frontier.set(start_idx);

    int dist = 0;
    while (current_frontier.any()) {
        ++dist;
        CellMask next_frontier;
        for (int i = 0; i < TOTAL_CELLS; ++i) {
            if (!current_frontier.test(i)) continue;
            next_frontier |= NEIGHBOR_MASKS[i];
        }
        next_frontier &= ~visited;
        next_frontier &= ~barriers;

        // 检查是否到达边界
        if ((next_frontier & BOUNDARY_MASK).any())
            return dist;

        if (!next_frontier.any()) return -1;
        visited |= next_frontier;
        current_frontier = next_frontier;
    }
    return -1;
}

/**
 * BFS 路径：需要记录前驱，用传统队列实现（路径重建需要 prev）
 */
std::vector<int> bfs_path(int start_idx, const CellMask& barriers) {
    if (barriers.test(start_idx)) return {};
    if (BOUNDARY_MASK.test(start_idx)) return {start_idx};

    std::array<int, TOTAL_CELLS> prev;
    prev.fill(-1);

    CellMask visited = barriers;
    visited.set(start_idx);

    std::queue<int> q;
    q.push(start_idx);
    int target = -1;

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        CellMask nb = NEIGHBOR_MASKS[u] & ~visited & ~barriers;
        for (int v = 0; v < TOTAL_CELLS; ++v) {
            if (!nb.test(v)) continue;
            prev[v] = u;
            if (BOUNDARY_MASK.test(v)) {
                target = v;
                goto found;
            }
            visited.set(v);
            q.push(v);
        }
    }
    return {};

found:
    std::vector<int> path;
    int cur = target;
    while (cur != start_idx) {
        path.push_back(cur);
        cur = prev[cur];
    }
    path.push_back(start_idx);
    std::reverse(path.begin(), path.end());
    return path;
}

int count_reachable(int idx, const CellMask& barriers) {
    CellMask visited = barriers;
    visited.set(idx);

    CellMask current;
    current.set(idx);
    int count = 1;

    while (current.any()) {
        CellMask next;
        for (int i = 0; i < TOTAL_CELLS; ++i) {
            if (!current.test(i)) continue;
            next |= NEIGHBOR_MASKS[i];
        }
        next &= ~visited;
        if (!next.any()) break;
        visited |= next;
        count += static_cast<int>(next.count());
        current = next;
    }
    return count;
}

int open_neighbor_count(int idx, const CellMask& barriers) {
    CellMask open_nb = NEIGHBOR_MASKS[idx] & ~barriers;
    return static_cast<int>(open_nb.count());
}

double calculate_score(int monster_idx, const CellMask& barriers, double weight_trap) {
    (void)weight_trap;  // 预留
    int d = bfs_distance(monster_idx, barriers);
    int reach = count_reachable(monster_idx, barriers);
    int onb = open_neighbor_count(monster_idx, barriers);

    if (d < 0)
        return 50000.0 + reach * 10.0;

    double score = d * 200.0;
    if (d > 3)
        score += reach * 5.0;
    else
        score -= reach * 2.0;
    score += (6 - onb) * 20.0;  // 奖励通道形成（开放邻居少=形成通道）
    return score;
}

CellMask mask_from_indices(const std::vector<int>& indices) {
    CellMask m;
    for (int i : indices)
        if (i >= 0 && i < TOTAL_CELLS)
            m.set(i);
    return m;
}

}  // namespace hex_engine
