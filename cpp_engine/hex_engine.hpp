/**
 * hex_engine.hpp - 六边形网格寻路 C++ 核心引擎
 * 位掩码 BFS、预计算邻居表、高性能评分
 * C++17, std::bitset<70>
 */

#pragma once

#include <array>
#include <bitset>
#include <cstdint>
#include <cstddef>
#include <vector>

namespace hex_engine {

// 棋盘常量：8-9-8-9-8-9-8-9 交错六边形
constexpr int ROWS_LAYOUT[8] = {8, 9, 8, 9, 8, 9, 8, 9};
constexpr int NUM_ROWS = 8;
constexpr int TOTAL_CELLS = 68;

// 位掩码类型：70 位（68 格 + 余量）
using CellMask = std::bitset<70>;

// 预计算表（init_backend 时填充）
extern std::array<CellMask, TOTAL_CELLS> NEIGHBOR_MASKS;
extern CellMask BOUNDARY_MASK;

// (r,c) -> idx 映射表
extern std::array<int, 8 * 10> RC_TO_IDX;  // 最大 8*10 足够

/**
 * 初始化后端：建立坐标映射，预计算 NEIGHBOR_MASKS 和 BOUNDARY_MASK
 */
void init_backend();

/**
 * (r, c) -> 平坦索引 [0, TOTAL_CELLS)
 * 无效坐标返回 -1
 */
int rc_to_idx(int r, int c);

/**
 * idx -> (r, c)，通过输出参数
 */
void idx_to_rc(int idx, int& r, int& c);

/**
 * 极速位运算 BFS：从 start_idx 到边界的最短步数
 * @return 步数，无法到达返回 -1
 */
int bfs_distance(int start_idx, const CellMask& barriers);

/**
 * BFS 路径：从 start 到边缘的步数最少路径
 * @return 路径索引列表（含 start），空表示无法到达
 */
std::vector<int> bfs_path(int start_idx, const CellMask& barriers);

/**
 * 可达区域计数
 */
int count_reachable(int idx, const CellMask& barriers);

/**
 * 开放邻居数量（未被障碍阻挡）
 */
int open_neighbor_count(int idx, const CellMask& barriers);

/**
 * 牧羊人策略评分
 * 高权重奖励 BFS 距离变长，奖励通道形成（开放邻居=2）
 */
double calculate_score(int monster_idx, const CellMask& barriers, double weight_trap = 1.0);

/**
 * 从索引列表构建 CellMask
 */
CellMask mask_from_indices(const std::vector<int>& indices);

}  // namespace hex_engine
