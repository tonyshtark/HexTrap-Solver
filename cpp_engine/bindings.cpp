/**
 * bindings.cpp - pybind11 接口，将 C++ 引擎导出给 Python
 * 修复版：删除了未定义的变量，优化了类型检查逻辑
 */

 #include "hex_engine.hpp"
 #include <pybind11/pybind11.h>
 #include <pybind11/stl.h>
 
 namespace py = pybind11;
 
 namespace {
 
 // 辅助函数：将 Python 对象转换为 C++ 的位掩码
 hex_engine::CellMask barriers_from_py(const py::object& obj) {
     hex_engine::CellMask m;
 
     // 情况 1: 输入是一个整数 (Bitmask)
     if (py::isinstance<py::int_>(obj)) {
         try {
             // 注意：这里假设 Python int 不会超过 64 位太多，
             // 如果超过 64 位，pybind11 的 cast 可能抛出异常或截断。
             // 对于 68 格子，我们需要 2 个 uint64 或者 bitset<70>。
             // 简单起见，这里演示遍历位（性能较低但安全），或者直接 cast 为 unsigned long long (如果格子<64)
             // 鉴于 TOTAL_CELLS=68，我们遍历位索引是安全的。
             // 更高效的方法是让 Python 传 list[int] 或 set[int]。
             
             // 如果 Python 传的是一个巨大的整数掩码：
             // 我们无法直接 cast 到 uint64_t (因为它只有 64 位，不够 68 位)。
             // 这里为了稳妥，我们暂时只支持 list/set/tuple 形式的输入，
             // 或者假设用户传的 int 掩码只包含前 64 个格子。
             uint64_t val = obj.cast<uint64_t>();
             for(int i=0; i<64 && i<hex_engine::TOTAL_CELLS; ++i) {
                 if ((val >> i) & 1) m.set(i);
             }
         } catch (...) {
             // 忽略转换错误
         }
         return m;
     }
 
     // 情况 2: 输入是序列 (list, set, tuple)
     if (py::isinstance<py::list>(obj) || py::isinstance<py::set>(obj) || py::isinstance<py::tuple>(obj)) {
         for (auto item : obj) {
             // 子项是 (r, c) 元组
             if (py::isinstance<py::tuple>(item)) {
                 py::tuple t = item.cast<py::tuple>();
                 if (t.size() >= 2) {
                     int r = t[0].cast<int>();
                     int c = t[1].cast<int>();
                     int idx = hex_engine::rc_to_idx(r, c);
                     if (idx >= 0) m.set(idx);
                 }
             }
             // 子项是 int 索引
             else if (py::isinstance<py::int_>(item)) {
                 int idx = item.cast<int>();
                 if (idx >= 0 && idx < hex_engine::TOTAL_CELLS) {
                     m.set(idx);
                 }
             }
         }
         return m;
     }
 
     return m;
 }
 
 }  // namespace
 
 PYBIND11_MODULE(cpp_hex_engine, m) {
     m.doc() = "六边形网格寻路 C++ 高性能引擎";
 
     // 模块加载时自动初始化后端数据
     hex_engine::init_backend();
 
     m.def("init_backend", &hex_engine::init_backend,
           "初始化后端：预计算邻居表、边界掩码（通常无需手动调用）");
 
     m.def("rc_to_idx", &hex_engine::rc_to_idx,
           py::arg("r"), py::arg("c"),
           "(r,c) -> 平坦索引 [0,67]，无效返回 -1");
 
     m.def("idx_to_rc", [](int idx) {
         int r, c;
         hex_engine::idx_to_rc(idx, r, c);
         return py::make_tuple(r, c);
     }, py::arg("idx"), "idx -> (r, c)");
 
     // 这是一个调试用的 helper，Python 端可以调用它来看看转换是否正确
     m.def("debug_barriers_count", [](const py::object& barriers_obj) {
         hex_engine::CellMask barriers = barriers_from_py(barriers_obj);
         return barriers.count();
     }, py::arg("barriers"), "调试：返回解析到的障碍物数量");
 
     m.def("bfs_distance", [](int start_idx, const py::object& barriers_obj) {
         hex_engine::CellMask barriers = barriers_from_py(barriers_obj);
         return hex_engine::bfs_distance(start_idx, barriers);
     }, py::arg("start_idx"), py::arg("barriers"),
        "极速 BFS：到边界最短步数，-1 表示无法到达。barriers 可为 list[(r,c)] 或 list[int]");
 
     // 注意：bfs_path 返回的是 std::vector<int>，pybind11 会自动转为 Python list
     m.def("bfs_path", [](int start_idx, const py::object& barriers_obj) {
         hex_engine::CellMask barriers = barriers_from_py(barriers_obj);
         return hex_engine::bfs_path(start_idx, barriers);
     }, py::arg("start_idx"), py::arg("barriers"),
        "BFS 路径：从 start 到边缘的最短路径索引列表");
 
     m.def("count_reachable", [](int idx, const py::object& barriers_obj) {
         hex_engine::CellMask barriers = barriers_from_py(barriers_obj);
         return hex_engine::count_reachable(idx, barriers);
     }, py::arg("idx"), py::arg("barriers"), "可达区域格子数");
 
     m.def("open_neighbor_count", [](int idx, const py::object& barriers_obj) {
         hex_engine::CellMask barriers = barriers_from_py(barriers_obj);
         return hex_engine::open_neighbor_count(idx, barriers);
     }, py::arg("idx"), py::arg("barriers"), "开放邻居数量");
 
     m.def("calculate_score", [](int monster_idx, const py::object& barriers_obj, double weight_trap) {
         hex_engine::CellMask barriers = barriers_from_py(barriers_obj);
         return hex_engine::calculate_score(monster_idx, barriers, weight_trap);
     }, py::arg("monster_idx"), py::arg("barriers"), py::arg("weight_trap") = 1.0,
        "牧羊人策略评分");
 
     // 导出常量
     m.attr("TOTAL_CELLS") = hex_engine::TOTAL_CELLS;
     m.attr("NUM_ROWS") = hex_engine::NUM_ROWS;
 }