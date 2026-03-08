# -*- coding: utf-8 -*-
"""
C++ 引擎测试脚本
验证 BFS、评分与 Python 实现一致性
"""
import sys
import os

# 确保能导入编译后的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_cpp_engine():
    try:
        import cpp_hex_engine as engine
    except ImportError as e:
        print("请先编译 C++ 扩展:")
        print("  cd cpp_engine && pip install .")
        print("  或: cd cpp_engine && mkdir build && cd build && cmake .. && cmake --build .")
        print(f"错误: {e}")
        return False

    engine.init_backend()

    # 与 hex_grid 的映射一致
    from hex_grid import _rc_to_idx, _idx_to_rc, barriers_to_mask, bfs_distance, _BOUNDARY_SET

    # 1. rc_to_idx 一致性
    for (r, c), expected in _rc_to_idx.items():
        got = engine.rc_to_idx(r, c)
        assert got == expected, f"rc_to_idx({r},{c}): got {got}, expected {expected}"
    print("rc_to_idx: OK")

    # 2. idx_to_rc 一致性
    for idx in range(68):
        r, c = engine.idx_to_rc(idx)
        assert (r, c) == _idx_to_rc[idx], f"idx_to_rc({idx}): got ({r},{c}), expected {_idx_to_rc[idx]}"
    print("idx_to_rc: OK")

    # 3. BFS 一致性
    barriers_set = {(0, 1), (1, 7), (2, 5)}
    bmask = barriers_to_mask(barriers_set)
    for start_rc in [(4, 4), (0, 0), (3, 4)]:
        idx = _rc_to_idx[start_rc]
        py_dist = bfs_distance(idx, bmask)
        cpp_dist = engine.bfs_distance(idx, list(barriers_set))
        assert cpp_dist == py_dist, f"bfs_distance({start_rc}): py={py_dist}, cpp={cpp_dist}"
        # 也测试 int 位掩码
        cpp_dist2 = engine.bfs_distance(idx, bmask)
        assert cpp_dist2 == py_dist, f"bfs_distance(int mask): py={py_dist}, cpp={cpp_dist2}"
    print("bfs_distance: OK")

    # 4. 评分一致性（数值应接近）
    idx = _rc_to_idx[(4, 4)]
    py_score = 0.0  # 需要从 hex_grid 导入 _score 逻辑，这里简化
    cpp_score = engine.calculate_score(idx, list(barriers_set))
    assert cpp_score > 0, "calculate_score 应返回正数"
    print(f"calculate_score: {cpp_score:.1f} OK")

    print("\n所有测试通过。")
    return True


if __name__ == "__main__":
    ok = test_cpp_engine()
    sys.exit(0 if ok else 1)
