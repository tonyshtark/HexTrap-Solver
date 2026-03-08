# 六边形网格 C++ 高性能引擎

将 BFS、评分等核心计算从 Python 迁移到 C++，通过 pybind11 供 Python 调用，目标 50–100 倍加速。

## 构建

### 方式一：pip（推荐）

```bash
cd cpp_engine
pip install .
```

### 方式二：CMake

```bash
cd cpp_engine
mkdir build && cd build
cmake ..
cmake --build .
# 生成的 cpp_hex_engine.pyd (Windows) 或 .so (Linux) 在 build/ 下
# 复制到项目根目录或将 build 加入 PYTHONPATH
```

## 依赖

- Python 3.8+
- pybind11: `pip install pybind11`
- C++17 编译器（MSVC / GCC / Clang）

## Python 用法

```python
import cpp_hex_engine as engine

# 1. 初始化（模块加载后调用一次）
engine.init_backend()

# 2. 坐标转换
idx = engine.rc_to_idx(4, 4)       # (r,c) -> 0-67
r, c = engine.idx_to_rc(idx)      # idx -> (r,c)

# 3. BFS 距离（barriers 可为 set[(r,c)]、list[int]、int 位掩码）
barriers = {(0,1), (1,7), (2,5)}
dist = engine.bfs_distance(idx, barriers)

# 4. BFS 路径
path = engine.bfs_path(idx, barriers)

# 5. 评分
score = engine.calculate_score(idx, barriers, weight_trap=1.0)
```

## 测试

```bash
cd cpp_engine
python test_cpp_engine.py
```

## 文件结构

- `hex_engine.hpp` - 头文件，bitset 别名、预计算表声明
- `hex_engine.cpp` - 位运算 BFS、预计算、评分实现
- `bindings.cpp` - pybind11 导出
- `CMakeLists.txt` - CMake 构建
- `setup.py` - setuptools 构建
