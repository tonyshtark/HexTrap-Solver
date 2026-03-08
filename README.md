# HexTrap Solver

一个基于 Python Tkinter 的六边形棋盘小游戏/求解器。支持单怪和双怪模式，使用多进程 Beam Search 自动寻找最优围困方案。

## 功能介绍

### 棋盘操作
- **左键**：放置永久障碍（棕色）
- **右键 1/2/3**：放置临时障碍（橙色，持续 1/2/3 回合后消失）
- **中键**：放置夹子（紫色，小怪踩上后停顿一回合）
- **左键拖拽小怪**：移动小怪起始位置（1号/2号，按 1/2 切换）
- **按 E**：指定第一步必须放置的格子（绿色高亮）

### 求解与回放
- **开始**：调用多进程 Beam Search 求解器，寻找用最少砖块困住小怪的方案
- **下一步**：逐步回放求解结果
- **自动播放**：自动逐步回放
- **重新开始**：从头重播当前方案（不重新求解）
- **全部重置**：回到编辑状态，清空求解结果

### 双怪模式
配置文件 `hex_grid_config.json` 中设置 `"two_monsters": true` 启用。两只小怪交替行动，目标是同时困住两只。

## 如何运行

### 纯 Python（推荐）

需要 Python 3.10+：

```bash
python hex_grid_gui.py
```

### 可选：C++ 加速引擎

`cpp_engine/` 目录提供了基于 pybind11 的 C++ 加速模块，可显著提升大规模搜索速度。

```bash
cd cpp_engine
pip install -r requirements.txt
pip install -e .
```

安装后程序会自动检测并启用 C++ 引擎（标题栏显示 `[C++]`）。

## 文件说明

| 文件 | 说明 |
|------|------|
| `hex_grid.py` | 核心逻辑：BFS 寻路、Beam Search 求解器、多进程调度 |
| `hex_grid_gui.py` | Tkinter 图形界面：编辑、求解、回放 |
| `hex_grid_config.json` | 用户配置（双怪模式、最大放砖数等） |
| `cpp_engine/` | 可选 C++ 加速模块（pybind11） |

## 依赖

- Python 3.10+
- tkinter（标准库，通常随 Python 附带）
- C++ 引擎（可选）：`pybind11`、`cmake`、C++17 编译器
