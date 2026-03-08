# -*- coding: utf-8 -*-
"""
六边形棋盘图形界面。
- 左键：永久障碍（棕色）
- 右键：临时障碍，按1次1回合、再按+1、再按+1、再按移除（橙色）
- 中键：夹子，小怪踩上停一回合（紫色）
- 小怪：鼠标移到格子上，按 1 生成小怪1，按 2 生成小怪2（再次按可移除）
- 无初始障碍、无初始小怪，完全由用户手动添加
"""

from __future__ import annotations

import json
import math
import os
import sys
import tkinter as tk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hex_grid_config.json")
from tkinter import ttk
from typing import Any

HEX_R = 22
HEX_H = math.sqrt(3) * HEX_R
OFFSET_X_EVEN = 0.75 * HEX_R
PADDING = 36


def row_col_to_xy(r: int, c: int) -> tuple[float, float]:
    if r % 2 == 0:
        x = OFFSET_X_EVEN + c * (1.5 * HEX_R)
    else:
        x = c * (1.5 * HEX_R)
    return (x + PADDING, r * HEX_H + PADDING)


def hex_vertices(cx: float, cy: float) -> list[tuple[float, float]]:
    out = []
    for i in range(6):
        angle = math.pi / 6 + i * (math.pi / 3)
        out.append((cx + HEX_R * math.cos(angle), cy + HEX_R * math.sin(angle)))
    return out


def _point_in_hex(px: float, py: float, cx: float, cy: float) -> bool:
    return (px - cx) ** 2 + (py - cy) ** 2 <= HEX_R * HEX_R


# ── 颜色 ──
C_EMPTY       = "#2d2d44"
C_EMPTY_O     = "#4a4a6a"
C_BARRIER     = "#5c4033"
C_BARRIER_O   = "#8b6914"
C_MONSTER     = "#c41e3a"
C_MONSTER_O   = "#ff6b6b"
C_MONSTER2    = "#c41e3a"   # 第二只小怪同色，用数字区分
C_MONSTER2_O  = "#ff6b6b"
C_PLACED      = "#1565c0"
C_PLACED_O    = "#42a5f5"
C_TEMP        = "#e65100"
C_TEMP_O      = "#ff9800"
C_TRAP        = "#7b1fa2"
C_TRAP_O      = "#ce93d8"
C_STUNNED     = "#ff6f00"


def run_gui_with_solver(
    initial_barriers: set[tuple[int, int]] | None = None,
    max_placements: int = 200,  # 障碍无上限，仅作求解搜索深度
) -> None:
    """无初始障碍、无初始小怪。鼠标移到格子上按 1/2 放置小怪。"""
    from hex_grid import (
        ROWS_LAYOUT, NUM_ROWS,
        get_monster_next_step, is_boundary,
        is_using_cpp_engine,
        solve_max_steps, solve_max_steps_two,
    )

    _engine_tag = " [C++引擎]" if is_using_cpp_engine() else " [Python]"

    if initial_barriers is None:
        initial_barriers = set()

    root = tk.Tk()
    root.title(f"六边形棋盘{_engine_tag}")
    root.resizable(False, False)

    canvas_w = int(9 * 1.5 * HEX_R + OFFSET_X_EVEN + 2 * PADDING)
    canvas_h = int(NUM_ROWS * HEX_H + 2 * PADDING)
    canvas = tk.Canvas(root, width=canvas_w, height=canvas_h, bg="#1a1a2e")
    canvas.pack(pady=6)

    # ── 状态 ──
    barriers: set[tuple[int, int]] = set(initial_barriers)
    temp_barriers: dict[tuple[int, int], int] = {}
    traps: set[tuple[int, int]] = set()
    monster_positions: list[tuple[int, int] | None] = [None, None]  # 小怪1、小怪2，初始无
    hovered_cell: tuple[int, int] | None = None
    move_count = 0
    placement_index = 0
    current_turn = 0  # 0=小怪1, 1=小怪2
    monster_stunned: int | None = None  # 0 或 1 表示哪只眩晕，None 表示无

    placed_cells: dict[tuple[int, int], int] = {}
    first_placement_cell: tuple[int, int] | None = None

    state: dict[str, Any] = {
        "placements": [],
        "total_steps": 0,
        "status_text": f"左键:障碍  右键:临时  中键:夹子  移格按1/2:小怪  按E:第一步  →「开始」{_engine_tag}",
        "editing": True,
        "game_over": False,
    }

    cell_centers: dict[tuple[int, int], tuple[float, float]] = {}
    for r in range(NUM_ROWS):
        for c in range(ROWS_LAYOUT[r]):
            cell_centers[(r, c)] = row_col_to_xy(r, c)

    def load_config() -> dict | None:
        if not os.path.isfile(CONFIG_PATH):
            return None
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def apply_config(cfg: dict) -> None:
        barriers.clear()
        barriers.update(tuple(x) for x in cfg.get("barriers", []))
        temp_barriers.clear()
        for k, v in cfg.get("temp_barriers", {}).items():
            r, c = json.loads(k)
            temp_barriers[(r, c)] = v
        traps.clear()
        traps.update(tuple(x) for x in cfg.get("traps", []))
        monsters = cfg.get("monster_positions", [None, None])
        for i, m in enumerate(monsters[:2]):
            monster_positions[i] = tuple(m) if m is not None else None

    def save_config() -> None:
        """只保存运行前的预设（障碍、临时、夹子、小怪）。"""
        if state["editing"]:
            b, t, tr, m = barriers, temp_barriers, traps, monster_positions
        else:
            b = set(initial_barriers) | state.get("snap_barriers", set())
            t = state.get("snap_temp", {})
            tr = state.get("snap_traps", set())
            m = state.get("snap_monsters", [None, None])
        data = {
            "barriers": [list(rc) for rc in b],
            "temp_barriers": {json.dumps(list(k)): v for k, v in t.items()},
            "traps": [list(rc) for rc in tr],
            "monster_positions": [list(p) if p is not None else None for p in m],
        }
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            state["status_text"] = "已保存配置"
        except OSError:
            state["status_text"] = "保存失败"
        draw_grid()

    def clear_all() -> None:
        nonlocal monster_positions, move_count, placement_index, current_turn, monster_stunned, first_placement_cell
        barriers.clear()
        barriers.update(initial_barriers)
        temp_barriers.clear()
        traps.clear()
        monster_positions[:] = [None, None]
        placed_cells.clear()
        state["placements"] = []
        state["total_steps"] = 0
        state["game_over"] = False
        state["editing"] = True
        state["status_text"] = "已清空所有预设"
        first_placement_cell = None
        monster_stunned = None
        current_turn = 0
        move_count = 0
        placement_index = 0
        btn_start.config(state=tk.NORMAL)
        btn_next.config(state=tk.DISABLED)
        btn_play.config(state=tk.DISABLED)
        root.title(f"六边形棋盘{_engine_tag}")
        draw_grid()

    def _monster_list() -> list[tuple[int, int]]:
        return [p for p in monster_positions if p is not None]

    def all_monster_cells() -> set[tuple[int, int]]:
        return set(_monster_list())

    def _two_monsters() -> bool:
        return len(_monster_list()) == 2

    # ── 绘制 ──
    def draw_grid() -> None:
        canvas.delete("all")
        monster_set = all_monster_cells()
        for (r, c), (cx, cy) in cell_centers.items():
            pts = hex_vertices(cx, cy)
            flat = [x for p in pts for x in p]
            cell = (r, c)
            if cell in placed_cells:
                fill, outline = C_PLACED, C_PLACED_O
            elif cell in temp_barriers:
                fill, outline = C_TEMP, C_TEMP_O
            elif cell in barriers:
                fill, outline = C_BARRIER, C_BARRIER_O
            elif cell in traps:
                fill, outline = C_TRAP, C_TRAP_O
            elif cell == first_placement_cell:
                fill, outline = "#2d4a2d", "#4caf50"
            elif cell in monster_set:
                idx = next((i for i, p in enumerate(monster_positions) if p == cell), 0)
                is_stunned = monster_stunned == idx
                fill = C_STUNNED if is_stunned else C_MONSTER
                outline = C_MONSTER_O
            else:
                fill, outline = C_EMPTY, C_EMPTY_O
            canvas.create_polygon(flat, fill=fill, outline=outline, width=2)

            if cell in placed_cells:
                canvas.create_text(cx, cy, text=str(placed_cells[cell]),
                                   fill="#fff", font=("Consolas", 10, "bold"))
            elif cell in temp_barriers:
                canvas.create_text(cx, cy, text=str(temp_barriers[cell]),
                                   fill="#fff", font=("Consolas", 11, "bold"))
            elif cell in traps:
                canvas.create_text(cx, cy, text="×",
                                   fill="#e1bee7", font=("Consolas", 14, "bold"))
            elif cell in monster_set:
                idx = next((i for i, p in enumerate(monster_positions) if p == cell), 0)
                canvas.create_text(cx, cy, text=str(idx + 1),
                                   fill="#fff", font=("Consolas", 12, "bold"))
            elif cell == first_placement_cell:
                canvas.create_text(cx, cy, text="E",
                                   fill="#4caf50", font=("Consolas", 11, "bold"))
            else:
                canvas.create_text(cx, cy, text=f"{r},{c}",
                                   fill="#888", font=("Consolas", 7))

        total_bricks = len(state["placements"])
        turn_info = f"  轮到:小怪{current_turn+1}" if _two_monsters() and not state["game_over"] else ""
        if state.get("solve_failed") and not state["status_text"]:
            # 求解失败时回放：隐藏误导性的总数，始终标注[未能困住]
            title = (
                f"小怪走了: {move_count} 步  |  已放砖: {placement_index}  [未能困住]"
                f"  |  临时: {len(temp_barriers)}  夹子: {len(traps)}{turn_info}"
            )
        else:
            title = state["status_text"] or (
                f"小怪走了: {move_count} 步  |  已放砖: {placement_index}/{total_bricks}"
                f"  |  临时: {len(temp_barriers)}  夹子: {len(traps)}{turn_info}"
            )
        canvas.create_text(canvas_w // 2, 14, text=title,
                           fill="#eee", font=("Consolas", 10), tags=("title",))

    # ── 编辑：左键/右键/中键 ──
    def _find_cell(event: tk.Event) -> tuple[int, int] | None:  # type: ignore[type-arg]
        px, py = event.x, event.y
        for (r, c), (cx, cy) in cell_centers.items():
            if _point_in_hex(px, py, cx, cy):
                return (r, c)
        return None

    def _is_monster_cell(cell: tuple[int, int]) -> bool:
        return cell in all_monster_cells()

    def on_left_click(event: tk.Event) -> None:  # type: ignore[type-arg]
        if not state["editing"]:
            return
        cell = _find_cell(event)
        if cell is None or _is_monster_cell(cell):
            return
        temp_barriers.pop(cell, None)
        traps.discard(cell)
        if cell in barriers:
            barriers.discard(cell)
        else:
            barriers.add(cell)
        draw_grid()

    def on_right_click(event: tk.Event) -> None:  # type: ignore[type-arg]
        if not state["editing"]:
            return
        cell = _find_cell(event)
        if cell is None or _is_monster_cell(cell):
            return
        barriers.discard(cell)
        traps.discard(cell)
        if cell not in temp_barriers:
            temp_barriers[cell] = 1
        elif temp_barriers[cell] < 3:
            temp_barriers[cell] += 1
        else:
            del temp_barriers[cell]
        draw_grid()

    def on_middle_click(event: tk.Event) -> None:  # type: ignore[type-arg]
        if not state["editing"]:
            return
        cell = _find_cell(event)
        if cell is None or _is_monster_cell(cell):
            return
        barriers.discard(cell)
        temp_barriers.pop(cell, None)
        if cell in traps:
            traps.discard(cell)
        else:
            traps.add(cell)
        draw_grid()

    canvas.bind("<Button-1>", on_left_click)
    canvas.bind("<Button-3>", on_right_click)
    canvas.bind("<Button-2>", on_middle_click)

    def on_motion(event: tk.Event) -> None:  # type: ignore[type-arg]
        nonlocal hovered_cell
        hovered_cell = _find_cell(event)

    def on_key_monster(event: tk.Event, idx: int) -> None:  # type: ignore[type-arg]
        if not state["editing"] or state["game_over"] or hovered_cell is None:
            return
        cell = hovered_cell
        if cell in barriers or cell in temp_barriers or cell in traps:
            return
        other_idx = 1 - idx
        if monster_positions[idx] == cell:
            monster_positions[idx] = None
        else:
            monster_positions[idx] = cell
            if monster_positions[other_idx] == cell:
                monster_positions[other_idx] = None
        draw_grid()

    def on_key_first_placement(event: tk.Event) -> None:  # type: ignore[type-arg]
        if not state["editing"] or state["game_over"] or hovered_cell is None:
            return
        cell = hovered_cell
        if cell in barriers or cell in temp_barriers or cell in traps or cell in all_monster_cells():
            return
        nonlocal first_placement_cell
        if first_placement_cell == cell:
            first_placement_cell = None
        else:
            first_placement_cell = cell
        draw_grid()

    canvas.bind("<Motion>", on_motion)
    root.bind("<KeyPress-1>", lambda e: on_key_monster(e, 0))
    root.bind("<KeyPress-2>", lambda e: on_key_monster(e, 1))
    root.bind("<KeyPress-e>", on_key_first_placement)
    root.bind("<KeyPress-E>", on_key_first_placement)
    root.focus_set()

    def all_blocking() -> set[tuple[int, int]]:
        return barriers | set(temp_barriers.keys())

    # ── 回放控制 ──
    def do_next() -> None:
        nonlocal monster_positions, move_count, placement_index, current_turn, monster_stunned
        if state["game_over"]:
            return

        blocking = all_blocking()
        pl = state["placements"]

        # 眩晕回合：仍放置一块砖、临时障碍衰减，该怪不移动，下一回合轮到另一只
        if monster_stunned is not None and monster_stunned == current_turn:
            if placement_index < len(pl):
                cell = pl[placement_index]
                placed_cells[cell] = placement_index + 1
                barriers.add(cell)
                placement_index += 1
            expired = []
            for c in list(temp_barriers):
                temp_barriers[c] -= 1
                if temp_barriers[c] <= 0:
                    expired.append(c)
            for c in expired:
                del temp_barriers[c]
            monster_stunned = None
            if _two_monsters():
                current_turn = (current_turn + 1) % 2
            _check_both_trapped(all_blocking())
            draw_grid()
            return

        # 1) 玩家放砖（有砖可放时放一块，无砖时跳过）
        if placement_index < len(pl):
            cell = pl[placement_index]
            placed_cells[cell] = placement_index + 1
            barriers.add(cell)
            placement_index += 1

        # 2) 临时障碍衰减
        expired = []
        for c in list(temp_barriers):
            temp_barriers[c] -= 1
            if temp_barriers[c] <= 0:
                expired.append(c)
        for c in expired:
            del temp_barriers[c]

        # 3) 当前小怪寻路（用放置+衰减后的障碍状态）
        blocking = all_blocking()
        cur_pos = monster_positions[current_turn]  # type: ignore[index]
        nxt = get_monster_next_step(cur_pos, blocking)
        if nxt is None:
            if is_boundary(*cur_pos):
                state["game_over"] = True
                state["status_text"] = f"小怪{current_turn+1}逃脱了！放了 {placement_index} 块砖，走了 {move_count} 步"
            else:
                if _two_monsters():
                    _check_both_trapped(blocking)
                    if not state["game_over"]:
                        current_turn = (current_turn + 1) % 2
                else:
                    state["game_over"] = True
                    state["status_text"] = f"小怪被困住了！放了 {placement_index} 块砖，走了 {move_count} 步"
            draw_grid()
            return

        # 4) 当前小怪走一步
        monster_positions[current_turn] = nxt
        move_count += 1

        # 5) 检查夹子
        if nxt in traps:
            traps.discard(nxt)
            monster_stunned = current_turn

        # 6) 走完后检查
        if is_boundary(*nxt):
            state["game_over"] = True
            state["status_text"] = f"小怪{current_turn+1}逃脱了！放了 {placement_index} 块砖，走了 {move_count} 步"
        else:
            _check_both_trapped(all_blocking())

        if _two_monsters():
            current_turn = (current_turn + 1) % 2

        draw_grid()

    def _check_both_trapped(blocking: set[tuple[int, int]]) -> None:
        ml = _monster_list()
        both = all(get_monster_next_step(p, blocking) is None for p in ml)
        if both:
            state["game_over"] = True
            state["status_text"] = f"小怪全部被困住！放了 {placement_index} 块砖，走了 {move_count} 步"

    def auto_play() -> None:
        def tick() -> None:
            do_next()
            if state["game_over"]:
                return
            root.after(400, tick)
        tick()

    def _restore_to_edit_state() -> None:
        nonlocal monster_positions, move_count, placement_index, current_turn, monster_stunned
        state["game_over"] = False
        state["placements"] = []
        state["total_steps"] = 0
        state["solve_failed"] = False
        state["editing"] = True
        monster_stunned = None
        current_turn = 0
        monster_positions[:] = list(state.get("snap_monsters", [None, None]))
        move_count = 0
        placement_index = 0
        placed_cells.clear()
        barriers.clear()
        barriers.update(initial_barriers)
        for rc in state.get("snap_barriers", set()):
            barriers.add(rc)
        temp_barriers.clear()
        temp_barriers.update(state.get("snap_temp", {}))
        traps.clear()
        traps.update(state.get("snap_traps", set()))
        btn_start.config(state=tk.NORMAL)
        btn_next.config(state=tk.DISABLED)
        btn_play.config(state=tk.DISABLED)
        root.title(f"六边形棋盘{_engine_tag}")

    def reset() -> None:
        _restore_to_edit_state()
        state["status_text"] = "已恢复到开始前的状态"
        draw_grid()

    def _init_replay_state() -> None:
        """重置到回放起始状态（与 restart_replay 一致）。"""
        nonlocal monster_positions, move_count, placement_index, current_turn, monster_stunned
        state["game_over"] = False
        monster_stunned = None
        move_count = 0
        placement_index = 0
        snap = list(state.get("snap_monsters", [None, None]))
        ml = [p for p in snap if p is not None]
        if len(ml) == 1:
            # 单怪归一化到 slot 0，与求解器一致
            monster_positions[:] = [ml[0], None]
        else:
            monster_positions[:] = snap
        current_turn = 0
        placed_cells.clear()
        barriers.clear()
        barriers.update(initial_barriers)
        barriers.update(state.get("snap_barriers", set()))
        temp_barriers.clear()
        temp_barriers.update(state.get("snap_temp", {}))
        traps.clear()
        traps.update(state.get("snap_traps", set()))

    def restart_replay() -> None:
        """还原搜索结果到起始，可再次播放/下一步（不重新求解）。"""
        if not state["placements"]:
            return
        _init_replay_state()
        state["editing"] = False
        btn_next.config(state=tk.NORMAL)
        btn_play.config(state=tk.NORMAL)
        btn_start.config(state=tk.DISABLED)
        state["status_text"] = None
        draw_grid()

    # ── 按钮 ──
    btn_frame = ttk.Frame(root)
    btn_frame.pack(pady=4)

    def run_solver() -> None:
        ml = _monster_list()
        if not ml:
            state["status_text"] = "请先放置至少一只小怪（鼠标移格上按 1 或 2）"
            draw_grid()
            return

        state["editing"] = False
        state["snap_barriers"] = set(barriers) - set(initial_barriers)
        state["snap_temp"] = dict(temp_barriers)
        state["snap_traps"] = set(traps)
        state["snap_monsters"] = list(monster_positions)
        btn_start.config(state=tk.DISABLED)

        use_two = len(ml) == 2
        state["status_text"] = "计算中…"
        draw_grid()
        root.update_idletasks()

        if use_two:
            m1, m2 = ml[0], ml[1]
            steps, pl = solve_max_steps_two(
                (m1, m2),
                set(barriers),
                max_placements=max_placements,
                exclude_cells=set(traps),
                temp_barriers=dict(temp_barriers),
                first_placement=first_placement_cell,
            )
        else:
            steps, pl = solve_max_steps(
                ml[0],
                set(barriers),
                max_placements=max_placements,
                exclude_cells=set(traps),
                temp_barriers=dict(temp_barriers),
                first_placement=first_placement_cell,
            )

        state["placements"][:] = pl
        state["total_steps"] = steps
        state["solve_failed"] = (steps < 0)
        _init_replay_state()
        btn_next.config(state=tk.NORMAL)
        btn_play.config(state=tk.NORMAL)
        if steps >= 0:
            bricks = len(pl)
            state["status_text"] = None
            root.title(f"六边形棋盘{_engine_tag} - 困住！放 {bricks} 砖")
        else:
            state["status_text"] = "无法困住，可回放查看"
            root.title(f"六边形棋盘{_engine_tag} - 无法困住（可回放）")
        draw_grid()

    btn_start = ttk.Button(btn_frame, text="开始", command=run_solver)
    btn_start.pack(side=tk.LEFT, padx=4)
    btn_next = ttk.Button(btn_frame, text="下一步", command=do_next, state=tk.DISABLED)
    btn_next.pack(side=tk.LEFT, padx=4)
    btn_play = ttk.Button(btn_frame, text="自动播放", command=auto_play, state=tk.DISABLED)
    btn_play.pack(side=tk.LEFT, padx=4)
    ttk.Button(btn_frame, text="重新开始", command=restart_replay).pack(side=tk.LEFT, padx=4)
    ttk.Button(btn_frame, text="全部重置", command=reset).pack(side=tk.LEFT, padx=4)
    ttk.Button(btn_frame, text="保存", command=save_config).pack(side=tk.LEFT, padx=4)
    ttk.Button(btn_frame, text="清空", command=clear_all).pack(side=tk.LEFT, padx=4)

    # ── 图例 ──
    legend = ttk.Frame(root)
    legend.pack(pady=2)
    for color, label in [
        (C_BARRIER, "永久障碍(左键)"),
        (C_TEMP, "临时(右键1/2/3/移除)"),
        (C_TRAP, "夹子(中键)"),
        (C_PLACED, "求解放置"),
        (C_MONSTER, "小怪(移格按1/2)"),
        ("#4caf50", "第一步(按E)"),
    ]:
        f = tk.Frame(legend)
        f.pack(side=tk.LEFT, padx=6)
        tk.Canvas(f, width=12, height=12, bg=color, highlightthickness=0).pack(side=tk.LEFT, padx=2)
        tk.Label(f, text=label, font=("Consolas", 8), fg="#ccc", bg="#1a1a2e").pack(side=tk.LEFT)

    cfg = load_config()
    if cfg:
        apply_config(cfg)
    draw_grid()
    root.mainloop()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    run_gui_with_solver(
        initial_barriers=set(),
        max_placements=50,
    )
