# -*- coding: utf-8 -*-
"""
六边形棋盘图形界面。
- 左键：永久障碍（棕色）
- 右键：临时障碍，耐久 2，每回合 -1，归零消失（橙色，显示耐久数字）
- 中键：夹子，小怪踩上停一回合（紫色，显示 ×）
- 点击「开始」后求解并回放
"""

from __future__ import annotations

import math
import os
import sys
import tkinter as tk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
C_PLACED      = "#1565c0"
C_PLACED_O    = "#42a5f5"
C_TEMP        = "#e65100"   # 临时障碍 - 橙色
C_TEMP_O      = "#ff9800"
C_TRAP        = "#7b1fa2"   # 夹子 - 紫色
C_TRAP_O      = "#ce93d8"
C_STUNNED     = "#ff6f00"   # 小怪被夹住时的颜色


def run_gui_with_solver(
    monster_start: tuple[int, int],
    initial_barriers: set[tuple[int, int]],
    max_placements: int = 40,
) -> None:
    from hex_grid import (
        ROWS_LAYOUT, NUM_ROWS,
        get_monster_next_step, is_boundary,
        solve_max_steps,
    )

    root = tk.Tk()
    root.title("六边形棋盘")
    root.resizable(False, False)

    canvas_w = int(9 * 1.5 * HEX_R + OFFSET_X_EVEN + 2 * PADDING)
    canvas_h = int(NUM_ROWS * HEX_H + 2 * PADDING)
    canvas = tk.Canvas(root, width=canvas_w, height=canvas_h, bg="#1a1a2e")
    canvas.pack(pady=6)

    # ── 状态 ──
    barriers: set[tuple[int, int]] = set(initial_barriers)
    temp_barriers: dict[tuple[int, int], int] = {}   # cell → 剩余耐久
    traps: set[tuple[int, int]] = set()               # 夹子位置
    monster_pos = monster_start
    move_count = 0
    placement_index = 0
    monster_stunned = False  # 小怪是否被夹住（本回合跳过移动）

    placed_cells: dict[tuple[int, int], int] = {}

    state: dict[str, Any] = {
        "placements": [],
        "total_steps": 0,
        "status_text": "左键:障碍  右键:临时障碍(2回合)  中键:夹子  →「开始」",
        "editing": True,
        "game_over": False,
    }

    cell_centers: dict[tuple[int, int], tuple[float, float]] = {}
    for r in range(NUM_ROWS):
        for c in range(ROWS_LAYOUT[r]):
            cell_centers[(r, c)] = row_col_to_xy(r, c)

    # ── 绘制 ──
    def draw_grid() -> None:
        canvas.delete("all")
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
            elif cell == monster_pos:
                fill = C_STUNNED if monster_stunned else C_MONSTER
                outline = C_MONSTER_O
            else:
                fill, outline = C_EMPTY, C_EMPTY_O
            canvas.create_polygon(flat, fill=fill, outline=outline, width=2)

            # 格子上的文字
            if cell in placed_cells:
                canvas.create_text(cx, cy, text=str(placed_cells[cell]),
                                   fill="#fff", font=("Consolas", 10, "bold"))
            elif cell in temp_barriers:
                dur = temp_barriers[cell]
                canvas.create_text(cx, cy, text=str(dur),
                                   fill="#fff", font=("Consolas", 11, "bold"))
            elif cell in traps:
                canvas.create_text(cx, cy, text="×",
                                   fill="#e1bee7", font=("Consolas", 14, "bold"))
            else:
                canvas.create_text(cx, cy, text=f"{r},{c}",
                                   fill="#888", font=("Consolas", 7))

        total_bricks = len(state["placements"])
        title = state["status_text"] or (
            f"小怪走了: {move_count} 步  |  已放砖: {placement_index}/{total_bricks}"
            f"  |  临时障碍: {len(temp_barriers)}  |  夹子: {len(traps)}"
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

    def on_left_click(event: tk.Event) -> None:  # type: ignore[type-arg]
        if not state["editing"]:
            return
        cell = _find_cell(event)
        if cell is None or cell == monster_pos:
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
        if cell is None or cell == monster_pos:
            return
        barriers.discard(cell)
        traps.discard(cell)
        if cell in temp_barriers:
            del temp_barriers[cell]
        else:
            temp_barriers[cell] = 2
        draw_grid()

    def on_middle_click(event: tk.Event) -> None:  # type: ignore[type-arg]
        if not state["editing"]:
            return
        cell = _find_cell(event)
        if cell is None or cell == monster_pos:
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

    # ── 小怪寻路用的障碍集 ──
    def all_blocking() -> set[tuple[int, int]]:
        """编辑模式用全部阻挡；回放模式用永久+当前临时障碍（临时障碍参与寻路，2回合后消失）。"""
        return barriers | set(temp_barriers.keys())

    # ── 回放控制 ──
    def do_next() -> None:
        nonlocal monster_pos, move_count, placement_index, monster_stunned
        if state["game_over"]:
            return

        # 眩晕回合：仍放置一块砖，然后小怪本回合不动
        if monster_stunned:
            pl = state["placements"]
            if placement_index < len(pl):
                cell = pl[placement_index]
                placed_cells[cell] = placement_index + 1
                barriers.add(cell)
                placement_index += 1
            monster_stunned = False
            if placement_index >= len(pl) and get_monster_next_step(monster_pos, all_blocking()) is None:
                state["game_over"] = True
                state["status_text"] = f"小怪被困住了！放了 {placement_index} 块砖，走了 {move_count} 步"
            draw_grid()
            return

        # 1) 玩家放砖
        pl = state["placements"]
        if placement_index < len(pl):
            cell = pl[placement_index]
            placed_cells[cell] = placement_index + 1
            barriers.add(cell)
            placement_index += 1

        # 2) 临时障碍耐久衰减（视觉效果，不影响回放寻路）
        expired = []
        for cell in list(temp_barriers):
            temp_barriers[cell] -= 1
            if temp_barriers[cell] <= 0:
                expired.append(cell)
        for cell in expired:
            del temp_barriers[cell]

        # 3) 检查小怪是否被困（只用永久障碍，与求解器一致）
        blocking = all_blocking()
        nxt = get_monster_next_step(monster_pos, blocking)
        if nxt is None:
            state["game_over"] = True
            if is_boundary(*monster_pos):
                state["status_text"] = f"小怪逃脱了！放了 {placement_index} 块砖，走了 {move_count} 步"
            else:
                state["status_text"] = f"小怪被困住了！放了 {placement_index} 块砖，走了 {move_count} 步"
            draw_grid()
            return

        # 4) 小怪走一步
        monster_pos = nxt
        move_count += 1

        # #region agent log
        import json as _json, time as _time
        _logpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug-ec7353.log")
        with open(_logpath, "a") as _f:
            _f.write(_json.dumps({"sessionId":"ec7353","hypothesisId":"B","location":"gui_do_next","message":"step","data":{"move_count":move_count,"placement_index":placement_index,"monster_pos":list(monster_pos),"is_boundary":is_boundary(*monster_pos),"num_temp":len(temp_barriers),"num_traps":len(traps)},"timestamp":int(_time.time()*1000)})+"\n")
        # #endregion

        # 5) 检查是否踩到夹子
        if monster_pos in traps:
            traps.discard(monster_pos)
            monster_stunned = True

        # 6) 走完后检查
        if is_boundary(*monster_pos):
            state["game_over"] = True
            state["status_text"] = f"小怪逃脱了！放了 {placement_index} 块砖，走了 {move_count} 步"
        elif get_monster_next_step(monster_pos, all_blocking()) is None and placement_index >= len(pl):
            state["game_over"] = True
            state["status_text"] = f"小怪被困住了！放了 {placement_index} 块砖，走了 {move_count} 步"

        # #region agent log
        if state["game_over"]:
            import json as _json2, time as _time2
            _logpath2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug-ec7353.log")
            with open(_logpath2, "a") as _f2:
                _f2.write(_json2.dumps({"sessionId":"ec7353","hypothesisId":"C","location":"gui_game_over","message":"game_ended","data":{"status":state["status_text"],"move_count":move_count,"placement_index":placement_index,"total_placements":len(pl),"solver_steps":state.get("total_steps",-99)},"timestamp":int(_time2.time()*1000)})+"\n")
        # #endregion

        draw_grid()

    def auto_play() -> None:
        def tick() -> None:
            do_next()
            if state["game_over"]:
                return
            root.after(400, tick)
        tick()

    def _restore_to_edit_state() -> None:
        """恢复到「开始」前的编辑状态（用 snapshot 快照）。"""
        nonlocal monster_pos, move_count, placement_index, monster_stunned
        state["game_over"] = False
        state["placements"] = []
        state["total_steps"] = 0
        state["editing"] = True
        monster_stunned = False
        monster_pos = monster_start
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
        root.title("六边形棋盘")

    def reset() -> None:
        """全部重置：恢复到「开始」前的编辑状态。"""
        _restore_to_edit_state()
        state["status_text"] = "已恢复到开始前的状态"
        draw_grid()

    def restart_keep() -> None:
        """重新开始：保留求解器放的砖为永久障碍，重新进入编辑模式。"""
        keep = set(placed_cells.keys())
        _restore_to_edit_state()
        for cell in keep:
            barriers.add(cell)
        # 更新快照，这样下次「全部重置」也包含这些砖
        state["snap_barriers"] = set(barriers) - set(initial_barriers)
        state["status_text"] = "已保留障碍，可继续编辑，再按「开始」"
        draw_grid()

    # ── 按钮 ──
    btn_frame = ttk.Frame(root)
    btn_frame.pack(pady=4)

    def run_solver() -> None:
        state["editing"] = False
        # 保存编辑快照（「全部重置」回到此刻）
        state["snap_barriers"] = set(barriers) - set(initial_barriers)
        state["snap_temp"] = dict(temp_barriers)
        state["snap_traps"] = set(traps)
        btn_start.config(state=tk.DISABLED)

        state["status_text"] = "多进程计算中…"
        draw_grid()
        root.update_idletasks()

        # 求解器：仅永久障碍；临时障碍与夹子由求解器内模拟（临时衰减2回合、夹子不挡寻路且踩到眩晕）
        steps, pl = solve_max_steps(
            monster_start,
            set(barriers),
            max_placements=max_placements,
            exclude_cells=set(traps),
            temp_barriers_cells=set(temp_barriers.keys()),
        )
        state["placements"][:] = pl
        state["total_steps"] = steps
        if steps >= 0:
            bricks = len(pl)
            state["status_text"] = None
            btn_next.config(state=tk.NORMAL)
            btn_play.config(state=tk.NORMAL)
            root.title(f"六边形棋盘 - 困住！放 {bricks} 砖，走 {steps} 步")
        else:
            state["status_text"] = f"无法困住（砖数不足），上限: {max_placements}"
            root.title("六边形棋盘 - 无法困住")
        draw_grid()

    btn_start = ttk.Button(btn_frame, text="开始", command=run_solver)
    btn_start.pack(side=tk.LEFT, padx=4)
    btn_next = ttk.Button(btn_frame, text="下一步", command=do_next, state=tk.DISABLED)
    btn_next.pack(side=tk.LEFT, padx=4)
    btn_play = ttk.Button(btn_frame, text="自动播放", command=auto_play, state=tk.DISABLED)
    btn_play.pack(side=tk.LEFT, padx=4)
    ttk.Button(btn_frame, text="重新开始", command=restart_keep).pack(side=tk.LEFT, padx=4)
    ttk.Button(btn_frame, text="全部重置", command=reset).pack(side=tk.LEFT, padx=4)

    # ── 图例 ──
    legend = ttk.Frame(root)
    legend.pack(pady=2)
    for color, label in [
        (C_BARRIER, "永久障碍(左键)"),
        (C_TEMP, "临时障碍(右键)"),
        (C_TRAP, "夹子(中键)"),
        (C_PLACED, "求解放置"),
        (C_MONSTER, "小怪"),
    ]:
        f = tk.Frame(legend)
        f.pack(side=tk.LEFT, padx=6)
        tk.Canvas(f, width=12, height=12, bg=color, highlightthickness=0).pack(side=tk.LEFT, padx=2)
        tk.Label(f, text=label, font=("Consolas", 8), fg="#ccc", bg="#1a1a2e").pack(side=tk.LEFT)

    draw_grid()
    root.mainloop()


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()

    from hex_grid import DEFAULT_BARRIERS

    run_gui_with_solver(
        monster_start=(4, 4),
        initial_barriers=set(DEFAULT_BARRIERS),
        max_placements=50,
    )
