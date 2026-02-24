# -*- coding: utf-8 -*-
"""
六边形棋盘图形界面。
- 开始前可点击格子切换障碍（左键添加/移除）
- 点击「开始」后在主线程边求解边刷新进度
- 求解完成后可步进/自动播放回放
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
    y = r * HEX_H
    return (x + PADDING, y + PADDING)


def hex_vertices(cx: float, cy: float) -> list[tuple[float, float]]:
    out = []
    for i in range(6):
        angle = math.pi / 6 + i * (math.pi / 3)
        out.append((cx + HEX_R * math.cos(angle), cy + HEX_R * math.sin(angle)))
    return out


def _point_in_hex(px: float, py: float, cx: float, cy: float) -> bool:
    """粗略判断点是否在六边形内（用圆近似）。"""
    return (px - cx) ** 2 + (py - cy) ** 2 <= HEX_R * HEX_R


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
    barriers = set(initial_barriers)
    monster_pos = monster_start
    move_count = 0
    placement_index = 0
    state: dict[str, Any] = {
        "placements": [],
        "total_steps": 0,
        "status_text": "点击格子增减障碍，然后按「开始」",
        "editing": True,  # 开始前允许编辑障碍
    }

    # 预计算格子中心，用于点击检测
    cell_centers: dict[tuple[int, int], tuple[float, float]] = {}
    for r in range(NUM_ROWS):
        for c in range(ROWS_LAYOUT[r]):
            cell_centers[(r, c)] = row_col_to_xy(r, c)

    # ── 绘制 ──
    COLOR_EMPTY = "#2d2d44"
    COLOR_EMPTY_OUTLINE = "#4a4a6a"
    COLOR_BARRIER = "#5c4033"
    COLOR_BARRIER_OUTLINE = "#8b6914"
    COLOR_MONSTER = "#c41e3a"
    COLOR_MONSTER_OUTLINE = "#ff6b6b"
    COLOR_PLACED = "#1565c0"
    COLOR_PLACED_OUTLINE = "#42a5f5"

    placed_cells: dict[tuple[int, int], int] = {}  # cell → 放置顺序（从 1 开始）

    def draw_grid() -> None:
        canvas.delete("all")
        for (r, c), (cx, cy) in cell_centers.items():
            pts = hex_vertices(cx, cy)
            flat = [x for p in pts for x in p]
            if (r, c) in placed_cells:
                fill, outline = COLOR_PLACED, COLOR_PLACED_OUTLINE
            elif (r, c) in barriers:
                fill, outline = COLOR_BARRIER, COLOR_BARRIER_OUTLINE
            elif (r, c) == monster_pos:
                fill, outline = COLOR_MONSTER, COLOR_MONSTER_OUTLINE
            else:
                fill, outline = COLOR_EMPTY, COLOR_EMPTY_OUTLINE
            canvas.create_polygon(flat, fill=fill, outline=outline, width=2)

            if (r, c) in placed_cells:
                order = placed_cells[(r, c)]
                canvas.create_text(cx, cy, text=str(order), fill="#fff", font=("Consolas", 10, "bold"))
            else:
                canvas.create_text(cx, cy, text=f"{r},{c}", fill="#888", font=("Consolas", 7))

        total_bricks = len(state["placements"])
        title = state["status_text"] or (
            f"小怪走了: {move_count} 步  |  已放砖: {placement_index}/{total_bricks}"
            f"  |  最优放砖: {total_bricks}"
        )
        canvas.create_text(canvas_w // 2, 14, text=title, fill="#eee", font=("Consolas", 11), tags=("title",))

    # ── 点击切换障碍 ──
    def on_canvas_click(event: tk.Event) -> None:  # type: ignore[type-arg]
        if not state["editing"]:
            return
        px, py = event.x, event.y
        for (r, c), (cx, cy) in cell_centers.items():
            if _point_in_hex(px, py, cx, cy):
                if (r, c) == monster_pos:
                    return
                if (r, c) in barriers:
                    barriers.discard((r, c))
                else:
                    barriers.add((r, c))
                draw_grid()
                return

    canvas.bind("<Button-1>", on_canvas_click)

    # ── 回放控制 ──
    def do_next() -> None:
        nonlocal monster_pos, move_count, placement_index
        if state.get("game_over"):
            return

        # 1) 玩家放砖（必须）
        pl = state["placements"]
        if placement_index < len(pl):
            cell = pl[placement_index]
            placed_cells[cell] = placement_index + 1
            barriers.add(cell)
            placement_index += 1

        # 2) 检查放砖后小怪是否已被困
        nxt = get_monster_next_step(monster_pos, barriers)
        if nxt is None:
            state["game_over"] = True
            if is_boundary(*monster_pos):
                state["status_text"] = f"小怪逃脱了！放了 {placement_index} 块砖，走了 {move_count} 步"
            else:
                state["status_text"] = f"小怪被困住了！放了 {placement_index} 块砖，走了 {move_count} 步"
            draw_grid()
            return

        # 3) 小怪走一步
        monster_pos = nxt
        move_count += 1

        # 4) 走完后再检查
        if is_boundary(*monster_pos):
            state["game_over"] = True
            state["status_text"] = f"小怪逃脱了！放了 {placement_index} 块砖，走了 {move_count} 步"
        elif get_monster_next_step(monster_pos, barriers) is None and placement_index >= len(pl):
            state["game_over"] = True
            state["status_text"] = f"小怪被困住了！放了 {placement_index} 块砖，走了 {move_count} 步"

        draw_grid()

    def auto_play() -> None:
        def tick() -> None:
            do_next()
            if state.get("game_over"):
                return
            root.after(350, tick)
        tick()

    def reset() -> None:
        nonlocal monster_pos, move_count, placement_index
        state["game_over"] = False
        state["status_text"] = None
        placed_cells.clear()
        barriers.clear()
        barriers.update(initial_barriers)
        for rc in state.get("user_added", set()):
            barriers.add(rc)
        monster_pos = monster_start
        move_count = 0
        placement_index = 0
        draw_grid()

    # ── 按钮 ──
    btn_frame = ttk.Frame(root)
    btn_frame.pack(pady=4)

    def on_start(btn_start: tk.Widget) -> None:
        state["editing"] = False
        state["user_added"] = set(barriers) - set(initial_barriers)
        btn_start.config(state=tk.DISABLED)  # type: ignore[arg-type]

        state["status_text"] = "计算中… 当前最优: 0 步"
        ids = canvas.find_withtag("title")
        if ids:
            canvas.itemconfig(ids[0], text=state["status_text"])
        root.update_idletasks()

        current_barriers = set(barriers)

        def on_progress(steps: int, pl: list[tuple[int, int]]) -> None:
            if pl:
                state["placements"][:] = pl
            state["total_steps"] = steps
            bricks = len(state["placements"])
            state["status_text"] = f"计算中… 当前最优: {steps} 步 ({bricks} 砖)"
            ids = canvas.find_withtag("title")
            if ids:
                canvas.itemconfig(ids[0], text=state["status_text"])
            root.update_idletasks()

        steps, pl = solve_max_steps(
            monster_start,
            current_barriers,
            max_placements=max_placements,
            on_progress=on_progress,
        )
        state["placements"][:] = pl
        state["total_steps"] = steps
        if steps >= 0:
            bricks = len(pl)
            state["status_text"] = None
            btn_next.config(state=tk.NORMAL)
            btn_play.config(state=tk.NORMAL)
            root.title(f"六边形棋盘 - 困住小怪！放 {bricks} 块砖，走 {steps} 步")
        else:
            state["status_text"] = f"无法困住小怪（砖数不足），放砖上限: {max_placements}"
            root.title("六边形棋盘 - 无法困住")
        draw_grid()

    btn_start = ttk.Button(btn_frame, text="开始", command=lambda: on_start(btn_start))
    btn_start.pack(side=tk.LEFT, padx=4)
    btn_next = ttk.Button(btn_frame, text="下一步", command=do_next, state=tk.DISABLED)
    btn_next.pack(side=tk.LEFT, padx=4)
    btn_play = ttk.Button(btn_frame, text="自动播放", command=auto_play, state=tk.DISABLED)
    btn_play.pack(side=tk.LEFT, padx=4)
    ttk.Button(btn_frame, text="重置", command=reset).pack(side=tk.LEFT, padx=4)

    draw_grid()
    root.mainloop()


if __name__ == "__main__":
    from hex_grid import DEFAULT_BARRIERS

    run_gui_with_solver(
        monster_start=(4, 4),
        initial_barriers=set(DEFAULT_BARRIERS),
        max_placements=40,
    )
