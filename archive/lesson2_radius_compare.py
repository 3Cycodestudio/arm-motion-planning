# -*- coding: utf-8 -*-
"""
第 2 课 加餐：障碍物半径 28 vs 45
=====================================
同一个机械臂、同一个起点终点，只改障碍物大小，结果一个能走通、一个走不通。
这张图同时把两件事讲明白：
  1. BFS 到底在干什么（看浅蓝色"水波"扩到哪）
  2. 为什么障碍物一大，路就被切断了

运行： python lesson2_radius_compare.py
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------- 固定参数 ----------------
L1, L2 = 140.0, 110.0
OX, OY = 170.0, 80.0
START = (-120.0, 80.0)
GOAL  = (150.0, -40.0)
RES = 2.0


def forward_kinematics(t1d, t2d):
    t1 = np.radians(t1d)
    t12 = np.radians(t1d + t2d)
    elbow = (L1 * np.cos(t1), L1 * np.sin(t1))
    end = (elbow[0] + L2 * np.cos(t12), elbow[1] + L2 * np.sin(t12))
    return elbow, end


def dist_point_to_segment(px, py, ax, ay, bx, by):
    abx, aby = bx - ax, by - ay
    L = abx ** 2 + aby ** 2
    if L == 0:
        return np.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / L))
    qx, qy = ax + t * abx, ay + t * aby
    return np.hypot(px - qx, py - qy)


def build_grid(r):
    n = int(360 / RES)
    grid = np.zeros((n, n), dtype=bool)
    margin = 3.0
    for i in range(n):
        t1 = -180.0 + i * RES
        for j in range(n):
            t2 = 180.0 - j * RES
            elbow, end = forward_kinematics(t1, t2)
            h1 = dist_point_to_segment(OX, OY, 0, 0, elbow[0], elbow[1]) < r + margin
            h2 = dist_point_to_segment(OX, OY, elbow[0], elbow[1], end[0], end[1]) < r + margin
            grid[i, j] = h1 or h2
    return grid


def angles_to_cell(t1, t2):
    return int(round((t1 + 180.0) / RES)), int(round((180.0 - t2) / RES))


def cell_to_angles(i, j):
    return -180.0 + i * RES, 180.0 - j * RES


def bfs(grid, sc, gc):
    """返回 (路径 or None, 探索格子数, 被访问到的格子掩码)"""
    visited = np.zeros_like(grid, dtype=bool)
    prev = {}
    q = deque([sc])
    visited[sc] = True
    explored = 0
    while q:
        cur = q.popleft()
        explored += 1
        if cur == gc:
            path = [cur]
            while cur in prev:
                cur = prev[cur]
                path.append(cur)
            return path[::-1], explored, visited
        i, j = cur
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                if not visited[ni, nj] and not grid[ni, nj]:
                    visited[ni, nj] = True
                    prev[(ni, nj)] = (i, j)
                    q.append((ni, nj))
    return None, explored, visited


# ---------------- 对两个半径各算一遍 ----------------
fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2))
EXTENT = [-180, 180, -180, 180]

for ax, r in zip(axes, [28.0, 45.0]):
    grid = build_grid(r)
    sc, gc = angles_to_cell(*START), angles_to_cell(*GOAL)
    path, explored, visited = bfs(grid, sc, gc)

    # 第一层：禁区（灰黑）
    ax.imshow(grid.T[:, ::-1], extent=EXTENT, origin="lower",
              cmap="Greys", vmin=0, vmax=1.5, interpolation="nearest", zorder=1)

    # 第二层：BFS 已经探索过的自由区域（浅蓝"水波"）
    vis = visited.T[:, ::-1].astype(float)
    ax.imshow(np.ma.masked_where(vis == 0, vis), extent=EXTENT, origin="lower",
              cmap="Blues", vmin=0, vmax=1, alpha=0.45,
              interpolation="nearest", zorder=2)

    # 第三层：路径 / 起终点
    if path is not None:
        pts = [cell_to_angles(i, j) for i, j in path]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-",
                color="#0969da", lw=2.4, zorder=4, label="BFS 找到的路")
    ax.plot(*START, "o", color="#1a7f37", ms=13, mec="white", mew=1.5, zorder=5, label="起点")
    ax.plot(*GOAL, "o", color="#8250df", ms=13, mec="white", mew=1.5, zorder=5, label="终点")

    ax.set_xlabel("关节角 θ1 (度)")
    ax.set_ylabel("关节角 θ2 (度)")
    ax.grid(alpha=0.2, zorder=0)
    ax.legend(loc="lower left", fontsize=9)

    ratio = grid.mean() * 100
    if path is not None:
        title = (f"半径 {r:.0f}：禁区占 {ratio:.1f}%\n"
                 f"[有解] 有水波通道，BFS 探索 {explored} 格后找到路")
        ax.set_title(title, color="#1a7f37", fontsize=11)
    else:
        title = (f"半径 {r:.0f}：禁区占 {ratio:.1f}%\n"
                 f"[无解] 水波被困在左侧，永远到不了紫点")
        ax.set_title(title, color="#cf222e", fontsize=11)
        ax.text(0, -150, "路断了：\n禁区上下封死", color="#cf222e",
                fontsize=12, ha="center", fontweight="bold", zorder=6)

fig.suptitle("只把障碍物半径从 28 改成 45，机械臂就从「有路」变成「无路」",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("lesson2_radius_compare.png", dpi=130)
print("已保存 lesson2_radius_compare.png")
print()
print("读图要点：浅蓝色 = BFS 已经'探索过'的区域（像水波扩散）；")
print("禁区（黑色）会挡住水波。水波能绕过去碰到紫点，就有路；被彻底挡住，就无解。")
