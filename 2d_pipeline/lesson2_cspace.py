# -*- coding: utf-8 -*-
"""
第 2 课：构型空间（C 空间）与路径规划入门
==========================================
本脚本做四件事：
  1. 遍历所有关节角组合 (θ1, θ2)，检查哪些姿势会撞到障碍物 -> 画出 C 空间地图
  2. 在这张地图上用 BFS（广度优先搜索）自动找一条"起点->终点"的无碰撞路径
  3. 生成双面板动画：左边机械臂沿路径运动，右边 C 空间的点同步移动
  4. 打印统计数字（这张图就是以后对比实验的雏形）

运行方法（在 VS Code 终端里）：
  python lesson2_cspace.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]  # 支持中文
plt.rcParams["axes.unicode_minus"] = False

# ---------------- 机械臂参数（和第 1 课一致） ----------------
L1, L2 = 140.0, 110.0          # 两根杆的长度
OBSTACLE = (175.0, 95.0, 43.0) # 障碍物：(圆心x, 圆心y, 半径)

START = (-120.0, 80.0)         # 起点姿势 (θ1, θ2)，单位：度
GOAL  = (150.0, -40.0)         # 终点姿势

# ---------------- 核心函数 1：正运动学（第 1 课学过的） ----------------
def forward_kinematics(theta1_deg, theta2_deg):
    """给定两个关节角（度），返回肘部坐标和末端坐标"""
    t1 = np.radians(theta1_deg)
    t12 = np.radians(theta1_deg + theta2_deg)
    elbow = (L1 * np.cos(t1), L1 * np.sin(t1))
    end = (elbow[0] + L2 * np.cos(t12), elbow[1] + L2 * np.sin(t12))
    return elbow, end

# ---------------- 核心函数 2：碰撞检测 ----------------
def dist_point_to_segment(px, py, ax, ay, bx, by):
    """点 (px,py) 到线段 (ax,ay)-(bx,by) 的最短距离"""
    abx, aby = bx - ax, by - ay
    length_sq = abx**2 + aby**2
    if length_sq == 0:  # 线段退化为一个点
        return np.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / length_sq))
    qx, qy = ax + t * abx, ay + t * aby
    return np.hypot(px - qx, py - qy)

def is_collision(theta1_deg, theta2_deg):
    """判断这个姿势下，任意一根杆是否撞到障碍物（含一点安全余量）"""
    ox, oy, r = OBSTACLE
    margin = 3.0  # 安全余量：别贴着障碍物边缘过
    elbow, end = forward_kinematics(theta1_deg, theta2_deg)
    hit_link1 = dist_point_to_segment(ox, oy, 0, 0, elbow[0], elbow[1]) < r + margin
    hit_link2 = dist_point_to_segment(ox, oy, elbow[0], elbow[1], end[0], end[1]) < r + margin
    return hit_link1 or hit_link2

# ---------------- 第一步：把整张 C 空间地图算出来 ----------------
RES = 2.0   # 分辨率：每 2 度采样一个格子
grid = np.zeros((int(360 / RES), int(360 / RES)), dtype=bool)  # True = 禁区

for i in range(grid.shape[0]):        # i -> θ1 = -180 + i*RES
    for j in range(grid.shape[1]):    # j -> θ2 = 180 - j*RES（j 越大角度越小）
        t1 = -180.0 + i * RES
        t2 = 180.0 - j * RES
        grid[i, j] = is_collision(t1, t2)

def angles_to_cell(t1, t2):
    """把角度转成网格下标"""
    i = int(round((t1 + 180.0) / RES))
    j = int(round((180.0 - t2) / RES))
    return i, j

def cell_to_angles(i, j):
    return -180.0 + i * RES, 180.0 - j * RES

collision_ratio = grid.mean() * 100
print(f"C 空间地图计算完成：{grid.shape[0]}x{grid.shape[1]} 个格子，"
      f"其中 {collision_ratio:.1f}% 是障碍物禁区")

# ---------------- 第二步：BFS 在自由区域里找路 ----------------
def bfs(start_cell, goal_cell):
    """经典的广度优先搜索：从起点一层层向外扩，直到碰到终点"""
    visited = np.zeros_like(grid, dtype=bool)
    prev = {}
    queue = deque([start_cell])
    visited[start_cell] = True
    explored = 0
    while queue:
        cur = queue.popleft()
        explored += 1
        if cur == goal_cell:
            # 回溯出完整路径
            path = [cur]
            while cur in prev:
                cur = prev[cur]
                path.append(cur)
            return path[::-1], explored
        i, j = cur
        for di, dj in ((1,0),(-1,0),(0,1),(0,-1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                if not visited[ni, nj] and not grid[ni, nj]:
                    visited[ni, nj] = True
                    prev[(ni, nj)] = (i, j)
                    queue.append((ni, nj))
    return None, explored

sc = angles_to_cell(*START)
gc = angles_to_cell(*GOAL)
assert not grid[sc], "起点姿势本身就在禁区里，请换一个起点"
assert not grid[gc], "终点姿势本身就在禁区里，请换一个终点"

path_cells, explored = bfs(sc, gc)
assert path_cells is not None, "找不到路！障碍物把自由区域切断了"
path = [cell_to_angles(i, j) for i, j in path_cells]

# 路径平滑：把格子路径细分成小步，动画才顺滑
smooth = []
for k in range(len(path) - 1):
    (a1, a2), (b1, b2) = path[k], path[k + 1]
    for t in np.linspace(0, 1, 12, endpoint=False):
        smooth.append((a1 + t * (b1 - a1), a2 + t * (b2 - a2)))
smooth.append(path[-1])

print(f"BFS 完成：探索了 {explored} 个格子，"
      f"找到一条 {len(path)} 个路标点、总转角 "
      f"{sum(abs(path[k+1][0]-path[k][0]) + abs(path[k+1][1]-path[k][1]) for k in range(len(path)-1)):.0f} 度的无碰撞路径")

# ---------------- 第三步：画静态的 C 空间地图（图 1） ----------------
fig1, ax1 = plt.subplots(figsize=(7, 6))
img = grid.T[:, ::-1]  # 转置+翻转，让横轴 θ1、纵轴 θ2 方向正确
extent = [-180, 180, -180, 180]
ax1.imshow(img, extent=extent, origin="lower",
           cmap="Greys", vmin=0, vmax=1.5, interpolation="nearest")
ax1.plot(START[0], START[1], "o", color="#1a7f37", markersize=12, label="起点姿势")
ax1.plot(GOAL[0], GOAL[1], "o", color="#8250df", markersize=12, label="终点姿势")
ax1.plot([p[0] for p in path], [p[1] for p in path], "-", color="#0969da",
         linewidth=2.2, label="BFS 找到的无碰撞路径")
ax1.set_xlabel("关节角 θ1 (度)")
ax1.set_ylabel("关节角 θ2 (度)")
ax1.set_title("C 空间地图：黑=碰撞禁区，白=自由区域")
ax1.legend(loc="upper right")
ax1.grid(alpha=0.25)
fig1.tight_layout()
fig1.savefig("lesson2_cspace_map.png", dpi=130)
print("已保存 lesson2_cspace_map.png")

# ---------------- 第四步：双面板动画（图 2 GIF） ----------------
fig2, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5.5))

# 左面板：工作空间里的机械臂
axL.set_xlim(-260, 260); axL.set_ylim(-260, 260)
axL.set_aspect("equal"); axL.grid(alpha=0.25)
ox, oy, orad = OBSTACLE
axL.add_patch(plt.Circle((ox, oy), orad, color="#6e7781", alpha=0.55, label="障碍物"))
line1, = axL.plot([], [], "-", color="#0969da", linewidth=6, solid_capstyle="round", label="杆1")
line2, = axL.plot([], [], "-", color="#0550ae", linewidth=5, solid_capstyle="round", label="杆2")
trace, = axL.plot([], [], "--", color="#cf222e", linewidth=1.2, alpha=0.8, label="末端轨迹")
axL.set_title("工作空间：机械臂真的动起来了")
axL.legend(loc="lower left", fontsize=8)

# 右面板：C 空间地图上的点同步移动
axR.imshow(img, extent=extent, origin="lower",
           cmap="Greys", vmin=0, vmax=1.5, interpolation="nearest")
axR.plot(START[0], START[1], "o", color="#1a7f37", markersize=10)
axR.plot(GOAL[0], GOAL[1], "o", color="#8250df", markersize=10)
axR.plot([p[0] for p in path], [p[1] for p in path], "-", color="#0969da", linewidth=2, alpha=0.6)
dot, = axR.plot([], [], "o", color="#dd8e1e", markersize=11, markeredgecolor="white")
axR.set_xlabel("θ1 (度)"); axR.set_ylabel("θ2 (度)")
axR.set_title("C 空间：同一个姿势，换个视角看")
axR.grid(alpha=0.0)

frames = smooth[::2]
def update_frame(k):
    t1, t2 = frames[k]
    elbow, end = forward_kinematics(t1, t2)
    line1.set_data([0, elbow[0]], [0, elbow[1]])
    line2.set_data([elbow[0], end[0]], [elbow[1], end[1]])
    trace.set_data([forward_kinematics(a, b)[1][0] for a, b in frames[:k+1]],
                   [forward_kinematics(a, b)[1][1] for a, b in frames[:k+1]])
    dot.set_data([t1], [t2])
    return line1, line2, trace, dot

anim = animation.FuncAnimation(fig2, update_frame, frames=len(frames),
                               interval=40, blit=True)
anim.save("lesson2_cspace_path.gif", writer=animation.PillowWriter(fps=25))
print("已保存 lesson2_cspace_path.gif")
print()
print("完成！注意看 GIF：左边机械臂绕开障碍物的同时，")
print("右边 C 空间里的橙点正沿着白色自由区域从绿点走到紫点。")
print("这就是'路径规划'的本质 —— 在 C 空间里找一条避开禁区的路。")
