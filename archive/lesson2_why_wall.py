# -*- coding: utf-8 -*-
"""
真正的答案：切断地图的不是"某一行"，而是"某一列"。

关键几何事实：大臂的位置只由 θ1 决定，跟 θ2 无关。
所以当障碍物大到某些 θ1 时大臂会撞，那这个 θ1 的"整列"都会撞 ——
这一整列就是一堵从上到下的墙，任何从左到右的路都跨不过去。

本脚本画两件事：
  图A：每个 θ1 列上，"撞的格子"占该列的比例（28 vs 45）
  图B：28 场景 BFS 路径的合法性检查（回应"蓝线是否碰到灰区"）
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

L1, L2 = 140.0, 110.0
OX, OY = 170.0, 80.0
RES = 2.0


def fk(t1d, t2d):
    t1 = np.radians(t1d); t12 = np.radians(t1d+t2d)
    e = (L1*np.cos(t1), L1*np.sin(t1))
    end = (e[0]+L2*np.cos(t12), e[1]+L2*np.sin(t12))
    return e, end


def dseg(px,py,ax,ay,bx,by):
    abx,aby = bx-ax, by-ay; L = abx**2+aby**2
    if L == 0: return np.hypot(px-ax, py-ay)
    t = max(0., min(1., ((px-ax)*abx+(py-ay)*aby)/L))
    qx,qy = ax+t*abx, ay+t*aby
    return np.hypot(px-qx, py-qy)


def is_coll(t1, t2, R):
    e, end = fk(t1, t2); m = 3.0
    return (dseg(OX,OY,0,0,e[0],e[1]) < R+m or
            dseg(OX,OY,e[0],e[1],end[0],end[1]) < R+m)


def build_grid(R):
    n = int(360/RES)
    g = np.zeros((n, n), dtype=bool)
    for i in range(n):
        t1 = -180.0 + i*RES
        for j in range(n):
            t2 = 180.0 - j*RES
            g[i, j] = is_coll(t1, t2, R)
    return g


def a2c(t1, t2):
    return int(round((t1+180)/RES)), int(round((180-t2)/RES))


def bfs(grid, sc, gc):
    visited = np.zeros_like(grid, dtype=bool)
    prev = {}
    q = deque([sc]); visited[sc] = True
    while q:
        cur = q.popleft()
        if cur == gc:
            path = [cur]
            while cur in prev:
                cur = prev[cur]; path.append(cur)
            return path[::-1], visited
        i, j = cur
        for di, dj in ((1,0),(-1,0),(0,1),(0,-1)):
            ni, nj = i+di, j+dj
            if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                if not visited[ni,nj] and not grid[ni,nj]:
                    visited[ni,nj] = True
                    prev[(ni,nj)] = (i,j)
                    q.append((ni,nj))
    return None, visited


# ================= 图 A：每一列"被撞满"的程度 =================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, R in zip(axes, [28.0, 45.0]):
    grid = build_grid(R)
    col_ratio = grid.mean(axis=1) * 100  # 每一列（每个 θ1）撞格子的百分比
    t1s = [-180 + i*RES for i in range(len(col_ratio))]

    ax.fill_between(t1s, 0, col_ratio, color="#cf222e", alpha=0.35)
    ax.plot(t1s, col_ratio, color="#cf222e", lw=1.5)
    ax.axvline(-120, color="#1a7f37", ls="--", lw=1.5, label="起点 θ1=-120°")
    ax.axvline(150, color="#8250df", ls="--", lw=1.5, label="终点 θ1=+150°")

    # 找出"整列全撞"（比例 ≥ 99%）的 θ1
    full_cols = [t1s[i] for i in range(len(col_ratio)) if col_ratio[i] >= 99]
    if full_cols:
        ax.axvspan(min(full_cols), max(full_cols), color="#cf222e", alpha=0.18)
        ax.annotate(f"θ1={full_cols[0]:.0f}° 整列全撞！\n这是『墙』",
                    xy=(full_cols[0], 100), xytext=(full_cols[0]+30, 70),
                    arrowprops=dict(arrowstyle="->", color="#cf222e", lw=1.6),
                    color="#cf222e", fontsize=11, fontweight="bold")

    ax.set_ylim(0, 108)
    ax.set_xlim(-180, 180)
    ax.set_xlabel("关节角 θ1 (度)", fontsize=11)
    ax.set_ylabel("该 θ1 列中撞格子的占比 (%)", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    if full_cols:
        ax.set_title(f"半径 {R:.0f}：存在整列全撞的 θ1（=上下贯通的墙）",
                     color="#cf222e", fontsize=12)
    else:
        ax.set_title(f"半径 {R:.0f}：没有任何一列全撞（墙没形成，可以绕）",
                     color="#1a7f37", fontsize=12)

fig.suptitle("为什么 45 走不通？因为 θ1=26° 这一整列被撞满了 —— 它是一堵竖着的墙",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("lesson2_why_wall.png", dpi=130)
print("已保存 lesson2_why_wall.png")
print()

# ================= 图 B：28 场景路径合法性检查 =================
print("=" * 60)
print("【检查】半径 28 场景里，BFS 找到的蓝线到底有没有真的碰到灰区？")
print("=" * 60)
grid28 = build_grid(28.0)
sc, gc = a2c(-120, 80), a2c(150, -40)
path, visited = bfs(grid28, sc, gc)
cell_to_ang = lambda i, j: (-180 + i*RES, 180 - j*RES)

# 1) 格子层面：路径上的每个格子是否在禁区里
bad_cells = [c for c in path if grid28[c]]
print(f"1) 路径共 {len(path)} 个格子，其中位于禁区内的格子数：{len(bad_cells)}")
if bad_cells:
    print(f"   有问题！这些格子撞了：{bad_cells[:10]}")
else:
    print(f"   ✅ 路径上每个格子都是自由的（BFS 保证不会走禁区格）")

# 2) 线段层面：相邻两格的中点在不在禁区里（离散网格的经典盲区）
bad_seg = []
for k in range(len(path)-1):
    (i1, j1), (i2, j2) = path[k], path[k+1]
    mid_i = (i1+i2)/2; mid_j = (j1+j2)/2
    # 因为 4 邻居，中点必然落在某条边上，直接用相邻格检查
    t1, t2 = cell_to_ang(*path[k])
    t1b, t2b = cell_to_ang(*path[k+1])
    t1m, t2m = (t1+t1b)/2, (t2+t2b)/2
    if is_coll(t1m, t2m, 28.0):
        bad_seg.append((t1m, t2m))
print(f"2) 相邻格子连线的中点：共有 {len(bad_seg)} 处中点落在禁区里")
if bad_seg:
    print(f"   注意！这些中点撞了（这是 4 邻居网格离散化的已知瑕疵）：{bad_seg[:8]}")
else:
    print(f"   ✅ 所有相邻格子连线的中点也都是自由的")

# 3) 路径是否贴着禁区走（最近距离）
min_clear = min(float(np.hypot(OX - e[0], OY - e[1])) for e in
                [fk(*cell_to_ang(*c))[0] for c in path])
print(f"3) 路径上所有姿势里，肘部离障碍物圆心的最近距离：{min_clear:.1f}")
print(f"   （障碍半径 28 + 余量 3 = 31；只要 ≥31 就不碰）")
print()
print("结论：")
if not bad_cells and not bad_seg:
    print("  蓝线在『格子层』和『连线中点』两个层面都完全合法。")
    print("  你在图上看到它『贴着』灰边走，是因为 BFS 找到的是步数最少的路径，")
    print("  最短路径必然紧贴障碍物边缘 —— 貌似'碰到'，其实是擦边而过。")
else:
    print("  发现离散化瑕疵，需要说明。")