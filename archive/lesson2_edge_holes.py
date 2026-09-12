# -*- coding: utf-8 -*-
"""
为什么 45 那张图看着'有路'，但 BFS 实际走不到？
答案藏在"天上那条边"上 —— 它看似通，其实有一个孤立的小洞。
这张图把那个小洞放大给你看。
"""
import numpy as np
import matplotlib.pyplot as plt

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
    n = int(360 / RES)
    g = np.zeros((n, n), dtype=bool)
    for i in range(n):
        t1 = -180.0 + i*RES
        for j in range(n):
            t2 = 180.0 - j*RES
            g[i, j] = is_coll(t1, t2, R)
    return g


fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))

for ax, R in zip(axes, [28.0, 45.0]):
    grid = build_grid(R)
    # 天上 (j=0, θ2=180°) + 地下 (j=179, θ2=-178°)
    top = grid[:, 0]
    bot = grid[:, 179]
    panel = np.vstack([top, bot]).astype(float)

    ax.imshow(panel, aspect="auto", cmap="Greys", vmin=0, vmax=1.5,
              extent=[-180, 180, 0, 2], interpolation="nearest")
    ax.set_yticks([1.5, 0.5])
    ax.set_yticklabels(["天上边 θ2=180°", "地下边 θ2=-178°"], fontsize=11)
    ax.set_xlabel("θ1 (度)", fontsize=11)

    # 标出每个撞点（panel[0]=top 显示在 y=1.5，panel[1]=bot 显示在 y=0.5）
    top_hits = [i*2-180 for i in range(180) if top[i]]
    bot_hits = [i*2-180 for i in range(180) if bot[i]]
    for h in top_hits:
        ax.scatter([h], [1.5], s=130, c="red", marker="X", zorder=3, edgecolors="white", linewidths=1.4)
        ax.annotate(f"θ1={h}°", (h, 1.5), xytext=(0, 12), textcoords="offset points",
                    ha="center", color="red", fontweight="bold", fontsize=10)
    for h in bot_hits:
        ax.scatter([h], [0.5], s=130, c="red", marker="X", zorder=3, edgecolors="white", linewidths=1.4)
        ax.annotate(f"θ1={h}°", (h, 0.5), xytext=(0, 12), textcoords="offset points",
                    ha="center", color="red", fontweight="bold", fontsize=10)

    # 把起点、终点对应的位置标出来（用蓝绿竖线）
    ax.axvline(-120, color="#1a7f37", ls="--", alpha=0.5, lw=1.2)
    ax.axvline(150, color="#8250df", ls="--", alpha=0.5, lw=1.2)
    ax.text(-120, 2.05, "起点θ1", color="#1a7f37", ha="center", fontsize=9)
    ax.text(150, 2.05, "终点θ1", color="#8250df", ha="center", fontsize=9)

    if not top_hits and not bot_hits:
        ax.set_title(f"半径 {R:.0f}：两条边都通 → 可以从天上/地下绕过去",
                     color="#1a7f37", fontsize=12)
        ax.text(0, 1.0, "两条边都是干净的白色，没有任何撞点\n水波可以畅通横穿",
                ha="center", va="center", fontsize=12, color="#1a7f37", alpha=0.9)
    else:
        ax.set_title(f"半径 {R:.0f}：天上 {len(top_hits)} 处撞、地下 {len(bot_hits)} 处撞 → 绕不过去",
                     color="#cf222e", fontsize=12)
    ax.grid(axis="x", alpha=0.3)

fig.suptitle("为什么 45 那图看着有路却走不通？把'天上'和'地下'两条边放大看",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("lesson2_edge_holes.png", dpi=130)
print("已保存 lesson2_edge_holes.png")
print()
print("结论：")
print("  28 那两条边（天上 j=0、地下 j=179）一条撞点都没有 → 可以绕过去")
print("  45 那两条边各出现 1 个孤立撞点 → BFS 必须连续走，一格断了就过不去")
print()
print("再进一步：为什么从撞点旁边绕一下也不行？看禁区在各 θ2 行上的宽度——")
grid45 = build_grid(45.0)
for j in range(0, 180, 6):
    hits = [i*2-180 for i in range(180) if grid45[i, j]]
    if hits:
        print(f"  θ2={180-j*2:>4}°: 撞点 θ1 范围 [{min(hits):>4}, {max(hits):>4}]，共 {len(hits)} 格")
    else:
        print(f"  θ2={180-j*2:>4}°: 整行自由")