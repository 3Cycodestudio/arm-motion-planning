"""
第 4 课：路径平滑（Path Smoothing by Random Shortcutting）。

问题：lesson3 的 RRT 给的路径是折线（一堆短边拼起来的），
      真上电机会频繁加减速、抖动。
解法：随机挑路径上两个节点，连直线，若中间无碰撞就把中间节点全删掉。
      反复迭代几千次，路径会越来越短、越来越直。

类比：隧道先按粗导线打一串小折线贯通，再上整平机来回刮几遍——
      这个"刮几遍"就是 shortcutting。

复用：lesson3_rrt 的 rrt_plan / is_collision_path / draw_background。
产出：lesson4_smoothed_path.png （5 个路径快照 + 1 张收敛曲线）
"""
import os
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

from lesson3_rrt import (
    L1, L2, OBSTACLE,
    rrt_plan, reconstruct_path, is_collision_path,
    build_obstacle_mask, draw_background, check_background_orientation,
)

OUT = os.path.dirname(os.path.abspath(__file__))
START = (-132.0, 16.0)
GOAL = (0.0, -178.0)


def path_length(path):
    return sum(
        np.linalg.norm(np.array(path[i+1]) - np.array(path[i]))
        for i in range(len(path) - 1)
    )


def shortcut_smoothing(path, n_iter=2000, seed=42, max_gap=8, verbose=False):
    """随机捷径平滑（带 max_gap 限制）。
    重复 n_iter 次：随机挑两个节点 i<j，j-i <= max_gap；
    如果它们之间走直线无碰撞，就把中间节点全删掉。
    max_gap 越小 → 越像"一段一段压平"（过程好看，但可能收敛不到最优）；
    max_gap 越大 → 越激进（一次能拉掉一大段）。
    """
    rng = np.random.default_rng(seed)
    path = [tuple(p) for p in path]
    accepted = 0
    for k in range(n_iter):
        n = len(path)
        if n < 3:
            break
        # 在 0..n-1 中选 i，再让 j 在 [i+2, i+max_gap] 范围内
        i = int(rng.integers(0, n))
        gap_max = min(max_gap, n - 1 - i)
        if gap_max < 2:
            continue
        j = i + int(rng.integers(2, gap_max + 1))
        a, b = path[i], path[j]
        if not is_collision_path(a[0], a[1], b[0], b[1]):
            path = path[:i+1] + path[j:]
            accepted += 1
        if verbose and (k + 1) % 500 == 0:
            print(f"    iter {k+1:4d}: 路径 {len(path):3d} 个点, "
                  f"长度 {path_length(path):.1f}°")
    return path, accepted


def shortcut_smoothing_full(path, n_iter=2000, seed=42, verbose=False):
    """无 max_gap 限制的版本——能拉到极限（直线），但中间过程看不到。
    用于最后做一个'终极平滑'对比。"""
    return shortcut_smoothing(path, n_iter=n_iter, seed=seed,
                              max_gap=10**9, verbose=verbose)


# =================== 主流程 ===================
print("=== 0. 自检 ===")
mask, n = build_obstacle_mask()
check_background_orientation(mask, n)
print(f"  起点->终点 的直线是否撞障碍："
      f"{is_collision_path(START[0], START[1], GOAL[0], GOAL[1])}  "
      f"(False = 直线安全，所以平滑的天花板就是这条直线)")
print()

print("=== 1. 先跑一次 RRT，拿到原始折线路径 ===")
nodes, parents, _, ok = rrt_plan(START, GOAL, max_iter=3000, step=6.0, seed=42)
assert ok, "RRT 失败，请检查起终点 / 障碍物"
raw_path = reconstruct_path(nodes, parents)
raw_len = path_length(raw_path)
print(f"  原始路径：{len(raw_path)} 个节点, 总长 {raw_len:.1f}°")
print()

print("=== 2. 平滑迭代（max_gap=8，每次最多砍 1 个中间节点）===")
snapshots = [(0, list(raw_path), raw_len)]
smoothed = list(raw_path)
for target in [50, 300, 1500]:
    extra = target - snapshots[-1][0]
    smoothed, acc = shortcut_smoothing(smoothed, n_iter=extra, seed=42, max_gap=8)
    plen = path_length(smoothed)
    snapshots.append((target, list(smoothed), plen))
    print(f"  累计 {target:5d} 次: {len(smoothed):3d} 个点, "
          f"长度 {plen:.1f}°  "
          f"(相比原始缩短 {(1 - plen/raw_len)*100:5.1f}%)")

print("\n=== 3. 终极平滑（无 max_gap 限制，砍到底）===")
ultra_path, ultra_acc = shortcut_smoothing_full(smoothed, n_iter=3000, seed=42)
ultra_len = path_length(ultra_path)
snapshots.append(("极限", ultra_path, ultra_len))
print(f"  极限版: {len(ultra_path):3d} 个点, 长度 {ultra_len:.1f}°  "
      f"(相比原始缩短 {(1 - ultra_len/raw_len)*100:5.1f}%)")
print()

# =================== 画图 ===================
fig, axes = plt.subplots(3, 2, figsize=(13, 16))
ax_curve = axes[2, 1]
ax_paths = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1], axes[2, 0]]
titles = ["原始 RRT（折线）",
          "平滑   50 次",
          "平滑  300 次",
          "平滑 1500 次",
          "极限（无 max_gap）"]

# 颜色：从蓝（原始）→ 橙（过程）→ 红（极限）
colors = ['tab:blue', 'tab:green', 'tab:olive', 'tab:orange', 'tab:red']
for ax, (iters, path, plen), title, col in zip(ax_paths, snapshots, titles, colors):
    draw_background(ax, mask, n)
    ax.plot(START[0], START[1], 'go', markersize=9, label='起点', zorder=4)
    ax.plot(GOAL[0], GOAL[1], 'r*', markersize=14, label='终点', zorder=4)
    xs = [p[0] for p in path]; ys = [p[1] for p in path]
    ax.plot(xs, ys, '-', color=col, linewidth=1.8, alpha=0.85, zorder=3)
    ax.plot(xs, ys, 'o', color=col, markersize=4, alpha=0.7, zorder=3)
    ax.set_title(f"{title}\n{len(path)} 个点 · 总长 {plen:.1f}°", fontsize=11)
    ax.set_xlabel('θ1 (°)'); ax.set_ylabel('θ2 (°)')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(-185, 185); ax.set_ylim(-185, 185)

# 收敛曲线（横轴用 index，更友好）
xs_c = list(range(len(snapshots)))
ys_c = [s[2] for s in snapshots]
labels = ["原始", "50次", "300次", "1500次", "极限"]
ax_curve.plot(xs_c, ys_c, 'o-', color='tab:blue', linewidth=2, markersize=8)
for x, y, lab in zip(xs_c, ys_c, labels):
    ax_curve.annotate(f"{lab}\n{y:.0f}°", (x, y), textcoords="offset points",
                      xytext=(0, 12), ha='center', fontsize=9)
ax_curve.set_xticks(xs_c)
ax_curve.set_xticklabels(labels, rotation=20)
ax_curve.set_xlabel('平滑阶段')
ax_curve.set_ylabel('路径总长度 (°)')
ax_curve.set_title('收敛曲线\n（1500 次已到底，再加就是浪费）')
ax_curve.grid(alpha=0.3)
ax_curve.set_ylim(min(ys_c) - 20, max(ys_c) + 20)

plt.suptitle('第 4 课：RRT 路径平滑（Random Shortcutting）',
             fontsize=15, fontweight='bold')
plt.tight_layout()
out_png = os.path.join(OUT, 'lesson4_smoothed_path.png')
plt.savefig(out_png, dpi=130, bbox_inches='tight')
print(f"图已存到: {out_png}")