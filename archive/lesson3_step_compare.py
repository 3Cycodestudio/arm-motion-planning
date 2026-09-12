"""
第3课 拓展：RRT step 大小对节点数和成功率的影响。

直接复跑 lesson3_rrt.py 里的 rrt_plan()，其它设置全部锁死：
    L1=200, L2=180, OBSTACLE=(175,95,43), START=(-132,16), GOAL=(0,-178)
    max_iter=3000, goal_bias=0.10, seed=42

只改一个变量：step
对比 step = 4 / 6 / 9 / 12 / 18 / 24 / 36 度
输出：lesson3_step_compare.png  （两张子图：节点数 vs step、路径 vs step）
      + 表格化打印每个 step 的节点数、是否找到路、路径长度
"""
import os
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

from lesson3_rrt import (
    L1, L2, OBSTACLE,
    rrt_plan, reconstruct_path,
    build_obstacle_mask, draw_background, check_background_orientation,
)

OUT = os.path.dirname(os.path.abspath(__file__))

# 与 lesson3_rrt.py 主脚本保持一致的起终点
START = (-132.0, 16.0)
GOAL = (0.0, -178.0)

# 跑一组不同的 step
STEPS = [4.0, 6.0, 9.0, 12.0, 18.0, 24.0, 36.0]
MAX_ITER = 3000
SEED = 42

results = []   # (step, n_nodes, success, path_len)
for s in STEPS:
    nodes, parents, _, ok = rrt_plan(START, GOAL, max_iter=MAX_ITER,
                                     step=s, seed=SEED)
    if ok:
        path = reconstruct_path(nodes, parents)
        plen = sum(np.linalg.norm(np.array(path[i+1]) - np.array(path[i]))
                   for i in range(len(path) - 1))
    else:
        plen = float('nan')
    results.append((s, len(nodes), ok, plen))
    print(f"step={s:5.1f}°  节点数={len(nodes):5d}  成功={ok}  "
          f"路径长度={plen if not np.isnan(plen) else '  -':>8}°")

# === 画两张图 ===
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左：节点数 vs step
ax = axes[0]
ok_steps = [r[0] for r in results if r[2]]
fail_steps = [r[0] for r in results if not r[2]]
ok_nodes = [r[1] for r in results if r[2]]
fail_nodes = [r[1] for r in results if not r[2]]
ax.plot([r[0] for r in results if r[2]], ok_nodes, 'o-', color='tab:blue',
        label='找到路径')
if fail_steps:
    ax.scatter(fail_steps, fail_nodes, marker='x', s=120, color='red',
               label='未找到（撞墙/撞边界）', zorder=5)
for s, n, ok, _ in results:
    if ok:
        ax.annotate(f"{n}", (s, n), textcoords="offset points",
                    xytext=(0, 8), ha='center', fontsize=9, color='tab:blue')
ax.set_xlabel('step（每次迈的角度，°）')
ax.set_ylabel('RRT 节点总数')
ax.set_title('step 越大，节点越少（每次迈得更远）')
ax.grid(alpha=0.3)
ax.legend()

# 右：把所有 step 的最终路径叠在同一张 C 空间背景上
ax = axes[1]
mask, n = build_obstacle_mask()
draw_background(ax, mask, n)
ax.plot(START[0], START[1], 'go', markersize=10, label='起点', zorder=3)
ax.plot(GOAL[0], GOAL[1], 'r*', markersize=14, label='终点', zorder=3)

colors = plt.cm.viridis(np.linspace(0, 1, len(STEPS)))
for (s, n_nodes, ok, _), c in zip(results, colors):
    if not ok:
        continue
    nodes, parents, _, _ = rrt_plan(START, GOAL, max_iter=MAX_ITER,
                                    step=s, seed=SEED)
    path = reconstruct_path(nodes, parents)
    xs = [p[0] for p in path]; ys = [p[1] for p in path]
    ax.plot(xs, ys, '-', color=c, linewidth=1.6, label=f'step={s:.0f}°', zorder=4)

ax.set_xlabel(r'$\theta_1$ (°)')
ax.set_ylabel(r'$\theta_2$ (°)')
ax.set_title('同一起终点，不同 step 的最终路径')
ax.legend(loc='upper right', fontsize=8)

plt.suptitle('RRT step 的影响：越大 → 节点越少，但可能撞墙失败',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'lesson3_step_compare.png'), dpi=130,
            bbox_inches='tight')
print(f"\n图已存到: {os.path.join(OUT, 'lesson3_step_compare.png')}")