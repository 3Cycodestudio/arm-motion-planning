"""
第 5 课：RRT* —— 让树在长大的同时自己修自己

运行方法：在 VS Code 终端切到本目录，执行
    python lesson5_rrt_star.py

本脚本做四件事：
1. 复用 RRT 算法跑出一条"够用但不是最优"的路径
2. 实现 RRT*（在每加一个新节点后做 rewire）
3. 同 seed 对比 RRT vs RRT* 的路径长度
4. 跑多次，看 RRT* 随迭代次数逼近最优的过程

核心思想（RRT 缺的这一步）：
  普通 RRT 只看"我接在哪根枝上最近"；
  RRT* 多看一步"我能不能让别的枝改接我后变短"。
  反复做 → 路径越来越短，但不保证全局最优（叫"渐进最优"）。
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt

# 中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 复用第 3 课的所有基础组件（场景、碰撞、画图函数全在这里）
from lesson3_rrt import (
    L1, L2, OBSTACLE,
    rrt_plan, reconstruct_path,
    is_collision, is_collision_path,
    build_obstacle_mask, draw_background, finish_axes,
    JOINT_RANGES,
)

# 起终点（在 lesson3 的 main 入口里，不在模块顶层；这里手写一份保持一致）
START = (-132.0, 16.0)
GOAL = (0.0, -178.0)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# 1. RRT* 算法
# ============================================================

def rrt_star_plan(start, goal, max_iter=3000, step=12.0,
                   goal_bias=0.10, seed=42, rewire_radius=30.0):
    """RRT*：每加一个新节点后做 choose_parent + rewire。

    返回 (nodes, parents, snapshots, success)。
    与 rrt_plan 接口一致，方便对比。
    """
    rng = np.random.default_rng(seed)
    nodes = [np.array(start, float)]
    parents = [-1]
    # 累计代价 cost[i] = 从根沿父链到 nodes[i] 的总欧氏距离
    cost = [0.0]
    edges = []
    snapshots = [[]]                      # 第 0 帧：只有根节点
    goal_np = np.array(goal, float)

    for _ in range(max_iter):
        if rng.random() < goal_bias:
            q_rand = goal_np.copy()
        else:
            q_rand = np.array([rng.uniform(*r) for r in JOINT_RANGES])

        # ---- 与 RRT 相同：找最近节点、朝它迈一步 ----
        tree = np.array(nodes)
        near_idx = int(np.argmin(np.linalg.norm(tree - q_rand, axis=1)))
        q_near = nodes[near_idx]
        d = q_rand - q_near
        nd = np.linalg.norm(d)
        if nd < 1e-6:
            continue
        q_new = q_near + step * d / nd

        if is_collision_path(q_near[0], q_near[1],
                             q_new[0], q_new[1]):
            continue                       # 撞墙就丢弃

        # ---- 加入树 ----
        new_idx = len(nodes)
        nodes.append(q_new)
        parents.append(near_idx)           # 先按 RRT 那样接到 q_near
        cost.append(cost[near_idx] + step)
        edges.append((tuple(q_near), tuple(q_new)))

        # ---- choose_parent：在 rewire_radius 内挑更优父节点 ----
        # 找 q_new 周围所有邻居
        dists_to_new = np.linalg.norm(tree - q_new, axis=1)
        neighbor_ids = np.where(dists_to_new <= rewire_radius)[0]

        # 在这些邻居里：哪一个当父节点，能让 cost(new) 最小？
        # 这里 q_new 已经按 step 大小走了一步，所以"以邻居 i 为父"的距离 ≈ dist(q_new, i)
        best_parent = near_idx
        best_cost = cost[near_idx] + step
        for nid in neighbor_ids:
            tentative = cost[nid] + dists_to_new[nid]
            if tentative < best_cost:
                # 边不能撞墙
                n_pos = nodes[nid]
                if not is_collision_path(n_pos[0], n_pos[1],
                                         q_new[0], q_new[1]):
                    best_parent = nid
                    best_cost = tentative

        # 把 q_new 的父节点换成最优的那个，边也要重画
        if best_parent != near_idx:
            # 撤掉旧边 q_near -> q_new，加上新边 best_parent -> q_new
            edges[-1] = (tuple(nodes[best_parent]), tuple(q_new))
            parents[new_idx] = best_parent
            cost[new_idx] = best_cost

        # ---- rewire：看邻居们"经过 q_new 中转"能不能更短 ----
        for nid in neighbor_ids:
            if nid == best_parent:
                continue                   # 已经是父节点，跳过
            n_pos = nodes[nid]
            # 经过 q_new 中转的累计代价 = cost[new] + |q_new - n_pos|
            new_cost_via = cost[new_idx] + np.linalg.norm(n_pos - q_new)
            if new_cost_via < cost[nid]:
                # 边 n_pos -> q_new 和 q_new -> n_pos 都不能撞墙
                if not is_collision_path(q_new[0], q_new[1],
                                         n_pos[0], n_pos[1]):
                    # 改写：n_pos 的父节点变成 new_idx
                    old_p = nodes[parents[nid]]
                    edges.append((tuple(q_new), tuple(n_pos)))
                    parents[nid] = new_idx
                    # 递归更新 nid 子树的 cost
                    _propagate_cost(nodes, parents, cost, nid, new_cost_via)

        snapshots.append(list(edges))

        # ---- 跟 RRT 一样：够近就接到终点 ----
        if np.linalg.norm(q_new - goal_np) < step:
            if not is_collision_path(q_new[0], q_new[1],
                                     goal_np[0], goal_np[1]):
                nodes.append(goal_np.copy())
                parents.append(new_idx)
                cost.append(cost[new_idx] + np.linalg.norm(goal_np - q_new))
                edges.append((tuple(q_new), tuple(goal_np)))
                snapshots.append(list(edges))
            return nodes, parents, snapshots, True

    return nodes, parents, snapshots, False


def _propagate_cost(nodes, parents, cost, idx, new_cost):
    """递归更新 idx 及其所有子节点的 cost。"""
    cost[idx] = new_cost
    for child in range(len(parents)):
        if parents[child] == idx:
            _propagate_cost(nodes, parents, cost, child,
                            new_cost + np.linalg.norm(nodes[child] - nodes[idx]))


def rrt_star_anytime(start, goal, max_iter=3000, step=12.0,
                     goal_bias=0.10, seed=42, rewire_radius=30.0,
                     query_every=50):
    """RRT* 的 'anytime'（随时可用）版本：跑满 max_iter，不提前退出。

    每隔 query_every 次迭代，查一次"当前树上能连到终点的最短路径长度"，
    返回 (迭代数列表, 当前最优长度列表)。

    这条单调下降的曲线就是 RRT* "渐进最优" 的直接证据：
    给它越多时间，它越接近真正的最优路径。
    """
    rng = np.random.default_rng(seed)
    nodes = [np.array(start, float)]
    parents = [-1]
    cost = [0.0]
    goal_np = np.array(goal, float)
    hist_it, hist_len = [], []

    def best_len_now():
        """扫描全树，返回当前能连到终点的最短路径长度（连不上返回 inf）。"""
        best = float('inf')
        for i, nd in enumerate(nodes):
            if np.linalg.norm(nd - goal_np) < step:
                if not is_collision_path(nd[0], nd[1], goal_np[0], goal_np[1]):
                    c = cost[i] + float(np.linalg.norm(nd - goal_np))
                    if c < best:
                        best = c
        return best

    for k in range(1, max_iter + 1):
        # ---- 生长：与 rrt_star_plan 完全相同的逻辑 ----
        if rng.random() < goal_bias:
            q_rand = goal_np.copy()
        else:
            q_rand = np.array([rng.uniform(*r) for r in JOINT_RANGES])

        tree = np.array(nodes)
        near_idx = int(np.argmin(np.linalg.norm(tree - q_rand, axis=1)))
        q_near = nodes[near_idx]
        d = q_rand - q_near
        nd = np.linalg.norm(d)
        if nd < 1e-6:
            continue
        q_new = q_near + step * d / nd
        if is_collision_path(q_near[0], q_near[1], q_new[0], q_new[1]):
            continue

        new_idx = len(nodes)
        nodes.append(q_new)
        parents.append(near_idx)
        cost.append(cost[near_idx] + step)

        dists_to_new = np.linalg.norm(tree - q_new, axis=1)
        neighbor_ids = np.where(dists_to_new <= rewire_radius)[0]
        best_parent, best_cost = near_idx, cost[near_idx] + step
        for nid in neighbor_ids:
            tentative = cost[nid] + dists_to_new[nid]
            if tentative < best_cost:
                n_pos = nodes[nid]
                if not is_collision_path(n_pos[0], n_pos[1],
                                         q_new[0], q_new[1]):
                    best_parent, best_cost = nid, tentative
        if best_parent != near_idx:
            parents[new_idx] = best_parent
            cost[new_idx] = best_cost

        for nid in neighbor_ids:
            if nid == best_parent:
                continue
            n_pos = nodes[nid]
            new_cost_via = cost[new_idx] + np.linalg.norm(n_pos - q_new)
            if new_cost_via < cost[nid]:
                if not is_collision_path(q_new[0], q_new[1],
                                         n_pos[0], n_pos[1]):
                    parents[nid] = new_idx
                    _propagate_cost(nodes, parents, cost, nid, new_cost_via)

        # ---- 每隔 query_every 次，记录当前最优路径长度 ----
        if k % query_every == 0:
            b = best_len_now()
            if b < float('inf'):
                hist_it.append(k)
                hist_len.append(b)

    return hist_it, hist_len


def path_length(path):
    """路径总长（累计欧氏距离，°）。"""
    s = 0.0
    for k in range(1, len(path)):
        s += math.hypot(path[k][0] - path[k-1][0],
                        path[k][1] - path[k-1][1])
    return s


# ============================================================
# 2. 同 seed 对比：RRT vs RRT*
# ============================================================

print("=== 0. 自检 ===")
mask, n = build_obstacle_mask()
START = np.array(START, float)
GOAL = np.array(GOAL, float)
print(f"  起点 {tuple(START)}, 终点 {tuple(GOAL)}")
print(f"  起点终点直线是否撞墙: {is_collision_path(START[0], START[1], GOAL[0], GOAL[1])}")

print("\n=== 1. 同 seed 对比 RRT vs RRT*（max_iter=1500）===")
SEED = 42
ITER = 1500
STEP = 12.0
RADIUS = 30.0

# RRT
nodes_r, parents_r, _, ok_r = rrt_plan(tuple(START), tuple(GOAL),
                                       max_iter=ITER, step=STEP, seed=SEED)
path_r = reconstruct_path(nodes_r, parents_r)
len_r = path_length(path_r)

# RRT*
nodes_s, parents_s, _, ok_s = rrt_star_plan(tuple(START), tuple(GOAL),
                                            max_iter=ITER, step=STEP,
                                            seed=SEED, rewire_radius=RADIUS)
path_s = reconstruct_path(nodes_s, parents_s)
len_s = path_length(path_s)

print(f"  RRT  : 节点 {len(nodes_r):4d}, 路径点数 {len(path_r):3d}, "
      f"长度 {len_r:7.2f}°  {'成功' if ok_r else '失败'}")
print(f"  RRT*: 节点 {len(nodes_s):4d}, 路径点数 {len(path_s):3d}, "
      f"长度 {len_s:7.2f}°  {'成功' if ok_s else '失败'}")
print(f"  RRT* 比 RRT 缩短: {(1 - len_s/len_r)*100:.1f}%")

# ============================================================
# 3. 收敛曲线：迭代越多越逼近最优
# ============================================================

print("\n=== 2. 渐进最优：给它更多时间，路径越来越短 ===")
hist_it, hist_len = rrt_star_anytime(tuple(START), tuple(GOAL),
                                     max_iter=4000, step=STEP,
                                     seed=SEED, rewire_radius=RADIUS,
                                     query_every=100)
if hist_len:
    print(f"  第一次能连到终点: 迭代 {hist_it[0]}, 长度 {hist_len[0]:.2f}°")
    for it, l in zip(hist_it[::5], hist_len[::5]):
        print(f"    iter={it:5d}  当前最短 {l:7.2f}°")
    print(f"  最终（迭代 {hist_it[-1]}）: {hist_len[-1]:.2f}°")
    print(f"  最长到最短改善: {(1 - hist_len[-1]/hist_len[0])*100:.1f}%")
else:
    print("  没连上终点，请调大 max_iter 或 step")

# ============================================================
# 4. rewire_radius 灵敏度扫描
# ============================================================

print("\n=== 3. rewire_radius 灵敏度扫描（iter=1500）===")
radii = [10.0, 20.0, 30.0, 50.0, 80.0]
rad_lens = []
for r in radii:
    nodes_sr, parents_sr, _, _ = rrt_star_plan(tuple(START), tuple(GOAL),
                                               max_iter=1500, step=STEP,
                                               seed=SEED, rewire_radius=r)
    ps = reconstruct_path(nodes_sr, parents_sr)
    plen = path_length(ps)
    rad_lens.append(plen)
    print(f"  rewire_radius={r:5.1f}°  节点 {len(nodes_sr):4d}  路径长度 {plen:7.2f}°")

# ============================================================
# 5. 画图
# ============================================================

def edges_in_tree(nodes, parents):
    """返回 (端点1, 端点2) 的边列表，方便画树。"""
    edges = []
    for i, p in enumerate(parents):
        if p == -1:
            continue
        edges.append((tuple(nodes[p]), tuple(nodes[i])))
    return edges


fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# (1) RRT 路径
ax = axes[0, 0]
draw_background(ax, mask, n)
ax.plot(START[0], START[1], 'go', markersize=10, label='起点', zorder=5)
ax.plot(GOAL[0], GOAL[1], 'r*', markersize=14, label='终点', zorder=5)
# 画所有树边
for edge in edges_in_tree(nodes_r, parents_r):
    (x1, y1), (x2, y2) = edge
    ax.plot([x1, x2], [y1, y2], '-', color='lightblue', linewidth=0.6, alpha=0.6, zorder=1)
# 画最终路径
xs = [p[0] for p in path_r]; ys = [p[1] for p in path_r]
ax.plot(xs, ys, 'b-', linewidth=2.5, label=f'RRT 路径 ({len_r:.1f}°)', zorder=3)
ax.plot(xs, ys, 'b.', markersize=4, zorder=3)
ax.set_title(f'RRT：节点 {len(nodes_r)} 个, 路径 {len(path_r)} 点, 长度 {len_r:.2f}°',
             fontsize=12)
finish_axes(ax)

# (2) RRT* 路径
ax = axes[0, 1]
draw_background(ax, mask, n)
ax.plot(START[0], START[1], 'go', markersize=10, label='起点', zorder=5)
ax.plot(GOAL[0], GOAL[1], 'r*', markersize=14, label='终点', zorder=5)
for edge in edges_in_tree(nodes_s, parents_s):
    (x1, y1), (x2, y2) = edge
    ax.plot([x1, x2], [y1, y2], '-', color='lightblue', linewidth=0.6, alpha=0.6, zorder=1)
xs = [p[0] for p in path_s]; ys = [p[1] for p in path_s]
ax.plot(xs, ys, 'm-', linewidth=2.5, label=f'RRT* 路径 ({len_s:.1f}°)', zorder=3)
ax.plot(xs, ys, 'm.', markersize=4, zorder=3)
ax.set_title(f'RRT*：节点 {len(nodes_s)} 个, 路径 {len(path_s)} 点, 长度 {len_s:.2f}°\n'
             f'(比 RRT 缩短 {(1-len_s/len_r)*100:.1f}%)',
             fontsize=12)
finish_axes(ax)

# (3) 渐进最优曲线
ax = axes[1, 0]
ax.plot(hist_it, hist_len, '-', color='magenta', linewidth=2.2, label='RRT* 当前最优')
ax.axhline(len_r, color='blue', linestyle='--', linewidth=2,
           label=f'RRT 最终结果 ({len_r:.1f}°)')
ax.set_xlabel('迭代次数（采样次数）', fontsize=11)
ax.set_ylabel('当前最好路径长度 (°)', fontsize=11)
ax.set_title('RRT* 的渐进最优：采样越多，路径越短', fontsize=12)
ax.grid(alpha=0.3)
ax.legend(fontsize=10)

# (4) rewire_radius 灵敏度
ax = axes[1, 1]
ax.plot(radii, rad_lens, 'g^-', linewidth=2, markersize=10)
ax.set_xlabel('rewire_radius (°)', fontsize=11)
ax.set_ylabel('最终路径长度 (°)', fontsize=11)
ax.set_title('rewire_radius 灵敏度扫描', fontsize=12)
ax.grid(alpha=0.3)
for r, l in zip(radii, rad_lens):
    ax.annotate(f'{l:.1f}', (r, l), textcoords='offset points',
                xytext=(8, 8), fontsize=9)

plt.tight_layout()
out_png = os.path.join(OUT_DIR, 'lesson5_rrt_star.png')
plt.savefig(out_png, dpi=130)
print(f"\n图片已保存到: {out_png}")
print(f"图：左上 RRT，右上 RRT*，左下收敛曲线，右下 rewire_radius 灵敏度")