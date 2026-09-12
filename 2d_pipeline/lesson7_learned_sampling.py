"""
第 7 课：学习的成分 —— 用神经网络学"哪里是通路"，引导 RRT 撒点。

对应课题题目《基于学习和抽样的混合运动规划算法研究》里的"基于学习"。

故事线（三步蒸馏）：
  1. 老师（BFS 穷举）  ：2° 网格全图搜索，找出起点到终点的最短路径走廊
  2. 学生（小神经网络）：输入关节角，输出"这里是通路"的概率，跟老师学
  3. 混合（学习引导 RRT）：撒点时按学生的建议撒（概率高的区域多撒），其余仍均匀撒
     —— 学习提供方向感，抽样保证完备性，这就是"混合"

运行方法：python lesson7_learned_sampling.py
本脚本做四件事：
  A. BFS 求路径 + 距离变换生成"走廊"训练标签
  B. 手写 2->48->1 神经网络（numpy，不用任何深度学习框架），加权梯度下降训练
  C. 学习引导 RRT（按网络概率分布撒点）与均匀 RRT 对比
  D. 20 个种子统计：首次成功各需要多少节点
产物：lesson7_learned_sampling.png
"""
import os
from collections import deque

import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from lesson3_rrt import (is_collision, is_collision_path, JOINT_RANGES,
                         rrt_plan, reconstruct_path)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- 场景（与前面各课一致，注意 lesson3 的 START/GOAL 在 __main__ 里 import 不到）----
START = (-132.0, 16.0)
GOAL = (0.0, -178.0)

# ---- 本课参数 ----
GRID = 2.0          # 老师用的网格分辨率（度）
N = int(round(360.0 / GRID))          # 180 x 180
H = 48              # 隐藏层神经元个数
EPOCHS = 3000       # 训练轮数
LR = 0.15           # 学习率
TAU = 6.0           # 走廊标签的衰减宽度（格）
W_CORRIDOR = 3.0    # 训练时给走廊样本的损失权重（强调"学通路"）
EPS_INF = 0.90      # 撒点时"听网络建议"的概率
GOAL_BIAS = 0.10    # 直接偏置到终点的概率（与第 3 课一致）
SAMPLE_EXPO = 5.0   # 采样权重 = 概率^指数（越大越集中在高概率区）
SEEDS = list(range(20))


# ============================================================
# 1. 老师：网格 + BFS 最短路 + 距离变换 -> 走廊标签
#    本课统一用"显示坐标约定"：格子 (p, q)：
#      p ↔ 关节1：θ1 = -180 + (p+0.5)*GRID （格子中心）
#      q ↔ 关节2：θ2 = -180 + (q+0.5)*GRID
# ============================================================
def cell_center(p, q):
    return -180.0 + (p + 0.5) * GRID, -180.0 + (q + 0.5) * GRID


def cell_of(t1, t2):
    return (int((t1 + 180.0) / GRID), int((t2 + 180.0) / GRID))


def build_obst():
    """显示约定下的障碍掩码 obst[p, q]。"""
    obst = np.zeros((N, N), dtype=bool)
    for p in range(N):
        for q in range(N):
            obst[p, q] = is_collision(*cell_center(p, q))
    return obst


def bfs_path(obst):
    """老师本人：4 邻域 BFS 找最短路（返回格子序列）。"""
    s, g = cell_of(*START), cell_of(*GOAL)
    parent = {s: None}
    dq = deque([s])
    while dq:
        c = dq.popleft()
        if c == g:
            break
        for dp, dq_ in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (c[0] + dp, c[1] + dq_)
            if (0 <= nb[0] < N and 0 <= nb[1] < N
                    and not obst[nb] and nb not in parent):
                parent[nb] = c
                dq.append(nb)
    if g not in parent:
        return None
    path, cur = [], g
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    return path


def corridor_label(obst, path):
    """多源 BFS 距离变换：离最短路的格距 d -> 标签 exp(-d/TAU)，障碍为 0。"""
    dist = np.full((N, N), 10 ** 9)
    dq = deque()
    for (p, q) in path:
        dist[p, q] = 0
        dq.append((p, q))
    while dq:
        p, q = dq.popleft()
        for dp, dq_ in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (p + dp, q + dq_)
            if (0 <= nb[0] < N and 0 <= nb[1] < N and not obst[nb]
                    and dist[nb] > dist[p, q] + 1):
                dist[nb] = dist[p, q] + 1
                dq.append(nb)
    y = np.exp(-dist / TAU)
    y[obst] = 0.0
    return y


# ============================================================
# 2. 学生：手写 2 -> H -> 1 网络（tanh + sigmoid，加权 MSE）
# ============================================================
def net_init(rng):
    return {'W1': rng.normal(0, 1.0, (2, H)) * 0.8,
            'b1': np.zeros(H),
            'W2': rng.normal(0, 1.0, (H, 1)) * 0.8,
            'b2': np.zeros(1)}


def net_forward(net, X):
    """X: (m,2) 归一化角。返回 (预测概率 (m,1), 隐藏层激活 (m,H))。"""
    A1 = X @ net['W1'] + net['b1']
    Z1 = np.tanh(A1)
    P = 1.0 / (1.0 + np.exp(-(Z1 @ net['W2'] + net['b2'])))
    return P, Z1


def net_train(net, X, Y, epochs, lr, wgt=None):
    """全批梯度下降（走廊样本损失加权）。返回每轮的普通 MSE（画曲线用）。"""
    if wgt is None:
        wgt = np.ones_like(Y)
    norm = wgt.sum()
    losses = []
    for ep in range(epochs):
        P, Z1 = net_forward(net, X)
        diff = P - Y
        losses.append(float(np.mean(diff ** 2)))
        dP = 2.0 * diff * wgt / norm
        dW2 = Z1.T @ dP
        db2 = dP.sum(axis=0)
        dZ1 = dP @ net['W2'].T * (1.0 - Z1 ** 2)
        dW1 = X.T @ dZ1
        db1 = dZ1.sum(axis=0)
        net['W1'] -= lr * dW1
        net['b1'] -= lr * db1
        net['W2'] -= lr * dW2
        net['b2'] -= lr * db2
    return losses


def net_predict_grid(net):
    """整张 C 空间的概率图 prob[p, q]（格心处推断，向量化一次算完）。"""
    centers = np.array([[t1 / 180.0, t2 / 180.0]
                        for t1 in (-180.0 + (np.arange(N) + 0.5) * GRID)
                        for t2 in (-180.0 + (np.arange(N) + 0.5) * GRID)])
    P, _ = net_forward(net, centers)
    return P.reshape(N, N)


# ============================================================
# 3. 混合：学习引导 RRT（与第 3 课 rrt_plan 唯一区别 = 撒点来源）
# ============================================================
def make_sampler(prob):
    """把概率图压平成累积分布，供 searchsorted 随机抽样。"""
    w = (prob ** SAMPLE_EXPO).ravel()
    cum = np.cumsum(w)
    return cum


def sample_from_table(cum, rng):
    u = rng.random() * cum[-1]
    idx = int(np.searchsorted(cum, u))
    idx = min(idx, N * N - 1)
    p, q = divmod(idx, N)
    # 格心 + 格内扰动，避免撒点都落在同一批格心上
    t1 = -180.0 + (p + rng.random()) * GRID
    t2 = -180.0 + (q + rng.random()) * GRID
    return t1, t2


def guided_rrt_plan(start, goal, cum, max_iter=4000, step=12.0,
                    eps_inf=EPS_INF, goal_bias=GOAL_BIAS, seed=42):
    """学习引导 RRT：撒点规则 = goal_bias 偏置终点 + eps_inf 听网络 + 其余均匀。"""
    rng = np.random.default_rng(seed)
    nodes = [np.array(start, float)]
    parents = [-1]
    goal_np = np.array(goal, float)

    for _ in range(max_iter):
        u = rng.random()
        if u < goal_bias:
            q_rand = goal_np.copy()
        elif u < goal_bias + eps_inf:
            q_rand = np.array(sample_from_table(cum, rng))
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
        nodes.append(q_new)
        parents.append(near_idx)

        if np.linalg.norm(q_new - goal_np) < step:
            if not is_collision_path(q_new[0], q_new[1], goal_np[0], goal_np[1]):
                nodes.append(goal_np.copy())
                parents.append(len(nodes) - 2)
            return nodes, parents, True
    return nodes, parents, False


# ============================================================
# 主流程
# ============================================================
if __name__ == '__main__':
    print('=' * 60)
    print('第 7 课：学习引导抽样')
    print('=' * 60)

    print('\n=== A. 老师：BFS 找最短路 + 生成走廊标签 ===')
    obst = build_obst()
    path = bfs_path(obst)
    assert path is not None, 'BFS 找不到路，检查场景'
    Y_grid = corridor_label(obst, path)
    print(f'  BFS 最短路 {len(path)} 格，'
          f'走廊标签覆盖 {int((Y_grid > 0.05).sum())} 格 '
          f'({(Y_grid > 0.05).sum() / (N * N) * 100:.1f}% 的 C 空间)')

    print('\n=== B. 学生：训练 2->48->1 网络（走廊样本加权）===')
    rng = np.random.default_rng(0)
    Xs, Ys = [], []
    for p in range(N):
        for q in range(N):
            t1, t2 = cell_center(p, q)
            Xs.append([t1 / 180.0, t2 / 180.0])
            Ys.append([Y_grid[p, q]])
    X = np.array(Xs); Y = np.array(Ys)
    wgt = 1.0 + (Y > 0.3) * (W_CORRIDOR - 1.0)
    net = net_init(rng)
    losses = net_train(net, X, Y, EPOCHS, LR, wgt)
    print(f'  训练 {EPOCHS} 轮：损失 {losses[0]:.4f} -> {losses[-1]:.4f}')
    prob = net_predict_grid(net)
    # 学得像不像：算概率图与真值的相关系数 + 概率质量集中度
    corr = float(np.corrcoef(prob.ravel(), Y_grid.ravel())[0, 1])
    w_mass = (prob ** SAMPLE_EXPO).ravel()
    in_cor = (Y_grid > 0.3).ravel()
    mass = float(w_mass[in_cor].sum() / w_mass.sum())
    print(f'  网络概率图 vs 老师走廊的相关系数：{corr:.3f}（1=完全一致）')
    print(f'  走廊只占面积 {in_cor.mean()*100:.1f}%，却分走 {mass*100:.0f}% 的撒点概率')

    print('\n=== C. 混合：学习引导 RRT vs 均匀 RRT ===')
    cum = make_sampler(prob)
    nodes_g, parents_g, ok_g = guided_rrt_plan(START, GOAL, cum, seed=42)
    nodes_u, parents_u, _, ok_u = rrt_plan(START, GOAL, max_iter=4000, step=12.0, seed=42)
    assert ok_g and ok_u
    len_u = sum(1 for p in parents_u if p != -1)
    print(f'  同一 seed=42：均匀 RRT 成功时树上 {len(nodes_u)} 节点；'
          f'学习引导 RRT 只要 {len(nodes_g)} 节点')

    print('\n=== D. 20 个种子的统计 ===')
    stats_u, stats_g = [], []
    for sd in SEEDS:
        nu, _, _, ok1 = rrt_plan(START, GOAL, max_iter=4000, step=12.0, seed=sd)
        ng, _, ok2 = guided_rrt_plan(START, GOAL, cum, seed=sd)
        assert ok1 and ok2, f'seed={sd} 有算法失败，需检查'
        stats_u.append(len(nu))
        stats_g.append(len(ng))
    mu_u, mu_g = np.mean(stats_u), np.mean(stats_g)
    print(f'  均匀 RRT   ：平均 {mu_u:6.1f} 节点（±{np.std(stats_u):.1f}）')
    print(f'  学习引导   ：平均 {mu_g:6.1f} 节点（±{np.std(stats_g):.1f}）')
    print(f'  --> 学习引导把探索量压到均匀版的 {mu_g / mu_u * 100:.0f}%')

    # ---------------- 画图 ----------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    edges = np.linspace(-180, 180, N + 1)

    # (0,0) 老师的走廊标签
    ax = axes[0, 0]
    ax.pcolormesh(edges, edges, np.where(obst, np.nan, Y_grid).T,
                  cmap='viridis', vmin=0, vmax=1)
    ax.plot(START[0], START[1], 'go', ms=10, zorder=5, label='起点')
    ax.plot(GOAL[0], GOAL[1], 'r*', ms=14, zorder=5, label='终点')
    ax.set_title('老师：BFS 最短路走廊标签\n（黄=通路，标签随距离衰减）', fontsize=11)
    ax.legend(loc='upper left', fontsize=8)

    # (0,1) 学生学到的概率图
    ax = axes[0, 1]
    ax.pcolormesh(edges, edges, prob.T, cmap='viridis', vmin=0, vmax=1)
    ax.plot(START[0], START[1], 'go', ms=10, zorder=5)
    ax.plot(GOAL[0], GOAL[1], 'r*', ms=14, zorder=5)
    ax.set_title(f'学生：网络学到的通路概率\n（相关系数 {corr:.2f}——形状像但更模糊）', fontsize=11)

    # (0,2) 训练损失曲线
    ax = axes[0, 2]
    ax.plot(losses, color='tab:blue', lw=2)
    ax.set_xlabel('训练轮数'); ax.set_ylabel('均方误差')
    ax.set_title(f'学习的过程：损失单调下降\n（{EPOCHS} 轮加权梯度下降，纯 numpy 手写）', fontsize=11)
    ax.grid(alpha=0.3)

    # (1,0) 均匀 RRT 的树
    ax = axes[1, 0]
    ax.pcolormesh(edges, edges, obst.T, cmap=ListedColormap(['white', '#B4B2A9']))
    for i, pa in enumerate(parents_u):
        if pa == -1:
            continue
        ax.plot([nodes_u[pa][0], nodes_u[i][0]], [nodes_u[pa][1], nodes_u[i][1]],
                color='tab:blue', lw=0.6, alpha=0.6)
    pu = reconstruct_path(nodes_u, parents_u)
    ax.plot([q[0] for q in pu], [q[1] for q in pu], 'k-', lw=2.2, zorder=5)
    ax.plot(START[0], START[1], 'go', ms=10, zorder=6)
    ax.plot(GOAL[0], GOAL[1], 'r*', ms=14, zorder=6)
    ax.set_title(f'均匀撒点 RRT 的树\n{len(nodes_u)} 节点才碰到终点（seed=42）', fontsize=11)

    # (1,1) 学习引导 RRT 的树（背景 = 学生概率图）
    ax = axes[1, 1]
    ax.pcolormesh(edges, edges, prob.T, cmap='viridis', vmin=0, vmax=1)
    for i, pa in enumerate(parents_g):
        if pa == -1:
            continue
        ax.plot([nodes_g[pa][0], nodes_g[i][0]], [nodes_g[pa][1], nodes_g[i][1]],
                color='w', lw=0.8, alpha=0.8)
    pg = reconstruct_path(nodes_g, parents_g)
    ax.plot([q[0] for q in pg], [q[1] for q in pg], 'r-', lw=2.2, zorder=5)
    ax.plot(START[0], START[1], 'go', ms=10, zorder=6)
    ax.plot(GOAL[0], GOAL[1], 'r*', ms=14, zorder=6)
    ax.set_title(f'学习引导 RRT 的树\n{len(nodes_g)} 节点就碰到终点（同 seed）', fontsize=11)

    # (1,2) 20 seed 统计
    ax = axes[1, 2]
    xs = [0, 1]
    ax.bar(xs, [mu_u, mu_g], width=0.5, color=['#85B7EB', '#5DCAA5'])
    ax.scatter([0] * len(SEEDS), stats_u, color='tab:blue', s=22, zorder=5,
               label='各 seed 实测')
    ax.scatter([1] * len(SEEDS), stats_g, color='#0F6E56', s=22, zorder=5)
    ax.set_xticks(xs)
    ax.set_xticklabels(['均匀撒点\nRRT', '学习引导\nRRT'])
    ax.set_ylabel('首次成功所需树节点数')
    ax.set_title(f'20 个种子对比：探索量压到 {mu_g / mu_u * 100:.0f}%\n'
                 f'（{mu_u:.0f} → {mu_g:.0f} 节点）', fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y')

    # 只给 4 张 C 空间图设"关节角"坐标；损失曲线和柱状图各自保留自己的轴
    for ax in [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]:
        ax.set_xlabel('关节1 (°)'); ax.set_ylabel('关节2 (°)')
        ax.set_xlim(-185, 185); ax.set_ylim(-185, 185)
    axes[1, 2].set_xlabel('')

    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, 'lesson7_learned_sampling.png')
    plt.savefig(out_png, dpi=130)
    print(f'\n图片已保存到: {out_png}')
    print('一句话总结：学习把"往哪撒点"的知识从昂贵的穷举老师手里蒸馏出来，'
          '喂给廉价的抽样算法——这就是"基于学习和抽样的混合规划"。')
