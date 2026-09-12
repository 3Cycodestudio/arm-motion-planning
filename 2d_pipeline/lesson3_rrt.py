"""
第 3 课：RRT（快速探索随机树），并在 2D C 空间上与 BFS 做对比。

复用了 lesson2 的运动学 / 碰撞函数（forward_kinematics / is_collision）。
本课新增：
  - rrt_plan()  基础 RRT：随机采样 → 找最近邻 → 迈一小步 → 碰撞检查 → 接上树
  - 生成树生长动画 lesson3_rrt_growth.gif
  - 生成 BFS vs RRT 对比图 lesson3_bfs_vs_rrt.png
  - 生成"连通性反例"图 lesson3_unreachable.png（起终点被禁区隔开时，两种算法都无解）
"""
import os
from collections import deque

import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.collections import LineCollection

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============ 与 lesson2 共用的运动学 / 碰撞 ============
L1, L2 = 200.0, 180.0
OBSTACLE = (175.0, 95.0, 43.0)      # (cx, cy, r)
JOINT_RANGES = [(-180.0, 180.0), (-180.0, 180.0)]
ANGLE_STEP = 2.0                     # 画背景网格用的角度分辨率


def forward_kinematics(t1, t2):
    a1 = np.radians(t1); a12 = np.radians(t1 + t2)
    elbow = (L1 * np.cos(a1), L1 * np.sin(a1))
    end = (elbow[0] + L2 * np.cos(a12), elbow[1] + L2 * np.sin(a12))
    return (0.0, 0.0), (float(elbow[0]), float(elbow[1])), (float(end[0]), float(end[1]))


def _seg_point_dist(px, py, ax, ay, bx, by):
    abx, aby = bx - ax, by - ay
    L2q = abx ** 2 + aby ** 2
    if L2q == 0:
        return np.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / L2q))
    return np.hypot(px - (ax + t * abx), py - (ay + t * aby))


def is_collision(t1, t2, margin=3.0):
    (_, elbow, end) = forward_kinematics(t1, t2)
    ox, oy, r = OBSTACLE
    rs = r + margin
    d1 = _seg_point_dist(ox, oy, 0, 0, elbow[0], elbow[1])
    d2 = _seg_point_dist(ox, oy, elbow[0], elbow[1], end[0], end[1])
    return (d1 < rs) or (d2 < rs)


def is_collision_path(t1a, t2a, t1b, t2b, sub=0.5, margin=3.0):
    """C 空间里从 A 到 B 的连线是否全程无碰撞（沿线段细分采样）。"""
    n = max(2, int(np.hypot(t1b - t1a, t2b - t2a) / sub) + 1)
    for s in np.linspace(0.0, 1.0, n):
        if is_collision(t1a + s * (t1b - t1a), t2a + s * (t2b - t2a), margin):
            return True
    return False


def build_obstacle_mask():
    """整个 C 空间的禁区掩码（用于背景渲染）。"""
    n = int(round(360.0 / ANGLE_STEP))
    m = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            m[i, j] = is_collision(-180.0 + i * ANGLE_STEP, 180.0 - j * ANGLE_STEP)
    return m, n


def check_background_orientation(mask, n):
    """自检：确认背景图的朝向 + 对齐都没错。

    做法：另外按"显示顺序"老老实实重建一张图——
      第 p 列代表 θ1 = -180 + (p-1)*step，第 q 行代表 θ2 = -180 + q*step
    再和 draw_background 里实际用的数组逐格比对。不一样就说明画反/画偏了。
    """
    step = 360.0 / n
    direct = np.zeros((n + 2, n + 2), dtype=bool)
    for q in range(n + 2):
        for p in range(n + 2):
            direct[q, p] = is_collision(-180.0 + (p - 1) * step,
                                        -180.0 + q * step)
    drawn = np.pad(np.flipud(mask.T), 1, mode='wrap')
    ok = bool(np.array_equal(direct, drawn))
    print(f"  [自检] 背景图朝向与对齐："
          f"{'通过（θ1 横轴 / θ2 纵轴，格子中心对得上角度）' if ok else '!!! 失败：图画反或画偏了 !!!'}")
    return ok


# ================= RRT =================
def rrt_plan(start, goal, max_iter=3000, step=12.0, goal_bias=0.10, seed=42):
    """基础 RRT。
    返回 (nodes, parents, snapshots, success)。
    snapshots[k] = 第 k 个成功节点加入时的边列表（用于回放生长过程）。
    """
    rng = np.random.default_rng(seed)
    nodes = [np.array(start, float)]
    parents = [-1]
    edges = []
    snapshots = [[]]                      # 第 0 帧：只有根节点
    goal_np = np.array(goal, float)

    for _ in range(max_iter):
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
            continue                       # 撞了，这一轮丢弃（不记录帧，动画更干净）

        nodes.append(q_new)
        parents.append(near_idx)
        edges.append((tuple(q_near), tuple(q_new)))
        snapshots.append(list(edges))

        if np.linalg.norm(q_new - goal_np) < step:
            if not is_collision_path(q_new[0], q_new[1], goal_np[0], goal_np[1]):
                nodes.append(goal_np.copy())
                parents.append(len(nodes) - 2)
                edges.append((tuple(q_new), tuple(goal_np)))
                snapshots.append(list(edges))
            return nodes, parents, snapshots, True

    return nodes, parents, snapshots, False


def reconstruct_path(nodes, parents):
    path, cur = [], len(nodes) - 1
    while cur != -1:
        path.append(nodes[cur]); cur = parents[cur]
    path.reverse()
    return path


# ================= BFS =================
def bfs_plan(start, goal, step=2.0, want_visited=False):
    """网格 + BFS。返回 (path, explored, success[, visited])。"""
    n = int(round(360.0 / step))
    grid = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            grid[i, j] = is_collision(-180.0 + i * step, 180.0 - j * step)

    def a2c(t1, t2):
        return (max(0, min(n - 1, int(round((t1 + 180.0) / step)))),
                max(0, min(n - 1, int(round((180.0 - t2) / step)))))

    si, sj = a2c(*start); gi, gj = a2c(*goal)
    if grid[si, sj] or grid[gi, gj]:
        return (None, 0, False, set()) if want_visited else (None, 0, False)

    parent = {(si, sj): None}
    q = deque([(si, sj)])
    while q:
        i, j = q.popleft()
        if (i, j) == (gi, gj):
            path, cur = [], (i, j)
            while cur is not None:
                path.append(cur); cur = parent[cur]
            path.reverse()
            return (path, len(parent), True, set(parent)) if want_visited \
                else (path, len(parent), True)
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < n and 0 <= nj < n and not grid[ni, nj] and (ni, nj) not in parent:
                parent[(ni, nj)] = (i, j); q.append((ni, nj))
    return (None, len(parent), False, set(parent)) if want_visited else (None, len(parent), False)


# ================= 可视化 =================
def draw_background(ax, mask, n):
    """把 C 空间禁区当背景铺上去。

    这里踩过两个坑，都记下来免得再犯：

    坑 1（图会转 90°）：mask 的下标是 mask[i, j]——i 数 θ1、j 数 θ2；
        而 imshow 规定"行在前、列在后"，所以必须先转置，否则 θ1 跑到纵轴上。
    坑 2（每格偏半格）：imshow 把数组平均铺满 extent，结果每个格子的中心
        会偏离它代表的角度半格（这里 1°）。看图时会误以为"路径贴着灰区"。
        办法是四周各补一圈（周期延拓）再把 extent 外扩半格，
        让格子中心正好落在它代表的角度上。
    """
    step = 360.0 / n
    mask_disp = np.pad(np.flipud(mask.T), 1, mode='wrap')
    xl = -180.0 - 1.5 * step
    xr = xl + (n + 2) * step
    yt = -180.0 - 0.5 * step
    yb = yt + (n + 2) * step
    ax.imshow(mask_disp, extent=(xl, xr, yb, yt), cmap='Greys',
              alpha=0.4, origin='upper', interpolation='nearest', zorder=1)


def finish_axes(ax):
    ax.set_xlim(-180, 180); ax.set_ylim(180, -180); ax.set_aspect('equal')
    ax.set_xlabel('θ1 (度)'); ax.set_ylabel('θ2 (度)')


def make_rrt_animation(start, goal, step=12.0, max_iter=3000, seed=42):
    nodes, parents, snaps, ok = rrt_plan(start, goal, max_iter=max_iter, step=step, seed=seed)
    mask, n = build_obstacle_mask()

    fig, ax = plt.subplots(figsize=(7.6, 7.6))
    draw_background(ax, mask, n)
    ax.plot(*start, 'o', color='#3B6D11', markersize=13, zorder=6)
    ax.plot(*goal, 'o', color='#8250df', markersize=13, zorder=6)
    ax.plot(*start, 'o', color='#3B6D11', markersize=13, zorder=6)
    ax.text(start[0], start[1], ' 起点', color='#27500A', fontsize=11, va='bottom')
    ax.text(goal[0], goal[1], ' 终点', color='#3C3489', fontsize=11, va='bottom')

    lc = LineCollection([], colors='#378ADD', linewidths=1.1, alpha=0.85, zorder=3)
    ax.add_collection(lc)
    sc = ax.scatter([], [], s=10, color='#185FA5', zorder=4)
    finish_axes(ax)
    title = ax.set_title('')

    nf = len(snaps)
    stride = max(1, nf // 220)                 # 限制总帧数，避免 GIF 过长
    frames = list(range(0, nf, stride))
    if frames[-1] != nf - 1:
        frames.append(nf - 1)

    def update(k):
        fi = frames[k]
        e = snaps[fi]
        lc.set_segments([[[a[0], a[1]], [b[0], b[1]]] for a, b in e])
        pts = np.array([b for _, b in e]) if e else np.empty((0, 2))
        sc.set_offsets(pts)
        if ok and fi == nf - 1:
            path = np.array(reconstruct_path(nodes, parents))
            ax.plot(path[:, 0], path[:, 1], '-', color='#A32D2D', lw=2.6, zorder=5)
            ax.plot(path[:, 0], path[:, 1], '-', color='#A32D2D', lw=2.6, zorder=5)
        title.set_text(f'RRT 生长中：第 {len(e)} / {len(snaps)-1} 个节点   '
                       f'{"（已抵达终点）" if (ok and fi == nf-1) else ""}')
        return lc, sc

    anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                   interval=60, blit=False, repeat=False)
    gif = os.path.join(OUT_DIR, 'lesson3_rrt_growth.gif')
    anim.save(gif, writer='pillow', fps=15)
    plt.close(fig)
    print(f"  已保存 {os.path.basename(gif)}（{len(frames)} 帧，树共 {len(nodes)} 节点）")
    return ok, len(nodes)


def make_compare(start, goal, bfs_step=2.0, rrt_step=12.0, seed=42):
    mask, n = build_obstacle_mask()
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.2))

    # ---- 左：BFS ----
    path, explored, ok, visited = bfs_plan(start, goal, step=bfs_step, want_visited=True)
    ax = axes[0]
    draw_background(ax, mask, n)
    ng = int(round(360.0 / bfs_step))
    for (i, j) in visited:
        ax.add_patch(mpatches.Rectangle((-180 + i * bfs_step - bfs_step / 2,
                                         180 - j * bfs_step - bfs_step / 2),
                                        bfs_step, bfs_step, facecolor='#378ADD',
                                        alpha=0.22, edgecolor='none', zorder=2))
    if path:
        for (i, j) in path:
            ax.add_patch(mpatches.Rectangle((-180 + i * bfs_step - bfs_step / 2,
                                             180 - j * bfs_step - bfs_step / 2),
                                            bfs_step, bfs_step, facecolor='#534AB7',
                                            alpha=0.75, edgecolor='none', zorder=3))
    ax.plot(*start, 'o', color='#3B6D11', markersize=13, zorder=6)
    ax.plot(*goal, 'o', color='#8250df', markersize=13, zorder=6)
    finish_axes(ax)
    ax.set_title(f'BFS（网格法，{bfs_step:.0f}° 一格）\n'
                 f'总格子 {ng*ng:,} 个 → 探索了 {explored:,} 个 '
                 f'（{100*explored/(ng*ng):.0f}%）',
                 fontsize=11)
    ax.text(start[0], start[1], ' 起点', color='#27500A', fontsize=10, va='bottom', zorder=7)
    ax.text(goal[0], goal[1], ' 终点', color='#3C3489', fontsize=10, va='bottom', zorder=7)

    # ---- 右：RRT ----
    nodes, parents, snaps, r_ok = rrt_plan(start, goal, step=rrt_step, seed=seed)
    ax = axes[1]
    draw_background(ax, mask, n)
    for a, b in snaps[-1]:
        ax.plot([a[0], b[0]], [a[1], b[1]], '-', color='#378ADD', lw=1.0, alpha=0.85, zorder=3)
    pts = np.array(nodes)
    ax.scatter(pts[:, 0], pts[:, 1], s=9, color='#185FA5', zorder=4)
    if r_ok:
        p = np.array(reconstruct_path(nodes, parents))
        ax.plot(p[:, 0], p[:, 1], '-', color='#A32D2D', lw=2.8, zorder=5, label='RRT 找到的路径')
        ax.legend(loc='lower right', fontsize=10)
    ax.plot(*start, 'o', color='#3B6D11', markersize=13, zorder=6)
    ax.plot(*goal, 'o', color='#8250df', markersize=13, zorder=6)
    finish_axes(ax)
    ax.set_title(f'RRT（随机抽样，步长 {rrt_step:.0f}°）\n'
                 f'只撒了 {len(nodes):,} 个节点就{"找到路" if r_ok else "未能找到"}',
                 fontsize=11)
    ax.text(start[0], start[1], ' 起点', color='#27500A', fontsize=10, va='bottom', zorder=7)
    ax.text(goal[0], goal[1], ' 终点', color='#3C3489', fontsize=10, va='bottom', zorder=7)

    plt.suptitle('同一张 C 空间、同一对起终点：BFS 扫全图 vs RRT 随机撒点', fontsize=13)
    out = os.path.join(OUT_DIR, 'lesson3_bfs_vs_rrt.png')
    plt.tight_layout(); plt.savefig(out, dpi=130, bbox_inches='tight'); plt.close(fig)
    print(f"  已保存 {os.path.basename(out)}")
    print(f"     BFS 探索 {explored:,} 格 / RRT 撒 {len(nodes):,} 点 "
          f"→ RRT 只用其 {100*len(nodes)/max(1,explored):.1f}%")
    return explored, len(nodes), ok, r_ok


def make_unreachable_demo():
    """起终点被禁区切在两个连通分量：两种算法都无解。"""
    mask, n = build_obstacle_mask()
    start, goal = (-120.0, 60.0), (150.0, -90.0)   # 故意跨分量
    _, _, bfs_ok, _ = bfs_plan(start, goal, step=2.0, want_visited=True)
    nodes, parents, snaps, r_ok = rrt_plan(start, goal, max_iter=4000, step=12.0, seed=7)

    fig, ax = plt.subplots(figsize=(8, 7.2))
    draw_background(ax, mask, n)
    for a, b in snaps[-1]:
        ax.plot([a[0], b[0]], [a[1], b[1]], '-', color='#378ADD', lw=0.9, alpha=0.8, zorder=3)
    pts = np.array(nodes)
    ax.scatter(pts[:, 0], pts[:, 1], s=8, color='#185FA5', zorder=4)
    ax.plot(*start, 'o', color='#3B6D11', markersize=13, zorder=6)
    ax.plot(*goal, 'o', color='#8250df', markersize=13, zorder=6)
    ax.text(start[0], start[1], ' 起点', color='#27500A', fontsize=11, va='bottom', zorder=7)
    ax.text(goal[0], goal[1], ' 终点', color='#3C3489', fontsize=11, va='bottom', zorder=7)
    # 标出那堵墙（竖着的，θ1≈16~42° 沿 θ2 方向整列堵死）
    ax.annotate('禁区的这堵"竖墙"\n把空间切成左右两半',
                xy=(29, 90), xytext=(95, 150),
                fontsize=11, color='#A32D2D', ha='center',
                arrowprops=dict(arrowstyle='->', color='#A32D2D', lw=1.4))
    finish_axes(ax)
    ax.set_title(f'反例：起终点被禁区隔开在两个连通分量里\n'
                 f'BFS {"找到" if bfs_ok else "找不到"}、RRT 撒了 {len(nodes):,} 点也 {"找到" if r_ok else "找不到"} '
                 f'→ 问题本身无解，任何算法都救不了',
                 fontsize=12)
    out = os.path.join(OUT_DIR, 'lesson3_unreachable.png')
    plt.tight_layout(); plt.savefig(out, dpi=130); plt.close(fig)
    print(f"  已保存 {os.path.basename(out)}")
    print(f"     结论：BFS {'有解' if bfs_ok else '无解'}，RRT {'有解' if r_ok else '无解'}"
          f" —— 两边一致，说明是问题不可行，不是算法不行")


if __name__ == "__main__":
    # 起终点选在同一连通分量内（下面 make_unreachable_demo 会证明它确实连通）
    # 注：终点特意离"竖墙"（θ1=16°~40°）留出余量，免得圆点压在灰区边上引起误会
    START = (-132.0, 16.0)
    GOAL = (0.0, -178.0)

    print("=== 0. 自检 ===")
    _mask, _n = build_obstacle_mask()
    check_background_orientation(_mask, _n)
    print(f"  起点 {START} 是否撞障碍：{is_collision(*START)}")
    print(f"  终点 {GOAL} 是否撞障碍：{is_collision(*GOAL)}")
    print()

    print("=== 1. RRT 树生长动画 ===")
    ok, cnt = make_rrt_animation(START, GOAL, step=12.0, seed=42)
    print(f"  结果：{'找到路径' if ok else '未找到'}，{cnt} 个节点")
    print()

    print("=== 2. BFS vs RRT 对比 ===")
    make_compare(START, GOAL, bfs_step=2.0, rrt_step=12.0, seed=42)
    print()

    print("=== 3. 反例：起终点跨连通分量 ===")
    make_unreachable_demo()
