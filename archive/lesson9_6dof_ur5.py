# -*- coding: utf-8 -*-
"""
lesson9_6dof_ur5.py —— 第9课：把规划对象升级为真实的六自由度 UR5 机械臂

运行方法（在 VS Code 终端里）：~
    python lesson9_6dof_ur5.py

本脚本做哪几件事：
    1) 用 UR5 官方 DH 参数（开源 ur_description 同源数据）搭 6 关节正运动学；
    2) 每根杆用"胶囊体"近似（线段 + 半径），与球形障碍做碰撞检测；
    3) 随机撒 2000 个关节构型，统计碰撞率（6 维 C 空间的"体检"）；
    4) 画出无碰构型和碰撞构型的机械臂三维图对照。

关键数字：DH 参数来自 UR5 官方文档 / ros-industrial/ur_description（开源）。
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------------- UR5 标准 DH 参数（开源数据，单位：米） ----------------
# 顺序：theta 偏置, d, a, alpha
DH = [
    (0.0,  0.089159,  0.0,      np.pi / 2),   # 关节1 肩部
    (0.0,  0.0,      -0.42500,  0.0),         # 关节2 大臂
    (0.0,  0.0,      -0.39225,  0.0),         # 关节3 小臂
    (0.0,  0.10915,   0.0,      np.pi / 2),   # 关节4 腕部1
    (0.0,  0.09465,   0.0,     -np.pi / 2),   # 关节5 腕部2
    (0.0,  0.0823,    0.0,      0.0),         # 关节6 腕部3（法兰）
]
JOINT_LIMIT = np.pi * 2.0        # UR5 各关节 ±360°
LINK_RADIUS = [0.07, 0.07, 0.06, 0.05, 0.05, 0.04]   # 各杆胶囊半径（近似）

# 场景：工作空间里放一个球形障碍
OBSTACLE = np.array([0.55, 0.25, 0.45])    # 圆心（米）
OBST_R   = 0.16                            # 半径


def dh_matrix(theta, d, a, alpha):
    """标准 DH 变换：Rot_z(theta)·Trans_z(d)·Trans_x(a)·Rot_x(alpha)"""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,  sa,      ca,      d],
        [0.0,  0.0,     0.0,     1.0],
    ])


def fk_all_joints(q):
    """返回 7 个点：基座 + 各关节原点（世界系），用于画胶囊杆"""
    pts = [np.zeros(3)]
    T = np.eye(4)
    for i, (th0, d, a, al) in enumerate(DH):
        T = T @ dh_matrix(q[i] + th0, d, a, al)
        pts.append(T[:3, 3].copy())
    return np.array(pts)


def seg_point_dist(p, a, b):
    """点 p 到线段 ab 的最短距离"""
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(p - (a + t * ab))


def is_collision(q):
    """6 关节构型是否与球形障碍相碰（逐杆胶囊检测）"""
    pts = fk_all_joints(q)
    for i in range(6):
        a, b = pts[i], pts[i + 1]
        if seg_point_dist(OBSTACLE, a, b) < OBST_R + LINK_RADIUS[i]:
            return True
    return False


def draw_arm(ax, q, color, label):
    pts = fk_all_joints(q)
    for i in range(6):
        a, b = pts[i], pts[i + 1]
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                color=color, linewidth=8 - i * 0.8, solid_capstyle='round')
    ax.scatter(pts[1:, 0], pts[1:, 1], pts[1:, 2], color=color, s=18)
    # 末端执行器
    ax.scatter(*pts[-1], color=color, marker='^', s=60)
    ax.text(pts[-1][0], pts[-1][1], pts[-1][2], ' 末端', fontsize=9, color=color)


if __name__ == '__main__':
    rng = np.random.default_rng(1)

    # --- 1. 随机 2000 构型，统计碰撞率 ---
    N = 2000
    coll = sum(is_collision(rng.uniform(-JOINT_LIMIT, JOINT_LIMIT, 6)) for _ in range(N))
    print(f'随机 {N} 个 6 关节构型：{coll} 个撞障碍（碰撞率 {coll / N * 100:.1f}%）')
    print(f'UR5 工作半径约 0.85 m，障碍球心 {OBSTACLE}，半径 {OBST_R} m')

    # --- 2. 找一个无碰构型和一个碰撞构型做对照 ---
    q_ok = rng.uniform(-JOINT_LIMIT, JOINT_LIMIT, 6)
    while is_collision(q_ok):
        q_ok = rng.uniform(-JOINT_LIMIT, JOINT_LIMIT, 6)
    q_bad = rng.uniform(-JOINT_LIMIT, JOINT_LIMIT, 6)
    while not is_collision(q_bad):
        q_bad = rng.uniform(-JOINT_LIMIT, JOINT_LIMIT, 6)
    print(f'无碰构型关节角(np.degrees)：{np.round(np.degrees(q_ok), 1)}')
    print(f'碰撞构型关节角(np.degrees)：{np.round(np.degrees(q_bad), 1)}')

    # --- 3. 三维对照图 ---
    fig = plt.figure(figsize=(11, 5))
    for k, (q, color, title) in enumerate([
            (q_ok,  '#2E70B8', '无碰撞构型（大臂/小臂从障碍旁绕过）'),
            (q_bad, '#C0392B', '碰撞构型（小臂穿过障碍球）')]):
        ax = fig.add_subplot(1, 2, k + 1, projection='3d')
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        ax.plot_surface(OBSTACLE[0] + OBST_R * np.cos(u) * np.sin(v),
                        OBSTACLE[1] + OBST_R * np.sin(u) * np.sin(v),
                        OBSTACLE[2] + OBST_R * np.cos(v),
                        color='#9AA4AE', alpha=0.45)
        draw_arm(ax, q, color, title)
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
        ax.set_xlim(-0.9, 0.9); ax.set_ylim(-0.9, 0.9); ax.set_zlim(0, 1.1)
        ax.set_title(f'UR5 六自由度机械臂\n{title}', fontsize=11)
        ax.view_init(elev=18, azim=-60)
    plt.tight_layout()
    plt.savefig('lesson9_ur5_fk.png', dpi=150, bbox_inches='tight')
    print('已保存：lesson9_ur5_fk.png')

    # --- 4. 下一步预告 ---
    print('\n下一步：把第3/5/7课的 RRT / RRT* / 学习引导撒点原样搬进这 6 维关节空间')
