"""
第 6 课：时间参数化 —— 把"几何路径"变成"电机能执行的轨迹"

运行方法：在 VS Code 终端切到本目录，执行
    python lesson6_time_param.py

前 5 课我们一直在解决"去哪"：BFS / RRT 找可行路径、平滑修形状、RRT* 逼近最优。
但那些结果都是一串关节点（几何路径），只回答了"经过哪些姿势"，
没回答"第几秒到哪、走多快、加减速多猛"。

本脚本补上这一维，分两步：
  第 1 步  弧长参数化：按累计弧长把折线均匀重采样 → 把"位置"和"进度"解耦
  第 2 步  梯形速度规划：加速 → 匀速 → 减速 → 速度连续、起止为零、加速度有界

产出：
  lesson6_time_param.png   6 张图（几何路径 / s(t) / v(t) / θ(t) / θ̇(t) / θ̈(t)）
  lesson6_trajectory.gif   机械臂按轨迹运动的动画 + 速度光标

想改参数做实验？看下面"规划参数"三行。
"""

import os
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

# 复用第 3 课的场景、碰撞检测、正运动学、画图函数
from lesson3_rrt import (
    OBSTACLE,
    rrt_plan, reconstruct_path, is_collision_path,
    build_obstacle_mask, draw_background, finish_axes,
    forward_kinematics,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
START = (-132.0, 16.0)
GOAL = (0.0, -178.0)

# ---------------- 规划参数（想改就改这三行）----------------
V_MAX = 90.0      # 路径速度上限 (°/s)：关节空间里每秒走多少度
A_MAX = 120.0     # 加速度上限 (°/s²)
W_LIM = 120.0     # 单个关节的速度上限 (°/s)，用来做安全校核
DOWN = 4          # 贪心化简时最多跳过几个 RRT 点（越大 waypoint 越少）


# ============================================================
# 1. 拿一条几何路径（沿用第 3 课的 RRT，再降采样成少量 waypoint）
# ============================================================
print("=== 1. 拿一条几何路径 ===")
nodes, parents, _, ok = rrt_plan(START, GOAL, max_iter=3000, step=12.0, seed=42)
assert ok, "RRT 没找到路，检查起终点/障碍"
raw = reconstruct_path(nodes, parents)


def simplify_greedy(path, max_gap):
    """贪心化简：能走直线就跳过中间点，但只用"验证过无碰撞"的捷径。
    这样化简后的每一段仍然安全——这是几何路径合法性的底线。"""
    out = [tuple(path[0])]
    i, n = 0, len(path)
    while i < n - 1:
        j = min(i + max_gap, n - 1)
        while j > i + 1 and is_collision_path(path[i][0], path[i][1],
                                              path[j][0], path[j][1]):
            j -= 1
        out.append(tuple(path[j]))
        i = j
    return out


P = np.array(simplify_greedy(raw, DOWN), float)   # waypoint 序列（几何路径）
# 验证：化简后每一段都无碰撞，否则这条"几何路径"根本不合法
assert all(not is_collision_path(P[k][0], P[k][1], P[k + 1][0], P[k + 1][1])
           for k in range(len(P) - 1)), "化简后有段撞障碍，请调小 DOWN"
print(f"  RRT 原始路径 {len(raw)} 个点 → 贪心化简成 {len(P)} 个 waypoint"
      f"（每段均已验证无碰撞）")

seg = np.linalg.norm(np.diff(P, axis=0), axis=1)   # 每段（直线）的长度
s_wp = np.concatenate([[0.0], np.cumsum(seg)])     # 每个 waypoint 的累计弧长
S = float(s_wp[-1])
print(f"  路径总长 S = {S:.1f}°（关节空间里的弧长）")
print(f"  各段长度(°): {np.round(seg, 1).tolist()}   ← 注意它们长短不一")


# ============================================================
# 2. 第 1 步：弧长参数化（把折线按累计长度均匀重采样）
# ============================================================
print("\n=== 2. 弧长参数化（均匀重采样）===")
N = 2000
s_grid = np.linspace(0.0, S, N)
th1_g = np.interp(s_grid, s_wp, P[:, 0])         # θ1(s)
th2_g = np.interp(s_grid, s_wp, P[:, 1])         # θ2(s)
ds = S / (N - 1)
dth1_ds = np.gradient(th1_g, ds)                 # dθ1/ds（路径切向的两个分量）
dth2_ds = np.gradient(th2_g, ds)
print(f"  重采样成 {N} 个等弧长点（每点间隔 {ds:.3f}°）")


# ============================================================
# 3. 第 2 步：梯形速度规划
# ============================================================
def trapezoid_info(S, vmax, amax):
    """先算出梯形曲线的几个关键量（不再画图，只算数）。"""
    d_acc = vmax ** 2 / (2 * amax)
    if 2 * d_acc >= S:                 # 路程太短，到不了 vmax → 三角形
        v_peak = np.sqrt(amax * S)
        t_a = v_peak / amax
        t_c = 0.0
        v_cruise = v_peak
        d_acc = 0.5 * amax * t_a ** 2
    else:                              # 正常梯形
        t_a = vmax / amax
        t_c = (S - 2 * d_acc) / vmax
        v_cruise = vmax
    return dict(t_a=t_a, t_c=t_c, d_acc=d_acc, v_cruise=v_cruise, T=2 * t_a + t_c)


def trapezoid_profile(t, S, info, amax):
    """给定时间，返回 (已走弧长 s, 速度 v, 加速度 a)。分段解析式。"""
    t = np.asarray(t, float)
    t_a, t_c, d_acc = info['t_a'], info['t_c'], info['d_acc']
    v_cruise, T = info['v_cruise'], info['T']
    s = np.empty_like(t); v = np.empty_like(t); a = np.empty_like(t)

    m1 = t <= t_a                      # 加速段：s = ½at²
    m3 = t >= t_a + t_c                # 减速段：从终点倒推
    m2 = ~(m1 | m3)                    # 匀速段

    s[m1] = 0.5 * amax * t[m1] ** 2
    v[m1] = amax * t[m1]
    a[m1] = amax

    s[m2] = d_acc + v_cruise * (t[m2] - t_a)
    v[m2] = v_cruise
    a[m2] = 0.0

    tau = T - t[m3]
    s[m3] = S - 0.5 * amax * tau ** 2
    v[m3] = amax * tau
    a[m3] = -amax
    return s, v, a


print("\n=== 3. 梯形速度规划 ===")
info = trapezoid_info(S, V_MAX, A_MAX)
T = info['T']
tg = np.linspace(0.0, T, 1200)
s_t, v_t, a_t = trapezoid_profile(tg, S, info, A_MAX)

# 由 s(t) 反查关节角 → 再求关节速度 / 加速度
th1_t = np.interp(s_t, s_grid, th1_g)
th2_t = np.interp(s_t, s_grid, th2_g)
dth1_t = np.interp(s_t, s_grid, dth1_ds) * v_t          # θ̇ = (dθ/ds)·v
dth2_t = np.interp(s_t, s_grid, dth2_ds) * v_t
ddth1_t = np.interp(s_t, s_grid, dth1_ds) * a_t         # θ̈ ≈ (dθ/ds)·a（略去拐角曲率项）
ddth2_t = np.interp(s_t, s_grid, dth2_ds) * a_t

print(f"  v_max={V_MAX}°/s, a_max={A_MAX}°/s²")
print(f"  加速段 {info['t_a']:.2f}s（走 {info['d_acc']:.1f}°）"
      f" → 匀速段 {info['t_c']:.2f}s（走 {S-2*info['d_acc']:.1f}°）"
      f" → 减速段 {info['t_a']:.2f}s")
print(f"  总时间 T = {T:.2f}s")
print(f"  关节峰值速度: 关节1 {np.max(np.abs(dth1_t)):5.1f}°/s  "
      f"关节2 {np.max(np.abs(dth2_t)):5.1f}°/s   （关节上限 {W_LIM}°/s）")
over = max(np.max(np.abs(dth1_t)), np.max(np.abs(dth2_t))) > W_LIM
print(f"  安全校核: {'!!! 超限，请调小 V_MAX !!!' if over else '通过，两个关节都没超速'}")


# ============================================================
# 4. 对照：朴素"等时间播放"（每个 waypoint 花一样的时间）
# ============================================================
print("\n=== 4. 对照：朴素等时播放（不做速度规划）===")
t_b = np.linspace(0.0, T, len(P))        # 把总时间均分给每一段
seg_dt = T / (len(P) - 1)

v_naive = np.zeros_like(tg)
dth_naive = np.zeros((2, len(tg)))
for k in range(len(P) - 1):
    m = (tg >= t_b[k]) & (tg <= t_b[k + 1])
    v_naive[m] = seg[k] / seg_dt
    dth_naive[0, m] = (P[k + 1, 0] - P[k, 0]) / seg_dt
    dth_naive[1, m] = (P[k + 1, 1] - P[k, 1]) / seg_dt

print(f"  朴素版峰值速度: 关节1 {np.max(np.abs(dth_naive[0])):5.1f}°/s  "
      f"关节2 {np.max(np.abs(dth_naive[1])):5.1f}°/s")
print(f"  注意：朴素版速度是一条条'台阶'，每级台阶都在拐点瞬间跳变")
print(f"        → 加速度在那些时刻理论上是无穷大（现实里就是抖动 / 异响 / 丢步）")


# ============================================================
# 5. 出图
# ============================================================
mask, n_ang = build_obstacle_mask()

fig, axes = plt.subplots(2, 3, figsize=(19, 9.6))

# ---- ① 几何路径 ----
ax = axes[0, 0]
draw_background(ax, mask, n_ang)
ax.plot(P[:, 0], P[:, 1], '-', color='gray', lw=1.4, alpha=0.7, zorder=2)
ax.plot(th1_g[::40], th2_g[::40], '.', color='#378ADD', ms=2.5, zorder=3,
        label='弧长均匀重采样')
ax.plot(P[:, 0], P[:, 1], 'o', color='#A32D2D', ms=6, zorder=4, label='waypoint')
ax.plot(*START, 'o', color='#3B6D11', ms=10, zorder=5)
ax.plot(*GOAL, 'r*', ms=15, zorder=5)
ax.text(START[0], START[1], ' 起点', color='#27500A', fontsize=9, va='bottom', zorder=6)
ax.text(GOAL[0], GOAL[1], ' 终点', color='#3C3489', fontsize=9, va='bottom', zorder=6)
finish_axes(ax)
ax.set_title('① 几何路径：只说了"经过哪些点"', fontsize=12)
ax.legend(loc='upper right', fontsize=8)

# ---- ② s(t) 位置-时间 ----
ax = axes[0, 1]
ax.axvspan(0, info['t_a'], color='tab:orange', alpha=0.15)
ax.axvspan(info['t_a'], info['t_a'] + info['t_c'], color='tab:green', alpha=0.10)
ax.axvspan(info['t_a'] + info['t_c'], T, color='tab:red', alpha=0.15)
ax.plot(tg, s_t, '-', color='#185FA5', lw=2.6)
ax.set_xlabel('时间 t (s)'); ax.set_ylabel('已走弧长 s (°)')
ax.set_title('② 时间律 s(t)：位置-时间曲线（梯形规划）', fontsize=12)
ax.grid(alpha=0.3)
ymid = S * 0.5
ax.text(info['t_a'] * 0.5, S * 0.05, '加速', ha='center', fontsize=9, color='#8A5A00')
ax.text(info['t_a'] + info['t_c'] * 0.5, S * 0.05, '匀速', ha='center', fontsize=9, color='#1B5E20')
ax.text(T - info['t_a'] * 0.5, S * 0.05, '减速', ha='center', fontsize=9, color='#8B1A1A')

# ---- ③ v(t) 速度-时间：梯形 vs 朴素 ----
ax = axes[0, 2]
ax.plot(tg, v_naive, '--', color='gray', lw=1.8, label='朴素等时播放')
ax.plot(tg, v_t, '-', color='#185FA5', lw=2.6, label='梯形规划（本课）')
ax.fill_between(tg, 0, v_t, color='#185FA5', alpha=0.12)
ax.set_xlabel('时间 t (s)'); ax.set_ylabel('路径速度 ds/dt (°/s)')
ax.set_title('③ 速度 v(t)：本课做到了"起止为零、加速度有界"', fontsize=12)
ax.grid(alpha=0.3); ax.legend(fontsize=9)

# ---- ④ 关节角 θ(t) ----
ax = axes[1, 0]
ax.plot(tg, th1_t, color='tab:blue', lw=2, label='关节1')
ax.plot(tg, th2_t, color='tab:orange', lw=2, label='关节2')
ax.set_xlabel('时间 t (s)'); ax.set_ylabel('关节角 (°)')
ax.set_title('④ 关节角随时间 θ(t)：由 s(t) 反查得到', fontsize=12)
ax.grid(alpha=0.3); ax.legend(fontsize=9)

# ---- ⑤ 关节速度 θ̇(t)：梯形 vs 朴素 ----
ax = axes[1, 1]
ax.plot(tg, dth_naive[0], '--', color='tab:blue', lw=1.5, alpha=0.6, label='关节1 朴素')
ax.plot(tg, dth_naive[1], '--', color='tab:orange', lw=1.5, alpha=0.6, label='关节2 朴素')
ax.plot(tg, dth1_t, color='tab:blue', lw=2.2, label='关节1 梯形')
ax.plot(tg, dth2_t, color='tab:orange', lw=2.2, label='关节2 梯形')
ax.axhline(W_LIM, color='red', ls=':', lw=1.4)
ax.axhline(-W_LIM, color='red', ls=':', lw=1.4)
ax.text(T * 0.02, W_LIM * 0.92, f'关节速度上限 {W_LIM:.0f}°/s', fontsize=8, color='red')
ax.set_xlabel('时间 t (s)'); ax.set_ylabel('关节速度 (°/s)')
ax.set_title('⑤ 关节速度：实线连续，虚线是"阶梯"', fontsize=12)
ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2)

# ---- ⑥ 关节加速度 θ̈(t) ----
ax = axes[1, 2]
for tb in t_b[1:-1]:
    ax.axvline(tb, color='gray', ls=':', lw=1.0)
ax.plot(tg, ddth1_t, color='tab:blue', lw=2, label='关节1')
ax.plot(tg, ddth2_t, color='tab:orange', lw=2, label='关节2')
ax.axhline(0, color='k', lw=0.6)
ax.set_xlabel('时间 t (s)'); ax.set_ylabel('关节加速度 (°/s²)')
ax.set_title('⑥ 关节加速度：只有 3 段常数\n灰虚线=朴素播放的拐点（速度突变→冲击）',
             fontsize=11)
ax.grid(alpha=0.3); ax.legend(fontsize=9)

plt.suptitle('第 6 课：时间参数化 —— 让几何路径变成电机能执行的轨迹',
             fontsize=16, fontweight='bold')
plt.tight_layout()
out_png = os.path.join(OUT_DIR, 'lesson6_time_param.png')
plt.savefig(out_png, dpi=125, bbox_inches='tight')
plt.close(fig)
print(f"\n图已保存: {out_png}")


# ============================================================
# 6. 动画：机械臂按轨迹运动 + 速度光标
# ============================================================
print("\n=== 5. 生成轨迹动画 ===")
ee = np.array([forward_kinematics(a, b)[2] for a, b in zip(th1_t, th2_t)])

fig2, (axL, axR) = plt.subplots(1, 2, figsize=(15, 7))

# 左：笛卡尔空间的机械臂
ox, oy, r = OBSTACLE
axL.add_patch(Circle((ox, oy), r, facecolor='gray', alpha=0.35, zorder=1))
axL.set_xlim(-420, 420); axL.set_ylim(-420, 420)
axL.set_aspect('equal'); axL.grid(alpha=0.25)
axL.set_xlabel('x (mm)'); axL.set_ylabel('y (mm)')
arm1, = axL.plot([], [], '-o', color='#185FA5', lw=7, solid_capstyle='round', zorder=4)
arm2, = axL.plot([], [], '-o', color='#378ADD', lw=6, solid_capstyle='round', zorder=4)
trail, = axL.plot([], [], '-', color='#A32D2D', lw=1.6, alpha=0.55, zorder=3)
ee_dot, = axL.plot([], [], 'o', color='#A32D2D', ms=9, zorder=5)
tL = axL.set_title('')

# 右：速度曲线 + 光标
axR.plot(tg, v_t, '-', color='#185FA5', lw=2.4)
axR.fill_between(tg, 0, v_t, color='#185FA5', alpha=0.12)
cursor = axR.axvline(0, color='#A32D2D', lw=2)
axR.set_xlim(0, T); axR.set_ylim(0, V_MAX * 1.15)
axR.set_xlabel('时间 t (s)'); axR.set_ylabel('路径速度 (°/s)')
axR.grid(alpha=0.3)
tR = axR.set_title('速度规划：v(t)（红线=当前时刻）')
infoTxt = axR.text(0.97, 0.95, '', transform=axR.transAxes,
                   ha='right', va='top', fontsize=11,
                   bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.9))

frame_idx = np.linspace(0, len(tg) - 1, 90).astype(int)


def update(k):
    i = frame_idx[k]
    t1, t2 = float(th1_t[i]), float(th2_t[i])
    (bx, by), (ex, ey), (fx, fy) = forward_kinematics(t1, t2)
    arm1.set_data([bx, ex], [by, ey])
    arm2.set_data([ex, fx], [ey, fy])
    trail.set_data(ee[:i + 1, 0], ee[:i + 1, 1])
    ee_dot.set_data([fx], [fy])
    tL.set_text(f'机械臂姿态：关节1={t1:6.1f}°  关节2={t2:6.1f}°')
    cursor.set_xdata([tg[i], tg[i]])
    phase = '加速' if tg[i] <= info['t_a'] else ('匀速' if tg[i] <= info['t_a'] + info['t_c'] else '减速')
    infoTxt.set_text(f't = {tg[i]:.2f} s\n'
                     f'v = {v_t[i]:5.1f} °/s（{phase}）\n'
                     f's = {s_t[i]:5.1f}° / {S:.1f}°')
    return arm1, arm2, trail, ee_dot


anim = animation.FuncAnimation(fig2, update, frames=len(frame_idx),
                               interval=50, blit=False, repeat=False)
gif = os.path.join(OUT_DIR, 'lesson6_trajectory.gif')
anim.save(gif, writer='pillow', fps=20)
plt.close(fig2)
print(f"动画已保存: {gif}（{len(frame_idx)} 帧）")

print("\n=== 完成 ===")
print("  一句话总结：几何路径回答'去哪'，时间参数化回答'什么时候到、走多快'。")
print("  第 1 步弧长重采样让'进度'均匀，第 2 步梯形规划让'速度'连续、起止为零、加速度有界。")
