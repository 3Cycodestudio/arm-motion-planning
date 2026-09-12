# -*- coding: utf-8 -*-
"""
第1课：二连杆机械臂的正运动学 + 轨迹动画
=================================================
这个脚本做什么：
1. 用两行三角函数公式（正运动学）算出机械臂末端位置
2. 让机械臂从"抬起"姿势平滑摆到"伸出"姿势，画出末端走过的曲线
3. 输出一张静态图 lesson1_poses.png 和一段动画 lesson1_swing.gif

怎么跑：python lesson1_forward_kinematics.py
需要：numpy、matplotlib（pip install numpy matplotlib）
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]  # 中文显示
plt.rcParams["axes.unicode_minus"] = False

# ---------- 1. 机械臂参数（想改就改，看看会发生什么） ----------
L1 = 140.0   # 连杆1长度 (mm)
L2 = 180.0   # 连杆2长度 (mm)

# ---------- 2. 正运动学：核心就这一个函数 ----------
def forward_kinematics(theta1_deg, theta2_deg):
    """输入两个关节角（度），返回 [关节2坐标, 末端坐标]"""
    t1 = np.radians(theta1_deg)
    t2 = np.radians(theta1_deg + theta2_deg)          # 注意：θ2 是相对角
    j2 = np.array([L1 * np.cos(t1), L1 * np.sin(t1)])  # 关节2位置
    ee = j2 + np.array([L2 * np.cos(t2), L2 * np.sin(t2)])  # 末端位置
    return j2, ee

# ---------- 3. 画几个典型姿势，验证公式 ----------
poses = [(90, 0), (30, 20), (0, 90), (60, -60), (-30, 100)]

fig, ax = plt.subplots(figsize=(8, 6))
for t1, t2 in poses:
    j2, ee = forward_kinematics(t1, t2)
    ax.plot([0, j2[0], ee[0]], [0, j2[1], ee[1]], "-o", lw=3, ms=8,
            label=f"θ1={t1}°, θ2={t2}° → 末端({ee[0]:.0f}, {ee[1]:.0f})")
    ax.plot(*ee, "r*", ms=14)

# 末端可达范围：以基座为圆心，半径 L1+L2 和 |L1-L2|
th = np.linspace(0, 2 * np.pi, 200)
ax.plot((L1 + L2) * np.cos(th), (L1 + L2) * np.sin(th), "k--", lw=0.8, label="可达边界")
ax.plot(0, 0, "ks", ms=10)
ax.set_aspect("equal"); ax.grid(alpha=0.3)
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
ax.set_title("第1课：不同关节角下的机械臂姿势（正运动学）")
ax.legend(fontsize=9, loc="upper left")
plt.tight_layout()
plt.savefig("lesson1_poses.png", dpi=130)
print("已保存 lesson1_poses.png")

# ---------- 4. 动画：机械臂从 (30°, 20°) 平滑摆到 (120°, 40°) ----------
t1_start, t2_start = 30, 20
t1_end, t2_end = 120, 40
frames = np.linspace(0, 1, 60)  # 60帧，均匀插值

fig2, ax2 = plt.subplots(figsize=(8, 6))
line, = ax2.plot([], [], "-o", lw=4, ms=8, color="#185FA5")
ee_dot, = ax2.plot([], [], "r*", ms=16)
trail, = ax2.plot([], [], "r--", lw=1)  # 末端轨迹

ee_hist_x, ee_hist_y = [], []

def draw_frame(k):
    global ee_hist_x, ee_hist_y
    t1 = t1_start + (t1_end - t1_start) * k
    t2 = t2_start + (t2_end - t2_start) * k
    j2, ee = forward_kinematics(t1, t2)
    line.set_data([0, j2[0], ee[0]], [0, j2[1], ee[1]])
    ee_dot.set_data([ee[0]], [ee[1]])
    ee_hist_x.append(ee[0]); ee_hist_y.append(ee[1])
    trail.set_data(ee_hist_x, ee_hist_y)
    ax2.set_title(f"机械臂摆动动画  θ1={t1:.0f}°  θ2={t2:.0f}°")
    return line, ee_dot, trail

ax2.set_xlim(-260, 260); ax2.set_ylim(-40, 280)
ax2.set_aspect("equal"); ax2.grid(alpha=0.3)
ax2.plot(0, 0, "ks", ms=10)
ax2.set_xlabel("x (mm)"); ax2.set_ylabel("y (mm)")

anim = FuncAnimation(fig2, draw_frame, frames=frames, blit=True)
anim.save("lesson1_swing.gif", writer=PillowWriter(fps=20))
print("已保存 lesson1_swing.gif")
print("\n完成！注意看：红色虚线是末端走过的轨迹——")
print("关节角匀速变化时，末端走的是一条曲线，这就是'轨迹生成'要处理的对象。")
