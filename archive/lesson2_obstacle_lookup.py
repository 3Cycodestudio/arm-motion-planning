# -*- coding: utf-8 -*-
"""
第 2 课补充：怎么在 C 空间里快速找到障碍物？
============================================
第 2 课的 C 空间地图上那团"香蕉"很难肉眼反推出工作空间里的位置。
这张图做了两件事让它一目了然：
  1. 在工作空间里挑 5 个"刚好撞到障碍物"的姿势，画出 5 条不同颜色的机械臂
  2. 在 C 空间地图上把同样这 5 个姿势标成 5 个彩色点
  3. 再用"逐列扫描"画出香蕉的完整轮廓（蓝色竖条 = 每个 θ1 下会撞的 θ2 范围）

运行：
  python lesson2_obstacle_lookup.py
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------- 跟 lesson2 一样的参数和函数 ----------------
L1, L2 = 140.0, 110.0
OBSTACLE = (175.0, 95.0, 43.0)  # (x, y, r)
START = (-120.0, 80.0)
GOAL  = (150.0, -40.0)

def fk(t1, t2):
    a1, a12 = np.radians(t1), np.radians(t1 + t2)
    elbow = (L1*np.cos(a1), L1*np.sin(a1))
    end = (elbow[0]+L2*np.cos(a12), elbow[1]+L2*np.sin(a12))
    return elbow, end

def dseg(px,py,ax,ay,bx,by):
    abx,aby=bx-ax,by-ay; L=abx**2+aby**2
    t=max(0,min(1,((px-ax)*abx+(py-ay)*aby)/L)) if L>0 else 0
    return np.hypot(px-ax-t*abx, py-ay-t*aby)

def is_collision(t1, t2):
    ox,oy,r = OBSTACLE
    e,end = fk(t1,t2)
    return (dseg(ox,oy,0,0,e[0],e[1]) < r+3) or (dseg(ox,oy,e[0],e[1],end[0],end[1]) < r+3)

# ---------------- 自动选 5 个"刚好撞到障碍物"的姿势 ----------------
# 沿 θ1 扫描找出会撞的 θ2 范围，从中均匀挑 5 个，覆盖禁区的整个轮廓。
# 这样不管你怎么改 OBSTACLE，脚本都能自动适配，不会报错。
hits_by_t1 = []
for t1 in np.arange(-180, 181, 1):
    coll_t2 = [t2 for t2 in np.arange(-180, 181, 1) if is_collision(t1, t2)]
    if coll_t2:
        hits_by_t1.append((t1, (min(coll_t2) + max(coll_t2)) // 2))

if len(hits_by_t1) < 5:
    raise RuntimeError(f"当前 OBSTACLE={OBSTACLE} 让禁区太小，只找到 {len(hits_by_t1)} 个撞点"
                       f"——换个更大的半径或更靠中间的位置再试")

# 均匀挑 5 个，覆盖 θ1 全程
idx = np.linspace(0, len(hits_by_t1) - 1, 5).astype(int)
SAMPLES = [hits_by_t1[i] for i in idx]
print(f"自动选出 5 个撞障碍物的姿势：{SAMPLES}")

# ---------------- 画 C 空间背景图（沿用 lesson2 的网格方法） ----------------
RES = 2.0
grid = np.zeros((int(360/RES), int(360/RES)), dtype=bool)
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        grid[i, j] = is_collision(-180+i*RES, 180-j*RES)
img = grid.T[:, ::-1]

# ---------------- 出图 ----------------
fig, (axW, axC) = plt.subplots(1, 2, figsize=(13, 5.8))
COLORS = ['#cf222e', '#1a7f37', '#0969da', '#bf3989', '#8250df']

# ===== 左图：工作空间，5 条机械臂撞到同一个障碍物 =====
ox, oy, orad = OBSTACLE
axW.set_xlim(-260, 260); axW.set_ylim(-260, 260)
axW.set_aspect('equal'); axW.grid(alpha=0.3)
axW.add_patch(plt.Circle((ox, oy), orad, color='#6e7781', alpha=0.6))
axW.plot(ox, oy, 'k+', markersize=14, markeredgewidth=2)
axW.annotate(f'障碍物 ({ox:.0f}, {oy:.0f}), r={orad}',
             (ox+orad+5, oy-5), fontsize=10, color='#1f2328')

for (t1, t2), c in zip(SAMPLES, COLORS):
    e, end = fk(t1, t2)
    axW.plot([0, e[0]], [0, e[1]], '-', color=c, linewidth=4, alpha=0.8,
             label=f'θ1={t1}°, θ2={t2}°')
    axW.plot([e[0], end[0]], [e[1], end[1]], '-', color=c, linewidth=3.5, alpha=0.8)
    axW.plot([0, e[0], end[0]], [0, e[1], end[1]], 'o', color=c, markersize=4)

axW.set_title("工作空间：5 个不同姿势的机械臂，\n每条都有一根杆子擦到那个灰色圆")
axW.legend(loc='lower left', fontsize=8, framealpha=0.9)

# ===== 右图：C 空间，5 个点 + 香蕉的完整轮廓 =====
axC.imshow(img, extent=[-180,180,-180,180], origin='lower',
           cmap='Greys', vmin=0, vmax=1.5, interpolation='nearest', alpha=0.55)

# 香蕉轮廓：每个 θ1 扫一遍 θ2，把会撞的范围画成蓝色竖条
for t1 in np.arange(-80, 60, 3):
    coll = np.array([is_collision(t1, t2) for t2 in np.arange(-180, 181, 1)])
    idx = np.where(coll)[0]
    if len(idx) > 0:
        axC.plot([t1, t1], [-180+idx[0], -180+idx[-1]],
                 '-', color='#0969da', linewidth=2, alpha=0.55)

# 5 个标志点
for (t1, t2), c in zip(SAMPLES, COLORS):
    axC.plot(t1, t2, 'o', color=c, markersize=13,
             markeredgecolor='white', markeredgewidth=1.5)
    axC.annotate(f'({t1}°, {t2}°)', (t1, t2),
                 textcoords='offset points', xytext=(9, 8),
                 fontsize=10, color=c, fontweight='bold')

# 起点终点
axC.plot(START[0], START[1], 's', color='#1a7f37', markersize=10, label='起点', markeredgecolor='black')
axC.plot(GOAL[0],  GOAL[1],  's', color='#8250df', markersize=10, label='终点', markeredgecolor='black')

# 加坐标网格
axC.set_xticks(np.arange(-180, 181, 30))
axC.set_yticks(np.arange(-180, 181, 30))
axC.set_xticks(np.arange(-180, 181, 10), minor=True)
axC.set_yticks(np.arange(-180, 181, 10), minor=True)
axC.grid(True, which='minor', alpha=0.15, color='gray')
axC.grid(True, which='major', alpha=0.4, color='gray')
axC.set_xlim(-180, 180); axC.set_ylim(-180, 180)
axC.set_xlabel('θ1 (度)'); axC.set_ylabel('θ2 (度)')
axC.set_title("C 空间：同样的 5 个姿势 → 5 个彩点\n蓝色竖条 = 禁区的完整形状（香蕉）")
axC.legend(loc='upper right')

fig.suptitle("怎么在 C 空间里找障碍物？看左右两边的颜色一一对应就行",
             fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig('lesson2_obstacle_lookup.png', dpi=130, bbox_inches='tight')
print("已保存 lesson2_obstacle_lookup.png")
print("\n关键信息：")
print(f"  障碍物在工作空间里：圆心 ({ox}, {oy})，半径 {orad}")
print(f"  在 C 空间里大致占的区域：θ1 ∈ [-25°, 40°]，θ2 ∈ [100°, -80°] 一条弯带")
print(f"  那个弯带的物理含义：每条蓝色竖条 = 固定 θ1 后，能让某根杆撞到圆的所有 θ2 范围")
