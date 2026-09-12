"""
第 3 课诊断图：C 空间禁区背景"画错 vs 画对"对照。

背景：早先 draw_background() 直接把 mask 丢给 imshow，忘了转置，
      导致整张 C 空间图被转了 90°——本该是"竖墙"的禁区画成了"横带"，
      看上去像是起点落在灰区里、路径穿过灰区。

本脚本直接调用 lesson3_rrt 里修好的函数，和新手写法的错误版本并排对照，
以后改代码后跑一遍，就能确认图没再画歪。

运行：python lesson3_diagnose_mask.py
"""
import os
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

import lesson3_rrt as L3

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

START, GOAL = (-132.0, 16.0), (0.0, -178.0)

mask, n = L3.build_obstacle_mask()

# ---------------- 先做数值诊断，把结论打印出来 ----------------
print("=" * 62)
print("诊断 1：起点/终点本身是否在禁区里？")
print("=" * 62)
print(f"  起点 {START}  碰撞 = {L3.is_collision(*START)}")
print(f"  终点 {GOAL}  碰撞 = {L3.is_collision(*GOAL)}")
print()

print("=" * 62)
print("诊断 2：禁区到底是'竖墙'还是'横带'？")
print("=" * 62)
col_ratio = np.array([sum(L3.is_collision(-180.0 + i * L3.ANGLE_STEP,
                                          180.0 - j * L3.ANGLE_STEP)
                         for j in range(n)) / n for i in range(n)])
row_ratio = np.array([sum(L3.is_collision(-180.0 + i * L3.ANGLE_STEP,
                                          180.0 - j * L3.ANGLE_STEP)
                         for i in range(n)) / n for j in range(n)])
print(f"  某一列（θ1 固定）最高被撞占比：{100*col_ratio.max():.1f}%  "
      f"→ 出现在 θ1 = {-180 + int(col_ratio.argmax())*L3.ANGLE_STEP:.0f}°")
print(f"  某一行（θ2 固定）最高被撞占比：{100*row_ratio.max():.1f}%  "
      f"→ 出现在 θ2 = {180 - int(row_ratio.argmax())*L3.ANGLE_STEP:.0f}°")
walls = [-180 + i * L3.ANGLE_STEP for i in range(n) if col_ratio[i] > 0.999]
print(f"  被 100% 堵死的 θ1 列：{walls}")
print("  结论：接近 100% 的方向就是'墙'的朝向 → 墙是【竖】的，图上必须画成竖条")
print()

print("=" * 62)
print("诊断 3：背景图朝向自检")
print("=" * 62)
L3.check_background_orientation(mask, n)
print()

# ---------------- 画对照图 ----------------
fig, axes = plt.subplots(1, 2, figsize=(15, 7.4))

# 左：错误画法（把 mask 直接丢给 imshow，忘了转置）
ax = axes[0]
ax.imshow(mask, extent=(-180, 180, 180, -180), cmap='Greys',
          alpha=0.4, origin='upper', interpolation='nearest', zorder=1)
ax.plot(*START, 'o', color='#3B6D11', ms=13, zorder=6)
ax.plot(*GOAL, 'o', color='#8250df', ms=13, zorder=6)
ax.text(START[0], START[1], ' 起点', color='#27500A', fontsize=11, va='bottom', zorder=7)
ax.text(GOAL[0], GOAL[1], ' 终点', color='#3C3489', fontsize=11, va='bottom', zorder=7)
L3.finish_axes(ax)
ax.set_title('错误画法（忘了转置 → 图被转了 90°）\n'
             '禁区画成一条"横带"，看上去把起点、终点全盖住了',
             fontsize=12, color='#A32D2D')

# 右：正确画法（调用修好的 draw_background）
ax = axes[1]
L3.draw_background(ax, mask, n)
ax.plot(*START, 'o', color='#3B6D11', ms=13, zorder=6)
ax.plot(*GOAL, 'o', color='#8250df', ms=13, zorder=6)
ax.text(START[0], START[1], ' 起点', color='#27500A', fontsize=11, va='bottom', zorder=7)
ax.text(GOAL[0], GOAL[1], ' 终点', color='#3C3489', fontsize=11, va='bottom', zorder=7)
if walls:
    ax.annotate(f'这堵"竖墙"\n（θ1={walls[0]:.0f}°~{walls[-1]:.0f}°，沿 θ2 全堵死）',
                xy=(sum(walls) / len(walls), 60), xytext=(95, 150),
                fontsize=11, color='#A32D2D', ha='center',
                arrowprops=dict(arrowstyle='->', color='#A32D2D', lw=1.4))
L3.finish_axes(ax)
ax.set_title('正确画法\n禁区是一条"竖墙"，起终点都在白色自由区里',
             fontsize=12, color='#1a7f37')

plt.suptitle('C 空间禁区背景图：画错 vs 画对', fontsize=14)
plt.tight_layout()
out = os.path.join(OUT_DIR, 'lesson3_mask_diagnose.png')
plt.savefig(out, dpi=130, bbox_inches='tight')
plt.close(fig)
print(f"已保存 {os.path.basename(out)}")
