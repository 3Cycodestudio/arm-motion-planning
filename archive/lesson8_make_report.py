# -*- coding: utf-8 -*-
"""
lesson8_make_report.py —— 第8课：自动装配研究报告初稿

运行方法（在 VS Code 终端里）：
    python lesson8_make_report.py

本脚本做哪几件事：
    1) 按"引言/建模/方法/实验/讨论结论"五章搭建报告骨架；
    2) 把第1~7课的关键数据写成草稿段落（可直接修改润色）；
    3) 把 12 张 png 按章节自动插入并编号（gif 不插入 Word，留给答辩演示）；
    4) 生成实验数据汇总表；
    5) 输出 lesson8_report_draft.docx。

注意：本脚本只"装配"，不重新跑实验。所有数字来自第1~7课的运行结果。
"""
import os
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, 'lesson8_report_draft.docx')

# ---------------- 基线参数（与 lesson2~7 一致） ----------------
L1, L2   = 200.0, 180.0
OBST     = (175.0, 95.0, 43.0)
START    = (-132.0, 16.0)
GOAL     = (0.0, -178.0)

def fig(doc, filename, caption, width=5.6):
    """插图 + 居中图注，缺图不报错只提示。"""
    path = os.path.join(HERE, filename)
    if os.path.exists(path):
        doc.add_picture(path, width=Inches(width))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap = doc.add_paragraph(caption)
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in cap.runs:
            r.font.size = Pt(9)
    else:
        doc.add_paragraph(f'[缺图：{filename}]')

def body(doc, text):
    return doc.add_paragraph(text)

# ---------------- 文档与全局字体 ----------------
doc = Document()
style = doc.styles['Normal']
style.font.name = 'Times New Roman'
style.font.size = Pt(12)
style.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')

# ---------------- 标题 / 摘要 ----------------
doc.add_heading('复杂受限环境多自由度机械臂基于学习和抽样的混合运动规划算法研究', level=0)
doc.add_paragraph('（科研训练项目结题报告 · 初稿框架）').alignment = WD_ALIGN_PARAGRAPH.CENTER

body(doc, '摘要（草稿）：针对复杂受限环境下多自由度机械臂的运动规划问题，'
          '本文以二连杆平面机械臂为对象，在构型空间（C 空间）中系统对比了'
          '基于网格搜索的 BFS 基准算法与基于抽样的 RRT/RRT* 算法，'
          '并提出了一种"教师—学生"式的学习引导抽样方法：'
          '以 BFS 最短路生成的走廊软标签训练一个小型手写神经网络（2-48-1），'
          '以网络输出的通路概率图作为 RRT 的非均匀采样先验。'
          '在单障碍典型场景下的 20 组随机种子实验中，学习引导 RRT 平均只需'
          '均匀撒点 RRT 约 72% 的树节点即可到达目标；同时验证了 RRT* 的'
          '渐近最优收敛值（235.64°）与几何路径平滑极限（234.6°）的一致性，'
          '两种独立优化路线收敛到同一数值，相互印证了该场景下的最优路径长度。')
body(doc, '关键词：运动规划；构型空间；抽样算法；RRT；RRT*；学习引导采样；神经网络')

# ---------------- 1 引言 ----------------
doc.add_heading('1 引言', level=1)
body(doc, '（草稿，可扩写）机械臂在高 clutter 受限环境（隧道支护、设备舱内检修等'
          '土木施工场景同样存在类似需求）中执行点到点运动时，需要在避开障碍的前提下'
          '求得一条可行的关节轨迹。经典网格类搜索方法（BFS/Dijkstra/A*）在维度'
          '增长时面临组合爆炸：以 6 关节机械臂按每关节 2° 分辨率离散计，'
          '构型空间网格数达 180^6 ≈ 3.4×10^12，网格法直接失效——这构成了'
          '抽样类算法（RRT 及其最优变体 RRT*）存在的现实理由。')
body(doc, '本课题的技术路线为"学习 + 抽样"的混合框架：利用确定性算法（BFS）'
          '离线生成高质量示范，经神经网络蒸馏为连续的概率先验，'
          '再用于加速在线抽样规划。')

# ---------------- 2 问题建模 ----------------
doc.add_heading('2 问题建模', level=1)
doc.add_heading('2.1 机械臂模型与正运动学', level=2)
body(doc, f'（草稿）采用二连杆平面机械臂，杆长 L1={L1}、L2={L2}，'
          f'两关节转角范围均为 [-180°, 180°]。末端位置由正运动学给出：'
          f'x = L1·cos(θ1) + L2·cos(θ1+θ2)，y = L1·sin(θ1) + L2·sin(θ1+θ2)。'
          f'障碍物为工作空间中的圆形障碍，圆心 ({OBST[0]}, {OBST[1]})，半径 {OBST[2]}；'
          f'起点 ({START[0]}, {START[1]})，终点 ({GOAL[0]}, {GOAL[1]})。')
fig(doc, 'lesson1_poses.png', '图 1  机械臂结构与几组典型姿态下的正运动学验证')

doc.add_heading('2.2 构型空间与碰撞检测', level=2)
body(doc, '（草稿）以 (θ1, θ2) 为坐标将问题映射到二维构型空间：每个"格子"'
          '对应一种姿态，通过正运动学 + 圆-线段距离检测判定碰撞，'
          '将工作空间的障碍"投影"为 C 空间禁区。分析表明：大臂位置仅由 θ1 决定，'
          '因此障碍方位角附近形成一堵沿 θ2 方向贯通的"竖墙"禁区；'
          'C 空间的横向连通性（是否存在被撞满的整列）决定解是否存在。')
fig(doc, 'lesson2_cspace_map.png', '图 2  工作空间障碍与 C 空间禁区对照图')
fig(doc, 'lesson2_why_wall.png', '图 3  竖墙禁区的成因分析（大臂仅受 θ1 控制）')

doc.add_heading('2.3 连通性与可行性', level=2)
body(doc, '（草稿）反例验证：当障碍半径增至临界值以上时，某列 θ1 被撞满，'
          'C 空间被切成互不连通的两块，此时 BFS 与 RRT 均无解'
          '（RRT 撒近 2000 点仍失败）。结论：算法只负责"找路"，'
          '可行性由问题本身（障碍布局）决定。')
fig(doc, 'lesson3_unreachable.png', '图 4  连通性反例：C 空间被竖墙切断时的无解情形')

# ---------------- 3 方法 ----------------
doc.add_heading('3 混合规划方法', level=1)
doc.add_heading('3.1 网格基准：BFS 最短路', level=2)
body(doc, '（草稿）以 2° 分辨率离散 C 空间（90×90 网格），4 邻域 BFS 求'
          '步数最短路径，作为后续所有算法的对照基准（"教师"信号也来源于此）。')
doc.add_heading('3.2 抽样规划：RRT 与步长灵敏度', level=2)
body(doc, '（草稿）RRT 三步循环：随机撒点（10% 概率偏置到目标）→ 最近邻 →'
          '以固定步长延伸并做碰撞检查。同等任务下 RRT 仅用 157 个节点即成功，'
          '探索量约为 BFS（16 965/32 400 格）的 0.9%。'
          '步长扫描显示节点数随步长增大而下降（6°:157 / 12°:80 / 36°:33），'
          '代价是路径更粗糙、绕障能力下降。')
fig(doc, 'lesson3_bfs_vs_rrt.png', '图 5  BFS 与 RRT 的探索量对比（0.9%）')
fig(doc, 'lesson3_step_compare.png', '图 6  RRT 步长参数扫描')

doc.add_heading('3.3 路径平滑：随机捷径法', level=2)
body(doc, '（草稿）Random Shortcutting：随机选取路径上 i<j 两点，'
          '若连线无碰撞则删除中间节点。原始 RRT 路径 52 点 / 300.3°，'
          '经平滑收敛到 2 点 / 234.6°，缩短 21.9%；因起终点连线本身无碰撞，'
          '234.6° 即为该场景平滑算法的几何极限。')
fig(doc, 'lesson4_smoothed_path.png', '图 7  随机捷径平滑前后的路径对比')

doc.add_heading('3.4 渐近最优：RRT*', level=2)
body(doc, '（草稿）RRT* 在 RRT 的每次延伸后额外执行 choose_parent 与 rewire：'
          '前者在新节点邻域内选择总代价最小的父节点，后者将邻域内邻居改挂到'
          '新节点名下并递归更新子树代价。同参数对比 RRT 332.17° → RRT* 278.94°'
          '（-16.0%）；anytime 曲线由 267.5°（200 次迭代）收敛到 235.64°'
          '（4000 次迭代）。')
fig(doc, 'lesson5_rrt_star.png', '图 8  RRT 与 RRT* 同参数对比及 anytime 收敛曲线')

doc.add_heading('3.5 时间参数化：弧长重采样 + 梯形速度规划', level=2)
body(doc, '（草稿）将几何路径按累计弧长均匀重采样（解耦位置与进度），'
          '再叠加梯形速度规划（加速-匀速-减速）。参数：总弧长 S=276.5°，'
          'v_max=90°/s，a_max=120°/s²，得总时间 T=3.82s，'
          '关节峰值速度 86.8 / 90.0°/s，速度连续、起止为零、加速度有界；'
          '相比"等时播放"的阶梯速度（加速度无穷、抖动异响）显著改善。')
fig(doc, 'lesson6_time_param.png', '图 9  弧长参数化与梯形速度规划结果')

doc.add_heading('3.6 学习引导抽样（本文核心贡献）', level=2)
body(doc, '（草稿）"教师—学生"蒸馏框架：①教师——多源 BFS 将 BFS 最短路上所有'
          '格子设为零距离源，同时向外扩散得到距离场，经 exp(-d/τ) 软化成走廊标签；'
          '②学生——手写 2-48-1 神经网络（numpy 实现，含 tanh/sigmoid 激活与'
          '反向传播）在走廊加权损失下训练 3000 轮，输出通路概率图'
          '（与教师标签相关系数 0.839）；③混合——概率图取 5 次幂并保留'
          '10% 均匀下限后作为 RRT 的采样分布（累加表反查实现）。'
          '走廊面积仅占 7.7%，却分走 79% 的采样概率。')
fig(doc, 'lesson7_learned_sampling.png',
    '图 10  教师-学生蒸馏与学习引导 RRT 全流程（2×3 组图）')

# ---------------- 4 实验与结果 ----------------
doc.add_heading('4 实验与结果', level=1)
body(doc, '（草稿）核心数据汇总见下表；20 组随机种子统计用于排除单次运气因素。')
doc.add_heading('4.1 数据汇总', level=2)
table = doc.add_table(rows=8, cols=4)
table.style = 'Light Grid Accent 1'
hdr = table.rows[0].cells
hdr[0].text = '实验'; hdr[1].text = '指标'; hdr[2].text = '基准'; hdr[3].text = '本文方法'
rows = [
    ('BFS vs RRT',     '探索量',        '16 965 格',        '157 节点（0.9%）'),
    ('步长扫描',       '节点数',        '6°: 157',          '12°: 80 / 36°: 33'),
    ('路径平滑',       '路径长度',      '300.3°（52 点）',  '234.6°（2 点，-21.9%）'),
    ('RRT*',           '路径长度',      '332.17°（RRT）',   '278.94°（-16.0%）'),
    ('RRT* anytime',   '收敛值',        '267.5°（iter200）','235.64°（iter4000）'),
    ('时间参数化',     '总运动时间',    '阶梯速度（不可用）','T=3.82s，速度连续'),
    ('学习引导（20 seed）','平均树节点', '80.2 ± 15.7',      '57.6 ± 12.6（72%）'),
]
for i, r in enumerate(rows, start=1):
    cells = table.rows[i].cells
    for j, v in enumerate(r):
        cells[j].text = v

doc.add_heading('4.2 双路线收敛的交叉验证', level=2)
body(doc, '（草稿，本文点睛之笔）两条彼此独立的技术路线收敛到同一数值：'
          '事后路径平滑的几何极限 234.6° = 起终点直线距离 √(132²+194²) = 234.6°；'
          'RRT* 的 anytime 收敛值 235.64°。生长时优化（RRT*）与事后修形（平滑）'
          '殊途同归，说明该值即本场景的最优路径长度，两种方法互相验证了正确性。')

# ---------------- 5 讨论 / 结论 ----------------
doc.add_heading('5 讨论', level=1)
body(doc, '（草稿）①连通性：抽样算法无法逾越被切断的 C 空间，可行性由问题决定；'
          '②学习引导的代价：需要离线教师信号，场景更换后需重新蒸馏，'
          '适合障碍布局相对固定的受限作业环境（如舱内检修、隧道台车作业）；'
          '③梯形规划在切换点加速度仍有突变（jerk 无穷），'
          '工程上可进一步采用 S 曲线 / jerk 限幅；④本研究的弧长与'
          '梯形规划均在 C 空间关节坐标下进行，未显式约束末端笛卡尔轨迹。')

doc.add_heading('6 结论与展望', level=1)
body(doc, '（草稿）本文在统一场景下完成了从建模、抽样、优化、时间参数化到'
          '学习引导的完整技术链验证：RRT 以 0.9% 的探索量达到网格基准的可行解；'
          'RRT* 与随机捷径平滑双路线收敛印证最优值；学习引导抽样将 20-seed '
          '平均树节点压缩至均匀版的 72%。后续工作：S 曲线 jerk 限幅、'
          '高维（6-DOF）场景验证、以末端笛卡尔约束为目标的混合规划。')

doc.add_heading('参考文献（待补）', level=1)
body(doc, '[1] LaValle S M. Rapidly-exploring random trees: A new tool for path planning. TR 98-11, 1998.\n'
          '[2] Karaman S, Frazzoli E. Sampling-based algorithms for optimal motion planning. IJRR, 2011.\n'
          '[3] Ichter B, et al. Learned sampling distributions for efficient planning. ICRA/WAFR, 2018.')

doc.save(OUT)
print('已生成：', OUT)
print('章节：摘要 / 1引言 / 2建模 / 3方法(6小节) / 4实验 / 5讨论 / 6结论')
print('插图：10 张 png 已按章节插入（gif 留作答辩演示）')
print('汇总表：7 行核心数据（0.9% / -21.9% / -16% / 72% / 3.82s ...）')
print('下一步：通读草稿 → 润色"（草稿）"段落 → 补摘要结论 → 导导师看')
