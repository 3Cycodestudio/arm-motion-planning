# -*- coding: utf-8 -*-
"""
运行方法:  python lesson10_6dof_rrt.py
本脚本做哪几件事:
1. 用 pybullet(开源物理引擎 Bullet)加载官方 Franka Panda 机械臂模型,
   冻结其腕部关节(第7轴), 等效为六自由度构型 —— 呼应课题"六自由度机械臂"
2. 在工作空间布置障碍墙 + 立柱, 碰撞检测全部交给 pybullet(精确网格碰撞体)
3. 在 6 维关节空间跑 RRT: 撒点 -> 最近邻 -> 迈步 -> pybullet 碰撞检查
4. 输出:
   - lesson10_6dof_rrt.png  (三联图: RRT 树的末端探索点云 / 末端工作空间轨迹 / 六关节角轨迹)
   - lesson10_6dof_anim.gif (机械臂沿规划轨迹运动的动画)
   - 命令行打印关键数据: 节点数 / 规划时间 / 路径长度(关节空间弧长)
5. 结尾打印关键结论数字。
注意: 固定随机种子, 结果可复现。
"""
import time
import numpy as np
import pybullet as p
import pybullet_data
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation, PillowWriter

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

RNG_SEED   = 42
STEP       = 0.35     # RRT 步长(弧度)
GOAL_BIAS  = 0.10     # 10% 概率直接朝终点撒
MAX_ITER   = 6000
MARGIN     = 0.015    # 安全裕量: 与障碍距离小于 1.5cm 即算碰撞(相当于把障碍加厚)

# ---------- 1. 启动物理引擎 + 加载 Panda ----------
cid = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)  # 只做运动学/碰撞, 不需要重力
robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

N_JOINTS = p.getNumJoints(robot)
# Panda 的 7 个转动关节在 pybullet 里的编号是 0..6
ALL_REV = [i for i in range(N_JOINTS)
           if p.getJointInfo(robot, i)[2] == p.JOINT_REVOLUTE]
FROZEN_IDX = 6                      # 冻结第 7 轴(腕部旋转), 等效六自由度
ACTIVE = [i for i in ALL_REV if i != FROZEN_IDX]

# 从模型里读关节限位(不硬编码, 体现"官方模型"的严谨性)
LO = np.array([p.getJointInfo(robot, j)[8] for j in ACTIVE], dtype=float)
HI = np.array([p.getJointInfo(robot, j)[9] for j in ACTIVE], dtype=float)
print("六自由度关节限位(度):")
for k, j in enumerate(ACTIVE):
    print("  关节%d: [%.1f, %.1f]" % (k + 1, np.degrees(LO[k]), np.degrees(HI[k])))

def set_config(q):
    """把机械臂摆到 6 维构型 q(弧度), 第7轴固定为 0"""
    for k, j in enumerate(ACTIVE):
        p.resetJointState(robot, j, q[k])
    p.resetJointState(robot, FROZEN_IDX, 0.0)

def ee_pos():
    """末端(link7)的世界坐标"""
    st = p.getLinkState(robot, ALL_REV[-1], computeForwardKinematics=True)
    return np.array(st[4])

# ---------- 2. 障碍: 一堵墙 + 一根立柱 ----------
wall = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.04, 0.30, 0.45])
wall_body = p.createMultiBody(0, wall, basePosition=[0.46, 0.0, 0.45])
pillar = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.35])
pillar_body = p.createMultiBody(0, pillar, basePosition=[0.28, 0.35, 0.30])
OBSTACLES = [wall_body, pillar_body]

OBST_MESHES = [  # 供 matplotlib 画图用: 每个障碍的 8 个顶点
    ([0.46, 0.0, 0.45], [0.04, 0.30, 0.45]),
    ([0.28, 0.35, 0.30], [0.05, 0.05, 0.35]),
]

# ---------- 3. 碰撞检测(交给 pybullet) ----------
N_COLLISION_CHECKS = 0   # 计数器: 一共调用多少次碰撞检查(报告里的关键数字)

def is_collision(q):
    """构型 q 是否碰撞: ①碰到障碍 ②自碰撞(非相邻连杆贴近)"""
    global N_COLLISION_CHECKS
    N_COLLISION_CHECKS += 1
    set_config(q)
    for obs in OBSTACLES:
        for pt in p.getClosestPoints(robot, obs, distance=MARGIN):
            if pt[8] < MARGIN:          # pt[8] = 两碰撞体最近距离
                return True
    # 自碰撞: 相邻连杆天然贴着, 只查"隔一个以上"的连杆对
    for pt in p.getClosestPoints(robot, robot, distance=0.0):
        a, b = pt[3], pt[4]             # 连杆编号
        if pt[8] < 0.0 and abs(a - b) > 1 and (a, b) not in SELF_OK:
            return True
    return False

EDGE_SUB = 12      # 每段路径的细分份数: 步长 0.35 rad / 12 ≈ 0.03 rad 一段

def is_edge_collision(q_a, q_b):
    """边检查: 起点 q_a 到 q_b 线段上细分成 12 小段, 逐一查碰撞。
    只查两个端点会漏掉'从障碍中穿过去'的情况(离散盲区)。"""
    for s in np.linspace(0.0, 1.0, EDGE_SUB + 1)[1:]:
        if is_collision(q_a + s * (q_b - q_a)):
            return True
    return False

# 白名单: 腕部+手爪组件的连杆对(它们是装配在一起的固定连接, 天然贴近, 不算自碰撞)
# Panda 的连杆编号: 0~6=七个臂杆, 7/8=法兰, 9/10=两根手指
SELF_OK = {(-1, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (-1, 1), (0, 2), (1, 3),
           (6, 8), (6, 9), (6, 10), (7, 8), (7, 9), (7, 10), (8, 9), (8, 10), (9, 10)}
SELF_OK |= {(b, a) for (a, b) in SELF_OK}

# ---------- 4. 起点 / 终点(已用诊断脚本验证均无碰撞) ----------
START = np.radians([ 50, -40,  10, -115,  10,  75 ])   # 收拢姿态, 末端在墙北侧
GOAL  = np.radians([-40,  30, -35, -140, -20, 150 ])   # 展开姿态, 末端在墙南侧 -> 必须绕行

print("\n起点碰撞?", is_collision(START), "  末端位置:", np.round(ee_pos(), 3))
print("终点碰撞?", is_collision(GOAL),  "  末端位置:", np.round(ee_pos(), 3))

def clamp(q):
    return np.clip(q, LO, HI)

def rrt_plan():
    rng = np.random.default_rng(RNG_SEED)
    nodes = [START.copy()]
    parent = [-1]
    t0 = time.time()
    for it in range(MAX_ITER):
        if rng.random() < GOAL_BIAS:
            q_rand = GOAL.copy()
        else:
            q_rand = LO + rng.random(len(ACTIVE)) * (HI - LO)
        dists = [np.linalg.norm(q_rand - n) for n in nodes]
        ni = int(np.argmin(dists))
        dirv = q_rand - nodes[ni]
        L = np.linalg.norm(dirv)
        q_new = nodes[ni] + (dirv / L) * min(STEP, L) if L > 1e-9 else q_rand
        q_new = clamp(q_new)
        if np.linalg.norm(q_new - nodes[ni]) < 1e-9:
            continue
        if is_edge_collision(nodes[ni], q_new):   # 边检查: 连线也不能穿障碍
            continue
        nodes.append(q_new.copy()); parent.append(ni)
        if np.linalg.norm(q_new - GOAL) < STEP:
            nodes.append(GOAL.copy()); parent.append(len(nodes) - 2)
            print("第 %d 次迭代找到路径!" % (it + 1))
            return nodes, parent, time.time() - t0
    return nodes, parent, time.time() - t0

print("\n开始 RRT 规划(6 维关节空间, 步长 %.2f rad, 目标偏置 %.0f%%)..." % (STEP, GOAL_BIAS * 100))
t0 = time.time()
nodes, parent, plan_time = rrt_plan()
n_nodes = len(nodes)
checks_planning = N_COLLISION_CHECKS   # 规划阶段用掉的碰撞检查次数

# 回溯路径
path = []
i = len(nodes) - 1
while i != -1:
    path.append(nodes[i]); i = parent[i]
path = path[::-1]
seg = [np.linalg.norm(path[k + 1] - path[k]) for k in range(len(path) - 1)]
arc = float(np.sum(seg))
print("\n===== 关键数据 =====")
print("树节点总数: %d    规划用时: %.2f s" % (n_nodes, plan_time))
print("路径点数: %d (含起终点)" % len(path))
print("关节空间路径长度: %.2f rad = %.1f 度" % (arc, np.degrees(arc)))
print("碰撞检查调用次数: %d 次(规划阶段)" % checks_planning)

# ---------- 路径严格复核: 相邻路径点之间再插 20 个中点, 逐一做碰撞检查 ----------
n_check, n_bad = 0, 0
for k in range(len(path) - 1):
    for s in np.linspace(0.0, 1.0, 22)[1:-1]:     # 段内 20 个中间构型
        q_mid = path[k] + s * (path[k + 1] - path[k])
        n_check += 1
        if is_collision(q_mid):
            n_bad += 1
print("路径严格复核: 插入 %d 个中间构型, 碰撞的 %d 个 -> %s"
      % (n_check, n_bad, "全部安全" if n_bad == 0 else "有碰撞,需改进!"))

# 维度灾难对照: 若用 2 度一格的网格穷举这个 6 维空间(按真实关节限位算)
cells = float(np.prod((HI - LO) / np.radians(2.0)))
print("对照: 若用 2 度网格穷举 6 维关节空间 = %.1e 格; RRT 只检查了 %d 次碰撞"
      % (cells, checks_planning))
print("      -> RRT 探索量约为网格穷举的 %.1e 分之一" % (cells / checks_planning))

# ---------- 5. 画三联图 ----------
fig = plt.figure(figsize=(16, 5))

def draw_box(ax, center, half, color):
    cx, cy, cz = center; hx, hy, hz = half
    x0, x1 = cx - hx, cx + hx; y0, y1 = cy - hy, cy + hy; z0, z1 = cz - hz, cz + hz
    v = np.array([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
                  [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]])
    faces = [[v[0],v[1],v[2],v[3]],[v[4],v[5],v[6],v[7]],
             [v[0],v[1],v[5],v[4]],[v[2],v[3],v[7],v[6]],
             [v[1],v[2],v[6],v[5]],[v[0],v[3],v[7],v[4]]]
    pc = Poly3DCollection(faces, facecolor=color, edgecolor="k", alpha=0.35)
    ax.add_collection3d(pc)

# (a) 探索点云: 每个树节点对应的末端位置
ax1 = fig.add_subplot(1, 3, 1, projection="3d")
ee_all = []
for q in nodes:
    set_config(q); ee_all.append(ee_pos())
ee_all = np.array(ee_all)
ax1.scatter(ee_all[:,0], ee_all[:,1], ee_all[:,2], s=3, c="gray", alpha=0.3)
draw_box(ax1, *OBST_MESHES[0], "#e67e22"); draw_box(ax1, *OBST_MESHES[1], "#e67e22")
ax1.set_title("(a) RRT 树节点的末端位置(探索的工作空间)")
ax1.set_xlabel("X(m)"); ax1.set_ylabel("Y(m)"); ax1.set_zlabel("Z(m)")

# (b) 末端工作空间轨迹
ax2 = fig.add_subplot(1, 3, 2, projection="3d")
ee_path = []
for q in path:
    set_config(q); ee_path.append(ee_pos())
ee_path = np.array(ee_path)
ax2.plot(ee_path[:,0], ee_path[:,1], ee_path[:,2], "r-", lw=2, label="末端轨迹")
ax2.scatter(*ee_path[0],  c="green", s=60, label="起点")
ax2.scatter(*ee_path[-1], c="blue",  s=60, label="终点")
draw_box(ax2, *OBST_MESHES[0], "#e67e22"); draw_box(ax2, *OBST_MESHES[1], "#e67e22")
ax2.set_title("(b) 末端工作空间轨迹(共 %d 个路径点)" % len(path))
ax2.set_xlabel("X(m)"); ax2.set_ylabel("Y(m)"); ax2.set_zlabel("Z(m)")
ax2.legend()

# (c) 六关节角随路径点的变化
ax3 = fig.add_subplot(1, 3, 3)
idx = np.arange(len(path))
for k in range(6):
    ax3.plot(idx, np.degrees([q[k] for q in path]), lw=1.5, label="关节%d" % (k + 1))
ax3.set_title("(c) 六个关节角沿路径的变化(度)")
ax3.set_xlabel("路径点序号"); ax3.set_ylabel("关节角(度)")
ax3.legend(fontsize=8, ncol=2); ax3.grid(alpha=0.3)

plt.tight_layout()
OUT_PNG = "C:/Users/csystudio/Desktop/挺好/arm_research/lesson10_6dof_rrt.png"
plt.savefig(OUT_PNG, dpi=130)
print("已保存", OUT_PNG)

# ---------- 6. 动画: 机械臂沿轨迹运动 ----------
# 把 16 个路径点细分成密集帧(每段 8 个中间构型), 动画才连续流畅
dense = []
for k in range(len(path) - 1):
    for s in np.linspace(0.0, 1.0, 9)[:-1]:
        dense.append(path[k] + s * (path[k + 1] - path[k]))
dense.append(path[-1])
print("正在渲染动画(%d 帧, 需要一点时间)..." % len(dense))
view = (2.2, -1.6, 1.2)   # 相机位置
target = (0.3, 0.0, 0.35)
view_mat = p.computeViewMatrix(cameraEyePosition=list(view),
                               cameraTargetPosition=list(target),
                               cameraUpVector=[0, 0, 1])
proj_mat = p.computeProjectionMatrixFOV(fov=50.0, aspect=1.0,
                                        nearVal=0.1, farVal=4.0)
frames = []
for q in dense:
    set_config(q)
    _, _, rgb, _, _ = p.getCameraImage(width=480, height=480,
                                       viewMatrix=view_mat,
                                       projectionMatrix=proj_mat,
                                       renderer=p.ER_TINY_RENDERER)
    frames.append(np.reshape(rgb, (480, 480, 4))[:, :, :3])

fig2 = plt.figure(figsize=(4.8, 4.8))
im = plt.imshow(frames[0]); plt.axis("off")
def update(k):
    im.set_data(frames[k]); return [im]
ani = FuncAnimation(fig2, update, frames=len(frames), interval=90)
ani.save("C:/Users/csystudio/Desktop/挺好/arm_research/lesson10_6dof_anim.gif",
         writer=PillowWriter(fps=12))
plt.close(fig2)
print("已保存 lesson10_6dof_anim.gif")

p.disconnect()
print("\n===== 结论 =====")
print("pybullet 官方 Panda 模型(冻结第7轴=六自由度) + pybullet 碰撞检测,")
print("在 6 维关节空间用 RRT 规划成功: %d 节点 / %.2f 秒 / 关节空间路径 %.1f 度" %
      (n_nodes, plan_time, np.degrees(arc)))
