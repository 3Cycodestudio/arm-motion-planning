# -*- coding: utf-8 -*-
"""
运行方法:  python lesson11_6dof_learned_sampling.py
本脚本做哪几件事(第11课: 把"学习引导撒点"搬到六自由度):
1. 老师换人: 6 维没法建 BFS 网格(180^6 格), 改用 Ichter 2018 的思路——
   先跑 20 个 seed 的纯 RRT, 收集所有成功路径当"经验路径"(老师)
2. 标签: 在 6 维关节空间均匀撒 6000 个查询点,
   每个点的标签 = exp(-离最近经验路径点的距离/TAU)  (离走廊越近标签越高)
3. 学生: 手写神经网络 6->48->1 (和第7课同一套, 只是输入从 2 维变 6 维),
   小批量梯度下降学"这个构型落在走廊里的概率"
4. 实战: 引导版 RRT(85% 概率按网络概率撒点) vs 均匀版 RRT, 各 20 个 seed,
   同 seed 对比树节点数 —— 验证学习成分在真实 Panda 上是否同样有效
5. 输出 lesson11_6dof_learned.png (2x3 六宫格) + 命令行关键数据
"""
import time
import numpy as np
import pybullet as p
import pybullet_data
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

STEP       = 0.35
GOAL_BIAS  = 0.10
MAX_ITER   = 6000
MARGIN     = 0.015
EDGE_SUB   = 12

# ---------- 1. 引擎 + Panda + 障碍(与第10课完全一致) ----------
cid = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)
robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
N_JOINTS = p.getNumJoints(robot)
ALL_REV = [i for i in range(N_JOINTS) if p.getJointInfo(robot, i)[2] == p.JOINT_REVOLUTE]
FROZEN_IDX = 6
ACTIVE = [i for i in ALL_REV if i != FROZEN_IDX]
LO = np.array([p.getJointInfo(robot, j)[8] for j in ACTIVE], dtype=float)
HI = np.array([p.getJointInfo(robot, j)[9] for j in ACTIVE], dtype=float)

def set_config(q):
    for k, j in enumerate(ACTIVE):
        p.resetJointState(robot, j, q[k])
    p.resetJointState(robot, FROZEN_IDX, 0.0)

def ee_pos():
    st = p.getLinkState(robot, ALL_REV[-1], computeForwardKinematics=True)
    return np.array(st[4])

wall = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.04, 0.30, 0.45])
wall_body = p.createMultiBody(0, wall, basePosition=[0.46, 0.0, 0.45])
pillar = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.35])
pillar_body = p.createMultiBody(0, pillar, basePosition=[0.28, 0.35, 0.30])
OBSTACLES = [wall_body, pillar_body]
OBST_MESHES = [([0.46, 0.0, 0.45], [0.04, 0.30, 0.45]),
               ([0.28, 0.35, 0.30], [0.05, 0.05, 0.35])]

CHECKS = {"n": 0}   # 每次规划前清零的碰撞检查计数器

SELF_OK = {(-1, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (-1, 1), (0, 2), (1, 3),
           (6, 8), (6, 9), (6, 10), (7, 8), (7, 9), (7, 10), (8, 9), (8, 10), (9, 10)}
SELF_OK |= {(b, a) for (a, b) in SELF_OK}

def is_collision(q):
    CHECKS["n"] += 1
    set_config(q)
    for obs in OBSTACLES:
        for pt in p.getClosestPoints(robot, obs, distance=MARGIN):
            if pt[8] < MARGIN:
                return True
    for pt in p.getClosestPoints(robot, robot, distance=0.0):
        a, b = pt[3], pt[4]
        if pt[8] < 0.0 and abs(a - b) > 1 and (a, b) not in SELF_OK:
            return True
    return False

def is_edge_collision(q_a, q_b):
    for s in np.linspace(0.0, 1.0, EDGE_SUB + 1)[1:]:
        if is_collision(q_a + s * (q_b - q_a)):
            return True
    return False

START = np.radians([ 50, -40,  10, -115,  10,  75 ])
GOAL  = np.radians([-40,  30, -35, -140, -20, 150 ])

def clamp(q):
    return np.clip(q, LO, HI)

# ---------- 2. RRT 骨架: 撒点方式做成参数(均匀 or 引导) ----------
def uniform_sampler(rng):
    return LO + rng.random(6) * (HI - LO)

def rrt_plan(seed, sampler, label=""):
    """返回 dict: success / nodes数 / path / 用时 / 碰撞检查次数"""
    rng = np.random.default_rng(seed)
    CHECKS["n"] = 0
    nodes = [START.copy()]
    parent = [-1]
    t0 = time.time()
    for it in range(MAX_ITER):
        if rng.random() < GOAL_BIAS:
            q_rand = GOAL.copy()
        else:
            q_rand = sampler(rng)
        dists = [np.linalg.norm(q_rand - n) for n in nodes]
        ni = int(np.argmin(dists))
        dirv = q_rand - nodes[ni]
        L = np.linalg.norm(dirv)
        q_new = nodes[ni] + (dirv / L) * min(STEP, L) if L > 1e-9 else q_rand
        q_new = clamp(q_new)
        if np.linalg.norm(q_new - nodes[ni]) < 1e-9:
            continue
        if is_edge_collision(nodes[ni], q_new):
            continue
        nodes.append(q_new.copy()); parent.append(ni)
        if np.linalg.norm(q_new - GOAL) < STEP:
            nodes.append(GOAL.copy()); parent.append(len(nodes) - 2)
            # 回溯路径
            path = []
            i = len(nodes) - 1
            while i != -1:
                path.append(nodes[i]); i = parent[i]
            return {"success": True, "n_nodes": len(nodes), "path": path[::-1],
                    "time": time.time() - t0, "checks": CHECKS["n"]}
    return {"success": False, "n_nodes": len(nodes), "path": None,
            "time": time.time() - t0, "checks": CHECKS["n"]}

# ---------- 3. 阶段A: 收集经验路径(老师) ----------
print("阶段A: 跑 20 个 seed 的纯 RRT, 收集经验路径(当老师)...")
TEACH_SEEDS = list(range(1000, 1020))
t0 = time.time()
exp_res = [rrt_plan(s, uniform_sampler) for s in TEACH_SEEDS]
n_ok = sum(1 for r in exp_res if r["success"])
print("  成功 %d/20 条, 用时 %.1f s" % (n_ok, time.time() - t0))

# 把每条路径细分, 得到密集的"走廊骨架点"
waypts = []
for r in exp_res:
    if not r["success"]:
        continue
    pa = r["path"]
    for k in range(len(pa) - 1):
        for s in np.linspace(0.0, 1.0, 5)[:-1]:
            waypts.append(pa[k] + s * (pa[k + 1] - pa[k]))
    waypts.append(pa[-1])
waypts = np.array(waypts)
print("  经验路径骨架点: %d 个(6 维)" % len(waypts))

# ---------- 4. 阶段B: 查询点 + 距离标签 ----------
NQ = 6000
rng = np.random.default_rng(7)
X = LO + rng.random((NQ, 6)) * (HI - LO)
dmin = np.empty(NQ)
for i0 in range(0, NQ, 1000):                       # 分块算到走廊骨架的最近距离
    Xc = X[i0:i0 + 1000]
    d2 = (Xc ** 2).sum(1)[:, None] + (waypts ** 2).sum(1)[None, :] - 2.0 * Xc @ waypts.T
    dmin[i0:i0 + 1000] = np.sqrt(np.maximum(d2, 0.0)).min(1)
# TAU 数据驱动(本课关键修正): 6 维空间里随机点到走廊的最近距离普遍 >1 rad,
# 拍脑袋定小 TAU 会让全图标签趋近 0、网络学不到东西 —— 改用距离分布的 15 分位数
TAU = float(np.percentile(dmin, 15))
Y = np.exp(-dmin / TAU)
print("阶段B: 最近距离中位数 %.2f rad, TAU(15分位)=%.2f, 标签>0.5 占 %.1f%% (走廊占比)"
      % (np.median(dmin), TAU, 100.0 * np.mean(Y > 0.5)))

# ---------- 5. 阶段C: 手写网络 6->48->1 学走廊概率 ----------
H = 48
def net_init(rng):
    return {"W1": rng.normal(0, 1, (H, 6)) * 0.5, "b1": np.zeros(H),
            "W2": rng.normal(0, 1, (1, H)) * 0.5, "b2": np.zeros(1)}

def to_norm(Q):
    return 2.0 * (Q - LO) / (HI - LO) - 1.0     # 关节角归一化到 [-1,1]

def forward(Qn, net):
    z1 = Qn @ net["W1"].T + net["b1"]
    a1 = np.tanh(z1)
    z2 = a1 @ net["W2"].T + net["b2"]
    return 1.0 / (1.0 + np.exp(-z2)), a1        # sigmoid 输出 + 隐藏层(反传要用)

Xn = to_norm(X)
W_CORR = 3.0                       # 走廊内样本权重 x3 (同第7课)
sw = 1.0 + (W_CORR - 1.0) * (Y > 0.5)

EPOCHS, BATCH, LR, MOM = 4000, 256, 0.05, 0.9
rng_t = np.random.default_rng(0)
net = net_init(rng_t)
vel = {k: np.zeros_like(v) for k, v in net.items()}
loss_hist = []
print("阶段C: 训练手写网络 6->48->1 (%d 轮, 小批量 %d)..." % (EPOCHS, BATCH))
t0 = time.time()
for ep in range(EPOCHS):
    idx = rng_t.choice(NQ, BATCH, replace=False)
    xb, yb, wb = Xn[idx], Y[idx][:, None], sw[idx][:, None]
    pr, a1 = forward(xb, net)
    loss = np.mean(wb * (pr - yb) ** 2)
    loss_hist.append(loss)
    # 反向传播(链式法则手推)
    dz2 = 2.0 * wb * (pr - yb) * pr * (1.0 - pr) / len(idx)      # sigmoid 导数
    dW2 = dz2.T @ a1; db2 = dz2.sum(0)
    da1 = dz2 @ net["W2"]
    dz1 = da1 * (1.0 - a1 ** 2)                                  # tanh 导数
    dW1 = dz1.T @ xb; db1 = dz1.sum(0)
    for k, g in (("W1", dW1), ("b1", db1), ("W2", dW2), ("b2", db2)):
        vel[k] = MOM * vel[k] - LR * g
        net[k] += vel[k]
print("  训练完成: %.1f s, 初始损失 %.4f -> 最终 %.4f" % (time.time() - t0, loss_hist[0], loss_hist[-1]))

pred_full = forward(Xn, net)[0].ravel()
corr = float(np.corrcoef(pred_full, Y)[0, 1])
print("  网络预测 vs 老师标签 相关系数: %.3f" % corr)

# ---------- 6. 阶段D: 引导版 vs 均匀版, 各 40 个 seed ----------
EPS_INF, EXPO, N_CAND = 0.90, 5.0, 64
def guided_sampler(rng):
    # EPS_INF 概率按网络概率挑"走廊候选", 其余回退均匀撒(防止网络过度自信卡死)
    if rng.random() > EPS_INF:
        return uniform_sampler(rng)
    cand = LO + rng.random((N_CAND, 6)) * (HI - LO)
    pr = forward(to_norm(cand), net)[0].ravel() ** EXPO
    s = pr.sum()
    if s <= 1e-12 or not np.isfinite(s):
        return cand[0]
    t = rng.random() * s
    return cand[int(np.searchsorted(np.cumsum(pr), t))]

TEST_SEEDS = list(range(40))
print("阶段D: 均匀版 vs 引导版, 各 %d 个 seed..." % len(TEST_SEEDS))
res_u = [rrt_plan(s, uniform_sampler) for s in TEST_SEEDS]
res_g = [rrt_plan(s, guided_sampler) for s in TEST_SEEDS]
nu = np.array([r["n_nodes"] for r in res_u], dtype=float)
ng = np.array([r["n_nodes"] for r in res_g], dtype=float)
cu = np.array([r["checks"] for r in res_u], dtype=float)
cg = np.array([r["checks"] for r in res_g], dtype=float)
ok_u = sum(1 for r in res_u if r["success"]); ok_g = sum(1 for r in res_g if r["success"])
print("\n===== 关键数据 =====")
print("均匀版: 成功 %d/%d, 节点 %.1f±%.1f, 碰撞检查 %.0f±%.0f" % (ok_u, len(res_u), nu.mean(), nu.std(), cu.mean(), cu.std()))
print("引导版: 成功 %d/%d, 节点 %.1f±%.1f, 碰撞检查 %.0f±%.0f" % (ok_g, len(res_g), ng.mean(), ng.std(), cg.mean(), cg.std()))
print("引导版节点数 = 均匀版的 %.1f%%" % (100.0 * ng.mean() / nu.mean()))
print("引导版碰撞检查 = 均匀版的 %.1f%%" % (100.0 * cg.mean() / cu.mean()))

# ---------- 7. 画 2x3 六宫格 ----------
fig = plt.figure(figsize=(17, 9.5))

def draw_box(ax, center, half, color):
    cx, cy, cz = center; hx, hy, hz = half
    x0, x1 = cx-hx, cx+hx; y0, y1 = cy-hy, cy+hy; z0, z1 = cz-hz, cz+hz
    v = np.array([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
                  [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]])
    faces = [[v[0],v[1],v[2],v[3]],[v[4],v[5],v[6],v[7]],
             [v[0],v[1],v[5],v[4]],[v[2],v[3],v[7],v[6]],
             [v[1],v[2],v[6],v[5]],[v[0],v[3],v[7],v[4]]]
    ax.add_collection3d(Poly3DCollection(faces, facecolor=color, edgecolor="k", alpha=0.35))

def ee_of_path(pa):
    out = []
    for q in pa:
        set_config(q); out.append(ee_pos())
    return np.array(out)

# (0,0) 老师: 20 条经验路径的末端轨迹
ax1 = fig.add_subplot(2, 3, 1, projection="3d")
for r in exp_res:
    if r["success"]:
        ee = ee_of_path(r["path"])
        ax1.plot(ee[:,0], ee[:,1], ee[:,2], "-", lw=1, color="steelblue", alpha=0.7)
draw_box(ax1, *OBST_MESHES[0], "#e67e22"); draw_box(ax1, *OBST_MESHES[1], "#e67e22")
ax1.set_title("(a) 老师: 20 条经验路径的末端轨迹")

# (0,1) 学生: 6 维概率图的切片(固定关节3~6=起点值, 看关节1-2平面)
ax2 = fig.add_subplot(2, 3, 2)
g1 = np.linspace(LO[0], HI[0], 80); g2 = np.linspace(LO[1], HI[1], 80)
G1, G2 = np.meshgrid(g1, g2)
slice_pts = np.column_stack([G1.ravel(), G2.ravel(),
                             np.full(G1.size, START[2]), np.full(G1.size, START[3]),
                             np.full(G1.size, START[4]), np.full(G1.size, START[5])])
P2 = forward(to_norm(slice_pts), net)[0].ravel().reshape(G1.shape)
im2 = ax2.imshow(P2, extent=[np.degrees(LO[0]), np.degrees(HI[0]),
                             np.degrees(LO[1]), np.degrees(HI[1])],
                 origin="lower", aspect="auto", cmap="viridis")
fig.colorbar(im2, ax=ax2, label="网络输出的走廊概率")
ax2.set_title("(b) 学生概率图(6维切片: 固定关节3~6)")
ax2.set_xlabel("关节1(度)"); ax2.set_ylabel("关节2(度)")

# (0,2) 损失曲线
ax3 = fig.add_subplot(2, 3, 3)
ax3.plot(loss_hist, color="crimson", lw=1.2)
ax3.set_title("(c) 训练损失(加权MSE)")
ax3.set_xlabel("训练轮数"); ax3.set_ylabel("损失")
ax3.grid(alpha=0.3)

# (1,0) 每 seed 节点数散点对比
ax4 = fig.add_subplot(2, 3, 4)
ax4.scatter(TEST_SEEDS, nu, color="gray", s=36, label="均匀版")
ax4.scatter(TEST_SEEDS, ng, color="crimson", s=36, label="引导版")
ax4.axhline(nu.mean(), color="gray", ls="--", lw=1)
ax4.axhline(ng.mean(), color="crimson", ls="--", lw=1)
ax4.set_title("(d) %d seed 节点数: 均匀 %.0f vs 引导 %.0f" % (len(TEST_SEEDS), nu.mean(), ng.mean()))
ax4.set_xlabel("seed"); ax4.set_ylabel("树节点数")
ax4.legend(); ax4.grid(alpha=0.3)

# (1,1) 引导版一条成功路径的末端轨迹
ax5 = fig.add_subplot(2, 3, 5, projection="3d")
best_g = min((r for r in res_g if r["success"]), key=lambda r: r["n_nodes"], default=None)
if best_g is not None:
    ee = ee_of_path(best_g["path"])
    ax5.plot(ee[:,0], ee[:,1], ee[:,2], "r-", lw=2)
    ax5.scatter(*ee[0], c="green", s=60); ax5.scatter(*ee[-1], c="blue", s=60)
draw_box(ax5, *OBST_MESHES[0], "#e67e22"); draw_box(ax5, *OBST_MESHES[1], "#e67e22")
ax5.set_title("(e) 引导版最优一次: %d 节点路径" % (best_g["n_nodes"] if best_g else -1))

# (1,2) 碰撞检查次数对比(计算量)
ax6 = fig.add_subplot(2, 3, 5 + 1)
bars = ax6.bar(["均匀版", "引导版"], [cu.mean(), cg.mean()],
               color=["gray", "crimson"], alpha=0.8)
ax6.errorbar([0, 1], [cu.mean(), cg.mean()], yerr=[cu.std(), cg.std()],
             fmt="none", ecolor="k", capsize=5)
for b, v in zip(bars, [cu.mean(), cg.mean()]):
    ax6.text(b.get_x() + b.get_width()/2, v, "%.0f" % v, ha="center", va="bottom")
ax6.set_title("(f) 平均碰撞检查次数(计算量对比)")
ax6.set_ylabel("次数")

OUT_PNG = "C:/Users/csystudio/Desktop/挺好/arm_research/lesson11_6dof_learned.png"
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=130)
print("已保存", OUT_PNG)
p.disconnect()
