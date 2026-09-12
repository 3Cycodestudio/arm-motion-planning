# -*- coding: utf-8 -*-
"""
运行方法:  python lesson10_scene_gui.py     (在 VS Code 终端跑, 会弹出 3D 窗口)
本脚本做哪几件事:
1. 在 pybullet 的 GUI 窗口里显示第10课的场景: Panda 机械臂 + 墙 + 立柱
2. 窗口右侧有 6 个滑块, 对应 6 个关节(单位: 度), 拖动就能"手动拧关节"
3. 实时碰撞提示: 机械臂碰到墙/柱(或自碰撞)时, 障碍变红 + 顶部显示警告;
   安全时障碍保持橙色、显示"安全"
关闭窗口即结束。鼠标: 左键拖=旋转视角, 滚轮=缩放, 中键拖=平移。
"""
import numpy as np
import pybullet as p
import pybullet_data
import time

cid = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)
p.resetDebugVisualizerCamera(cameraDistance=2.2, cameraYaw=55,
                             cameraPitch=-25, cameraTargetPosition=[0.3, 0, 0.5])
robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
N_JOINTS = p.getNumJoints(robot)
ALL_REV = [i for i in range(N_JOINTS) if p.getJointInfo(robot, i)[2] == p.JOINT_REVOLUTE]
FROZEN_IDX = 6
ACTIVE = [i for i in ALL_REV if i != FROZEN_IDX]
LO = np.array([p.getJointInfo(robot, j)[8] for j in ACTIVE], dtype=float)
HI = np.array([p.getJointInfo(robot, j)[9] for j in ACTIVE], dtype=float)

SELF_OK = {(-1, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (-1, 1), (0, 2), (1, 3),
           (6, 8), (6, 9), (6, 10), (7, 8), (7, 9), (7, 10), (8, 9), (8, 10), (9, 10)}
SELF_OK |= {(b, a) for (a, b) in SELF_OK}
MARGIN = 0.015

# ---------- 障碍: 墙 + 立柱(和第10课同一位置尺寸), 带可见外观 ----------
def make_box(center, half, color):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=list(color) + [0.55])
    return p.createMultiBody(0, col, baseVisualShapeIndex=vis, basePosition=center)

wall_body   = make_box([0.46, 0.0, 0.45], [0.04, 0.30, 0.45], [0.95, 0.55, 0.15])
pillar_body = make_box([0.28, 0.35, 0.30], [0.05, 0.05, 0.35], [0.95, 0.55, 0.15])
OBSTACLES = [wall_body, pillar_body]

# ---------- 6 个关节滑块(度), 初始摆成第10课的起点姿态 ----------
START = np.radians([50, -40, 10, -115, 10, 75])
sliders = []
for k in range(6):
    sid = p.addUserDebugParameter("关节%d (度)" % (k + 1),
                                  np.degrees(LO[k]), np.degrees(HI[k]),
                                  np.degrees(START[k]))
    sliders.append(sid)
warn_id = p.addUserDebugText("安全", [0, 0, 1.15], textColorRGB=[0, 0.7, 0],
                             textSize=1.5)

def check_collision():
    for obs in OBSTACLES:
        for pt in p.getClosestPoints(robot, obs, distance=MARGIN):
            if pt[8] < MARGIN:
                return True
    for pt in p.getClosestPoints(robot, robot, distance=0.0):
        a, b = pt[3], pt[4]
        if pt[8] < 0.0 and abs(a - b) > 1 and (a, b) not in SELF_OK:
            return True
    return False

# ---------- 主循环 ----------
# 连接自检: GUI 连接失败时 connect 可能返回 -1, 先拦下来给出人话提示
if cid < 0:
    print("!! GUI 连接建立失败(可能被远程桌面/显卡驱动拦了)。")
    print("   换在 VS Code 本地终端重跑; 仍失败就告诉我, 我改成无界面版。")
    raise SystemExit(1)

print("GUI 已打开: 拖右侧滑块拧关节, 鼠标左键转视角/滚轮缩放/中键平移。关闭窗口结束。")
n_frames = 0
try:
    while True:
        q = np.radians([p.readUserDebugParameter(s) for s in sliders])
        for k, j in enumerate(ACTIVE):
            p.resetJointState(robot, j, q[k])
        p.resetJointState(robot, FROZEN_IDX, 0.0)
        hit = check_collision()
        col = [1.0, 0.1, 0.1, 0.7] if hit else [0.95, 0.55, 0.15, 0.55]
        for b in OBSTACLES:
            p.changeVisualShape(b, -1, rgbaColor=col)
        try:
            p.removeUserDebugItem(warn_id)
            warn_id = p.addUserDebugText(
                "!! 碰撞 !!" if hit else "安全", [0, 0, 1.15],
                textColorRGB=[1, 0, 0] if hit else [0, 0.7, 0], textSize=1.5)
        except Exception:
            pass                      # 文字刷新失败不拖垮主循环
        n_frames += 1
        time.sleep(1.0 / 60.0)
except KeyboardInterrupt:
    print("手动中断(Ctrl+C), 退出。")
except Exception as e:
    # 引擎连接断开(窗口被关/被系统收走)都走这里: 给人话, 不给一屏红字
    print("仿真窗口已关闭(共运行 %d 帧), 正常退出。" % n_frames)
    print("(技术细节: %s)" % e)
finally:
    try:
        p.disconnect()
    except Exception:
        pass
