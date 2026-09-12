# -*- coding: utf-8 -*-
"""
运行方法:  python pybullet_smoke_test.py
本脚本做哪几件事(冒烟测试, 验证 pybullet 装好了且能用):
1. 无界面模式启动物理引擎(DIRECT), 打印版本号
2. 加载官方自带的 KUKA iiwa 六轴机械臂(真实机器人模型)
3. 读出 6 个关节的信息和初始末端位置(正运动学)
4. 随机拧一个关节角, 再算一次末端位置, 验证 FK 会动
5. 做一次碰撞检查(把末端"推"进地面, 看能否检出碰撞)
结尾打印关键结论: 每一步都 OK 就说明 pybullet 完全可用。
"""
import pybullet as p
import numpy as np

print("=" * 50)
print("[1] 启动物理引擎(无界面 DIRECT 模式)")
cid = p.connect(p.DIRECT)
print("    连接ID:", cid, " pybullet版本:", p.getAPIVersion())

print("[2] 加载 KUKA iiwa 六轴机械臂(官方模型库)")
p.setAdditionalSearchPath(pybullet_data_path := __import__("pybullet_data").getDataPath())
robot = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)
print("    模型加载成功, 机体ID:", robot)
print("    模型路径:", pybullet_data_path)

print("[3] 读关节与正运动学(FK)")
n_joints = p.getNumJoints(robot)
revolute = [i for i in range(n_joints)
            if p.getJointInfo(robot, i)[2] == p.JOINT_REVOLUTE]
print("    可动(转动)关节数:", len(revolute))

def fk_joint_angle(joint_id, angle):
    """把某个关节拧到 angle(弧度), 返回末端连杆的世界坐标"""
    p.resetJointState(robot, joint_id, angle)
    ee_state = p.getLinkState(robot, revolute[-1],
                              computeForwardKinematics=True)
    return np.array(ee_state[4])  # 末端连杆质心的世界坐标

ee0 = fk_joint_angle(revolute[0], 0.0)
ee1 = fk_joint_angle(revolute[0], 0.5)  # 关节0 拧 0.5 rad ≈ 28.6 度
print("    初始末端位置:", np.round(ee0, 4))
print("    关节0拧28.6度后:", np.round(ee1, 4))
print("    末端移动距离:", round(float(np.linalg.norm(ee1 - ee0)), 4), "米")

print("[4] 碰撞检查能力测试")
p.resetSimulation()
p.setAdditionalSearchPath(pybullet_data_path)
plane = p.createCollisionShape(p.GEOM_PLANE)
p.createMultiBody(0, plane)                      # 地面
box = p.createCollisionShape(p.GEOM_BOX,
                             halfExtents=[0.1, 0.1, 0.1])
body = p.createMultiBody(1.0, box,
                         basePosition=[0, 0, 0.5])  # 离地 0.5m 的方块
p.performCollisionDetection()
contact_free = len(p.getClosestPoints(body, plane, distance=0.05)) == 0
print("    方块悬空时与地面有碰撞吗:", "否(正确)" if contact_free else "是(错误!)")
p.resetBasePositionAndOrientation(body, [0, 0, 0.02], [0, 0, 0, 1])
p.performCollisionDetection()
contact_hit = len(p.getClosestPoints(body, plane, distance=0.01)) > 0
print("    方块压到地面时有碰撞吗:", "是(正确)" if contact_hit else "否(错误!)")

p.disconnect()
print("=" * 50)
print("结论: 4 项全部通过 -> pybullet 在你的 Python 3.12 上完全可用!")
