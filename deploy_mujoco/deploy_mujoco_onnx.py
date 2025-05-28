import time
import mujoco.viewer
import mujoco
import numpy as np
import torch
import onnx
import onnxruntime
import yaml
from pathlib import Path
from typing import Dict, Any

class MuJoCoConfig:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # 基本配置
        self.xml_path = config_data.get('xml_path', '')
        self.policy_path = config_data.get('policy_path', '')
        self.msg_type = config_data.get('msg_type', 'hg')
        
        # 控制参数
        self.control_dt = config_data.get('control_dt', 0.01)
        self.simulation_dt = config_data.get('simulation_dt', 0.001)
        self.control_decimation = int(self.control_dt / self.simulation_dt)
        
        # 机器人参数
        self.num_actions = config_data.get('num_actions', 23)
        self.num_obs = config_data.get('num_obs', 52)
        self.history_length = config_data.get('history_length', 4)
        
        # 默认角度和增益
        self.default_angles = np.array(config_data.get('default_angles', [0.0] * self.num_actions), dtype=np.float32)
        self.kps = np.array(config_data.get('kps', [100.0] * self.num_actions), dtype=np.float32)
        self.kds = np.array(config_data.get('kds', [2.5] * self.num_actions), dtype=np.float32)
        self.tau_limit = np.array(config_data.get('tau_limit', [50.0] * self.num_actions), dtype=np.float32)
        
        # 关节限制
        self.dof_pos_limit_low = np.array(config_data.get('dof_pos_limit_low', [-3.14] * self.num_actions), dtype=np.float32)
        self.dof_pos_limit_up = np.array(config_data.get('dof_pos_limit_up', [3.14] * self.num_actions), dtype=np.float32)
        
        # 观测缩放
        self.dof_pos_scale = config_data.get('dof_pos_scale', 1.0)
        self.dof_vel_scale = config_data.get('dof_vel_scale', 0.05)
        self.ang_vel_scale = config_data.get('ang_vel_scale', 0.25)
        self.action_scale = config_data.get('action_scale', 0.25)
        
        # 运动相位参数
        self.ref_motion_phase_step = config_data.get('ref_motion_phase_step', 0.00314)
        
        # 观测模式设置
        self.single_frame = config_data.get('single_frame', False)
        self.use_linear_velocity = config_data.get('use_linear_velocity', False)
        self.linear_velocity_scale = config_data.get('linear_velocity_scale', 2.0)
        
        # 仿真参数
        self.simulation_duration = config_data.get('simulation_duration', 60)

def get_gravity_orientation(quaternion):
    """从四元数计算重力方向"""
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """PD控制计算扭矩"""
    return (target_q - q) * kp + (target_dq - dq) * kd

class MuJoCoController:
    def __init__(self, config: MuJoCoConfig):
        self.config = config
        
        # 初始化ONNX模型
        self.ort_session = onnxruntime.InferenceSession(config.policy_path)
        self.input_name = self.ort_session.get_inputs()[0].name
        
        # 初始化状态变量
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()
        self.ref_motion_phase = 0.0
        self.counter = 0
        
        # 初始化历史缓冲区
        self.init_history_buffers()
        
        # 加载MuJoCo模型
        self.m = mujoco.MjModel.from_xml_path(config.xml_path)
        self.d = mujoco.MjData(self.m)
        self.m.opt.timestep = config.simulation_dt
        
        print(f"Policy path: {config.policy_path}")
        print(f"XML path: {config.xml_path}")
        print(f"Number of actions: {config.num_actions}")
        print(f"Control decimation: {config.control_decimation}")
    
    def init_history_buffers(self):
        """初始化历史缓冲区"""
        config = self.config
        self.lin_vel_buf = np.zeros(3 * config.history_length, dtype=np.float32)
        self.ang_vel_buf = np.zeros(3 * config.history_length, dtype=np.float32)
        self.proj_g_buf = np.zeros(3 * config.history_length, dtype=np.float32)
        self.dof_pos_buf = np.zeros(config.num_actions * config.history_length, dtype=np.float32)
        self.dof_vel_buf = np.zeros(config.num_actions * config.history_length, dtype=np.float32)
        self.action_buf = np.zeros(config.num_actions * config.history_length, dtype=np.float32)
        self.ref_motion_phase_buf = np.zeros(1 * config.history_length, dtype=np.float32)
    
    def get_observation(self):
        """构建观测向量"""
        config = self.config
        
        # 从MuJoCo状态获取数据
        qj = self.d.qpos[7:]  # 关节位置 (跳过base的7个自由度)
        dqj = self.d.qvel[6:]  # 关节速度 (跳过base的6个自由度)
        quat = self.d.qpos[3:7]  # 四元数
        lin_vel = self.d.qvel[:3]  # 线速度
        ang_vel = self.d.qvel[3:6]  # 角速度
        
        # 处理观测数据
        projected_gravity = get_gravity_orientation(quat)
        dof_pos = qj * config.dof_pos_scale
        dof_vel = dqj * config.dof_vel_scale
        base_ang_vel = ang_vel * config.ang_vel_scale
        base_lin_vel = lin_vel * config.linear_velocity_scale
        
        # 更新运动相位
        self.ref_motion_phase += config.ref_motion_phase_step
        
        # 构建观测向量
        if config.single_frame:
            # 单帧观测
            obs_buf = np.concatenate((
                self.action,
                base_ang_vel,
                base_lin_vel,
                dof_pos,
                dof_vel,
                projected_gravity,
                np.array([self.ref_motion_phase])
            ), axis=-1, dtype=np.float32)
        else:
            # 多帧历史观测
            if config.use_linear_velocity:
                history_obs_buf = np.concatenate((
                    self.action_buf, 
                    self.ang_vel_buf, 
                    self.lin_vel_buf, 
                    self.dof_pos_buf, 
                    self.dof_vel_buf, 
                    self.proj_g_buf, 
                    self.ref_motion_phase_buf
                ), axis=-1, dtype=np.float32)
                
                obs_buf = np.concatenate((
                    self.action, 
                    base_ang_vel, 
                    base_lin_vel, 
                    dof_pos, 
                    dof_vel, 
                    history_obs_buf, 
                    projected_gravity, 
                    [self.ref_motion_phase]
                ), axis=-1, dtype=np.float32)
            else:
                history_obs_buf = np.concatenate((
                    self.action_buf, 
                    self.ang_vel_buf, 
                    self.dof_pos_buf, 
                    self.dof_vel_buf, 
                    self.proj_g_buf, 
                    self.ref_motion_phase_buf
                ), axis=-1, dtype=np.float32)
                
                obs_buf = np.concatenate((
                    self.action, 
                    base_ang_vel, 
                    dof_pos, 
                    dof_vel, 
                    history_obs_buf, 
                    projected_gravity, 
                    [self.ref_motion_phase]
                ), axis=-1, dtype=np.float32)
        
        # 更新历史缓冲区
        self.update_history_buffers(base_ang_vel, base_lin_vel, projected_gravity, dof_pos, dof_vel)
        
        return obs_buf
    
    def update_history_buffers(self, base_ang_vel, base_lin_vel, projected_gravity, dof_pos, dof_vel):
        """更新历史缓冲区"""
        config = self.config
        
        self.ang_vel_buf = np.concatenate((base_ang_vel, self.ang_vel_buf[:-3]), axis=-1, dtype=np.float32)
        self.lin_vel_buf = np.concatenate((base_lin_vel, self.lin_vel_buf[:-3]), axis=-1, dtype=np.float32)
        self.proj_g_buf = np.concatenate((projected_gravity, self.proj_g_buf[:-3]), axis=-1, dtype=np.float32)
        self.dof_pos_buf = np.concatenate((dof_pos, self.dof_pos_buf[:-config.num_actions]), axis=-1, dtype=np.float32)
        self.dof_vel_buf = np.concatenate((dof_vel, self.dof_vel_buf[:-config.num_actions]), axis=-1, dtype=np.float32)
        self.action_buf = np.concatenate((self.action, self.action_buf[:-config.num_actions]), axis=-1, dtype=np.float32)
        self.ref_motion_phase_buf = np.concatenate((np.array([self.ref_motion_phase]), self.ref_motion_phase_buf[:-1]), axis=-1, dtype=np.float32)
    
    def get_action(self, obs_buf):
        """从观测获取动作"""
        obs_tensor = torch.from_numpy(obs_buf).unsqueeze(0).cpu().numpy()
        action = np.squeeze(self.ort_session.run(None, {self.input_name: obs_tensor})[0])
        return action
    
    def run_simulation(self):
        """运行仿真"""
        config = self.config
        
        with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
            start = time.time()
            
            while viewer.is_running() and time.time() - start < config.simulation_duration:
                step_start = time.time()
                
                # PD控制
                tau = pd_control(
                    self.target_dof_pos, 
                    self.d.qpos[7:], 
                    config.kps, 
                    np.zeros_like(config.kds), 
                    self.d.qvel[6:], 
                    config.kds
                )
                tau = np.clip(tau, -config.tau_limit, config.tau_limit)
                self.d.ctrl[:] = tau
                
                # 步进仿真
                mujoco.mj_step(self.m, self.d)
                
                self.counter += 1
                
                # 控制频率抽取
                if self.counter % config.control_decimation == 0:
                    # 获取观测
                    obs_buf = self.get_observation()
                    
                    # 获取动作
                    self.action = self.get_action(obs_buf)
                    
                    # 转换为目标关节位置
                    self.target_dof_pos = self.action * config.action_scale + config.default_angles
                    
                    # 限制关节位置
                    self.target_dof_pos = np.clip(
                        self.target_dof_pos, 
                        config.dof_pos_limit_low, 
                        config.dof_pos_limit_up
                    )
                
                # 同步viewer
                viewer.sync()
                
                # 时间控制
                time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='MuJoCo ONNX Deployment')
    parser.add_argument('config', type=str, help='Path to YAML config file')
    args = parser.parse_args()
    
    # 加载配置
    config = MuJoCoConfig(args.config)
    
    # 创建控制器并运行仿真
    controller = MuJoCoController(config)
    controller.run_simulation()

if __name__ == "__main__":
    main()