"""
# @Time    : 2021/7/2 5:22 下午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : env.py
"""
import fmpy
import shutil
import numpy as np
from fmpy.fmi2 import FMU2Slave


class Env(object):
    """
    # 环境中的智能体
    """

    def __init__(self, i):
        self.agent_num = 2  # 设置智能体(小飞机)的个数，这里设置为两个
        self.obs_dim = 1  # 设置智能体的观测纬度
        self.action_dim = 1  # 设置智能体的动作纬度，这里假定为一个五个纬度的

        # define the model name and simulation parameters
        self.fmu_filename = 'plant.fmu'
        self.start_time = 0.0
        # threshold = 2.0
        self.stop_time = 20.0
        self.step_size = 0.1
        self.time = 0
        self.rows = []

        # read the model description
        self.model_description = fmpy.read_model_description(self.fmu_filename)

        # collect the value reference
        vrs = {}
        for variable in self.model_description.modelVariables:
            vrs[variable.name] = variable.valueReference

        # get the value references for the variables we want to get/set
        self.vr_servoventil = vrs['servoventil']
        self.vr_schaltventil1 = vrs['schaltventil1']
        self.vr_schaltventil2 = vrs['schaltventil2']
        self.vr_position = vrs['position']

        # extract the FMU
        unzipdir = fmpy.extract(self.fmu_filename)

        self.fmu = FMU2Slave(guid=self.model_description.guid,
                             unzipDirectory=unzipdir,
                             modelIdentifier=self.model_description.coSimulation.modelIdentifier,
                             instanceName='instance1')

    def reset(self):
        """
        # self.agent_num设定为2个智能体时，返回值为一个list，每个list里面为一个shape = (self.obs_dim, )的观测数据
        """

        # initialize
        self.fmu.instantiate()
        self.fmu.setupExperiment(startTime=self.start_time)
        self.fmu.enterInitializationMode()
        self.fmu.exitInitializationMode()
        self.time = self.start_time
        self.rows = []

        sub_agent_obs = []
        for i in range(self.agent_num):
            sub_obs = np.random.random(size=(self.obs_dim,))
            sub_agent_obs.append(sub_obs)
        return sub_agent_obs

    def step(self, actions):
        """
        # self.agent_num设定为2个智能体时，actions的输入为一个2纬的list，每个list里面为一个shape = (self.action_dim, )的动作数据
        # 默认参数情况下，输入为一个list，里面含有两个元素，因为动作纬度为5，所里每个元素shape = (5, )
        """
        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []
        for i in range(self.agent_num):
            self.fmu.setReal([self.vr_servoventil, self.vr_schaltventil1, self.vr_schaltventil2], [-1, 1, 0])
            self.fmu.doStep(currentCommunicationPoint=self.time, communicationStepSize=self.step_size)
            self.time += self.step_size
            schaltventil1, servoventil, schaltventil2, position = self.fmu.getReal(
                [self.vr_schaltventil1, self.vr_servoventil, self.vr_schaltventil2, self.vr_position])
            self.rows.append((self.time, schaltventil1, servoventil, schaltventil2, position))

            sub_agent_obs.append([position])
            sub_agent_reward.append([pow(position, 2)])
            sub_agent_done.append(self.time >= self.stop_time)
            sub_agent_info.append({})

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]
