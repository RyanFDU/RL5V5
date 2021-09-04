"""
@FileName：config.py
@Description：
@Author：wubinxing
@Time：2021/5/9 下午8:08
@Department：AIStudio研发部
@Copyright：©2011-2021 北京华如科技股份有限公司
"""
from agent.demo_agent import DemoAgent
from agent.FDU2021_agent import FDU2021_agent

# 是否启用host模式,host仅支持单个xsim
ISHOST = True

# 为态势显示工具域分组ID  1-1000
HostID = 1

IMAGE = "xsim:v6.0"


config = {
    "episode_time": 20000000,   # 训练次数
    "step_time": 1, # 想定步长
    'agents': {
            'red': FDU2021_agent,
            'blue': DemoAgent
              }
}

# 进程数量
POOL_NUM = 10

# 启动XSIM的数量
XSIM_NUM = 2


ADDRESS = {
    "ip": "10.98.126.118",
    "port": 50051
}
