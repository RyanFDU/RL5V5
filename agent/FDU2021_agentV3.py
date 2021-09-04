"""
@FileName：demo_agent.py
@Description：
@Author：
@Time：2021/6/17 上午9:21
@Department：AIStudio研发部
@Copyright：©2011-2021 北京华如科技股份有限公司
"""
from typing import List
from agent.agent import Agent
from env.env_cmd import CmdEnv
from utils.utils_math import TSVector3
import numpy as np
import  copy
import  random
from collections import defaultdict

"""
选手需要重写继承自基类Agent中的step(self,sim_time, obs_red, **kwargs)去实现自己的策略逻辑，注意，这个方法每个步长都会被调用
"""
num2id = {'red':{0:1, 1:2, 2:11, 3:12, 4:13},'blue':{0:6, 1:14, 2:15, 3:16, 4:17}}
obs_own_missile_nums = 2
obs_enemy_missile_nums = 2
x_y_threshold = 1000
z_threshold = 300
class FDU2021_agent(Agent):
    """
         自定义的Demo智能体
     @Examples:
         添加使用示例
         >>> 填写使用说明
         ··· 填写简单代码示例
     """
    def __init__(self, name, config):
        """
        初始化信息
        :param name:阵营名称
        :param config:阵营配置信息
        """
        super(FDU2021_agent, self).__init__(name, config["side"])
        if self.side == 'blue':
            self.my_numids = num2id['blue']
            self.enemy_numids = num2id['red']
        else:
            self.my_numids = num2id['red']
            self.enemy_numids = num2id['blue']
        self._init() #调用用以定义一些下文所需变量

    def _init(self):
        """对下文中使用到的变量进行定义和初始化"""
        self.my_uvas_infos = []         #该变量用以保存己方所有无人机的态势信息
        self.my_manned_info = []        #该变量用以保存己方有人机的态势信息
        self.my_allplane_infos = []     #该变量用以保存己方所有飞机的态势信息
        self.enemy_uvas_infos = []      #该变量用以保存敌方所有无人机的态势信息
        self.enemy_manned_info = []     #该变量用以保存敌方有人机的态势信息
        self.enemy_allplane_infos = []  #该变量用以保存敌方所有飞机的态势信息
        self.enemy_missile_infos = []   #该变量用以保存敌方导弹的态势信息

        self.attack_handle_enemy = {}   #该变量用于记录已经去攻击的飞机
        self.in_range_enemy = defaultdict(list)  # 该变量用于解算当前所有飞机的相对态势
        self.my_attacking_enemy = {} # 该变量用于记录正在去攻击的飞机

        "对环境的初始化"
        self.n_agents = 5
        self.n_enemies = 5
        self.n_actions = 13
        self.map_len = 300000
        self.map_width = 300000
        self.enemy_features= {'rel_x':0,'rel_y':0, 'rel_z':0,'head':0,'pitch': 0,'speed':0,'islocked':0}
        self.ally_features = {'rel_x':0,'rel_y':0, 'rel_z':0,'head':0,'pitch': 0,'speed':0,'islocked':0}
        self.own_features = {'x':0,'y':0, 'z':0,'head':0,'pitch':0,'roll':0,'speed':0,'islocked':0,'leftweapon':0}
        self.enemy_missile_features = {'rel_x':0,'rel_y':0, 'rel_z':0,'head':0,'pitch':0,'speed':0}
        self.own_missile_features = {'rel_x': 0, 'rel_y': 0, 'rel_z': 0, 'head': 0, 'pitch':0,'speed': 0}
        self.type_features = [0,0]

    def reset(self, **kwargs):
        """当引擎重置会调用,选手需要重写此方法,来实现重置的逻辑"""
        self.attack_handle_enemy.clear() #重置已经去攻击的飞机的信息
        self.in_range_enemy.clear()
        self.my_attacking_enemy.clear()
        pass

    def step(self,sim_time, obs_side, **kwargs) -> List[dict]:
        """ 步长处理
        此方法继承自基类中的step(self,sim_time, obs_red, **kwargs)
        选手通过重写此方法，去实现自己的策略逻辑，注意，这个方法每个步长都会被调用
        :param sim_time: 当前想定已经运行时间
        :param obs_side:当前方所有的态势信息信息，包含了所有的当前方所需信息以及探测到的敌方信息
				obs_side 包含 platforminfos，trackinfos，missileinfos三项Key值
				obs_side['platforminfos'] 为己方飞机信息，字典列表，其中包含的字典信息如下（以下为Key值）

                        Name 			# 飞机的名称
                        Identification 	# 飞机的标识符（表示飞机是红方还是蓝方）
                        ID 				# 飞机的ID（表示飞机的唯一编号）
                        Type 			# 飞机的类型（表示飞机的类型，其中有人机类型为 1，无人机类型为2）
						Availability 	# 飞机的可用性（表示飞机的可用性，范围为0到1,为1表示飞机存活，0表示飞机阵亡）
						X 				# 飞机的当前X坐标（表示飞机的X坐标）
						Y 				# 飞机的当前Y坐标（表示飞机的Y坐标）
						Lon 			# 飞机的当前所在经度（表示飞机的所在经度）
						Lat 			# 飞机的当前所在纬度（表示飞机的所在纬度）
						Alt 			# 飞机的当前所在高度（表示飞机的所在高度）
						Heading 		# 飞机的当前朝向角度（飞机的当前朝向,范围为-180°到180° 朝向0°表示正北,逆时针方向旋转为正数0°到180°，顺时针方向为正数0°到-180°）
						Pitch 			# 飞机的当前俯仰角度（飞机的当前俯仰,范围为-90°到90°,朝向高处为正,低处为负）
						Roll 			# 飞机的当前滚转角度（飞机的当前滚转,范围为-180°到180° ）
						Speed 			# 飞机的当前速度（飞机的当前速度）
						CurTime 		# 当前时间（当前时间）
						AccMag 			# 飞机的指令加速度（飞机的指令加速度）
						NormalG 		# 飞机的指令过载（飞机的指令过载）
						IsLocked 		# 飞机是否被敌方导弹锁定（飞机是否被敌方导弹锁定）
						Status 			# 飞机的当前状态（飞机的当前状态）
						LeftWeapon 		# 飞机的当前剩余导弹数（飞机的当前剩余导弹数）

				obs_side['trackinfos'] 为敌方飞机信息，字典列表，其中包含的字典信息如下（以下为Key值）

						Name 			# 敌方飞机的名称
						Identification 	# 敌方飞机的标识符（表示敌方飞机是红方还是蓝方）
						ID 				# 敌方飞机的ID（表示飞机的唯一编号）
						Type 			# 敌方飞机的类型（表示飞机的类型，其中有人机类型为 1，无人机类型为2）
						Availability 	# 敌方飞机的可用性（表示飞机的可用性，范围为0到1,为1表示飞机存活，0表示飞机阵亡）
						X 				# 敌方飞机的当前X坐标（表示飞机的X坐标）
						Y 				# 敌方飞机的当前Y坐标（表示飞机的Y坐标）
						Lon 			# 敌方飞机的当前所在经度（表示飞机的所在经度）
						Lat 			# 敌方飞机的当前所在纬度（表示飞机的所在纬度）
						Alt 			# 敌方飞机的当前所在高度（表示飞机的所在高度）
						Heading 		# 敌方飞机的当前朝向角度（飞机的当前朝向,范围为-180°到180° 朝向0°表示正北,逆时针方向旋转为正数0°到180°，顺时针方向为正数0°到-180°）
						Pitch 			# 敌方飞机的当前俯仰角度（飞机的当前俯仰,范围为-90°到90°,朝向高处为正,低处为负）
						Roll 			# 敌方飞机的当前滚转角度（飞机的当前滚转,范围为-180°到180° ）
						Speed 			# 敌方飞机的当前速度（飞机的当前速度）
						CurTime 		# 当前时间（当前时间）
						IsLocked 		# 敌方飞机是否被敌方导弹锁定（飞机是否被己方导弹锁定）

				obs_side['missileinfos']为空间中所有未爆炸的双方导弹信息，字典列表，其中包含的字典信息如下（以下为Key值）

						Name 			# 导弹的名称
						Identification 	# 导弹的标识符（表示导弹是红方还是蓝方）
						ID 				# 导弹的ID（表示导弹的唯一编号）
						Type 			# 导弹的类型（表示导弹的类型，其中导弹类型为 3）
						Availability 	# 导弹的可用性（表示导弹的可用性，范围为0到1,为1表示飞机存活，0表示导弹已爆炸）
						X 				# 导弹的当前X坐标（表示导弹的X坐标）
						Y 				# 导弹的当前Y坐标（表示导弹的Y坐标）
						Lon 			# 导弹的当前所在经度（表示导弹的所在经度）
						Lat 			# 导弹的当前所在纬度（表示导弹的所在纬度）
						Alt 			# 导弹的当前所在高度（表示导弹的所在高度）
						Heading 		# 导弹的当前朝向角度（导弹的当前朝向,范围为-180°到180° 朝向0°表示正北,逆时针方向旋转为正数0°到180°，顺时针方向为正数0°到-180°）
						Pitch 			# 导弹的当前俯仰角度（导弹的当前俯仰,范围为-90°到90°,朝向高处为正,低处为负）
						Roll 			# 导弹的当前滚转角度（导弹的当前滚转,范围为-180°到180° ）
						Speed 			# 导弹的当前速度（导弹的当前速度）
						CurTime 		# 当前时间（当前时间）
						LauncherID 		# 导弹的发射者ID（敌方导弹的发射者ID）
						EngageTargetID 	# 导弹攻击目标的ID（敌方导弹攻击目标的ID）

        :param kwargs:保留的变量
        :return: 决策完毕的任务指令列表
        """
        cmd_list = []   # 此变量为 保存所有的决策完毕任务指令列表

        self.process_decision(sim_time, obs_side, cmd_list) # 调用决策函数进行决策判断

        return cmd_list # 返回决策完毕的任务指令列表

    def process_decision(self, sim_time, obs_side, cmd_list):
        """处理决策
        :param sim_time: 当前想定已经运行时间
        :param obs_side: 当前方所有的Observation信息，包含了所有的当前方所需信息以及探测到的敌方信息
        :param cmd_list保存所有的决策完毕任务指令列表
				可用指令有六种
					1.初始化实体指令 （初始化实体的信息，注意该指令只能在开始的前3秒有效）
						make_entityinitinfo(receiver: int,x: float,y: float,z: float,init_speed: float,init_heading: float)
						参数含义为
							:param receiver:飞机的唯一编号，即上文中飞机的ID
							:param x: 初始位置为战场x坐标
							:param y: 初始位置为战场y坐标
							:param z: 初始位置为战场z坐标
							:param init_speed: 初始速度(单位：米/秒，有人机取值范围：[150,400]，无人机取值范围：[100,300])
							:param init_heading: 初始朝向(单位：度，取值范围[0,360]，与正北方向的夹角)
					2.航线巡逻控制指令（令飞机沿航线机动）
						make_linepatrolparam(receiver: int,coord_list: List[dict],cmd_speed: float,cmd_accmag: float,cmd_g: float)
						参数含义为
							:param receiver: 飞机的唯一编号，即飞机的ID
							:param coord_list: 路径点坐标列表 -> [{"x": 500, "y": 400, "z": 2000}, {"x": 600, "y": 500, "z": 3000}]
											   区域x，y不得超过作战区域,有人机高度限制[2000,15000]，无人机高度限制[2000,10000]
							:param cmd_speed: 指令速度
							:param cmd_accmag: 指令加速度
							:param cmd_g: 指令过载
					3.区域巡逻控制指令	（令飞机沿区域巡逻）
						make_areapatrolparam(receiver: int,x: float,y: float,z: float,area_length: float,area_width: float,cmd_speed: float,cmd_accmag: float,cmd_g: float)
						    :param receiver: 飞机的唯一编号，即飞机的ID
							:param x: 区域中心坐标x坐标
							:param y: 区域中心坐标y坐标
							:param z: 区域中心坐标z坐标
							:param area_length: 区域长
							:param area_width: 区域宽
							:param cmd_speed: 指令速度
							:param cmd_accmag: 指令加速度
							:param cmd_g: 指令过载
					4.机动参数调整控制指令（调整飞机的速度、加速度和过载）
						make_motioncmdparam(receiver: int, update_motiontype: int,cmd_speed: float,cmd_accmag: float,cmd_g: float)
						    :param receiver: 飞机的唯一编号，即飞机的ID
							:param update_motiontype: 调整机动参数,其中 1为设置指令速度，2为设置指令加速度，3为设置指令速度和指令加速度
							:param cmd_speed: 指令速度
							:param cmd_accmag: 指令加速度
							:param cmd_g: 指令过载
					5.跟随目标指令 （令飞机跟随其他飞机）
						make_followparam(receiver: int,tgt_id: int,cmd_speed: float,cmd_accmag: float,cmd_g: float)
							:param receiver:飞机的唯一编号，即飞机的ID
							:param tgt_id: 目标ID,友方敌方均可
							:param cmd_speed: 指令速度
							:param cmd_accmag: 指令加速度
							:param cmd_g: 指令过载
					6.打击目标指令（令飞机使用导弹攻击其他飞机）
						make_attackparam(receiver: int,tgt_id: int,fire_range: float )
							:param receiver:飞机的唯一编号，即飞机的ID
							:param tgt_id: 目标ID
							:param fire_range: 开火范围，最大探测范围的百分比，取值范围[0, 1]

        """
        self.process_observation(obs_side)  # 获取态势信息,对态势信息进行处理

        if sim_time == 1:  # 当作战时间为1s时,初始化实体位置,注意,初始化位置的指令只能在前三秒内才会被执行
            self.init_pos(cmd_list) # 将实体放置到合适的初始化位置上

        if sim_time == 2 : # 当作战时间为2s时,开始进行任务开始,并保存任务指令;
            self.mission_start(sim_time, cmd_list)# 调用任务判断以及处理，进入决策阶段
        
        if sim_time > 300:  # 当作战时间大于10s时,开始进行任务控制,并保存任务指令;
            for plane_info in self.my_allplane_infos:  # 遍历所有的己方飞机
                if self.get_available_action(plane_info)[10] and (plane_info['Type']==1 or plane_info['ID'] not in self.my_attacking_enemy):
                    self.get_action(10, plane_info, cmd_list)
                elif self.get_available_action(plane_info)[11]:
                    self.get_action(11, plane_info, cmd_list)
                elif self.get_available_action(plane_info)[8]:
                    self.get_action(8, plane_info, cmd_list)
                elif self.get_available_action(plane_info)[9]:
                    self.get_action(9,plane_info,cmd_list)
                else:
                    self.get_action(7, plane_info, cmd_list)

    def process_observation(self, obs_side):
        """
        初始化飞机态势信息
        :param obs_red: 当前方所有的态势信息信息，包含了所有的当前方所需信息以及探测到的敌方信息
        """
        my_entity_infos = obs_side['platforminfos'] # 拿到己方阵营有人机、无人机在内的所有飞机信息
        if len(my_entity_infos) < 1:
            return
        my_manned_info = [] # 用以保存当前己方有人机信息
        my_uvas_infos = []  # 用以保存当前己方无人机信息
        my_allplane_infos = []    # 用以保存当前己方所有飞机信息
        for uvas_info in my_entity_infos:
            if uvas_info['ID'] != 0 and uvas_info['Availability'] > 0.0001: # 判断飞机是否可用 飞机的ID即为飞机的唯一编号 飞机的Availability为飞机当前生命值
                uvas_info["Z"] = uvas_info["Alt"]     # 飞机的 Alt 即为飞机的当前高度
                if uvas_info['Type'] == 1:           # 所有类型为 1 的飞机是 有人机
                    my_manned_info.append(uvas_info) # 将有人机保存下来 一般情况，每方有人机只有1架
                if uvas_info['Type'] == 2:           # 所有类型为 2 的飞机是 无人机
                    my_uvas_infos.append(uvas_info)  # 将无人机保存下来 一般情况，每方无人机只有4架
                my_allplane_infos.append(uvas_info)        # 将己方所有飞机信息保存下来 一般情况，每方飞机实体总共5架

        if len(my_manned_info) < 1:       #  判断己方有人机是否被摧毁
            return

        enemy_entity_infos = obs_side['trackinfos']   # 拿到敌方阵营的飞机信息,包括敌方有人机、无人机在内的所有飞机信息
        enemy_manned_info = []  # 用以保存当前敌方有人机信息
        enemy_uvas_infos = []   # 用以保存当前敌方无人机信息
        enemy_allplane_infos = []     # 用以保存当前敌方所有飞机信息
        for uvas_info in enemy_entity_infos:
            if uvas_info['ID'] != 0 and uvas_info['Availability'] > 0.0001:  # 判断飞机是否可用 飞机的ID即为飞机的唯一编号 飞机的Availability为飞机当前生命值
                uvas_info['Z'] = uvas_info['Alt']         # 飞机的 Alt 即为飞机的当前高度
                if uvas_info['Type'] == 1:               # 所有类型为 1 的飞机是 有人机
                    enemy_manned_info.append(uvas_info)  # 将有人机保存下来 一般情况，每方有人机只有1架
                if uvas_info['Type'] == 2:               # 所有类型为 2 的飞机是 无人机
                    enemy_uvas_infos.append(uvas_info)   # 将无人机保存下来 一般情况，每方无人机只有4架
                enemy_allplane_infos.append(uvas_info)         # 将己方所有飞机信息保存下来 一般情况，每方飞机实体总共5架

        my_allplane_maps = {}
        for input_entity in my_allplane_infos:
            my_allplane_maps[int(input_entity['ID'])] = input_entity

        missile_infos = obs_side['missileinfos']  # 拿到空间中已发射且尚未爆炸的导弹信息
        enemy_missile_infos = [] #  用以保存敌方已发射且尚未爆炸的导弹信息
        my_missile_infos = []
        for missile_info in missile_infos:
            if missile_info['LauncherID'] in my_allplane_maps:#  判断导弹是否为己方导弹 导弹的LauncherID即为导弹的发射者
                my_missile_infos.append(missile_info)
                continue
            if (missile_info['ID'] != 0 and missile_info['Availability'] > 0.0001): # 判断导弹是否可用 导弹的ID即为导弹的唯一编号 导弹的Availability为导弹当前生命值
                missile_info["Z"] = missile_info["Alt"]     # 导弹的 Alt 即为导弹的当前高度
                enemy_missile_infos.append(missile_info)    # 保存敌方已发射且尚未爆炸的导弹信息

        self.my_uvas_infos = my_uvas_infos              # 保存当前己方无人机信息
        self.my_manned_info = my_manned_info            # 保存当前己方有人机信息
        self.my_allplane_infos = my_allplane_infos            # 保存当前己方所有飞机信息
        self.enemy_uvas_infos = enemy_uvas_infos        # 保存当前敌方无人机信息
        self.enemy_manned_info = enemy_manned_info      # 保存当前敌方有人机信息
        self.enemy_allplane_infos = enemy_allplane_infos      # 保存当前敌方所有飞机信息
        self.enemy_missile_infos = enemy_missile_infos  # 保存敌方已发射且尚未爆炸的导弹信息
        self.my_missile_infos = my_missile_infos # 保存我方已发射且尚未爆炸的导弹信息
        self.get_in_attack_range()
        for my_id, enemy_id in list(self.my_attacking_enemy.items()):
            if not self.enemy_is_alive(enemy_id):
                self.my_attacking_enemy.pop(my_id)
                
    def cal_heading(self,direction):
        # 计算heading,注意这里的范围是[-pi,pi] heading是与y周的angle
        if direction["X"] == 0 and direction["Y"] == 0 and direction["Z"] == 0:
            return 0
        else:
            heading = np.arctan2(direction["X"], direction["Y"])
            return heading

    def get_in_attack_range(self):
        # 获取所有在攻击范围内的飞机
        self.in_range_enemy.clear()
        for my_plane_info in self.my_allplane_infos:
            data = self.get_move_data(my_plane_info)  # 根据拿到飞机获取对应的机动数据
            for enemy_plane in self.enemy_allplane_infos:
                # 计算两个飞机之前的方位角与偏航角差
                direction = {'X':enemy_plane['X'] - my_plane_info['X'],'Y':enemy_plane['Y'] - my_plane_info['Y'],'Z':enemy_plane['Z'] - my_plane_info['Z']}
                v_h = self.cal_heading(direction)
                v_p = TSVector3.calpitch(direction)
                d_h = my_plane_info['Heading']-v_h
                d_p = my_plane_info['Pitch']-v_p
                distance = TSVector3.distance(my_plane_info,enemy_plane)# 计算空间两点距离
                if distance < data['launch_range'] and np.abs(d_h) < data['max_heading_range'] and np.abs(d_p) < data['max_pitch_range']:
                    self.in_range_enemy[my_plane_info['ID']].append(enemy_plane)

    def my_is_alive(self,my_id):
        my_ids = [my_plane['ID'] for myplane in self.my_allplane_infos]
        return my_id in my_ids

    def enemy_is_alive(self,enemy_id):
        my_enemy_ids = [enemy['ID'] for enemy in self.enemy_allplane_infos]
        return enemy_id in my_enemy_ids
    
    def init_pos(self, cmd_list):
        """
        初始化飞机部署位置
        :param cmd_list:所有的决策完毕任务指令列表
        """
        leader_original_pos = {}    # 用以初始化当前方的位置
        if self.name == "red":
            leader_original_pos = {"X": -130000, "Y": -135000, "Z": 9500}
        else :
            leader_original_pos = {"X": 130000, "Y": 135000, "Z": 9500}

        interval_distance = 5000   # 间隔 5000米排列
        for leader in self.my_manned_info: # 为己方有人机设置初始位置
            # CmdEnv.make_entityinitinfo 指令，可以将 飞机实体置于指定位置点，参数依次为 实体ID，X坐标，Y坐标，Z坐标，初始速度，初始朝向
            cmd_list.append(CmdEnv.make_entityinitinfo(leader['ID'], leader_original_pos['X'], leader_original_pos['Y'], leader_original_pos['Z'],400 * 0.6, 45))

        #己方无人机在有人机的y轴上分别以9500的间距进行部署
        sub_index = 0  # 编号 用以在有人机左右位置一次排序位置点
        for sub in self.my_uvas_infos: # 为己方每个无人机设置初始位置
            sub_pos = copy.deepcopy(leader_original_pos)  # 深拷贝有人机的位置点
            if sub_index & 1 == 0: # 将当前编号放在有人机的一侧
                # CmdEnv.make_entityinitinfo 指令，可以将 飞机实体置于指定位置点，参数依次为 实体ID，X坐标，Y坐标，Z坐标，初始速度，初始朝向
                cmd_list.append(CmdEnv.make_entityinitinfo(sub['ID'], sub_pos['X'], sub_pos['Y'] + interval_distance, sub_pos['Z'],300 * 0.6, 45))
            else:                   # 将当前编号放在有人机的另一侧
                # CmdEnv.make_entityinitinfo 指令，可以将 飞机实体置于指定位置点，参数依次为 实体ID，X坐标，Y坐标，Z坐标，初始速度，初始朝向
                cmd_list.append(CmdEnv.make_entityinitinfo(sub['ID'], sub_pos['X'], sub_pos['Y'] - interval_distance, sub_pos['Z'],300 * 0.6, 45))
                interval_distance *= 2 # 编号翻倍
            sub_index += 1 # 编号自增

    def mission_start(self, sim_time, cmd_list):
        """
         任务开始，根据敌方信息朝敌方有人机前进
         :param sim_time: 当前想定已经运行时间
         :param cmd_list保存所有的决策完毕任务指令列表
         """
        for plane_info in self.my_uvas_infos:  # 遍历所有的己方无人飞机
            self.get_action(7,plane_info,cmd_list)
            
    def mission_test(self, sim_time, cmd_list):
        """
         任务开始，根据敌方信息朝敌方有人机前进
         :param sim_time: 当前想定已经运行时间
         :param cmd_list保存所有的决策完毕任务指令列表
         """
        for plane_info in self.my_allplane_infos:  # 遍历所有的己方无人飞机
            if plane_info['Type'] == 1:
                data = self.get_move_data(plane_info)  # 根据拿到飞机获取对应的机动数据
                #
                # curr_pos = [{"X":plane_info['X'] + rel_pos[0], "Y": plane_info['Y'] + rel_pos[1],
                #                            "Z": plane_info['Alt'] + rel_pos[2]}, ]
                # if int(sim_time/10) %2 == 0:
                #     cmd_list= self.get_action(6,plane_info,cmd_list)
                # else:
                #     cmd_list = self.get_action(5,plane_info,cmd_list)
                if int(sim_time/10) %2 == 0:
                    cmd_list= self.get_action(3,plane_info,cmd_list)
                # elif int(sim_time/10) %2 == 1:
                #     cmd_list = self.get_action(6,plane_info,cmd_list)
                # else:
                #     cmd_list = self.get_action(2, plane_info, cmd_list)

                plan_obs = self.get_obs_agent(plane_info['ID'])
                o_s = self.get_obs_size()
                state = self.get_state()

    def get_plane_by_id(self, planeID) -> {}:
        """
        根据飞机ID获取己方飞机态势信息
        :param planeID:飞机ID
        :return:己方飞机态势信息
        """
        for plane in self.my_allplane_infos:
            if plane['ID']== planeID:
                return plane
        return None # 己方飞机态势信息

    def get_move_data(self, plane) -> {}:
        """
        获取己方飞机对应的机动数据
        :param plane:己方飞机态势信息
        :return:飞机获取对应的机动数据
        """
        data = {} # 保存己方机动数据
        if plane['Type'] == 1:  # 所有类型为 1 的飞机是 有人机
            data['move_min_speed'] = 150     # 当前类型飞机的最小速度
            data['move_max_speed'] = 400     # 当前类型飞机的最大速度
            data['move_max_acc'] = 1         # 当前类型飞机的最大加速度
            data['move_max_g'] = 6           # 当前类型飞机的最大超载
            data['area_max_alt'] = 15000     # 当前类型飞机的最大高度
            data['area_min_alt'] = 2000  # 当前类型飞机的最小高度
            data['attack_range'] = 1         # 当前类型飞机的最大导弹射程百分比
            data['launch_range'] = 80000     # 当前类型飞机的最大雷达探测范围
            data['max_weapon_num'] = 4      # 当前飞机的导弹树木
            data['max_heading_range'] = np.pi/3
            data['max_pitch_range'] = np.pi / 3
        elif plane['Type'] == 2:                # 所有类型为 2 的飞机是 无人机
            data['move_min_speed'] = 100     # 当前类型飞机的最小速度
            data['move_max_speed'] = 300     # 当前类型飞机的最大速度
            data['move_max_acc'] = 2         # 当前类型飞机的最大加速度
            data['move_max_g'] = 12          # 当前类型飞机的最大超载
            data['area_max_alt'] = 10000     # 当前类型飞机的最大高度
            data['area_min_alt'] = 2000  # 当前类型飞机的最小高度
            data['attack_range'] = 1         # 当前类型飞机的最大导弹射程百分比
            data['launch_range'] = 60000     # 当前类型飞机的最大雷达探测范围
            data['max_weapon_num'] = 2  # 当前飞机的导弹树木
            data['max_heading_range'] = np.pi / 6
            data['max_pitch_range'] = np.pi / 18
        else:                                # 所有类型为 3 的飞机是 导弹
            data['move_min_speed'] = 400
            data['move_max_speed'] = 1000     # 当前类型飞机的最大速度
            data['move_max_acc'] = 10         # 当前类型飞机的最大加速度
            data['move_max_g'] = 20         # 当前类型飞机的最大超载
            data['area_max_alt'] = 30000     # 当前类型飞机的最大高度
            data['area_min_alt'] = 2000  # 当前类型飞机的最小高度
            data['attack_range'] = 100000         # 雷达的最大飞行距离
        return data # 保存己方机动数据

    def get_unit_by_id(self, agent_id):
        # 获取当前agent_id 的飞机的所有信息，返回的是一个dict里面包含了所有的agent的信息
        for agent_info in self.my_allplane_infos:
            if agent_info['ID'] == agent_id:
                return agent_info
        return None
    
    def get_enemy_by_id(self, agent_id):
        # 获取当前agent_id 的飞机的所有信息，返回的是一个dict里面包含了所有的agent的信息
        for agent_info in self.enemy_allplane_infos:
            if agent_info['ID'] == agent_id:
                return agent_info
        return None
    
    def get_own_missile_by_id(self, agent_id):
        # 获取当前agent_id 的飞机发射的导弹的所有信息
        my_missiles = []
        for missile_info in self.my_missile_infos:
            if missile_info['LauncherID'] == agent_id:
                my_missiles.append(missile_info)
        return my_missiles
    
    def get_enemy_missile_by_id(self, agent_id):
        # 获取attack当前agent_id 的飞机的导弹的所有信息
        enemy_missiles = []
        for missile_info in self.enemy_missile_infos:
            if missile_info['EngageTargetID'] == agent_id:
                enemy_missiles.append(missile_info)
        return enemy_missiles
    
    def cal_distance(self,lista,listb):
        return np.linalg.norm(np.array(lista) - np.array(listb))
    
    def cal_rel_state(self, unit_agent, unit_enemy):
        # 计算相对态势,返回[qr,qb,d,belta,rel_h,rel_speed,speed,h]
        chi_r =  np.pi/2 - unit_agent['Heading']
        tau_r = unit_agent['Pitch']
        x_r = unit_agent['X']
        y_r = unit_agent['Y']
        z_r = unit_agent['Alt']
        speed_r = unit_agent['Speed']

        chi_b = np.pi/2 - unit_enemy['Heading']
        tau_b = unit_enemy['Pitch']
        x_b = unit_enemy['X']
        y_b = unit_enemy['Y']
        z_b = unit_enemy['Alt']
        speed_b = unit_enemy['Speed']
        
        dis = self.cal_distance([x_r,y_r,z_r],[x_b,y_b,z_b])
        qr = np.arccos(((x_b - x_r) * np.cos(chi_r) * np.cos(tau_r) + (y_b-y_r) * np.sin(chi_r) * np.cos(tau_r) + (z_b-z_r) * np.sin(tau_r))/dis)
        qb = np.arccos(((x_r - x_b) * np.cos(chi_b) * np.cos(tau_b) + (y_r - y_b) * np.sin(chi_b) * np.cos(tau_b) + (z_r - z_b) * np.sin(tau_b)) / dis)
        belta = np.arccos(np.cos(chi_r)*np.cos(tau_r)*np.cos(chi_b)*np.cos(tau_b) + np.cos(tau_r)*np.sin(tau_r)*np.cos(tau_b)*np.sin(tau_b) + np.sin(tau_r) * np.sin(tau_b))
        rel_h = z_r - z_b
        rel_speed = speed_r - speed_b
        return [qr,qb,dis,belta,rel_h,rel_speed,speed_r,z_r]

    def cal_rel_state_by_id(self, my_id, enemy_id):
        unit_agent = self.get_unit_by_id(my_id)
        unit_enemy = self.get_enemy_by_id(enemy_id)
        rel_state = self.cal_rel_state(unit_agent, unit_enemy)
        return rel_state

    def get_obs_agent(self, agent_id):
        # obs 里包括了 1. enemy特征 2. ally的相对特征 3. 自己发射的missile 4. attack 自己的missile 5. 自己的特征
        unit = self.get_unit_by_id(agent_id)

        enemy_feats_dim = [self.n_enemies, len(self.enemy_features)]
        ally_feats_dim = [self.n_agents-1, len(self.ally_features)]
        own_feats_dim = [1,len(self.own_features)]
        enemy_missile_dim = [obs_enemy_missile_nums, len(self.enemy_missile_features)]
        own_missile_dim = [obs_own_missile_nums, len(self.own_missile_features)]
        type_feats_dim = [1, len(self.type_features)]

        enemy_feats = np.zeros(enemy_feats_dim, dtype=np.float32)
        ally_feats = np.zeros(ally_feats_dim, dtype=np.float32)
        own_feats = np.zeros(own_feats_dim, dtype=np.float32)
        enemy_missile_feats = np.zeros(enemy_missile_dim, dtype=np.float32)
        own_missile_feats = np.zeros(own_missile_dim, dtype=np.float32)
        type_feats = np.zeros(type_feats_dim, dtype=np.float32)
        own_move_data = self.get_move_data(unit)
        if unit and unit['Availability'] > 0.01 : # 当前飞机还活着
            for i in range(self.n_enemies):
                enemy_id = self.enemy_numids[i]
                enemy_unit = self.get_enemy_by_id(enemy_id)
                if enemy_unit and enemy_unit['Availability'] > 0.01: # 这个飞机还存在
                    enemy_move_data = self.get_move_data(enemy_unit)
                    enemy_feats[i][0] = (unit['X'] - enemy_unit['X']) / self.map_len
                    enemy_feats[i][1] = (unit['Y'] - enemy_unit['Y']) / self.map_width
                    enemy_feats[i][2] = (unit['Alt'] - enemy_unit['Alt']) / (own_move_data['area_max_alt'] - enemy_move_data['area_min_alt'])
                    enemy_feats[i][3] = enemy_unit['Heading'] / np.pi
                    enemy_feats[i][4] = enemy_unit['Pitch'] / np.pi
                    enemy_feats[i][5] = (enemy_unit['Speed'] - enemy_move_data['move_min_speed']) /  (enemy_move_data['move_max_speed'] - enemy_move_data['move_min_speed'])
                    enemy_feats[i][6] = enemy_unit['IsLocked']
            inc = 0
            for i in range(self.n_agents):
                ally_id = self.my_numids[i]
                ally_unit = self.get_unit_by_id(ally_id)
                if ally_unit and ally_unit['Availability'] > 0.01 and ally_id != agent_id:
                    ally_move_data = self.get_move_data(ally_unit)
                    ally_feats[inc][0] = (unit['X'] - ally_unit['X']) / self.map_len
                    ally_feats[inc][1] = (unit['Y'] - ally_unit['Y']) / self.map_width
                    ally_feats[inc][2] = (unit['Alt'] - ally_unit['Alt']) / (own_move_data['area_max_alt'] - ally_move_data['area_min_alt'])
                    ally_feats[inc][3] = ally_unit['Heading'] / np.pi
                    ally_feats[inc][4] = ally_unit['Pitch'] / np.pi
                    ally_feats[inc][5] = (ally_unit['Speed'] - ally_move_data['move_min_speed']) /  (ally_move_data['move_max_speed'] - ally_move_data['move_min_speed'])
                    ally_feats[inc][6] = ally_unit['IsLocked']
                    inc += 1

            own_feats[0][0] = 2 * unit['X'] / self.map_len
            own_feats[0][1] = 2 * unit['Y'] / self.map_width
            own_feats[0][2] = (unit['Alt'] - own_move_data['area_min_alt']) / (own_move_data['area_max_alt'] - own_move_data['area_min_alt'])
            own_feats[0][3] = unit['Heading'] / np.pi
            own_feats[0][4] = unit['Pitch'] / np.pi
            own_feats[0][5] = unit['Roll'] / np.pi
            own_feats[0][6] = (unit['Speed'] - own_move_data['move_min_speed']) / (own_move_data['move_max_speed'] - own_move_data['move_min_speed'])
            own_feats[0][7] = unit['IsLocked']
            own_feats[0][8] = unit['LeftWeapon'] / own_move_data['max_weapon_num']

            own_missiles = self.get_own_missile_by_id(agent_id)
            for i in range(obs_own_missile_nums):
                if i<len(own_missiles) and own_missiles[i]['Availability'] > 0.001:
                    missile_unit = own_missiles[i]
                    missile_move_data = self.get_move_data(missile_unit)
                    own_missile_feats[i][0] = (unit['X'] - missile_unit['X']) / self.map_len
                    own_missile_feats[i][1] = (unit['Y'] - missile_unit['Y']) / self.map_width
                    own_missile_feats[i][2] = (unit['Alt'] - missile_unit['Alt']) / (own_move_data['area_max_alt'] - missile_move_data['area_min_alt'])
                    own_missile_feats[i][3] = missile_unit['Heading'] / np.pi
                    own_missile_feats[i][4] = missile_unit['Pitch'] / np.pi
                    own_missile_feats[i][5] = (missile_unit['Speed'] - missile_move_data['move_min_speed']) / (missile_move_data['move_max_speed'] - missile_move_data['move_min_speed'])

            enemy_missiles = self.get_enemy_missile_by_id(agent_id)
            for i in range(obs_enemy_missile_nums):
                if i < len(enemy_missiles) and enemy_missiles[i]['Availability'] > 0.001:
                    missile_unit = enemy_missiles[i]
                    missile_move_data = self.get_move_data(missile_unit)
                    enemy_missile_feats[i][0] = (unit['X'] - missile_unit['X']) / self.map_len
                    enemy_missile_feats[i][1] = (unit['Y'] - missile_unit['Y']) / self.map_width
                    enemy_missile_feats[i][2] = (unit['Alt'] - missile_unit['Alt']) / (own_move_data['area_max_alt'] - missile_move_data['area_min_alt'])
                    enemy_missile_feats[i][3] = missile_unit['Heading'] / np.pi
                    enemy_missile_feats[i][4] = missile_unit['Pitch'] / np.pi
                    enemy_missile_feats[i][5] = (missile_unit['Speed'] - missile_move_data['move_min_speed']) / (missile_move_data['move_max_speed'] - missile_move_data['move_min_speed'])

            if unit['Type'] == 1:
                type_feats[0][0] = 1
            else:
                type_feats[0][1] = 1

            return np.concatenate((enemy_feats.flatten(),
                                   ally_feats.flatten(),
                                   own_feats.flatten(),
                                   enemy_missile_feats.flatten(),
                                   own_missile_feats.flatten(),
                                   type_feats.flatten()))

    def get_obs_size(self):
        enemy_feats_dim = self.n_enemies * len(self.enemy_features)
        ally_feats_dim = (self.n_agents-1) * len(self.ally_features)
        own_feats_dim = 1 * len(self.own_features)
        enemy_missile_dim = obs_enemy_missile_nums *  len(self.enemy_missile_features)
        own_missile_dim = obs_own_missile_nums * len(self.own_missile_features)
        type_feats_dim = 1 * len(self.type_features)
        return (enemy_feats_dim + ally_feats_dim + own_feats_dim + enemy_missile_dim + own_missile_dim + type_feats_dim)

    def get_state(self):  # 全局的状态
        s_size = self.get_obs_size()
        state = np.zeros(s_size)
        if len(self.my_manned_info) > 0:
            planeinfo = self.my_manned_info[0]
            state = self.get_obs_agent(planeinfo['ID'])
        return state

    def get_rotation_matrix_gb(self,theta, psi, phi):
        matri = np.array([[np.cos(theta)*np.cos(psi), np.sin(phi)* np.sin(theta)*np.cos(psi) - np.cos(phi)*np.sin(psi), np.cos(phi)*np.sin(theta)*np.cos(psi)+np.sin(phi)*np.sin(psi)],
                          [np.cos(theta)*np.sin(psi), np.sin(phi)* np.sin(theta)*np.sin(psi) + np.cos(phi)*np.cos(psi), np.cos(phi)*np.sin(theta)*np.sin(psi)-np.sin(phi)*np.cos(psi)],
                          [-np.sin(theta), np.sin(phi)*np.cos(theta), np.cos(phi)*np.cos(theta)]])
        return matri

    def get_rotation_matrix_xy(self, theta):
        matri = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        return matri

    '''
    def get_action(self, action_num,theta, psi, phi):
        # psi = np.pi/2 - psi
        if action_num == 0: # 定常飞行，速度不变
            target_pos = np.array([400,0,0])
            Rgb =  self.get_rotation_matrix_gb(theta, psi, phi)
            return Rgb.dot(target_pos)
        elif action_num == 1: # 加速平飞
            target_pos = np.array([400,0,0])
            Rgb =  self.get_rotation_matrix_gb(theta, psi, phi)
            return Rgb.dot(target_pos)
        elif action_num == 2: # 减速平飞
            target_pos = np.array([400,0,0])
            Rgb =  self.get_rotation_matrix_gb(theta, psi, phi)
            return Rgb.dot(target_pos)
        elif action_num == 3: # zuo转弯
            target_pos = np.array([50,50,0])
            Rgb =  self.get_rotation_matrix_gb(theta, psi, phi)
            return Rgb.dot(target_pos)
        elif action_num == 4: # 右转弯
            target_pos = np.array([50,-50,0])
            Rgb =  self.get_rotation_matrix_gb(theta, psi, phi)
            return Rgb.dot(target_pos)
        elif action_num == 5: # 拉升向上
            target_pos = np.array([0,800,400])
            Rgb =  self.get_rotation_matrix_gb(theta, psi, phi)
            return Rgb.dot(target_pos)
        elif action_num == 6: # 俯冲乡下
            target_pos = np.array([0,800,-400])
            Rgb =  self.get_rotation_matrix_gb(theta, psi, phi)
            return Rgb.dot(target_pos)
        else:
            pass
    '''
    def select_nearest_target(self, own_plane,enemy_list):
        """
        选择离己方飞机最近的敌方飞机
        :param own_plane:己方飞机信息
        :param enemy_list:敌方飞机信息列表
        :return:最近的敌方飞机信息
        """
        min_distance = np.inf # 超过此距离的敌方飞机不在考虑之内
        # print(len(enemy_list))
        # assert len(enemy_list) == 0
        close_enemy = None
        for enemy in enemy_list:
            distance = TSVector3.distance(own_plane,enemy)# 计算空间两点距离
            if enemy['Type'] == 1 and distance < min_distance: # 除非是有人挤才允许同时被多个我方飞机在攻击
                min_distance = distance
                close_enemy = enemy
            elif own_plane['ID'] not in self.my_attacking_enemy and distance < min_distance: # 计算出一个离得最近的敌方实体
                if enemy['ID'] not in list(self.my_attacking_enemy.values()):
                    min_distance = distance
                    close_enemy = enemy
            else:
                pass
        return close_enemy # 返回最近的敌方飞机信息
    
    def get_action(self, action_num, plane_info, cmd_list):
        data = self.get_move_data(plane_info)  # 根据拿到飞机获取对应的机动数据
        x, y, z = [plane_info['X'],plane_info['Y'], plane_info['Alt']]
        
        if action_num == 0: # 定常飞行
            speed = data['move_max_speed'] * 0.7
            # CmdEnv.make_motioncmdparam 指令，可以将修改当前飞机的速度和超载，参数依次为 飞机ID，调整机动参数（1为调整速度，2调整加速度，3调整速度和加速度） ，速度，加速度，超载（与飞机转向角度有关）
            cmd_list.append(CmdEnv.make_motioncmdparam(plane_info['ID'], 1, speed, data['move_max_acc'], 0))
        
        elif action_num == 1: # 加速飞行
            speed =  data['move_max_speed']
            # CmdEnv.make_motioncmdparam 指令，可以将修改当前飞机的速度和超载，参数依次为 飞机ID，调整机动参数（1为调整速度，2调整加速度，3调整速度和加速度） ，速度，加速度，超载（与飞机转向角度有关）
            cmd_list.append(CmdEnv.make_motioncmdparam(plane_info['ID'], 1, speed, data['move_max_acc'], data['move_max_g']))
        
        elif action_num == 2: # 减速飞行
            speed = data['move_min_speed'] * 1.1
            # CmdEnv.make_motioncmdparam 指令，可以将修改当前飞机的速度和超载，参数依次为 飞机ID，调整机动参数（1为调整速度，2调整加速度，3调整速度和加速度） ，速度，加速度，超载（与飞机转向角度有关）
            cmd_list.append(CmdEnv.make_motioncmdparam(plane_info['ID'], 1, speed, data['move_max_acc'], data['move_max_g']))
        
        elif action_num == 3: # 右
            rgb = self.get_rotation_matrix_xy(-np.pi/4)
            target = rgb.dot(np.array([x,y]))
            [tar_x, tar_y, tar_z] = [target[0],target[1],z]
            curr_pos = [{"X": tar_x, "Y": tar_y, "Z": tar_z}, ]
            cmd_list.append(
                CmdEnv.make_linepatrolparam(plane_info['ID'], curr_pos, data['move_max_speed']*0.7,
                                            data['move_max_acc'], data["move_max_g"]))

        elif action_num == 4: # 左
            rgb = self.get_rotation_matrix_xy(np.pi/4)
            target = rgb.dot(np.array([x,y]))
            [tar_x, tar_y, tar_z] = [target[0],target[1],z]
            curr_pos = [{"X": tar_x, "Y": tar_y, "Z": tar_z}, ]
            cmd_list.append(
                CmdEnv.make_linepatrolparam(plane_info['ID'], curr_pos, data['move_max_speed']*0.7,
                                            data['move_max_acc'], data["move_max_g"]))

        elif action_num == 5: # 上 yz
            # rgb = self.get_rotation_matrix_xy(-np.pi/4)
            # target = rgb.dot(np.array([x,y]))
            # [tar_x, tar_y, tar_z] = [target[0],target[1],data['area_max_alt']*0.8]
            [tar_x, tar_y, tar_z] = [x,y,data['area_max_alt']*0.8]
            curr_pos = [{"X": tar_x, "Y": tar_y, "Z": tar_z}, ]
            cmd_list.append(
                CmdEnv.make_linepatrolparam(plane_info['ID'], curr_pos, data['move_max_speed']*0.7,
                                            data['move_max_acc'], data["move_max_g"]))

        elif action_num == 6: # 下xz
            # rgb = self.get_rotation_matrix_xy(np.pi/4)
            # target = rgb.dot(np.array([x,y]))
            # [tar_x, tar_y, tar_z] = [target[0], target[1], data['area_min_alt']*1.2]
            [tar_x, tar_y, tar_z] = [x, y, data['area_min_alt'] * 1.2]
            curr_pos = [{"X": tar_x, "Y": tar_y, "Z": tar_z}, ]
            cmd_list.append(
                CmdEnv.make_linepatrolparam(plane_info['ID'], curr_pos, data['move_max_speed']*0.7,
                                            data['move_max_acc'], data["move_max_g"]))

        elif action_num == 7: # 追踪敌方有人机
            if len(self.enemy_manned_info) > 0:
                enemy_leader = self.enemy_manned_info[0]  # 拿到敌方有人机信息
                speed = enemy_leader['Speed']
                heading = enemy_leader['Heading']
                pitch = enemy_leader['Pitch']
                offset_x = speed * np.cos(pitch) * np.sin(heading)
                offset_y = speed * np.cos(pitch) * np.cos(heading)
                offset_z = speed * np.sin(pitch)
                Alt = np.clip(enemy_leader['Z'] + offset_z, data['area_min_alt'], data['area_max_alt'])
                leader_fire_route_list = [{"X":enemy_leader['X'] + offset_x, "Y":enemy_leader['Y'] + offset_y, "Z":Alt}, ]#获取一个路径点,将敌方有人机的所在点设置为目标点
                # EnvCmd.make_linepatrolparam 指令，可以令飞机沿给定路线机动，参数依次为 飞机ID，路线，速度，加速度，超载（与飞机转向角度有关）
                cmd_list.append(CmdEnv.make_linepatrolparam(plane_info['ID'], leader_fire_route_list, data['move_max_speed'],data['move_max_acc'],data["move_max_g"]));
            else: # 如果敌有人机阵亡
                # EnvCmd.make_areapatrolparam 指令，可以令飞机绕给定点巡逻，参数依次为 飞机ID，给定点X坐标，给定点Y坐标，给定点Z坐标，给定区域的长度，给定区域的宽度，速度，加速度，超载（与飞机转向角度有关）
                cmd_list.append(
                    CmdEnv.make_areapatrolparam(plane_info['ID'], plane_info['X'], plane_info['Y'], data['area_max_alt'], 200, 100,
                                                data['move_max_speed'], data['move_max_acc'], data['move_max_g']))
        
        elif action_num == 8:  # 追踪当前飞机已经攻击的飞机
            if plane_info['ID'] in self.my_attacking_enemy:
                attack_enemy_id = self.my_attacking_enemy[plane_info['ID']]
                attack_info = self.get_enemy_by_id(attack_enemy_id)
                speed = attack_info['Speed']
                heading = attack_info['Heading']
                pitch = attack_info['Pitch']
                offset_x = speed * np.cos(pitch) * np.sin(heading)
                offset_y = speed * np.cos(pitch) * np.cos(heading)
                offset_z = speed * np.sin(pitch)
                Alt = np.clip(attack_info['Alt'] + offset_z, data['area_min_alt'], data['area_max_alt'])
                target_fire_route_list = [{"X":attack_info['X'] + offset_x, "Y":attack_info['Y'] + offset_y, "Z":Alt}, ]
                cmd_list.append(
                    CmdEnv.make_linepatrolparam(plane_info['ID'], target_fire_route_list, data['move_max_speed'],
                                                data['move_max_acc'], data["move_max_g"]))
        
        elif action_num == 9: # 贪心攻击,攻击距离我最近,in_range
            enemy_list = self.in_range_enemy[plane_info['ID']]
            close_enemy = self.select_nearest_target(plane_info, enemy_list)
            cmd_list.append(CmdEnv.make_attackparam(plane_info['ID'], close_enemy['ID'], 1))
            self.my_attacking_enemy[plane_info['ID']] = close_enemy['ID'] # save attacking
        
        elif action_num == 10: # 主动防御，转90，降高度
            for missle in self.get_enemy_missile_by_id(plane_info['ID']):
                launcher_id = missle['LauncherID']
                rel_state = self.cal_rel_state_by_id(launcher_id, plane_info['ID'])
                if rel_state[8] < data['max_heading_range'] and rel_state[9] < data['max_pitch_range'] and rel_state[2] < max_radarrange:
                    # 转90，降高度
                    
        elif action_num == 11: # 主动规避，除非对面是有人机，否则不相向对头
            for enemy in self.enemy_allplane_infos:
                if np.abs(enemy['Heading'] - plane_info['Heading']) > np.pi * (5/6):
                    rel_vec = BaseTSVector3.minus(enemy, plane_info)
                    # 判断与当前速度的相对方向
                    direction = BaseTSVector3.cross({"X": plane_info['Speed'] * np.cos(plane_info['Pitch']) * np.sin(plane_info['Heading']), \
                        "Y": plane_info['Speed'] * np.cos(plane_info['Pitch']) * np.cos(plane_info['Heading']), \
                            "Z": plane_info['Speed'] * np.sin(plane_info['Pitch'])}, rel_vec)
                    if BaseTSVector3.dot(direction, {"X": 0, "Y": 0, "Z": 1}) > 0:
                        self.get_action(3, plane_info, cmd_list)
                    else:
                        self.get_action(4, plane_info, cmd_list)
        
        elif action_num == 12:  # 最后一个动作表示的是只有已经死亡的飞机才可以执行的动作
            pass
        else:
            pass
        
        return cmd_list

    def get_available_action(self, plane_info):
        data = self.get_move_data(plane_info)  # 根据拿到飞机获取对应的机动数据
        available_action = [0] * self.n_actions
        x,y,z,heading =  [plane_info['X'], plane_info['Y'],plane_info['Alt'],plane_info['Heading']]
        corner_x = self.map_len/2
        corner_y = self.map_width / 2
        if x <  (-corner_x+x_y_threshold) and heading < 0:
            available_action[4] = 1
            available_action[3] = 1
        elif x >  (corner_x - x_y_threshold) and heading > 0:
            available_action[4] = 1
            available_action[3] = 1
        elif y < (-corner_y+x_y_threshold) and np.abs(heading) > np.pi/2:
            available_action[4] = 1
            available_action[3] = 1
        elif y > (corner_y-x_y_threshold) and np.abs(heading) < np.pi/2:
            available_action[4] = 1
            available_action[3] = 1
        else:
            available_action[0] = 1
            available_action[1] = 1
            available_action[2] = 1
            available_action[3] = 1
            available_action[4] = 1
        if z < data['area_max_alt'] - z_threshold:
            available_action[5] = 1
        if z > data['area_min_alt'] + z_threshold:
            available_action[6] = 1
        available_action[7] = 1
        ## 判断是否可以追逐对象
        if plane_info['ID'] in self.my_attacking_enemy:
            available_action[8] = 1
        # 判断有可以攻击的对象
        if plane_info['ID'] in self.in_range_enemy and plane_info['LeftWeapon'] > 0:
            enemy_list = self.in_range_enemy[plane_info['ID']]
            close_enemy = self.select_nearest_target(plane_info, enemy_list)
            if close_enemy:
                available_action[9] = 1
        # 判断是否被攻击
        if self.get_enemy_missile_by_id(plane_info['ID']):
            for missle in self.get_enemy_missile_by_id(plane_info['ID']):
                if self.cal_distance([missle['X'], missle['Y'], missle['Alt']], [x, y, z]) < 20:
                    break
                launcher_id = missle['LauncherID']
                rel_state = self.cal_rel_state_by_id(launcher_id, plane_info['ID'])
                if rel_state[9] < data['max_heading_range'] and rel_state[10] < data['max_pitch_range'] and rel_state[2] < max_radarrange:
                    available_action[10] = 1
        # 判断是否需要回避，避免火拼
        for enemy in self.enemy_allplane_infos:
            if np.abs(enemy['Heading'] - plane_info['Heading']) > np.pi * (5/6):
                if plane_info['Type'] == 1: # 我机为有人机，直接回避
                    available_action[11] = 1
                elif enemy['Type'] == 2:    # 敌我均为无人机，避免战损
                    available_action[11] = 1
        
        return available_action
    
    def get_available_action_by_id(self, id_num):
        plane_info = self.get_unit_by_id(self.my_numids[id_num])
        if plane_info:
            return self.get_available_action(plane_info)
        return None
    
    def get_cmd_from_action(self, a_id, action_num):
        # a_id是我方agent的id,[0,4], action表示执行的动作编号
        plane_id = self.my_numids[a_id]
        plane_info = self.get_unit_by_id(plane_id)
        assert plane_info == None
        cmd_list = []
        cmd_list = self.get_action(action_num,plane_info, cmd_list)
        return cmd_list
