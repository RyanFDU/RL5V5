from multiagentenv import MultiAgentEnv
from env.xsim_env import XSimEnv
from config import IMAGE
import math
import numpy as np


uav_cost = 500
manned_cost = 2500
class AirbatEnv(XSimEnv):
    def __init__(self, agents, address, mode: str = 'host'):
        """
        对战环境初始化
        @param agents: 智能体列表
        @author:wubinxing
        @create data:2021/05/22 15.00
        @change date:
        """
        print("初始化 EnvRunner")
        XSimEnv.__init__(self, 80, address, IMAGE, mode)
        self.agents = {}
        self.__init_agents(agents)

        self.heuristic_ai = False
    def __init_agents(self, agents):
        """
        根据配置信息构建红蓝双方智能体
        @param agents: 智能体列表
        @return:
        @author:wubinxing
        @create data:2021/05/22 15.00
        @change date:
        """
        red_cls = agents['red']
        blue_cls = agents['blue']
        red_agent = red_cls('red', {"side": 'red'})
        self.agents["red"] = red_agent
        blue_agent = blue_cls('blue', {"side": 'blue'})
        self.agents["blue"] = blue_agent

        self.my_side = 'red'
        self.enemy_agent_identification = 'blue'
        self.my_agent = self.agents[self.my_side]
        self.pre_my_uav_nums = self.my_agent.n_agents-1 # 前一时刻 我方无人机的数量
        self.pre_enemy_uav_nums = self.my_agent.n_agents-1 # 前一时刻 敌方无人机的数量
        self.pre_dis = {} # distance to enemy_manned
        for i in range(self.my_agent.n_agents):
            id = self.my_agent.my_numids[i]
            self.pre_dis[id] == 400000

        self._episode_steps = 0
        self.obs = None
        self.red_score = 0
        self.blue_score = 0

    def _end(self):

        self.end()

    def _reset(self):
        """
        智能体、环境重置
        @return:
        @author:wubinxing
        @create data:2021/05/22 15.00
        @change date:
        """

        # 智能体重置
        for side, agent in self.agents.items():
            agent.reset()

        # 环境重置
        self.reset()
        obs = self.step([])
        # while obs["sim_time"] > 10:
        #     obs = self.step([])
        return obs

    def get_enemy_action(self, obs):
        """
        从对手智能体中获取动作
        @param obs: 状态信息
        @return: 动作指令
        @author:wubinxing
        @create data:2021/05/22 15.00
        @change date:
        """
        actions = []
        cur_time = obs["sim_time"]
        for side, agent in self.agents.items():
            if side != self.my_side:
                cmd_list = self._agent_step(agent, cur_time, obs[side])
                # print(cmd_list)
                actions.extend(cmd_list)

        return actions

    def _agent_step(self, agent, cur_time, obs_side):
        """
        获取一方智能体动作指令
        @param agent:
        @param cur_time:
        @param obs_side:
        @return:
        @author:wubinxing
        @create data:2021/05/22 15.00
        @change date:
        """
        cmd_list = agent.step(cur_time, obs_side)
        return cmd_list

    def _step(self, actions):
        """Returns reward, terminated, info."""
        # 这里的actions指的是红方的action
        terminated = False
        bad_transition = False
        infos = [{} for i in range(self.my_agent.n_agents)]
        dones = np.zeros((self.my_agent.n_agents), dtype=bool)

        actions_int = [int(a) for a in actions]
        self.last_action = np.eye(self.n_actions)[np.array(actions_int)]

        # Collect individual actions
        sc_actions = []
        for a_id, action in enumerate(actions_int):
            if action != -1: # action是-1表示这个agent deaded 不需要采取任何动作，
                sc_action = self.get_agent_action(a_id, action)
            if sc_action:
                sc_actions = sc_actions + sc_action
        sc_actions = sc_actions + self.get_enemy_action(self.obs)
        self.obs = self.step(sc_actions)
        self._episode_steps += 1
        # Update units
        self.update_agent_observation(self.obs)
        done = self.get_done(self.obs)
        terminated = done[0]
        reward = self.get_sparse_reward(done) + self.get_dense_reward()
        available_actions = []
        for i in range(self.my_agent.n_agents):
            avail_act = self.my_agent.get_available_action_by_id(i)
            if not avail_act:
                dones[i] = 1
                available_actions.append([0]*(self.my_agent.n_actions-1) + [1])
            else:
                available_actions.append(avail_act)






        raise NotImplementedError
    def update_agent_observation(self,obs):
        cur_time = obs["sim_time"]
        for side, agent in self.agents.items():
            agent.process_observation(obs[side])


    def get_agent_action(self,a_id, action):
        cmd_list = self.my_agent.get_cmd_from_action(a_id,action)
        return cmd_list

    def get_obs(self):
        """Returns all agent observations in a list."""
        raise NotImplementedError

    # def get_rel_reward(self,agentA):

    def get_done(self, obs):
            """
            推演是否结束
            @param obs: 环境状态信息
            @return: done列表信息
            """
            # print(obs)
            done = [0, 0, 0]  # 终止标识， 红方战胜利， 蓝方胜利

            # 时间超时，终止
            cur_time = obs["sim_time"]
            print("get_done cur_time:", cur_time)
            if cur_time >= 20 * 60 - 1:
                done[0] = 1
                # 当战损相同时，判断占领中心区域时间
                if len(obs["red"]["platforminfos"]) == len(obs["blue"]["platforminfos"]):
                    if self.red_score > self.blue_score:
                        print("红方占领中心区域时间更长")
                        done[1] = 1
                    elif self.red_score < self.blue_score:
                        print("蓝方占领中心区域时间更长")
                        done[2] = 1
                # 当战损不同时，判断战损更少一方胜
                else:
                    if len(obs["red"]["platforminfos"]) > len(obs["blue"]["platforminfos"]):
                        print("红方战损更少")
                        done[1] = 1
                    else:
                        print("蓝方战损更少")
                        done[2] = 1
                # 重置分数
                self.red_score = 0
                self.blue_score = 0
                return done

            # 红方有人机全部战损就终止
            red_obs_units = obs["red"]["platforminfos"]
            has_red_combat = False
            for red_obs_unit in red_obs_units:
                red_obs_unit_name = red_obs_unit["Name"]
                if red_obs_unit_name.split("_")[0] == "红有人机":
                    has_red_combat = True
                    # 判断红方有人机是否在中心区域
                    distance_to_center = math.sqrt(
                        red_obs_unit["X"] * red_obs_unit["X"]
                        + red_obs_unit["Y"] * red_obs_unit["Y"]
                        + (red_obs_unit["Alt"] - 9000) * (red_obs_unit["Alt"] - 9000))
                    # print("Red distance:", distance_to_center)
                    if distance_to_center <= 50000 and red_obs_unit["Alt"] >= 2000 and red_obs_unit['Alt'] <= 16000:
                        self.red_score = self.red_score + 1
                        print("Red Score:", self.red_score)
                    break
            if not has_red_combat:
                print("红方有人机阵亡")
                done[0] = 1
                done[2] = 1

            # 蓝方有人机全部战损就终止
            blue_obs_units = obs["blue"]["platforminfos"]
            has_blue_combat = False
            for blue_obs_unit in blue_obs_units:
                blue_obs_unit_name = blue_obs_unit["Name"]
                if blue_obs_unit_name.split("_")[0] == "蓝有人机":
                    has_blue_combat = True
                    # 判断蓝方有人机是否在中心区域
                    distance_to_center = math.sqrt(
                        blue_obs_unit["X"] * blue_obs_unit["X"]
                        + blue_obs_unit["Y"] * blue_obs_unit["Y"]
                        + (blue_obs_unit["Alt"] - 9000) * (blue_obs_unit["Alt"] - 9000))
                    # print("Blue distance:", distance_to_center)
                    if distance_to_center <= 50000 and blue_obs_unit["Alt"] >= 2000 and blue_obs_unit['Alt'] <= 16000:
                        self.blue_score = self.blue_score + 1
                        print("Blue Score:", self.blue_score)
                    break
            # print("get_done has_blue_combat:", has_blue_combat)
            if not has_blue_combat:
                print("蓝方有人机阵亡")
                done[0] = 1
                done[1] = 1

            if done[0] == 1:
                self.red_score = 0
                self.blue_score = 0
                return done

            # 红方没有导弹就终止
            has_red_missile = False
            for red_obs_unit in red_obs_units:
                if red_obs_unit["LeftWeapon"] > 0:
                    has_red_missile = True
                    break
            if not has_red_missile:
                if len(obs["red"]["missileinfos"]) == 0:
                    print("红方无弹")
                    done[0] = 1
                    done[2] = 1
                else:
                    flag = True
                    for red_missile in obs['red']['missileinfos']:
                        if red_missile["Identification"] == "红方":
                            flag = False
                            break
                    if flag:
                        print("红方无弹")
                        done[0] = 1
                        done[2] = 1

            # 蓝方没有导弹就终止
            has_blue_missile = False
            for blue_obs_unit in blue_obs_units:
                if blue_obs_unit["LeftWeapon"] > 0:
                    has_blue_missile = True
                    break
            if not has_blue_missile:
                if len(obs["blue"]["missileinfos"]) == 0:
                    print("蓝方无弹")
                    done[0] = 1
                    done[1] = 1
                else:
                    flag = True
                    for blue_missile in obs['blue']['missileinfos']:
                        if blue_missile["Identification"] == "蓝方":
                            flag = False
                            break
                    if flag:
                        print("蓝方无弹")
                        done[0] = 1
                        done[1] = 1

            if done[0] == 1:
                self.red_score = 0
                self.blue_score = 0
                return done

            if done[0] == 1:
                self.red_score = 0
                self.blue_score = 0
            return done
    def get_sparse_reward(self,done):
        reward = 0
        uav_lost = self.pre_my_uav_nums - len(self.my_agent.my_uvas_infos)
        if uav_lost > 0:
            reward -= uav_cost * uav_lost
            self.pre_my_uav_nums = len(self.my_agent.my_uvas_infos)
        enemy_uav_lost = self.pre_enemy_uav_nums - len(self.my_agent.enemy_uvas_infos)
        if enemy_uav_lost > 0:
            reward += uav_cost * enemy_uav_lost
            self.pre_enemy_uav_nums = len(self.my_agent.enemy_uvas_infos)
        if len(self.my_agent.my_manned_info ) == 0:
            reward -= manned_cost
        if len(self.my_agent.enemy_manned_info ) == 0:
            reward += manned_cost
        if done[0]==1 and done[1] == 1:
            if self.my_side == 'red': # 胜利的奖励
                reward += manned_cost
            else:
                reward -= manned_cost
        if done[0]==1 and done[2] == 1:
            if self.my_side == 'red': #
                reward -= manned_cost
            else:
                reward += manned_cost
        return reward
    def cal_distance(self,lista,listb):
        return np.linalg.norm(np.array(lista) - np.array(listb))
    def cal_rel_state(self,unit_agent, unit_enemy):
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
    def get_dense_reward(self,sim_time):
        sum_reward = 0
        enemy_manned_info = self.my_agent.enemy_manned_info[0]
        for plane_info in self.my_agent.my_allplane_infos:
            reward = 0
            [qr, qb, dis, belta, rel_h, rel_speed, speed_r, z_r] = self.cal_rel_state(plane_info,enemy_manned_info)
            if dis < 80000 and qr< np.pi/6 and qb > np.pi/6 and belta < np.pi/3:
                reward += 10
            elif dis < 80000  and qb < np.pi/6:
                reward -= 10
            else:
                pass
            if dis > 80000 and sim_time>2:
                reward += (dis - self.dis_pre[plane_info['ID']])/400
            self.dis_pre[plane_info['ID']] = dis
            sum_reward += reward
        return sum_reward

