import numpy as np
import torch
from torch.distributions import Categorical


# Agent no communication
class Agents:
    def __init__(self, args):
        self.n_actions = args.n_actions # 动作空间的大小
        self.n_agents = args.n_agents # 智能体的数量
        self.state_shape = args.state_shape # 全局状态的维度 todo
        self.obs_shape = args.obs_shape # 每个智能体的观测维度 todo
        # todo 其他未看过的算法后续继续观看
        if args.alg == 'vdn':
            from policy.vdn import VDN
            self.policy = VDN(args)
        elif args.alg == 'iql':
            from policy.iql import IQL
            self.policy = IQL(args)
        elif args.alg == 'qmix':
            from policy.qmix import QMIX
            self.policy = QMIX(args)
        elif args.alg == 'coma':
            from policy.coma import COMA
            self.policy = COMA(args)
        elif args.alg == 'qtran_alt':
            from policy.qtran_alt import QtranAlt
            self.policy = QtranAlt(args)
        elif args.alg == 'qtran_base':
            from policy.qtran_base import QtranBase
            self.policy = QtranBase(args)
        elif args.alg == 'maven':
            from policy.maven import MAVEN
            self.policy = MAVEN(args)
        elif args.alg == 'central_v':
            from policy.central_v import CentralV
            self.policy = CentralV(args)
        elif args.alg == 'reinforce':
            from policy.reinforce import Reinforce
            self.policy = Reinforce(args)
        else:
            raise Exception("No such algorithm")
        self.args = args

    def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon, maven_z=None):
        '''
        obs: 当前的agent的观察
        last_action: 上一次选择的动作
        agent_num: agent的编号
        avail_actions: 可以使用的动作列表
        epsilon： 当前的epsilon的值
        maven_z：todo 作用未明
        '''
        inputs = obs.copy()
        avail_actions_ind = np.nonzero(avail_actions)[0]  # index of actions which can be choose

        # transform agent_num to onehot vector
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1. # 这里是构建了一个one-hot的编码，代表是哪一个agent

        if self.args.last_action:
            # 这个参数代表是否将上一次执行的动作和当前的观察合并在一起
            inputs = np.hstack((inputs, last_action))
        if self.args.reuse_network:
            # 将agent_id融入到输入中，目前输入已经合并了观察-上一次的动作-agentid
            # 如果采用了循环神经网络大概率要这个，毕竟是有历史记忆
            inputs = np.hstack((inputs, agent_id))
        hidden_state = self.policy.eval_hidden[:, agent_num, :] # 提取对应隐藏层的状态，主要针对循环神经网络

        # transform the shape of inputs from (42,) to (1,42)
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
            hidden_state = hidden_state.cuda()

        # get q value
        if self.args.alg == 'maven':
            # todo 后面再看
            maven_z = torch.tensor(maven_z, dtype=torch.float32).unsqueeze(0)
            if self.args.cuda:
                maven_z = maven_z.cuda()
            q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state, maven_z)
        else:
            # 将输入和隐藏层的状态输入后到RNN中，然后获取动作Q值的分布以及最新的隐藏层状态，存储到eval_hidden中
            q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state)

        # choose action from q value
        if self.args.alg == 'coma' or self.args.alg == 'central_v' or self.args.alg == 'reinforce':
            # todo 后面再说
            action = self._choose_action_from_softmax(q_value.cpu(), avail_actions, epsilon)
        else:
            # avail_actions：这里可以看出可用的动作列表是一个tensor，有点像[1,1,1,0,0,1]这种形式，1的地方表示有动作、0表示没有动作
            # 将无意义的动作的Q值设置为最小
            q_value[avail_actions == 0.0] = - float("inf")
            # 随机采集动作或者采集最大Q值的动作
            # 从这里可以看出对于Q值的预测依旧还是让神经网络预测一个动作Q值的分布
            if np.random.uniform() < epsilon:
                action = np.random.choice(avail_actions_ind)  # action是一个整数
            else:
                action = torch.argmax(q_value)
        return action

    def _choose_action_from_softmax(self, inputs, avail_actions, epsilon):
        """
        :param inputs: # q_value of all actions
        """
        action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, avail_actions.shape[-1])  # num of avail_actions
        # 先将Actor网络的输出通过softmax转换成概率分布
        prob = torch.nn.functional.softmax(inputs, dim=-1)
        # add noise of epsilon
        prob = ((1 - epsilon) * prob + torch.ones_like(prob) * epsilon / action_num)
        prob[avail_actions == 0] = 0.0  # 不能执行的动作概率为0

        """
        不能执行的动作概率为0之后，prob中的概率和不为1，这里不需要进行正则化，因为torch.distributions.Categorical
        会将其进行正则化。要注意在训练的过程中没有用到Categorical，所以训练时取执行的动作对应的概率需要再正则化。
        """
        
        action = Categorical(prob).sample().long()
        return action

    def _get_max_episode_len(self, batch):
        # 这里的作用是找到当前训练的batch中最长的episode的长度，因为每个episode的长度不一样，所以要找到最长的episode的长度，然后在训练的时候只用到这个长度的数据，避免填充的数据对训练的影响
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        if max_episode_len == 0:  # 防止所有的episode都没有结束，导致terminated中没有1
            max_episode_len = self.args.episode_limit
        return max_episode_len

    def train(self, batch, train_step, epsilon=None):  # coma needs epsilon for training

        # different episode has different length, so we need to get max length of the batch
        max_episode_len = self._get_max_episode_len(batch) # 获取本次训练的batch中最长的episode的长度
        for key in batch.keys():
            if key != 'z':
                batch[key] = batch[key][:, :max_episode_len] # 只用到最长的episode的长度的数据，避免填充的数据对训练的影响，看来是因为有循环神经网络的关系
        self.policy.learn(batch, max_episode_len, train_step, epsilon)
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            self.policy.save_model(train_step) # 每隔save_cycle步保存一次模型


# Agent for communication
class CommAgents:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        alg = args.alg
        if alg.find('reinforce') > -1:
            from policy.reinforce import Reinforce
            self.policy = Reinforce(args)
        elif alg.find('coma') > -1:
            from policy.coma import COMA
            self.policy = COMA(args)
        elif alg.find('central_v') > -1:
            from policy.central_v import CentralV
            self.policy = CentralV(args)
        else:
            raise Exception("No such algorithm")
        self.args = args
        print('Init CommAgents')

    # 根据weights得到概率，然后再根据epsilon选动作
    def choose_action(self, weights, avail_actions, epsilon):
        weights = weights.unsqueeze(0)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, avail_actions.shape[-1])  # 可以选择的动作的个数
        # 先将Actor网络的输出通过softmax转换成概率分布
        prob = torch.nn.functional.softmax(weights, dim=-1)
        # 在训练的时候给概率分布添加噪音
        prob = ((1 - epsilon) * prob + torch.ones_like(prob) * epsilon / action_num)
        prob[avail_actions == 0] = 0.0  # 不能执行的动作概率为0

        """
        不能执行的动作概率为0之后，prob中的概率和不为1，这里不需要进行正则化，因为torch.distributions.Categorical
        会将其进行正则化。要注意在训练的过程中没有用到Categorical，所以训练时取执行的动作对应的概率需要再正则化。
        """

        action = Categorical(prob).sample().long()
        return action

    def get_action_weights(self, obs, last_action):
        obs = torch.tensor(obs, dtype=torch.float32)
        last_action = torch.tensor(last_action, dtype=torch.float32)
        inputs = list()
        inputs.append(obs)
        # 给obs添加上一个动作、agent编号
        if self.args.last_action:
            inputs.append(last_action)
        if self.args.reuse_network:
            inputs.append(torch.eye(self.args.n_agents))
        inputs = torch.cat([x for x in inputs], dim=1)
        if self.args.cuda:
            inputs = inputs.cuda()
            self.policy.eval_hidden = self.policy.eval_hidden.cuda()
        weights, self.policy.eval_hidden = self.policy.eval_rnn(inputs, self.policy.eval_hidden)
        weights = weights.reshape(self.args.n_agents, self.args.n_actions)
        return weights.cpu()

    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        if max_episode_len == 0:  # 防止所有的episode都没有结束，导致terminated中没有1
            max_episode_len = self.args.episode_limit
        return max_episode_len

    def train(self, batch, train_step, epsilon=None):  # coma在训练时也需要epsilon计算动作的执行概率
        # 每次学习时，各个episode的长度不一样，因此取其中最长的episode作为所有episode的长度
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            batch[key] = batch[key][:, :max_episode_len]
        self.policy.learn(batch, max_episode_len, train_step, epsilon)
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            self.policy.save_model(train_step)










