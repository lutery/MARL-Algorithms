import torch
import torch.nn as nn
import torch.nn.functional as F


# counterfactual joint networks, 输入state、所有agent的hidden_state、其他agent的动作、自己的编号，输出自己所有动作对应的联合Q值
class QtranQAlt(nn.Module):
    def __init__(self, args):
        super(QtranQAlt, self).__init__()
        self.args = args

        # 对每个agent的action进行编码
        self.action_encoding = nn.Sequential(nn.Linear(self.args.n_actions, self.args.n_actions),
                                             nn.ReLU(),
                                             nn.Linear(self.args.n_actions, self.args.n_actions))

        # 对每个agent的hidden_state进行编码
        self.hidden_encoding = nn.Sequential(nn.Linear(self.args.rnn_hidden_dim, self.args.rnn_hidden_dim),
                                             nn.ReLU(),
                                             nn.Linear(self.args.rnn_hidden_dim, self.args.rnn_hidden_dim))

        # 编码求和之后输入state、所有agent的hidden_state之和、其他agent的action之和, state包括当前agent的编号
        q_input = self.args.state_shape + self.args.n_actions + self.args.rnn_hidden_dim + self.args.n_agents
        self.q = nn.Sequential(nn.Linear(q_input, self.args.qtran_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.args.qtran_hidden_dim, self.args.qtran_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.args.qtran_hidden_dim, self.args.n_actions))

    # 因为所有时刻所有agent的hidden_states在之前已经计算好了，所以联合Q值可以一次计算所有transition的，不需要一条一条计算。
    def forward(self, state, hidden_states, actions):  # (episode_num, max_episode_len, n_agents, n_actions)
        # state的shape为(episode_num, max_episode_len, n_agents, state_shape+n_agents)，包括了当前agent的编号
        episode_num, max_episode_len, n_agents, n_actions = actions.shape

        # 对每个agent的action进行编码
        action_encoding = self.action_encoding(actions.reshape(-1, n_actions))
        action_encoding = action_encoding.reshape(episode_num, max_episode_len, n_agents, n_actions)

        # 对每个agent的hidden_state进行编码
        hidden_encoding = self.hidden_encoding(hidden_states.reshape(-1, self.args.rnn_hidden_dim))
        hidden_encoding = hidden_encoding.reshape(episode_num, max_episode_len, n_agents, self.args.rnn_hidden_dim)

        # 所有agent的hidden_encoding相加
        hidden_encoding = hidden_encoding.sum(dim=-2)  # (episode_num, max_episode_len, rnn_hidden_dim)
        hidden_encoding = hidden_encoding.unsqueeze(-2).expand(-1, -1, n_agents, -1)  # (episode_num, max_episode_len, n_agents， rnn_hidden_dim)

        # 对于每个agent，其他agent的action_encoding相加
        # 先让最后一维包含所有agent的动作
        action_encoding = action_encoding.reshape(episode_num, max_episode_len, 1, n_agents * n_actions)
        action_encoding = action_encoding.repeat(1, 1, n_agents, 1)  # 此时每个agent都有了所有agent的动作
        # 把每个agent自己的动作置0
        action_mask = (1 - torch.eye(n_agents))  # th.eye（）生成一个二维对角矩阵
        action_mask = action_mask.view(-1, 1).repeat(1, n_actions).view(n_agents, -1)
        if self.args.cuda:
            action_mask = action_mask.cuda()
        action_encoding = action_encoding * action_mask.unsqueeze(0).unsqueeze(0)
        # 因为现在所有agent的动作都在最后一维，不能直接加。所以先扩展一维，相加后再去掉
        action_encoding = action_encoding.reshape(episode_num, max_episode_len, n_agents, n_agents, n_actions)
        action_encoding = action_encoding.sum(dim=-2)  # (episode_num, max_episode_len, n_agents， rnn_hidden_dim)

        inputs = torch.cat([state, hidden_encoding, action_encoding], dim=-1)
        q = self.q(inputs)
        return q


# Joint action-value network， 输入state,所有agent的hidden_state，所有agent的动作，输出对应的联合Q值
class QtranQBase(nn.Module):
    def __init__(self, args):
        super(QtranQBase, self).__init__()
        self.args = args
        # 这里接收的输入是RNN特征提取后的
        ae_input = self.args.rnn_hidden_dim + self.args.n_actions
        # 编码器：用于把每个智能体的hidden_state和动作进行编码，输出的维度也是ae_input，后续会将所有智能体的编码结果进行求和，所以输出维度要和输入维度一样
        self.hidden_action_encoding = nn.Sequential(nn.Linear(ae_input, ae_input),
                                             nn.ReLU(),
                                             nn.Linear(ae_input, ae_input))

        # ===== Q 网络：全局 state + 所有智能体的编码 → 标量 Q =====
        q_input = self.args.state_shape + self.args.n_actions + self.args.rnn_hidden_dim
        self.q = nn.Sequential(nn.Linear(q_input, self.args.qtran_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.args.qtran_hidden_dim, self.args.qtran_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.args.qtran_hidden_dim, 1))

    def forward(self, state, hidden_states, actions):  # (episode_num, max_episode_len, n_agents, n_actions)
        '''
        state: 全局的obs观察
        hidden_states: 循环神经网络每一步的状态 # 形状：(E, T, n_agents, 64 + 12) = (E, T, n_agents, 76)
        actions: 采集动作时的one-hot编码，这里为什么还要进一步输入动作，是因为之前的动作是单个智能体局与环境的交互动作，而这里的动作是所有智能体的动作
        所以输入再次输入动作在评估整体的价值情况时是必要的
        '''
        # todo 这里每一个输入的是啥？
        episode_num, max_episode_len, n_agents, _ = actions.shape
        hidden_actions = torch.cat([hidden_states, actions], dim=-1) # 将隐藏状态和动作合并，RNN的隐藏状态输入的是状态、动作、agent id
        hidden_actions = hidden_actions.reshape(-1, self.args.rnn_hidden_dim + self.args.n_actions)
        # 它把 (hidden, action) 的原始拼接空间映射到一个"更适合求和"的编码空间。
        # 为什么需要编码？因为直接把所有智能体的 (hidden, action) 求和太粗糙了——编码器让网络自己学会"用什么样的方式对每个智能体编码，使得求和后能最好地预测联合 Q 值"。
        hidden_actions_encoding = self.hidden_action_encoding(hidden_actions) # 对隐藏状态+动作的特征进一步提取特征
        hidden_actions_encoding = hidden_actions_encoding.reshape(episode_num * max_episode_len, n_agents, -1)  # 变回n_agents维度用于求和
        hidden_actions_encoding = hidden_actions_encoding.sum(dim=-2) # 这里相当于maxpool\\sumpool等操作，求和之后就将每一步所有智能体的特征融合在一起了，其余维度保持不变
        # 把所有智能体的信息"融合"成一个向量。用求和而非拼接，保证了排列不变性（智能体的顺序不影响结果）。

        inputs = torch.cat([state.reshape(episode_num * max_episode_len, -1), hidden_actions_encoding], dim=-1)
        q = self.q(inputs) # 最后预测联合Q值，输出的维度是(episode_num * max_episode_len, 1)，每一行对应一个transition的联合Q值
        return q # 这里就是每一个transition的联合Q值


class QtranV(nn.Module):
    def __init__(self, args):
        super(QtranV, self).__init__()
        self.args = args

        hidden_input = self.args.rnn_hidden_dim
        #  # ===== 编码器：编码每个智能体的 hidden state =====
        self.hidden_encoding = nn.Sequential(nn.Linear(hidden_input, hidden_input),
                                             nn.ReLU(),
                                             nn.Linear(hidden_input, hidden_input))

        v_input = self.args.state_shape + self.args.rnn_hidden_dim
        # ===== V 网络：全局 state + 所有 hidden 的编码和 → 标量 V =====
        self.v = nn.Sequential(nn.Linear(v_input, self.args.qtran_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.args.qtran_hidden_dim, self.args.qtran_hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.args.qtran_hidden_dim, 1))

    def forward(self, state, hidden):
        '''
        state: 全局的obs观察
        hidden: 循环神经网络每一步的状态 # 形状：(E, T, n_agents, 64 + 12) = (E, T, n_agents, 76)
        '''
        episode_num, max_episode_len, n_agents, _ = hidden.shape
        state = state.reshape(episode_num * max_episode_len, -1)
        # # Step 1: 编码每个智能体的 hidden
        hidden_encoding = self.hidden_encoding(hidden.reshape(-1, self.args.rnn_hidden_dim))
        # # Step 2: 求和所有智能体的编码，得到一个融合了所有智能体信息的向量
        hidden_encoding = hidden_encoding.reshape(episode_num * max_episode_len, n_agents, -1).sum(dim=-2)
        # Step 3: 拼接全局 state 后过 MLP 输出标量
        inputs = torch.cat([state, hidden_encoding], dim=-1)
        v = self.v(inputs)
        return v
