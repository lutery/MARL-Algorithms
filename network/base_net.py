import torch.nn as nn
import torch.nn.functional as f


class RNN(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    # 预测的是动作的Q值分布
    def __init__(self, input_shape, args):
        super(RNN, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, obs, hidden_state):
        # 输入的一般是智能体局部的观察，再根据配置决定是否输入智能体之前的动作以及智能体的id
        # 因为这里的RNN网络不通的智能体都会使用，所以输入不通的智能体id非常重要
        x = f.relu(self.fc1(obs)) # 通过全连接层进行一次特征提取
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim) # 对隐藏层进行一次reshape
        h = self.rnn(x, h_in) # 输入到rnn网络中
        q = self.fc2(h) # 最后预测动作
        return q, h # 返回预测的动作Q值分布以及新的隐藏层


# Critic of Central-V
class Critic(nn.Module):
    def __init__(self, input_shape, args):
        super(Critic, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.critic_dim)
        self.fc2 = nn.Linear(args.critic_dim, args.critic_dim)
        self.fc3 = nn.Linear(args.critic_dim, 1)

    def forward(self, inputs):
        x = f.relu(self.fc1(inputs))
        x = f.relu(self.fc2(x))
        q = self.fc3(x)
        return q
