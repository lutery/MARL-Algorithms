import numpy as np
import threading


class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.n_actions = self.args.n_actions
        self.n_agents = self.args.n_agents
        self.state_shape = self.args.state_shape
        self.obs_shape = self.args.obs_shape
        self.size = self.args.buffer_size
        self.episode_limit = self.args.episode_limit
        # memory management
        self.current_idx = 0
        self.current_size = 0
        # create the buffer to store info
        self.buffers = {'o': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape]),
                        'u': np.empty([self.size, self.episode_limit, self.n_agents, 1]),
                        's': np.empty([self.size, self.episode_limit, self.state_shape]),
                        'r': np.empty([self.size, self.episode_limit, 1]),
                        'o_next': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape]),
                        's_next': np.empty([self.size, self.episode_limit, self.state_shape]),
                        'avail_u': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
                        'avail_u_next': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
                        'u_onehot': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
                        'padded': np.empty([self.size, self.episode_limit, 1]),
                        'terminated': np.empty([self.size, self.episode_limit, 1])
                        }
        if self.args.alg == 'maven':
            self.buffers['z'] = np.empty([self.size, self.args.noise_dim])
        # thread lock
        self.lock = threading.Lock()

        # store the episode
    def store_episode(self, episode_batch):
        '''
        episode_batch是一个字典，包含了所有局的采样数据，每一项的维度都是(episode个数, max_episode_len， n_agents， 具体维度)，比如obs的维度是(episode个数, max_episode_len， n_agents， obs维度)，action的维度是(episode个数, max_episode_len， n_agents， 1)，reward的维度是(episode个数, max_episode_len， 1)
        '''
        batch_size = episode_batch['o'].shape[0]  # episode_number
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size) # 获取存储这些经验的索引，idxs是一个数组，长度为batch_size，每个元素都是一个整数，表示要存储经验的位置
            # store the informations
            self.buffers['o'][idxs] = episode_batch['o']
            self.buffers['u'][idxs] = episode_batch['u']
            self.buffers['s'][idxs] = episode_batch['s']
            self.buffers['r'][idxs] = episode_batch['r']
            self.buffers['o_next'][idxs] = episode_batch['o_next']
            self.buffers['s_next'][idxs] = episode_batch['s_next']
            self.buffers['avail_u'][idxs] = episode_batch['avail_u']
            self.buffers['avail_u_next'][idxs] = episode_batch['avail_u_next']
            self.buffers['u_onehot'][idxs] = episode_batch['u_onehot']
            self.buffers['padded'][idxs] = episode_batch['padded']
            self.buffers['terminated'][idxs] = episode_batch['terminated']
            if self.args.alg == 'maven':
                self.buffers['z'][idxs] = episode_batch['z']

    def sample(self, batch_size):
        '''
        从经验回放池中随机抽取一个mini_batch进行学习
        batch_size: 抽取的经验数量
        '''
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffers.keys():
            temp_buffer[key] = self.buffers[key][idx]
        return temp_buffer

    def _get_storage_idx(self, inc=None):
        # inc表示这次要存储多少条经验，如果不指定则默认为1
        inc = inc or 1
        if self.current_idx + inc <= self.size: # 如果缓冲区的大小足够存储新的经验，则直接存储，为每一个经验分配一个位置
            idx = np.arange(self.current_idx, self.current_idx + inc)
            self.current_idx += inc
        elif self.current_idx < self.size: # 如果缓冲区的大小不够存储新的经验了，但是当前的索引还没有到达缓冲区的末尾，则先把能存储的经验存储了，然后再从缓冲区的开头开始存储剩下的经验
            overflow = inc - (self.size - self.current_idx)
            idx_a = np.arange(self.current_idx, self.size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])
            self.current_idx = overflow
        else: # 如果缓冲区的大小不够存储新的经验了，并且当前的索引已经到达缓冲区的末尾，则从缓冲区的开头开始存储新的经验
            idx = np.arange(0, inc)
            self.current_idx = inc
        self.current_size = min(self.size, self.current_size + inc) # 进一步确保不会溢出，但是上面不是已经确保了不会溢出了吗？这里是为了防止上面代码的bug导致的溢出
        if inc == 1: # 如果只存储一条经验，则返回一个整数，否则返回一个数组
            idx = idx[0]
        return idx
