QMixNet 是 QMIX 算法的**灵魂**——它把多个 agent 的个体 Q 值"混合"成一个全局的 $Q_{tot}$，训练时用全局 reward 做 TD 学习。从头到尾串一遍：

---

## 1. 数据从哪里来

`rollout.py` 采集一条 episode，存入 `ReplayBuffer`，每条 transition 记录：

| key | 含义 | shape |
|-----|------|-------|
| `o` | 每个 agent 的局部观测 | `(episode, max_len, n_agents, obs_dim)` |
| `s` | **全局状态** | `(episode, max_len, state_dim)` |
| `u` | 每个 agent 的实际动作 | `(episode, max_len, n_agents, 1)` |
| `r` | 全局 reward | `(episode, max_len, 1)` |
| ... | ... | |

**关键点**：`s` 是全局 state，`r` 也是全局 reward——但 agent 的个体 Q 值（`q_evals`）是各自用 RNN 独立算的。

---

## 2. 个体 Q 值怎么算的

`policy/qmix.py:140-162` 的 `get_q_values`：

```python
for transition_idx in range(max_episode_len):
    inputs = _get_inputs(batch, transition_idx)   # (ep*agents, feature_dim)
    q_eval, eval_hidden = eval_rnn(inputs, eval_hidden)  # GRUCell
    # reshape: (ep*agents, n_actions) → (ep, n_agents, n_actions)
    q_eval = q_eval.view(episode_num, self.n_agents, -1)
```

循环完 60 个 time step 后 `stack` → `q_evals.shape = (ep, 60, n_agents, n_actions)`。

然后 `gather` 取出实际动作对应的 Q 值：

```python
q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)
# shape: (ep, 60, n_agents)
```

**到此为止和标准 DQN 一样——只是我们现在有 N 个 agent 的 N 条 Q 值。**

---

## 3. QMixNet 做了什么

`policy/qmix.py:93` 这一行调用：

```python
q_total_eval = self.eval_qmix_net(q_evals, s)
# q_evals: (ep, 60, n_agents)   ← 每个agent的个体Q
# s:       (ep, 60, state_dim)  ← 全局状态
# 输出:    (ep, 60, 1)           ← 全局Q_tot
```

进入 `QMixNet.forward`（`network/qmix_net.py:36-58`）：

### Step A — 拍平准备

```python
episode_num = q_values.size(0)                    # 32
q_values = q_values.view(-1, 1, n_agents)         # (1920, 1, 5)
states = states.reshape(-1, state_shape)           # (1920, state_dim)
```

把 `(episode, time_step)` 两个维度合并成 1920 个独立样本，每个样本是 5 个 agent 的 Q 值 + 全局 state。

### Step B — Hypernetwork 生成第一层权重

```python
w1 = torch.abs(self.hyper_w1(states))    # (1920, 5*32) = (1920, 160)
b1 = self.hyper_b1(states)               # (1920, 32)
```

**Hypernetwork**：不是直接学权重矩阵，而是用另一个网络 `hyper_w1` 把 **全局 state 映射成权重的每一行**。

`w1` 是 160 维向量，reshape 成 `(1920, 5, 32)`——每一行是 5 个 agent 各自的 32 维特征变换向量。

`torch.abs` 是 QMIX 的核心约束：

$$
\frac{\partial Q_{tot}}{\partial Q_i} \geq 0
$$

权重要求非负，保证 $Q_{tot}$ 对任意一个 agent 的 $Q_i$ **单调递增**。这是 QMIX 相比 VDN（直接求和）的唯一数学约束。

### Step C — 第一层混合

```python
hidden = F.elu(torch.bmm(q_values, w1) + b1)
# q_values: (1920, 1, 5)  @  w1: (1920, 5, 32)  →  (1920, 1, 32)
```

Batch matmul 完成矩阵乘法。5 个 agent 的 Q 值通过 weight 映射成 32 维隐藏表示。

用 `ELU`（不是 ReLU）——ELU 在负数区域是光滑的，梯度更好。

### Step D — Hypernetwork 生成第二层权重，出最终值

```python
w2 = torch.abs(self.hyper_w2(states))    # (1920, 32) → reshape → (1920, 32, 1)
b2 = self.hyper_b2(states)               # (1920, 1)
q_total = torch.bmm(hidden, w2) + b2     # (1920, 1, 1) → reshape → (32, 60, 1)
```

同样 `torch.abs` 保单调性，第二层把 32 维隐藏表示压成一个标量 $Q_{tot}$。

---

## 4. 为什么这样设计

直觉类比：

> 5 个队员各自有自信程度 $Q_i$（"我觉得能赢得分"）。教练根据场上全局局势（state）决定**怎么综合**这 5 个人的判断。如果某个队员特别关键，教练就放大他的权重；如果某个队员的 Q 值不准，教练就压低。但教练不能"反转"队员的判断——队员觉得好，综合后不能变差。这个"不能反转"就是 `torch.abs`。

对比：

| 方法 | $Q_{tot}$ | 约束 |
|------|-----------|------|
| VDN | $\sum Q_i$ | 强制线性求和 |
| **QMIX** | $\text{NN}([Q_1..Q_n], s)$ | $\partial Q_{tot}/\partial Q_i \ge 0$ |
| QTRAN | $\text{NN}([Q_1..Q_n], s)$ | 无单调约束，额外 loss 项 |

QMIX 在 VDN 的简单求和和 QTRAN 的无约束之间取了一个平衡——保持单调性（保证最优个体动作 = 最优全局动作），同时用神经网络让混合方式更灵活。

---

## 5. 回到 loss 计算

```python
q_total_eval   = eval_qmix_net(q_evals, s)           # (32, 60, 1)
q_total_target = target_qmix_net(q_targets, s_next)   # (32, 60, 1)
targets = r + gamma * q_total_target * (1 - terminated)
loss = MSE(q_total_eval, targets)
```

**全局 reward `r` 只对应全局 $Q_{tot}$**，$Q_{tot}$ 的梯度通过 `torch.abs` 的行列拆分回传到每个 agent 的 RNN 网络，以此实现集中训练。


## `torch.abs` 在做什么

`hyper_w1(states)` 是 hypernetwork 从全局 state 生成的**原始权重值**，可正可负。如果直接当权重用：

```
Q_tot = w1 * Q1 + w2 * Q2 + ...
```

万一 $w_2 = -3$，那么 agent_2 觉得"这步很好""（$Q_2$ 很大），混进 $Q_{tot}$ 反而变小——**单调性被破坏**。

QMIX 要求 $Q_{tot}$ 对每个 $Q_i$ 单调递增，所以权重必须非负。`torch.abs` 是最简单的做法：把负数强行翻正。

## 关于求导

**`torch.abs` 是可导的**，不会断梯度传播。PyTorch 对 `abs` 的梯度定义为：

$$
\frac{d}{dx}|x| = 
\begin{cases}
1  & x > 0 \\
-1 & x < 0 \\
0  & x = 0
\end{cases}
$$

导数图：

```
abs 函数      导数
  \  /|       -1 | 1
   \/ |           |
    \ |      -----+-----
     \|            |
```

在 $x=0$ 处有个不可导的"尖点"，PyTorch 直接返回 0。实践中权重几乎不会精确停在 0 上，所以不影响训练。

QMIX 原论文用的其实是 **平方再开方**（等价于 abs）、或者 `elu(x)+1` 这种保证正数的激活。本质都一样：**需要一个处处非负、几乎处处可导的函数把权重钳制在正数域**。`torch.abs` 是代码实现中最直接的写法。


用具体数值来演示。

## 如果不用 `torch.abs`，会发生什么

简化到一层混合（只讲原理，结构同 qmix_net 的第二层）：

```
Q_tot = w1·Q1 + w2·Q2 + w3·Q3 + b
```

假设某时刻算出来的权重是：

```
w1 = 0.5,  w2 = -3,  w3 = 2,  b = 0
```

再假设三个 agent 各自认为"这步不错"，Q 值分别是：

```
Q1 = 10,  Q2 = 10,  Q3 = 10
```

不取绝对值的结果：

```
Q_tot = 0.5×10 + (-3)×10 + 2×10 = 5 - 30 + 20 = -5
```

agent_2 觉得这步很好（Q₂=10），但它贡献的是 **-30**，直接把总分拉到负数。

如果把 agent_2 的 Q 值增大到 20（它觉得更好），结果更惨：

```
Q_tot = 0.5×10 + (-3)×20 + 2×10 = 5 - 60 + 20 = -35
```

**Q₂ 越大 → Q_tot 越小**，单调性被打破。

## 这在强化学习中为什么致命

QMIX 这种多智能体算法依赖一个假设——**IGM（Individual-Global-Max，个体全局最大一致性）**：

> 每个 agent 选自己 Q 值最大的动作，组合起来就是全局 Q 值最大的联合动作。

如果权重可以为负，这个假设就崩塌了。agent_2 选 Q 最大的动作，结果让 Q_tot 变小，那 argmax 个体的 Q 和 argmax 全局的 Q 就不一致——训练出来的策略会互相冲突，无法收敛。

## `torch.abs` 怎么修好它

```
原始 w2 = -3
abs(w2) = 3
Q_tot = 0.5×10 + 3×10 + 2×10 = 5 + 30 + 20 = 55
```

现在 agent_2 贡献的就是正的 30，Q₂ 越大 Q_tot 越大——`∂Q_tot/∂Q₂ = 3 ≥ 0`，单调性成立。

## 回到代码中的两层

`network/qmix_net.py` 的实际计算：

```python
w1 = torch.abs(self.hyper_w1(states))      # (1920, 5, 32)，≥0
hidden = elu(Q @ w1 + b1)                  # (1920, 1, 32)，elu 也是单调的
w2 = torch.abs(self.hyper_w2(states))      # (1920, 32, 1)，≥0  
q_total = hidden @ w2 + b2                 # (1920, 1, 1)
```

两层权重都取绝对值 → 复合函数保持单调 → 每个 agent 的 `Q_i` 增大必定导致 `Q_total` 不减小。这就是 QMIX 论文中 **"monotonic mixing"** 的全部含义。


你的理解完全正确。QMixNet 做的事就是**把 5 个 agent 的个体 Q 值 + 全局 state → 一个标量 $Q_{tot}$**。

这个 $Q_{tot}$ 接下来就做一件事——**用来算 TD error**。把它放回 `learn()` 的完整流程里：

```python
# policy/qmix.py:93-106

# 1. 混合出 全局Q —— 这行就是 QMixNet
q_total_eval = self.eval_qmix_net(q_evals, s)          # (32,60,1)

# 2. target网络也用同样的方式，但用的是 s_next + 下一步的max Q
q_total_target = self.target_qmix_net(q_targets, s_next) # (32,60,1)

# 3. 标准 DQN 的 TD target
targets = r + self.args.gamma * q_total_target * (1 - terminated)
# targets: (32,60,1) —— "这个状态到底值多少分"

# 4. TD error = 当前估值 - 目标值
td_error = (q_total_eval - targets.detach())

# 5. loss = MSE，反向传播
loss = (td_error ** 2).sum() / mask.sum()
loss.backward()    # 梯度通过 QMixNet 传回 RNN
```

本质是**把多智能体问题变成了单智能体问题**：

```
单智能体 DQN:      s → RNN → Q(s,a)        →  MSE(Q, r + γ·max Q')
多智能体 QMIX:  每个 agent: obs_i → RNN → Q_i(obs_i, a_i)
                       ↓
            QMixNet(state, [Q₁..Q₅]) → Q_tot →  MSE(Q_tot, r + γ·max Q_tot')
```

全局 reward `r` 只对应全局 `Q_tot`。loss 的梯度从 `Q_tot` 沿着 QMixNet 的权重反向传回每个 agent 的 RNN——这就是**集中训练**：训练时能看全局信息，执行时各 agent 只要自己的 obs。

