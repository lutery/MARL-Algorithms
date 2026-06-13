from runner import Runner
from smac.env import StarCraft2Env
from common.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, get_reinforce_args, get_commnet_args, get_g2anet_args


if __name__ == '__main__':
    for i in range(8):
        args = get_common_args()
        if args.alg.find('coma') > -1:
            args = get_coma_args(args)
        elif args.alg.find('central_v') > -1:
            args = get_centralv_args(args)
        elif args.alg.find('reinforce') > -1:
            args = get_reinforce_args(args)
        else:
            args = get_mixer_args(args) # qtran强化学习算法会走这条分支
        if args.alg.find('commnet') > -1:
            args = get_commnet_args(args)
        if args.alg.find('g2anet') > -1:
            args = get_g2anet_args(args)
        env = StarCraft2Env(map_name=args.map,
                            step_mul=args.step_mul,
                            difficulty=args.difficulty,
                            game_version=args.game_version,
                            replay_dir=args.replay_dir)
        env_info = env.get_env_info()
        args.n_actions = env_info["n_actions"] # 该环境有多少个有效的动作
        args.n_agents = env_info["n_agents"] # 有多少个Agent 
        args.state_shape = env_info["state_shape"] # 全局环境观察空间的shape
        args.obs_shape = env_info["obs_shape"] # Agent局部观察空间的shape
        args.episode_limit = env_info["episode_limit"] # 一局游戏的步数限制
        runner = Runner(env, args)
        if not args.evaluate: # 这个参数用来控制是训练还是评估，训练的话就调用runner.run(i)，评估的话就调用runner.evaluate()，评估的时候不需要训练，所以不需要调用runner.run(i)
            # evaluate 这个参数也是错的
            runner.run(i) 
        else:
            win_rate, _ = runner.evaluate() # todo 后续再看
            print('The win rate of {} is  {}'.format(args.alg, win_rate))
            break
        env.close()
