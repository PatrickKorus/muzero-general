import gym

from muzero import MuZero


#env = gym.make("MountainCar-v0")

#for it in range(2000):
#    if it == 0 or done:
#        env.reset()
#    a = env.action_space.sample()
#    env.render()
#    o, r, done, _ = env.step(a)
#env.close()

muzero = MuZero("cartpole-swingup")
muzero.train()
muzero.test(render=True, opponent="self", muzero_player=None)

