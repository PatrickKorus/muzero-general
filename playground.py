import gym

from muzero import MuZero
env = gym.make('MountainCar-v0')
r = 0
env.reset()
for it in range(1000):
    action = env.action_space.sample()
    _, reward, done, _ = env.step(action)
    r += 0.995 * reward
    if done:
        break
print(r)


muzero = MuZero('mountaincar')

muzero.train()

muzero.test(render=True)
