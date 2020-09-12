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
muzero = MuZero("mountaincar") #, weights_load_path="results/pendulum_discrete/2020-07-09--22-08-16/model_peak.weights")
                #weights_load_path="results/mountaincar/2020-06-18--13-20-41/model_peak.weights",
                #replay_buffer_load_path="results/mountaincar/2020-06-17--20-18-49/replay_buffer.pkl")
muzero.train()
muzero.test(render=True, opponent="self", muzero_player=None)
