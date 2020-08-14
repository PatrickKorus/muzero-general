from muzero import MuZero

muzero = MuZero("cartpole")
muzero.train()
muzero.test(render=True, opponent="self")
