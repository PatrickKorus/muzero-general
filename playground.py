from muzero import MuZero

muzero = MuZero("cartpole",)
                #weights_load_path="results/mountaincar/2020-06-18--13-20-41/model_peak.weights",
                #replay_buffer_load_path="results/mountaincar/2020-06-17--20-18-49/replay_buffer.pkl")
muzero.train()
muzero.test(render=True, opponent="self", muzero_player=None)
