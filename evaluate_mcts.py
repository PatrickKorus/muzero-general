import math
from copy import deepcopy

import gym
import numpy



# Game independent
from gym.spaces import Discrete

from common.games import DeepCopyableGymGame
from common.gym_wrapper import DiscreteActionWrapper, DeepCopyableWrapper, ScaledRewardWrapper


def select_action(node, temperature):
    """
    Select action according to the visit count distribution and the temperature.
    The temperature is changed dynamically with the visit_softmax_temperature function
    in the config.
    """
    visit_counts = numpy.array(
        [child.visit_count for child in node.children.values()], dtype="int32"
    )
    print(visit_counts)
    values = numpy.array([child.value() for child in node.children.values()])
    print(values)
    actions = [action for action in node.children.keys()]
    if temperature == 0:
        action = actions[numpy.argmax(visit_counts)]
    elif temperature == float("inf"):
        action = numpy.random.choice(actions)
    else:
        # See paper appendix Data Generation
        visit_count_distribution = visit_counts ** (1 / temperature)
        visit_count_distribution = visit_count_distribution / sum(
            visit_count_distribution
        )
        action = numpy.random.choice(actions, p=visit_count_distribution)

    return action



class MCTS:
    """
    Core Monte Carlo Tree Search algorithm.
    To decide on an action, we run N simulations, always starting at the root of
    the search tree and traversing the tree according to the UCB formula until we
    reach a leaf node.
    """

    def __init__(self, config):
        self.config = config

    def run(
        self,
        observation,
        reward,
        game,
        legal_actions,
        add_exploration_noise,
        override_root_with=None,
    ):
        """
        At the root of the search tree we use the representation function to obtain a
        hidden state given the current observation.
        We then run a Monte Carlo Tree Search using only action sequences and the model
        learned by the network.
        """
        if override_root_with:
            root = override_root_with
            # root_predicted_value = None
        else:
            root = Node(prior=0, rew=reward, obs=observation, done=False, game=game)
            root.expand(legal_actions)

        if add_exploration_noise:
            root.add_exploration_noise(
                dirichlet_alpha=self.config.root_dirichlet_alpha,
                exploration_fraction=self.config.root_exploration_fraction,
            )

        min_max_stats = MinMaxStats()

        max_tree_depth = 0
        for _ in range(self.config.num_simulations):
            node = root
            search_path = [node]
            current_tree_depth = 0

            while node.expanded():
                current_tree_depth += 1
                action, node = self.select_child(node, min_max_stats)
                search_path.append(node)

            # Inside the search tree we use the dynamics function to obtain the next hidden
            # state given an action and the previous hidden state
            # parent = search_path[-2]
            node.expand(
                self.config.action_space,
            )

            # TODO What is value?
            value = node.reward + self.config.discount * node.value()

            self.backpropagate(search_path, value, min_max_stats)

            max_tree_depth = max(max_tree_depth, current_tree_depth)

        extra_info = {
            "max_tree_depth": max_tree_depth,
            # "root_predicted_value": root_predicted_value,
        }
        return root, extra_info

    def select_child(self, node, min_max_stats):
        """
        Select the child with the highest UCB score.
        """
        max_ucb = max(
            self.ucb_score(node, child, min_max_stats)
            for action, child in node.children.items()
        )
        action = numpy.random.choice(
            [
                action
                for action, child in node.children.items()
                if self.ucb_score(node, child, min_max_stats) == max_ucb
            ]
        )
        return action, node.children[action]

    def ucb_score(self, parent, child, min_max_stats):
        """
        The score for a node is based on its value, plus an exploration bonus based on the prior.
        """
        pb_c = (
            math.log(
                (parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base
            )
            + self.config.pb_c_init
        )
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior    # TODO

        if child.visit_count > 0:
            # Mean value Q
            value_score = min_max_stats.normalize(
                child.reward
                + self.config.discount
                * child.value()
            )
        else:
            value_score = 0

        ucb_score = prior_score + value_score
        child.last_ucb_score = ucb_score
        return ucb_score

    def backpropagate(self, search_path, value, min_max_stats):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            min_max_stats.update(node.reward + self.config.discount * node.value())
            value = node.reward + self.config.discount * value


class Node:

    def __init__(self, prior, obs, rew, done, game):
        self.visit_count = 0
        # self.to_play = -1 removing due to control task
        self.value_sum = 0
        self.prior = prior
        self.children = {}
        self.observation = obs
        self.game_copy: DeepCopyableGame = game
        self.reward = rew
        self.done = done
        self.last_ucb_score = 0

    def expanded(self):
        return len(self.children) > 0

    def __str__(self):
        return "Node value {}, last ucb {}".format(self.value(), self.last_ucb_score)

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, actions):
        """
        We expand a node using the value, reward and policy prediction obtained from the
        neural network.
        """
        if self.done:
            #self.value_sum = -100
            #self.reward = 0.0
            #self.value_sum = 0.0
            self.children = {}
        else:
            roll_out_action = numpy.random.choice(actions)
            for action in actions:
                game_copy = self.game_copy.get_copy()
                obs, rew, done = game_copy.step(action)
                if action == roll_out_action:
                    visit_count = 10
                    value_sum = self.roll_out(game=game_copy, gamma=0.99, max_depth=20, num_roll_outs=visit_count)
                else:
                    visit_count = 1
                    value_sum = 1.0
                self.children[action] = Node(prior=0.0, obs=obs, rew=rew, done=done, game=game_copy)
                self.children[action].visit_count = visit_count
                self.children[action].value_sum = value_sum

    def roll_out(self, game: "DeepCopyableGame", gamma, max_depth, num_roll_outs):
        reward_sum = 0
        for it1 in range(num_roll_outs):
            game = game.get_copy()
            reward = 0
            done = False
            for it in range(max_depth):
                action = numpy.random.choice(2)
                if done:
                    break
                _, rew, done = game.step(action)
                reward += gamma * rew
            reward_sum += reward
        return reward_sum

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        """
        At the start of each search, we add dirichlet noise to the prior of the root to
        encourage the search to explore new actions.
        """
        actions = list(self.children.keys())
        noise = numpy.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac


class MinMaxStats:
    """
    A class that holds the min-max values of the tree.
    """

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class MCTSEvalConfig:

    def __init__(self):
        self.pb_c_base = 19652
        self.pb_c_init = 1.25
        self.discount = 0.99
        self.action_space = [i for i in range(2)]
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25
        self.num_simulations = 500

class DeepCopyableGame:

    def __init__(self, env: gym.Env, seed=0):
        self.env = env
        self.env.seed(1)
        # numpy.random.seed(0)
        self.previous_steps = numpy.array([0 for i in range(5)], dtype="float64")

    def reset(self):
        return self.env.reset()

    def sample_action(self):
        return self.env.action_space.sample()

    def step(self, action):
        observation, rew, done, _ = self.env.step(action)
        # adding past steps to state
        numpy.roll(self.previous_steps, 1)
        self.previous_steps[0] = float(action)
        #print((rew + 16.2736044)/16.2736044)
        return numpy.concatenate((self.previous_steps, observation)), ((rew + 16.2736044)/16.2736044) * 2 - 1, done
        #def step_multiple(action, n, gamma=0.3):
        #    observation, reward, done, _ = self.env.step(action)
        #    reward += 11
        #    for it in range(n-1):
        #
        #        reward = reward_add + 11 + gamma * reward_add
        #    return observation, reward, done
        #if action == 0:
        #    return step_multiple(0, 8)
        #if action == 1:
        #    return step_multiple(0, 4)
        #elif action == 2:
        #    return step_multiple(0, 2)
        #elif action == 3:
        #    return step_multiple(0, 1)
        #elif action == 4:
        #    return step_multiple(1, 1)
        #elif action == 5:
        #    return step_multiple(1, 2)
        #elif action == 6:
        #    return step_multiple(1, 4)
        #elif action == 7:
        #    return step_multiple(1, 8)
        #else:
        #    raise ValueError("Invalid Action in Wrapper, Action was {}".format(action))#

    def get_copy(self):
        return DeepCopyableGame(deepcopy(self.env))






if __name__ == "__main__":
    config = MCTSEvalConfig()
    mcts = MCTS(config)
    #env = gym.make("CartPole-v1")
    env = gym.make("Pendulum-v0")
    env = DiscreteActionWrapper(env)
    env = ScaledRewardWrapper(env, min_rew=-16.2736044, max_rew=0)
    env = DeepCopyableWrapper(env)
    game = DeepCopyableGymGame(env)
    game_copy = None
    obs = game.reset()
    done = False
    reward = 0
    render = True
    it = 0
    next_node = None
    while not done:
        it += 1
        result_node, info = mcts.run(
            observation=obs,
            reward=reward,
            game=game,
            legal_actions=config.action_space,
            add_exploration_noise=False,
            override_root_with=next_node
        )
        print(info)
        action = select_action(
            node=result_node,
            temperature=0,
        )
        next_node = result_node.children[action]
        print(action)
        if render:
            if game_copy is not None:
                game_copy.env.close()
            game_copy = game.get_copy()
            game_copy.env.render()
        obs, reward, done = game.step(action)
        print (reward)
    print(it)