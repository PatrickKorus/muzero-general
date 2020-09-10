import copy
import math
import sys
import time

import gym
import numpy
import ray
import torch

import models
from common.games import DeepCopyableGymGame
from common.gym_wrapper import DeepCopyableWrapper, ScaledRewardWrapper
from evaluate_mcts import DeepCopyableGame
from evaluate_mcts_discrete import MCTSExperimentConfig


def select_action(node, temperature):
    """
    Select action according to the visit count distribution and the temperature.
    The temperature is changed dynamically with the visit_softmax_temperature function
    in the config.
    """
    visit_counts = numpy.array(
        [child.visit_count for child in node.children.values()], dtype="int32"
    )
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


# Game independent
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
            game: DeepCopyableGame,
            add_exploration_noise,
            override_root_with=None):
        """
        At the root of the search tree we use the representation function to obtain a
        hidden state given the current observation.
        We then run a Monte Carlo Tree Search using only action sequences and the model
        learned by the network.
        """
        if override_root_with:
            root = override_root_with
            root_predicted_value = None
        else:
            root = Node(0)
            # root_predicted_value = 0  # TODO: possibily initial roll out?
            reward = reward
            # policy_params = 0   # TODO
            # observation = (
            #     torch.tensor(observation)
            #     .float()
            #     .unsqueeze(0)
            #     .to(next(model.parameters()).device)
            # )
            # (
            #     root_predicted_value,
            #     reward,
            #     policy_params,
            #     hidden_state,
            # ) = model.initial_inference(observation)
            # root_predicted_value = models.support_to_scalar(
            #     root_predicted_value, self.config.support_size
            # ).item()
            # reward = models.support_to_scalar(reward, self.config.support_size).item()
            mu = 0   # TODO
            sigma = 2   # TODO

            root.expand(
                # legal_actions,
                reward,
                mu,
                sigma,
                observation,
                game
            )
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
            parent = search_path[-2]
            game_copy = parent.game.get_copy()

            observation, reward, done = game_copy.step(action)
            mu = 0
            sigma = 1
            value = reward if done else 0   # TODO: Value estimate over roll outs?
            node.expand(
                reward,
                mu,
                sigma,
                observation,
                game_copy
            )

            self.backpropagate(search_path, value, min_max_stats)

            max_tree_depth = max(max_tree_depth, current_tree_depth)

        return root, max_tree_depth

    def select_child(self, node, min_max_stats):
        """
        Select the child with the highest UCB score.
        """
        # Progressive widening (See https://hal.archives-ouvertes.fr/hal-00542673v2/document)
        # TODO: Move prog widening out of select child
        C = self.config.C
        alpha = self.config.alpha
        while len(node.children) < math.ceil(C * node.visit_count ** alpha):
            action = sample_action(node.mu, node.sigma)
            node.children[action] = Node()

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

        # Uniform prior for continuous action space
        prior_score = pb_c * (1 / len(parent.children))

        if child.visit_count > 0:
            # Mean value Q
            value_score = min_max_stats.normalize(
                child.reward + self.config.discount * child.value()
            )
        else:
            value_score = 0

        return prior_score + value_score

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
    def __init__(self, prior=1):
        self.visit_count = 0
        self.prior = prior # Unused prior for continuous action space
        self.value_sum = 0
        self.children = {}
        self.observation = None
        self.reward = 0
        self.game = None

        # progressve widening
        self.mu = None
        self.sigma = None

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, reward, mu, sigma, observation, game):
        """
        We expand a node using the value, reward and policy prediction obtained from the
        neural network.
        """
        self.reward = reward
        self.observation = observation
        self.game = game

        self.mu, self.sigma = mu, sigma

        action = sample_action(self.mu, self.sigma)
        self.children[action] = Node()

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


def sample_action(mu, sigma):
    # mu = torch.tanh(mu)
    # sigma = torch.exp(sigma)

    # m = torch.distributions.normal.Normal(mu, sigma)
    # return m.sample()
    return numpy.random.normal(mu, sigma)


def get_log_prob(mu, sigma, action):
    mu = torch.tanh(mu)
    sigma = torch.exp(sigma)

    m = torch.distributions.normal.Normal(mu, sigma)
    log_prob = m.log_prob(action)
    return log_prob

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


class MCTSExperimentConfigContinuous(MCTSExperimentConfig):

    def __init__(self):
        self.pb_c_base = 19652
        self.pb_c_init = 1.25
        self.discount = 0.999
        self.action_space = [i for i in range(2)]
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25
        self.num_simulations = 200

        # progressive widening
        self.C = 1
        self.alpha = 0.25

        super().__init__()


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    config = MCTSExperimentConfigContinuous()
    mcts = MCTS(config)
    #env = gym.make("CartPole-v1")
    env = gym.make("Pendulum-v0")
    #env = DiscreteActionWrapper(env)
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
            add_exploration_noise=False,
            override_root_with=None  # next_node
        )
        print(info)
        print(["{}: {}, ".format(action, child.visit_count) for action, child in result_node.children.items()])
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
        print(reward)
    print(it)
