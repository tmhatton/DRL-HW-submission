import gym
import numpy as np
import ray
from really import SampleManager
from gridworlds import GridWorld

"""
Your task is to solve the provided Gridword with tabular Q learning!
In the world there is one place where the agent cannot go, the block.
There is one terminal state where the agent receives a reward.
For each other state the agent gets a reward of 0.
The environment behaves like a gym environment.
Have fun!!!!


Disclaimer: We did not really use the framework to train the agent in
            this task but built a little around it.

"""


class TabularQ(object):
    def __init__(self, h, w, action_space, q_table):
        self.h = h
        self.w = w
        self.action_space = action_space
        self.q_table = q_table              # define q-table (shape 10x10x4)
        pass

    def __call__(self, state):
        output = {}
        state = state.astype(int)
        output["q_values"] = self.q_table[state[0][0], state[0][1]].reshape((1,4))
        return output

    def get_weights(self):
        return None

    def set_weights(self, q_vals):
        pass

    def train(self, env, episodes, steps, alpha, discount, epsilon):
        # train the agent by updating q-values
        all_rewards = []

        # Q-learning algorithm
        for episode in range(episodes):
            state = env.reset()

            done = False
            current_reward = 0

            for step in range(steps):
                # check if we take a random action (explore vs. greedy)
                epsilon_threshold = np.random.uniform(0, 1)
                if epsilon_threshold < epsilon:
                    # if epsilon is larger than a random threshold, we exploit
                    action = np.argmax(self.q_table[state[0], state[1],:])
                else:
                    # if epsilon is smaller, we explore
                    action = env.action_space.sample()

                # take action
                new_state, reward, done, info = env.step(action)

                # update q-values
                curr_qval = self.q_table[state[0], state[1], action]
                next_qval = self.q_table[new_state[0], new_state[1], :]

                self.q_table[state[0], state[1], action] += (
                    alpha * (reward + discount * np.max(next_qval) - curr_qval)
                    )

                state = new_state
                current_reward += reward

                if done == True:
                    break

            all_rewards.append(current_reward)



if __name__ == "__main__":
    action_dict = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}

    # define environment
    env_kwargs = {
        "height": 10,
        "width": 10,
        "action_dict": action_dict,
        "start_position": (2, 0),
        "reward_position": (0, 3),
    }

    env = GridWorld(**env_kwargs)

    # define model (with empty q-table)
    model_kwargs = {
        "h": env.height,
        "w": env.width,
        "action_space": 4,
        "q_table": np.zeros((env.height, env.width, 4))
    }

    # define SampleManager
    kwargs = {
        "model": TabularQ,
        "environment": GridWorld,
        "num_parallel": 2,
        "total_steps": 100,
        "model_kwargs": model_kwargs,
        "env_kwargs" : env_kwargs
    }

    # initialize ray & SampleManager
    ray.init(log_to_driver=False)
    manager = SampleManager(**kwargs)

    
    ## test run with empty q-table
    print("test before training: ")
    manager.test(
        max_steps=100,
        test_episodes=10,
        render=True,
        do_print=True,
        evaluation_measure="time_and_reward",
    )
    
    
    # train an agent
    training_kwargs = {
        "env" : env,
        "episodes": 100,
        "steps": 100,
        "alpha": 0.1,
        "discount": 0.99,
        "epsilon": 0.95,
    }

    agent_q = TabularQ(**model_kwargs)
    agent_q.train(**training_kwargs)

    # update model & SampleManager arguments
    updated_model_kwargs = {
        "h": env.height,
        "w": env.width,
        "action_space": 4,
        "q_table": agent_q.q_table
    }

    updated_kwargs = {
        "model": TabularQ,
        "environment": GridWorld,
        "num_parallel": 2,
        "total_steps": 100,
        "model_kwargs": updated_model_kwargs,
        "env_kwargs" : env_kwargs
    }

    manager = SampleManager(**updated_kwargs)


    # test run with trained q-values
    print("test after training: ")
    manager.test(
        max_steps=100,
        test_episodes=10,
        render=True,
        do_print=True,
        evaluation_measure="time_and_reward",
    )


    
    
