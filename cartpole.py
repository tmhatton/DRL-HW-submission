import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import gym
import ray
from really import SampleManager  # important !!
from really.utils import (
    dict_to_dict_of_datasets,
)  # convenient function for you to create tensorflow datasets


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.hidden_layer_1 = tf.keras.layers.Dense(units=32,
                                                    activation=tf.nn.leaky_relu
                                                    )
        self.hidden_layer_2 = tf.keras.layers.Dense(units=32,
                                                    activation=tf.nn.leaky_relu
                                                    )
        self.output_layer = tf.keras.layers.Dense(units=2,
                                                  activation=None, use_bias=False)

    def call(self, x_in):
        output = {}
        x = self.hidden_layer_1(x_in)
        x = self.hidden_layer_2(x)
        x = self.output_layer(x)
        output["q_values"] = x
        return output




class ModelContunous(tf.keras.Model):
    def __init__(self, output_units=2):
        super(ModelContunous, self).__init__()

        self.layer_mu = tf.keras.layers.Dense(output_units)
        self.layer_sigma = tf.keras.layers.Dense(output_units, activation=None)
        self.layer_v = tf.keras.layers.Dense(1)

    def call(self, x_in):
        output = {}
        mus = self.layer_mu(x_in)
        sigmas = tf.exp(self.layer_sigma(x_in))
        v = self.layer_v(x_in)
        output["mu"] = mus
        output["sigma"] = sigmas

        return output


if __name__ == "__main__":

    buffer_size = 5000
    test_steps = 1000
    epochs = 20
    sample_size = 20000
    optim_batch_size = 8
    saving_after = 5
    alpha = .001
    gamma = .95
    optimizer = tf.keras.optimizers.SGD(alpha)
    loss_function = loss_function = tf.keras.losses.MSE
    epsilon = 1

    kwargs = {
        "model": MyModel,
        "environment": "CartPole-v0",
        "num_parallel": 5,
        "total_steps": 2000,
        "action_sampling_type": "epsilon_greedy",
        "num_episodes": 20,
        "epsilon": epsilon,
    }

    ray.init(log_to_driver=False)

    manager = SampleManager(**kwargs)
    # where to save your results to: create this directory in advance!
    saving_path = os.getcwd() + "/progress_cartpole"



    # keys for replay buffer -> what you will need for optimization
    optim_keys = ["state", "action", "reward", "state_new", "not_done"]

    # initialize buffer
    manager.initilize_buffer(buffer_size, optim_keys)

    # initilize progress aggregator
    manager.initialize_aggregator(
        path=saving_path, saving_after=5, aggregator_keys=["loss", "time_steps"]
    )

    # initial testing:
    print("test before training: ")
    manager.test(test_steps, test_episodes=10, do_print=True, render=True)

    # get initial agent
    agent = manager.get_agent()

    for e in range(epochs):

        # training core

        # experience replay
        print("collecting experience..")
        data = manager.get_data(total_steps=100)
        manager.store_in_buffer(data)

        # sample data to optimize on from buffer
        sample_dict = manager.sample(sample_size)
        print(f"collected data for: {sample_dict.keys()}")
        # create and batch tf datasets
        data_dict = dict_to_dict_of_datasets(sample_dict, batch_size=64)
        print("optimizing...")

        # for each batch
        for state, action, reward, state_new, not_done in zip(data_dict['state'], data_dict['action'],
                                                              data_dict['reward'], data_dict['state_new'],
                                                              data_dict['not_done']):
            q_target = tf.cast(reward, tf.float64) + (
                        tf.cast(not_done, tf.float64) * tf.cast(gamma * agent.max_q(state_new), tf.float64))

            with tf.GradientTape() as tape:
                prediction = agent.q_val(state, action)
                loss = loss_function(prediction, q_target)
                gradients = tape.gradient(loss, agent.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, agent.model.trainable_variables))

        new_weights = agent.model.get_weights()

        # set new weights
        agent.set_weights(new_weights)
        manager.set_agent(new_weights)
        # get new weights
        agent = manager.get_agent()
        # update aggregator
        time_steps = manager.test(test_steps)
        manager.update_aggregator(loss=loss, time_steps=time_steps)
        # print progress
        print(
            f"epoch ::: {e}  loss ::: {loss}   avg env steps ::: {np.mean(time_steps)}"
        )

        # you can also alter your managers parameters
        if e % 5 == 0:
            epsilon = epsilon * .9
            manager.set_epsilon(epsilon=epsilon)
            print(f"New epsilon: {epsilon}")

        # if e % saving_after == 0:
        #     #you can save models
        #     manager.save_model(saving_path, e)

    # and load mmodels
    # manager.load_model(saving_path)

    print("done")
    print("testing optimized agent")
    manager.test(test_steps, test_episodes=10, render=True, do_print=True)
