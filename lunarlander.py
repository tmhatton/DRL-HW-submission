import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import gym
import ray
import statistics

from really import SampleManager  # important !!
from really.utils import (
    dict_to_dict_of_datasets,
)  # convenient function for you to create tensorflow datasets

"""
    -first, disclaimer -> you can probably reuse some code of your deep q implementation
    - you probably want to implement a policy network which outputs a normal distribution so think about what keys you need for the output dictionary
        and also think about what method you need to sample actions (tip: look up 'continuous-normal-diagonal')
    -if you want to be on-policy (thus no replay buffer) you can still use the manager's sampling method but just pass on
        from_buffer=False ->  manager.sample(sample_size, from_buffer=False)
    -if you want to also use a state value estimate have two things in mind:
        - you can write one model that outputs both your state value and your policy with separate internal layers. However, if you want to optimize,
          you have to make sure to only optimize the part of your model you want to optimize (the policy part or the state value estimate part).
          You can for example achieve that by naming your layers in the model initialization and then filter model.trainable_variables according
        to these names to compute and apply your gradients separately
        - the state values returned by the manager can't be used to backpropagate directly (as the gradient flow is interrupted) so you might need
          to compute the state values again when optimizing
-if you are on policy ant need log probabilities to train, be aware you cannot make use of the collected log prob values of the sample manager,
    as a) you might need gradient flow and b) if you change your policy the future probabilities will also change

"""




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

    #output_units = 2, bc action has two real values vectors
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
    epochs = 50 # epochs to train
    sample_size = 250
    optim_batch_size = 8
    saving_after = 5 # saving model after x epochs
    alpha = .001 # learning rate
    gamma = .9 # discount factor gamma
    optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)
    epsilon = 1

    kwargs = {
        "model": ModelContunous,
        "environment": "LunarLanderContinuous-v2",
        "num_parallel": 4, # runner boxes
        "total_steps": 2000, # amouint of maximal steps of each runner
        "action_sampling_type": "continuous_normal_diagonal",
        "num_episodes": 50, # num_episodes per runner box
        "epsilon": epsilon,
    }

    ray.init(log_to_driver=False)

    manager = SampleManager(**kwargs)
    # where to save your results to: create this directory in advance!
    saving_path = os.getcwd() + "/progress_lunar"



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

    manager.test(test_steps, test_episodes=10, do_print=True, render=True, evaluation_measure="time_and_reward")

    # get initial agent
    agent = manager.get_agent()

    for e in range(epochs):

        # training core

        # experience replay off (-> on policy)
        print("collecting experience..")
        data = manager.get_data()
        manager.store_in_buffer(data)
        # sample data to optimize on from buffer
        sample_dict = manager.sample(sample_size, from_buffer=False)
        print(f"collected data for: {sample_dict.keys()}")


        episodes_reward_discounted = []
        counter = 0
        last_sample = 0

        done_samples = 0
        for not_done in sample_dict["not_done"]:
            # list for the sample we looking into atm
            reward_discounted = []

            # if one episodes is done...
            if not_done==0:

                # need this for correct array size
                if(last_sample==0):
                    x=0
                else:
                    x=1

                # iterate through the sample
                for i in range(counter-last_sample-x):
                    if(i==0):
                        # first entry in the discounted reward list
                        reward_discounted.insert(0,gamma**(counter-last_sample) * sample_dict["reward"][counter])
                    # other entrys in discounted reward list
                    reward_discounted.insert(0,gamma**(counter-i-last_sample) * sample_dict["reward"][counter-i] + reward_discounted[0])
                # save the index of the terminal state of last sample
                last_sample = counter

                #normalize discounted rewards
                reward_discounted = (reward_discounted - statistics.mean(reward_discounted) / statistics.stdev(reward_discounted) + 1e-9)
                #extend list
                episodes_reward_discounted.extend(reward_discounted)
            counter+=1


        #change reward dict to discounted reward
        sample_dict["reward"] = episodes_reward_discounted

        # size of state,action ,rewarwd etc in for loop
        data_dict = dict_to_dict_of_datasets(sample_dict, batch_size=64)

        print("optimizing...")

        # loss function
        def a_loss(flowing_log_prob, reward):

            reward = tf.cast(reward, tf.float32)
            reward = tf.reshape(reward, (-1,1))
            loss = tf.multiply(-flowing_log_prob,reward)


            return loss

        # one episode in correct order
        # need discounted reward here already
        # rest is the more or less the same as in the cartpole.py
        for state, action, reward, state_new, not_done in zip(data_dict['state'], data_dict['action'],
                                                            data_dict['reward'], data_dict['state_new'],
                                                              data_dict['not_done']):

            with tf.GradientTape() as tape:#
                #print("STUFF:")
                #print(len(agent.flowing_log_prob(state, action)))
                #print(len(reward))

                loss = a_loss(agent.flowing_log_prob(state, action), reward)
                loss = tf.keras.backend.mean(loss)
                gradients = tape.gradient(loss, agent.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, agent.model.trainable_variables))


        new_weights = agent.model.get_weights()

        # set new weights
        agent.set_weights(new_weights)
        manager.set_agent(new_weights)
        # get new weights
        agent = manager.get_agent()
        # update aggregator
        #print(test_steps)
        time_steps = manager.test(test_steps)
        manager.update_aggregator(loss=loss, time_steps=time_steps)
        # print progress
        print(
            f"epoch ::: {e}  loss ::: {loss} avg env steps ::: {np.mean(time_steps)}"
        )
        manager.test(test_steps, test_episodes=10, render=True, do_print=True, evaluation_measure="time_and_reward" )
        # you can also alter your managers parameters
        if e % 5 == 0:
            epsilon = epsilon * .9
            manager.set_epsilon(epsilon=epsilon)
            print(f"New epsilon: {epsilon}")
            print("testing optimized agent")
            manager.test(test_steps, test_episodes=10, render=True, do_print=True, evaluation_measure="time_and_reward" )

        if e % saving_after == 0:
        #     #you can save models
            manager.save_model(saving_path, e)

    # and load mmodels
    manager.load_model(saving_path)

    print("done")
    print("testing optimized agent")
    manager.test(test_steps, test_episodes=100, render=True, do_print=True, evaluation_measure="time_and_reward" )
