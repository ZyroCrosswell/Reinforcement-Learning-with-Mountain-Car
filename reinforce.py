import gym
import numpy as np
env = gym.make("MountainCar-v0")
print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space.n)

LEARNING_RATE = 0.1
DISCOUNT = 0.95  # measure of how important are the future actions***
EPISODES = 25000

DISCRETE_OS_SIZE = [20]*len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high -
                        env.observation_space.low)/DISCRETE_OS_SIZE

# initializing the random Q table
q_table = np.random.uniform(
    low=-2,  high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
print(q_table.shape)


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


# env.reset() returns initial state
discrete_state = get_discrete_state(env.reset())


done = False
while not done:
    action = np.argmax(discrete_state)
    new_state, reward, done, _ = env.step(action)

    env.render()
env.close()
