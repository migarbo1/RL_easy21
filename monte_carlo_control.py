import matplotlib.pyplot as plt
import Easy21_env as env
import numpy as np
import random

N_0 = 100
N_sa = np.zeros((11,22,2))
Q_sa = np.zeros((11,22,2))
actions = [env.Actions.HIT, env.Actions.STICK]

def get_epsilon(state):
    n_st = np.sum(N_sa[state.broker_hand][state.player_hand])
    return N_0/(N_0 + n_st)

def get_alpha(state, action):
    n_sta = N_sa[state.broker_hand][state.player_hand][int(action)]
    return 1/n_sta

def update_N(state, action):
    #N_sa[state.broker_hand][state.player_hand] += 1
    N_sa[state.broker_hand][state.player_hand][int(action)] += 1

def update_q_value(state, action, reward):
    q_stat = Q_sa[state.broker_hand][state.player_hand][int(action)]
    Q_sa[state.broker_hand][state.player_hand][int(action)] += get_alpha(state, action) * (reward - q_stat)

def select_action(state):
    eps = get_epsilon(state)
    if eps/len(actions) + 1-eps > random.random():
        action = get_argmax_q_value(state)
    else:
        action = random.choice(actions)
    return action

def get_argmax_q_value(state):
    max = -10000
    argmax = None
    for action in actions:
        local_max = Q_sa[state.broker_hand][state.player_hand][int(action)]
        if  local_max > max:
            max = local_max
            argmax = action
    return argmax

def monte_carlo_control(num_episodes):
    i = 0
    environment = env.Easy21Environment()
    while i < num_episodes:
        state_action_pair = []
        rewards = []
        st = env.State(environment.draw(black_only=True), environment.draw(black_only=True))
        while not st.is_terminal:
            act = select_action(st)

            st_, reward = environment.step(st, act)
            
            rewards.append(reward)
            state_action_pair.append((st, act))
            st = st_

        #update N and Q
        for j, state_action in enumerate(state_action_pair):
            state = state_action[0]
            action = state_action[1]
            update_N(state, action)
            update_q_value(state, action, rewards[j])
        i+=1

def plot_q():
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    aux = np.zeros((11,22))
    for i in range(1,11):
        for j in range(1,22):
            aux[i,j] = max(Q_sa[i,j,:])
    aux = aux[1:, 1:]

    x = [i for i in range(1,11)]
    y = [i for i in range(1,22)]

    x, y = np.meshgrid(x, y) 

    aux = aux.transpose()
    ax.plot_surface(x, y, aux, rstride=1, cstride=1,
                    cmap='viridis')
    ax.set_title('easy 21')
    plt.savefig('MC_control')

if __name__ == '__main__':
    monte_carlo_control(5000000)
    plot_q()