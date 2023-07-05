import matplotlib.pyplot as plt
import Easy21_env as env
import monte_carlo_control as mc
import numpy as np
import random

N_0 = 100
N_sa = np.zeros((10,21,2))
Q_sa = np.zeros((10,21,2))
actions = [env.Actions.HIT, env.Actions.STICK]

def get_epsilon(state):
    n_st = np.sum(N_sa[state.broker_hand - 1][state.player_hand - 1])
    return N_0/(N_0 + n_st)

def get_alpha(state, action):
    n_sta = N_sa[state.broker_hand - 1][state.player_hand - 1][int(action)]
    return 1/n_sta

def update_N(state, action):
    N_sa[state.broker_hand - 1][state.player_hand - 1][int(action)] += 1

def select_action(state):
    eps = get_epsilon(state)
    if eps/len(actions) + 1-eps > random.random():
        action = get_argmax_q_value(state)
    else:
        action = random.choice(actions)
    return action

def get_argmax_q_value(state):
    max = -np.inf
    argmax = None
    for action in actions:
        local_max = Q_sa[state.broker_hand - 1][state.player_hand - 1][int(action)]
        if  local_max > max:
            max = local_max
            argmax = action
    return argmax

def SARSA_control(num_episodes, lam):
    i = 0
    environment = env.Easy21Environment()
    while i < num_episodes:

        #initialize S
        st = env.State(environment.draw(black_only=True), environment.draw(black_only=True))
        #choose action for selected S
        act = select_action(st)

        while not st.is_terminal:
            
            #take action and observe Reward and New state
            st_, reward = environment.step(st, act)
            
            #choose new action from derived new state
            if not st_.is_terminal:
                act_ = select_action(st_)
                update_N(st, act)
                delta = (1 - lam) * Q_sa[st_.broker_hand - 1][st_.player_hand - 1][int(act_)] - Q_sa[st.broker_hand - 1][st.player_hand - 1][int(act)]
                Q_sa[st.broker_hand - 1][st.player_hand - 1][int(act)] += get_alpha(st, act) * (reward + delta)
                act = act_
            else:
                update_N(st, act)
                Q_sa[st.broker_hand - 1][st.player_hand - 1][int(act)] += get_alpha(st, act) * (reward - Q_sa[st.broker_hand - 1][st.player_hand - 1][int(act)])
                act = None

            st = st_

        i += 1

def plot_q(q, lam):
    plt.clf()
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    aux = np.zeros((10,21))
    for i in range(10):
        for j in range(21):
            aux[i,j] = max(q[i,j,:])

    x = [i for i in range(10)]
    y = [i for i in range(21)]

    x, y = np.meshgrid(x, y) 

    aux = aux.transpose()
    ax.plot_surface(x, y, aux, rstride=1, cstride=1,
                    cmap='viridis')
    ax.set_title('easy 21')
    plt.savefig('SARSA_control_l{}'.format(lam).replace('.', ''))

if __name__ == '__main__':
    for lam in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        N_sa = np.zeros((10,21,2))
        Q_sa = np.zeros((10,21,2))
        SARSA_control(50000, lam)
        plot_q(Q_sa, lam)