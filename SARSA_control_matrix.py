import matplotlib.pyplot as plt
import Easy21_env as env
import monte_carlo_control as mc
import numpy as np
import random

N_0 = 100
N_sa = np.zeros((10,21,2))
Q_sa = np.zeros((10,21,2))
actions = [env.Actions.HIT, env.Actions.STICK]
gamma = 0.9

def get_epsilon(state):
    n_st = np.sum(N_sa[state.broker_hand - 1][state.player_hand - 1])
    return N_0/(N_0 + n_st)

def get_alpha(state, action):
    n_sta = N_sa[state.broker_hand - 1][state.player_hand - 1][int(action)]
    return 1/n_sta

def update_N(state, action):
    #N_sa[state.broker_hand - 1][state.player_hand - 1] += 1
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

def SARSA_control(num_episodes, lamda, m1):
    i = 0
    environment = env.Easy21Environment()
    mse_over_episodes = []
    while i < num_episodes:
        
        E_sa = np.zeros((10,21,2))
        st = env.State(environment.draw(black_only=True), environment.draw(black_only=True))
        
        act = select_action(st)
        while not st.is_terminal:
            
            st_, reward = environment.step(st, act)
            update_N(st, act)

            E_sa[st.broker_hand - 1][st.player_hand - 1][int(act)] += 1
            if not st_.is_terminal:
                act_ = select_action(st_)
                delta = reward + gamma * Q_sa[st_.broker_hand - 1][st_.player_hand - 1][int(act_)] - Q_sa[st.broker_hand - 1][st.player_hand - 1][int(act)]
            else:
                act_ = None
                delta = reward - Q_sa[st.broker_hand - 1][st.player_hand - 1][int(act)]

            for bh in range(10):
                for ph in range(21):
                    for action in actions:
                        Q_sa[bh][ph][int(action)] += get_alpha(st, act) * delta * E_sa[bh][ph][int(action)]
                        E_sa[bh][ph][int(action)] = gamma * lamda * E_sa[bh][ph][int(action)]

            st = st_
            act = act_
        i += 1
        mse_over_episodes.append(mse(m1, Q_sa))
    return mse_over_episodes

def mse(m1, m2):
    res = 0
    for i in range(10):
        for j in range(21):
            for action in actions:
                res += pow(m1[i][j][int(action)] - m2[i][j][int(action)],2)

    return res

def plot_q(q):
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
    plt.savefig('SARSA_control')

def plot_sarsa(mse_list, mse_progressions):
    plt.clf()
    #plot mse over lambda
    plt.title('MSE over values of Lambda')
    plt.plot([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], mse_list)
    plt.savefig('sarsa_mse_ev_over_lambda')

    #plot mse over episodes for L=0 and L=1
    plt.clf()
    plt.title('MSE over episodes')
    plt.plot([i for i in range(len(mse_progressions[0]))], mse_progressions[0], label = "mse lam = 0")
    plt.plot([i for i in range(len(mse_progressions[1]))], mse_progressions[1], label = "mse lam = 1")
    plt.legend()
    plt.savefig('sarsa_mse_ev_over_episodes')

if __name__ == '__main__':
    mc.monte_carlo_control(1000)
    m1 = mc.Q_sa
    mse_list = []
    mse_progressions = []
    min_sme = np.inf
    argmin_sme = 0
    for l in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        Q_sa = np.zeros((10,21,2))
        local_mse = SARSA_control(1000, l, m1)
        if local_mse[-1] < min_sme:
            argmin_sme = Q_sa
        mse_list.append(mse(m1, Q_sa))
        if l == 0 or l == 1:
            mse_progressions.append(local_mse)
    print(mse_list)
    plot_q(argmin_sme)
    plot_sarsa(mse_list, mse_progressions)