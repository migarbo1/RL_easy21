import matplotlib.pyplot as plt
import Easy21_env as env
import numpy as np
import random

N_0 = 100
N_s = {}
N_sa = {}
Q_sa = {}
actions = [env.Actions.HIT, env.Actions.STICK]

def get_epsilon(state):
    n_st = N_s.get(state, 0)
    return N_0/(N_0 + n_st)

def get_alpha(state, action):
    n_sta = N_sa.get((state, action), 0)
    return 1/n_sta

def update_N(state, action):
    n_st = N_s.get(state, 0)
    n_stat = N_sa.get((state, action), 0)
    N_s[state] = n_st + 1
    N_sa[(state, action)] = n_stat + 1

def update_q_value(state, action, reward):
    q_stat = Q_sa.get((state, action), 0)
    Q_sa[(state, action)] = q_stat + get_alpha(state, action) * (reward - q_stat)

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
        local_max = Q_sa.get((state, action),0)
        if  local_max > max:
            max = local_max
            argmax = action
    return argmax

def monte_carlo_control(num_episodes):
    i = 0
    environment = env.Easy21Environment()
    while i < num_episodes:
        #print('EPISODE: {}'.format(i))
        state_action_pair = []
        rewards = []
        st = env.State(environment.draw(black_only=True), environment.draw(black_only=True))
        # print('INITIAL STATE:\n{}\n'.format(st))
        while not st.is_terminal:
            act = select_action(st)

            st_, reward = environment.step(st, act)
            
            #print('Action chosen:{}\nReward:{}\nNew state:{}\n\n'.format(act, reward, st_))

            rewards.append(reward)
            state_action_pair.append((st, act))
            st = st_
        # print('WIN' if reward > 0 else 'DEFEAT')

        #update N and Q
        for j, state_action in enumerate(state_action_pair):
            state = state_action[0]
            action = state_action[1]
            update_N(state, action)
            update_q_value(state, action, rewards[j])
        # print(Q_sa)
        # print()
        # print()
        # print('----------------------------------')
        i+=1

def plot_q():
    mat_3d = []
    for key in Q_sa.keys():
        state = key[0]
        mat_3d.append([state.broker_hand, state.player_hand, Q_sa[key]])
    mat_3d = np.array(mat_3d)
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(mat_3d[:,0], mat_3d[:,1], mat_3d[:,2], linewidth=0, antialiased=False,
                    cmap='viridis', edgecolor='none')
    ax.set_title('easy 21')
    plt.savefig('MC_control')

if __name__ == '__main__':
    monte_carlo_control(5000)
    plot_q()