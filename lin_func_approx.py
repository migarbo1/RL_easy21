import matplotlib.pyplot as plt
import Easy21_env as env
import monte_carlo_control as mc
import numpy as np
import random

theta = np.zeros((36,))
actions = [env.Actions.HIT, env.Actions.STICK]
broker_pairs = [(1,4), (4,7), (7,10)]
player_pairs = [(1,6), (4,9), (7,12), (10,15), (13,18), (16,21)]

def get_epsilon():
    return 0.05

def get_alpha():
    return 0.01

def is_in_range(number, lower_range, upper_range):
    return lower_range <= number <= upper_range

def get_feature_vector(state, action):
    bh = state.broker_hand
    ph = state.player_hand
    feature_vector = np.zeros((36,))

    feature_index = 0
    for bplr, bpur in broker_pairs:
        for pplr, ppur in player_pairs:
            for act in actions:
                if( is_in_range(bh, bplr, bpur) and
                   is_in_range(ph, pplr, ppur) and
                   action == act
                ):
                    feature_vector[feature_index] = 1
                feature_index +=1
    return feature_vector

def get_q_hat(state, action):
    x = np.array(get_feature_vector(state, action))
    w = theta.transpose()
    return np.dot(x, w)

def select_action(state):
    eps = get_epsilon()
    if eps/len(actions) + 1-eps > random.random():
        action = get_argmax_q_value(state)
    else:
        action = random.choice(actions)
    return action

def get_argmax_q_value(state):
    max = -np.inf
    argmax = None
    for action in actions:
        local_max = get_q_hat(state, action)
        if  local_max > max:
            max = local_max
            argmax = action
    return argmax

def SARSA_control(num_episodes, lam):
    global theta
    theta = np.zeros((36,))
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
                delta = (1 - lam) * get_q_hat(st_, act_) - get_q_hat(st, act)
                theta += get_alpha() * (reward + delta) * get_feature_vector(st, act)
                act = act_
            else:
                theta += get_alpha() * (reward - get_q_hat(st, act)) * get_feature_vector(st, act)
                act = None
            st = st_

        i += 1

def plot_q(lam):
    plt.clf()
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    aux = np.zeros((10,21))
    for i in range(10):
        for j in range(21):
            aux[i,j] = max([get_q_hat(env.State(i,j), env.Actions.HIT),
                            get_q_hat(env.State(i,j), env.Actions.STICK)])

    x = [i for i in range(10)]
    y = [i for i in range(21)]

    x, y = np.meshgrid(x, y) 

    aux = aux.transpose()
    ax.plot_surface(x, y, aux, rstride=1, cstride=1,
                    cmap='viridis')
    ax.set_title('easy 21')
    plt.savefig('lin_func_aprox_l{}'.format(lam).replace('.', ''))

if __name__ == '__main__':
    for lam in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        SARSA_control(50000, lam)
        plot_q(lam)