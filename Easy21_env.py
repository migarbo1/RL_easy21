from enum import Enum
import random
import math

sign = lambda x: math.copysign(1, x)

class Actions(Enum):
    STICK = 0,
    HIT = 1

class State():

    def __init__(self, broker_hand = 0, player_hand = 0, is_terminal = False):
        self.broker_hand = broker_hand
        self.player_hand = player_hand
        self.is_terminal = is_terminal

class Easy21Environment():

    def __init__(self):
        pass

    def step(self, state: State, action: Actions):
        reward = 0
        new_state = state
        if action == Actions.HIT:
            new_state.player_hand = state.player_hand + self.draw()
            if new_state.player_hand > 21 or new_state.player_hand < 1:
                reward = -1
                new_state.is_terminal = True

            return new_state, reward
        
        if action == Actions.STICK:
            new_state = self.sim_broker_turn(state)
            new_state.is_terminal = True
            if new_state.broker_hand > 21 or new_state.broker_hand < 1:
                reward = 1
            else:
                reward = sign(state[1] - new_state.broker_hand)
            
            return new_state, reward

    
    def sim_broker_turn(self, state: State):
        new_state = state
        while new_state.broker_hand > 1 or new_state.broker_hand < 17:
            new_state.broker_hand += self.draw()
        return new_state

    def draw(self):
        card = random.randint(1, 10)
        color = random.choices([-1, 1], [1/3, 2/3], k=1)[0]
        return card * color
