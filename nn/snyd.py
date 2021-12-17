import random
import torch
from torch import nn
import itertools
import numpy as np
import math
from collections import Counter


class Net(torch.nn.Module):
    def __init__(self, d_pri, d_pub, hiddens=(100,)):
        super().__init__()

        # Bilinear can't be used inside nn.Sequantial
        # https://github.com/pytorch/pytorch/issues/37092
        self.layer0 = torch.nn.Bilinear(d_pri, d_pub, hiddens[0])

        layers = [torch.nn.ReLU()]
        for size0, size1 in zip(hiddens, hiddens[1:]):
            layers += [torch.nn.Linear(size0, size1), torch.nn.ReLU()]
        layers += [torch.nn.Linear(hiddens[-1], 1), nn.Tanh()]
        self.seq = nn.Sequential(*layers)

    def forward(self, priv, pub):
        joined = self.layer0(priv, pub)
        return self.seq(joined)

def calc_args(d1, d2, sides, variant):
    # Maximum call is (D1+D2) 6s
    D_PUB = (d1 + d2) * sides
    if variant == "stairs":
        D_PUB = 2 * (d1 + d2) * sides
    # We also have a feature that means "game has been called"
    D_PUB += 1

    # Total number of actions in the game
    N_ACTIONS = D_PUB
    LIE_ACTION = N_ACTIONS - 1

    # Add one extra feature to keep track of who's to play next
    CUR_INDEX = D_PUB
    D_PUB += 1

    # One player technically may have a smaller private space than the other,
    # but we just take the maximum for simplicity
    D_PRI = max(d1, d2) * sides

    # And then a feature to describe from whos perspective we
    # are given the private information
    PRI_INDEX = D_PRI
    D_PRI += 1

    return D_PUB, D_PRI, N_ACTIONS, LIE_ACTION, CUR_INDEX, PRI_INDEX

class Game:
    def __init__(self, model, d1, d2, sides, variant):
        self.model = model
        self.D1 = d1
        self.D2 = d2
        self.SIDES = sides
        self.VARIANT = variant

        self.D_PUB, self.D_PRI, self.N_ACTIONS, self.LIE_ACTION, self.CUR_INDEX, self.PRI_INDEX = \
                calc_args(d1, d2, sides, variant)

    def make_regrets(self, priv, state, last_call):
        """
            priv: Private state, including the perspective for the scores
            state: Public statte
            last_call: Last action taken by a player. Returned regrets will be for actions after this one.
        """

        if priv[self.PRI_INDEX] != state[self.CUR_INDEX]:
            print('Warning: Regrets are not with respect to current player')

        # Number of child nodes
        n_actions = self.N_ACTIONS - last_call - 1

        # One for the current state, and one for each child
        batch = state.repeat(n_actions + 1, 1)

        for i in range(n_actions):
            batch[i + 1][self.CUR_INDEX] *= -1
            batch[i + 1][i + last_call + 1] = 1

        priv_batch = priv.repeat(n_actions + 1, 1)

        v, *vs = list(self.model(priv_batch, batch))
        return [max(vi - v, 0) for vi in vs]
        # The Hedge method
        #return [math.exp(10*(vi - v)) for vi in vs]


    def evaluate(self, r1, r2, last_call):
        # Players have rolled r1, and r2.
        # Previous actions are `state`
        # Player `caller` just called lie. (This is not included in last_call)
        # Returns True if the call is good, false otherwise

        # Calling lie immediately is an error, so we pretend the
        # last call was good to punish the player.
        if last_call == -1:
            return True

        n, d = divmod(last_call, self.SIDES)
        n, d = n + 1, d + 1  # (0, 0) means 1 of 1s

        cnt = Counter(r1 + r2)
        if self.VARIANT == "normal":
            actual = cnt[d]
        if self.VARIANT == "joker":
            actual = cnt[d] + cnt[1] if d != 1 else cnt[d]
        if self.VARIANT == "stairs":
            if all(r == i + 1 for r, i in zip(r1, range(self.SIDES))):
                actual += 2 * len(r1) - r1.count(d)
            if all(r == i + 1 for r, i in zip(r2, range(self.SIDES))):
                actual += 2 * len(r2) - r1.count(d)
        #print(f'{r1=}, {r2=}, {last_call=}, {(n, d)=}, {actual=}', actual >= n)
        return actual >= n

    def sample_action(self, priv, state, last_call, eps):
        regrets = self.make_regrets(priv, state, last_call)
        for i in range(len(regrets)):
            regrets[i] += eps
        action = next(
            iter(torch.utils.data.WeightedRandomSampler(regrets, num_samples=1))
        ) + last_call + 1
        return action

    def apply_action(self, state, action):
        new_state = state.clone()
        new_state[action] = 1
        new_state[self.CUR_INDEX] *= -1
        return new_state

    def make_priv(self, roll, player):
        priv = torch.zeros(self.D_PRI)
        assert player in [0, 1]
        priv[self.PRI_INDEX] = 1 - 2*player
        for i, r in enumerate(roll):
            priv[i * self.SIDES + r - 1] = 1
        return priv

    def make_state(self, player=1):
        state = torch.zeros(self.D_PUB)
        state[self.CUR_INDEX] = 1
        return state

    def get_cur(self, state):
        # cur is the current player, in {1,-1}
        return (1-int(state[self.CUR_INDEX]))//2 # now in {0,1}

    def rolls(self, player):
        assert player in [0, 1]
        n_faces = self.D1 if player == 0 else self.D2
        return itertools.product(range(1, self.SIDES + 1), repeat=n_faces)

    def get_last_call(self, state):
        ids = tuple((state[:self.CUR_INDEX] == 1).nonzero(as_tuple=True)[0])
        if not ids:
            return -1
        return int(ids[-1])

