# -*- coding: utf-8 -*-
import random
import torch
from torch import nn
import itertools
import numpy as np
import math
from collections import Counter
import argparse

from snyd import Net, Game, calc_args

parser = argparse.ArgumentParser()
parser.add_argument('d1', type=int, help='Number of dice for player 1')
parser.add_argument('d2', type=int, help='Number of dice for player 2')
parser.add_argument('--sides', type=int, default=6, help='Number of sides on the dice')
parser.add_argument('--variant', type=str, default='normal', help='one of normal, joker, stairs')
parser.add_argument('--eps', type=float, default=1e-2, help='Added to regrets for exploration')

parser.add_argument('--layers', type=int, default=4, help='Number of fully connected layers')
parser.add_argument('--layer-size', type=int, default=100, help='Number of neurons per layer')
parser.add_argument('--lr-decay', type=float, default=.999, help='LR = (lr-decay)**t')
parser.add_argument('--w', type=float, default=.01, help='weight decay')

parser.add_argument('--path', type=str, default='model.pt', help='Where to save checkpoints')

args = parser.parse_args()

# Model : (private state, public state) -> value
D_PUB, D_PRI, *_ = calc_args(args.d1, args.d2, args.sides, args.variant)
print(f'{D_PUB=}, {D_PRI=}')
model = Net(D_PRI, D_PUB, [args.layer_size] * args.layers)
game = Game(model, args.d1, args.d2, args.sides, args.variant)
value_loss = torch.nn.MSELoss()


def mean_score(state):
    # Mean score in state over all rolls
    total = 0
    for r1 in game.rolls(0):
        priv = game.make_priv(r1, 0)
        v = model(priv, state)
        rs = np.array(game.make_regrets(priv, state, last_call=-1))
        if rs.sum() != 0:
            rs /= rs.sum()
        print(r1, v, ' '.join(f'{r:.2f}' for r in rs))
        total += v
    return total / len(list(game.rolls(0)))


@torch.no_grad()
def play(r1, r2, replay_buffer):
    privs = [
        game.make_priv(r1, 0),
        game.make_priv(r2, 1)
    ]

    def play_inner(last_call, state):
        # last_call is -1 if there has been no calls
        cur = game.get_cur(state)
        action = game.sample_action(privs[cur], state, last_call, args.eps)

        # Create state for after action has been done, and cur switched
        new_state = game.apply_action(state, action)

        if action == game.N_ACTIONS - 1:  # If the player calls "lie"
            if game.evaluate(r1, r2, last_call):
                # If the last_call was actually good, that means our calling "lie" failed.
                res = -1
            else: res = 1

            # If the call was good, the player that just played won (otherwise lost)
            if last_call == -1:
                assert res == -1
                assert cur == 0

            #if last_call == N_ACTIONS-2:
                #print(f'{privs[cur]=}, {state=}, {res=}')

            replay_buffer.append((privs[cur], state, res))

            # The player whos turn it was about to be (after the call) lost (otherwise won)
            replay_buffer.append((privs[1-cur], new_state, -res))

            # Extras?
            replay_buffer.append((privs[1-cur], state, -res))

            # Actually, this is the one that's going to discourage this
            #if last_call == -1:
                #print('cpnr', cur, privs[cur], new_state, res)
            replay_buffer.append((privs[cur], new_state, res))

            return res

        # Just classic min/max stuff
        res = -play_inner(action, new_state)

        # The back-propegated result is stored with the original state
        replay_buffer.append((privs[cur], state, res))

        # TODO: If we start including perspective in the state, we should
        # perhaps save the result seen from both sides here?
        replay_buffer.append((privs[1-cur], state, -res))
        return res

    with torch.no_grad():
        play_inner(-1, game.make_state())




def train():
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=args.w)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    for t in range(100_000):
        replay_buffer = []
        for r1 in game.rolls(0):
            for r2 in game.rolls(1):
                play(r1, r2, replay_buffer)

        # Just some debug stuff
        with torch.no_grad():
            for priv, state, res in replay_buffer:
                assert state[:-1].sum() % 2 == (1-state[-1])//2
                # Check cases where lie was called immediately
                if all(state[i] == 0 for i in range(game.N_ACTIONS-1)) and state[game.N_ACTIONS-1] == 1:
                    assert state[-1] == -1 # Must be second players move, since only one action is made
                    if priv[game.PRI_INDEX] == 1:
                        assert res == -1
                    if priv[game.PRI_INDEX] == -1:
                        assert res == 1

        random.shuffle(replay_buffer)
        privs, states, y = zip(*replay_buffer)
        y = torch.tensor(y, dtype=torch.float).reshape(-1, 1)
        privs = torch.vstack(privs)  # Must be tensor
        states = torch.vstack(states)  #
        y_pred = model(privs, states)

        # Compute and print loss
        loss = value_loss(y_pred, y)
        with torch.no_grad():
            print(t, loss.item(), mean_score(game.make_state()))

            # Debugging a call problem
            # pub[N_ACTIONS-1] = 1 # Call
            # pub[CUR_INDEX] *= -1 # Next player
            # call_vs = []
            # for i in range(SIDES):
            #     priv = torch.zeros(D_PRI)
            #     priv[PRI_INDEX] = 1
            #     priv[i] = 1
            #     v = model(priv, pub)
            #     call_vs.append(v)
            # print(call_vs) # These should all be lost from the perspective of player 1

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (t+1) % 10 == 0:
            print(f'Saving to {args.path}')
            torch.save({
                        'epoch': t,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'args': args
                        }, args.path)

train()
