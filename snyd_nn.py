# -*- coding: utf-8 -*-
import random
import torch
from torch import nn
import itertools
import numpy as np
import math
from collections import Counter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('d1', type=int, help='Number of dice for player 1')
parser.add_argument('d2', type=int, help='Number of dice for player 1')
parser.add_argument('--sides', type=int, default=6, help='Number of sides on the dice')
parser.add_argument('--joker', action='store_true', help='1s count for anything')
parser.add_argument('--stairs', action='store_true', help='A chain of numbers count for twice than many anything')

parser.add_argument('--eps', type=float, default=1e-2, help='Added to regrets for exploration')
parser.add_argument('--layers', type=int, default=4, help='Number of fully connected layers')
parser.add_argument('--layer-size', type=int, default=100, help='Number of neurons per layer')
parser.add_argument('--lr-decay', type=float, default=.999, help='LR = (lr-decay)**t')
parser.add_argument('--w', type=float, default=.01, help='weight decay')

args = parser.parse_args()



D1 = args.d1
D2 = args.d2
SIDES = args.sides
VARIANT = "normal"
if args.joker:
    VARIANT = "joker"
if args.stairs:
    VARIANT = "stairs"

EPSILON = args.eps # Added to all regrets when computing policy

# Maximum call is (D1+D2) 6s
D_PUB = (D1 + D2) * SIDES
if VARIANT == "stairs":
    D_PUB = 2 * (D1 + D2) * SIDES
# We also have a feature that means "game has been called"
D_PUB += 1

# Total number of actions in the game
N_ACTIONS = D_PUB

# Add one extra feature to keep track of who's to play next
CUR_INDEX = D_PUB
D_PUB += 1


# One player technically may have a smaller private space than the other,
# but we just take the maximum for simplicity
D_PRI = max(D1, D2) * SIDES

# And then a feature to describe from whos perspective we
# are given the private information
PRI_INDEX = D_PRI
D_PRI += 1

# Model : (private state, public state) -> value
class Net(torch.nn.Module):
    def __init__(self, hiddens=(100,)):
        super().__init__()

        # Bilinear can't be used inside nn.Sequantial
        # https://github.com/pytorch/pytorch/issues/37092
        self.layer0 = torch.nn.Bilinear(D_PRI, D_PUB, hiddens[0])

        layers = [torch.nn.ReLU()]
        for size0, size1 in zip(hiddens, hiddens[1:]):
            layers += [torch.nn.Linear(size0, size1), torch.nn.ReLU()]
        layers += [torch.nn.Linear(hiddens[-1], 1), nn.Tanh()]
        self.seq = nn.Sequential(*layers)

    def forward(self, priv, pub):
        joined = self.layer0(priv, pub)
        return self.seq(joined)

model = Net([100] * args.layers)
value_loss = torch.nn.MSELoss()


def evaluate(r1, r2, last_call):
    # Players have rolled r1, and r2.
    # Previous actions are `state`
    # Player `caller` just called lie. (This is not included in last_call)
    # Returns True if the call is good, false otherwise

    # Calling lie immediately is an error, so we pretend the
    # last call was good to punish the player.
    if last_call == -1:
        return True

    n, d = divmod(last_call, SIDES)
    n, d = n + 1, d + 1  # (0, 0) means 1 of 1s

    cnt = Counter(r1 + r2)
    if VARIANT == "normal":
        actual = cnt[d]
    if VARIANT == "joker":
        actual = cnt[d] + cnt[1] if d != 1 else cnt[d]
    if VARIANT == "stairs":
        if all(r == i + 1 for r, i in zip(r1, range(SIDES))):
            actual += 2 * len(r1) - r1.count(d)
        if all(r == i + 1 for r, i in zip(r2, range(SIDES))):
            actual += 2 * len(r2) - r1.count(d)
    #print(f'{r1=}, {r2=}, {last_call=}, {(n, d)=}, {actual=}', actual >= n)
    return actual >= n


def make_regrets(priv, state, last_call):
    # Number of child nodes
    n_actions = N_ACTIONS - last_call - 1

    # One for the current state, and one for each child
    batch = state.repeat(n_actions + 1, 1)

    for i in range(n_actions):
        batch[i + 1][CUR_INDEX] *= -1
        batch[i + 1][i + last_call + 1] = 1

    priv_batch = priv.repeat(n_actions + 1, 1)

    v, *vs = list(model(priv_batch, batch))
    return [max(vi - v, 0) for vi in vs]
    # The Hedge method
    #return [math.exp(10*(vi - v)) for vi in vs]


def mean_score(state):
    # Mean score in state over all rolls
    total = 0
    for r1 in rolls(D1):
        priv = torch.zeros(D_PRI)
        priv[PRI_INDEX] = 1 # The private state is from the perspective of player 1
        for i, r in enumerate(r1):
            priv[i * SIDES + r - 1] += 1
        v = model(priv, state)
        rs = np.array(make_regrets(priv, state, last_call=-1))
        if rs.sum() != 0:
            rs /= rs.sum()
        print(r1, v, ' '.join(f'{r:.2f}' for r in rs))
        total += v
    return total / len(list(rolls(D1)))


@torch.no_grad()
def play(r1, r2, replay_buffer):
    privs = [torch.zeros(D_PRI) for _ in range(2)]
    privs[0][PRI_INDEX] = 1
    privs[1][PRI_INDEX] = -1
    for priv, roll in zip(privs, (r1, r2)):
        for i, r in enumerate(roll):
            priv[i * SIDES + r - 1] += 1

    def play_inner(last_call, state):
        # last_call is -1 if there has been no calls
        # cur=state[0] is the current player, in {1,-1}
        cur = (1-int(state[CUR_INDEX]))//2 # now in {0,1}
        regrets = make_regrets(privs[cur], state, last_call)
        #print(regrets)
        # Add some noise for exploration
        #print([float(r) for r in regrets])
        for i in range(len(regrets)):
            regrets[i] += EPSILON
        action = next(
            iter(torch.utils.data.WeightedRandomSampler(regrets, num_samples=1))
        )
        action += last_call + 1

        # Create state for after action has been done, and cur switched
        new_state = state.clone()
        new_state[action] = 1
        new_state[CUR_INDEX] *= -1

        if action == N_ACTIONS - 1:  # If the player calls "lie"
            if evaluate(r1, r2, last_call):
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
        state = torch.zeros(D_PUB)
        state[CUR_INDEX] = 1
        play_inner(-1, state)


def rolls(n_faces):
    return itertools.product(range(1, SIDES + 1), repeat=n_faces)


optimizer = torch.optim.AdamW(model.parameters(), weight_decay=args.w)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
for t in range(100_000):
    replay_buffer = []
    for r1 in rolls(D1):
        for r2 in rolls(D2):
            play(r1, r2, replay_buffer)

    for priv, state, res in replay_buffer:
        assert state[:-1].sum() % 2 == (1-state[-1])//2
        # Check cases where lie was called immediately
        if all(state[i] == 0 for i in range(N_ACTIONS-1)) and state[N_ACTIONS-1] == 1:
            assert state[-1] == -1 # Must be second players move, since only one action is made
            if priv[PRI_INDEX] == 1:
                assert res == -1
            if priv[PRI_INDEX] == -1:
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
        pub = torch.zeros(D_PUB)
        pub[CUR_INDEX] = 1

        print(t, loss.item(), mean_score(pub))

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
