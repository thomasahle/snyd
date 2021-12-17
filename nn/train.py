# -*- coding: utf-8 -*-
import random
import torch
from torch import nn
import itertools
import numpy as np
import math
from collections import Counter
import argparse
import os

from snyd import Net, Game, calc_args

parser = argparse.ArgumentParser()
parser.add_argument("d1", type=int, help="Number of dice for player 1")
parser.add_argument("d2", type=int, help="Number of dice for player 2")
parser.add_argument("--sides", type=int, default=6, help="Number of sides on the dice")
parser.add_argument("--variant", type=str, default="normal", help="one of normal, joker, stairs")
parser.add_argument("--eps", type=float, default=1e-2, help="Added to regrets for exploration")
parser.add_argument("--layers", type=int, default=4, help="Number of fully connected layers")
parser.add_argument("--layer-size", type=int, default=100, help="Number of neurons per layer")
parser.add_argument("--lr-decay", type=float, default=0.999, help="LR = (lr-decay)**t")
parser.add_argument("--w", type=float, default=0.01, help="weight decay")
parser.add_argument("--path", type=str, default="model.pt", help="Where to save checkpoints")

args = parser.parse_args()


# Check if there is a model we should continue training
if os.path.isfile(args.path):
    checkpoint = torch.load(args.path)
    print(f"Using args from {args.path}")
    old_path = args.path
    args = checkpoint["args"]
    args.path = old_path
else:
    checkpoint = None

# Model : (private state, public state) -> value
D_PUB, D_PRI, *_ = calc_args(args.d1, args.d2, args.sides, args.variant)
model = Net(D_PRI, D_PUB, [args.layer_size] * args.layers)
game = Game(model, args.d1, args.d2, args.sides, args.variant)

if checkpoint is not None:
    print("Loading previous model for continued training")
    model.load_state_dict(checkpoint["model_state_dict"])


@torch.no_grad()
def play(r1, r2, replay_buffer):
    privs = [game.make_priv(r1, 0), game.make_priv(r2, 1)]

    def play_inner(last_call, state):
        cur = game.get_cur(state)
        action = game.sample_action(privs[cur], state, last_call, args.eps)

        # Create state for after action has been done, and cur switched
        new_state = game.apply_action(state, action)

        if action == game.LIE_ACTION:
            # If the last call is good it mean we lost (because we called lie)
            res = -1 if game.evaluate(r1, r2, last_call) else 1

            replay_buffer.append((privs[cur], new_state, res))
            replay_buffer.append((privs[1 - cur], new_state, -res))

            replay_buffer.append((privs[cur], state, res))
            replay_buffer.append((privs[1 - cur], state, -res))

            return res

        # Just classic min/max stuff
        res = -play_inner(action, new_state)

        replay_buffer.append((privs[cur], state, res))
        replay_buffer.append((privs[1 - cur], state, -res))

        return res

    with torch.no_grad():
        # last_call is -1 if there has been no calls
        play_inner(-1, game.make_state())


def print_strategy(state):
    total_v = 0
    total_cnt = 0
    for r1, cnt in sorted(Counter(game.rolls(0)).items()):
        priv = game.make_priv(r1, 0)
        v = model(priv, state)
        rs = np.array(game.make_regrets(priv, state, last_call=-1))
        if rs.sum() != 0:
            rs /= rs.sum()
        strat = []
        for action, prob in enumerate(rs):
            n, d = divmod(action, game.SIDES)
            n, d = n + 1, d + 1
            if d == 1:
                strat.append(f"{n}:")
            strat.append(f"{prob:.2f}")
        print(r1, f"{float(v):.4f}".rjust(7), f"({cnt})", " ".join(strat))
        total_v += v
        total_cnt += cnt
    print(f"Mean value: {total_v / total_cnt}")


def train():
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=args.w)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    value_loss = torch.nn.MSELoss()
    all_rolls = list(itertools.product(game.rolls(0), game.rolls(1)))
    for t in range(100_000):
        replay_buffer = []

        BS = 100  # Number of rolls to include
        for r1, r2 in (all_rolls if len(all_rolls) <= BS else random.sample(all_rolls, BS)):
            play(r1, r2, replay_buffer)

        random.shuffle(replay_buffer)
        privs, states, y = zip(*replay_buffer)
        y_pred = model(torch.vstack(privs), torch.vstack(states))

        # Compute and print loss
        y = torch.tensor(y, dtype=torch.float).reshape(-1, 1)
        loss = value_loss(y_pred, y)
        print(t, loss.item())

        if t % 5 == 0:
            with torch.no_grad():
                print_strategy(game.make_state())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (t + 1) % 10 == 0:
            print(f"Saving to {args.path}")
            torch.save(
                {
                    "epoch": t,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": args,
                },
                args.path,
            )


train()
