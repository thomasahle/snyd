from collections import Counter
from collections import defaultdict
import itertools
import numpy as np
import scipy.optimize
import scipy.linalg
import sys

if len(sys.argv) != 3:
    print('Run {} [dice] [sides]'.format(sys.argv[0]))
    sys.exit()
else:
    DICE = int(sys.argv[1])
    SIDES = int(sys.argv[2])

CALLS = [(count, side)
        for count in range(1, 2*DICE+1)
        for side in range(SIDES)]

ROLLS = list(itertools.product(range(SIDES), repeat=DICE))

SNYD = None

def possible_calls(hist):
    if not hist:
        return CALLS
    if hist[-1] is SNYD:
        return []
    return [call for call in CALLS if call > hist[-1]] + [SNYD]

def is_correct_call(ds, call):
    count, side = call
    return not bool(Counter({side: count}) - Counter(ds))

def is_leaf(hist):
    assert not hist or hist[0] is not SNYD, "SNYD can't be first call"
    return hist and hist[-1] is SNYD

def histories(hist=()):
    yield hist
    if not is_leaf(hist):
        for call in possible_calls(hist):
            yield from histories(hist+(call,))
histories = list(histories())

leafs = [hist for hist in histories if is_leaf(hist)]
non_leafs = [hist for hist in histories if not is_leaf(hist)]

# xs are the states after a move by player 1
xs = [h for h in histories if not h or len(h)%2==1]
# lxs are basically every state but the root.
# This is because they include leafs where player 2 made the last move.
# In these cases we want y_last*x_last and every x but root is an x_last
lxs = [h if len(h)%2==1 else h[:-1] for h in leafs]
# Because we can nearly always call snyd,
# the x leafs should be all the xs, but the root
assert set(xs) - set(lxs) == set([()])

ids = defaultdict(lambda: len(ids))
ids['l'] = 0

# Now for the linear programming
# Minimize: c^T * x
# Subject to: A_ub * x <= b_ub
# A_eq * x == b_eq

x_cnt = len(xs)
var_cnt = (1 + x_cnt) * len(ROLLS)

# The thing to maximize/minimize: all the lambdas
c = np.zeros(var_cnt)
for i in range(0, var_cnt, 1 + x_cnt):
    c[i] = 1

# Equalities
eqs = []
# Making sure probabilities sum to 1.
# Variables are associated with the history they lead to.
# Top variables are set to sum to state(the empty history),
# which we should make sure is 1.
for hist in non_leafs:
    if len(hist) % 2 == 0:
        lhs = [hist+(call,) for call in possible_calls(hist)]
        eqs.append((lhs, [hist[:-1]]))
Es = []
es = []
for d1 in ROLLS:
    E = np.zeros((len(eqs)+1, 1+x_cnt))
    Es.append(E)
    e = np.zeros(len(eqs)+1)
    es.append(e)
    for i, (lhs, rhs) in enumerate(eqs):
        for h in lhs:
            E[i, ids[h]] = 1
        for h in rhs:
            E[i, ids[h]] = -1
    # x0 = 1
    E[len(eqs), ids[()]] = 1
    e[len(eqs)] = 1
E = scipy.linalg.block_diag(*Es)
ev = np.hstack(es)
print(E)
print(ev)

# The bounds
Us = []
for d1 in ROLLS:
    Us.append([])
    for d2 in ROLLS:
        U = np.zeros((len(lxs), 1+x_cnt))
        print((len(lxs), 1+x_cnt))
        for i, h in enumerate(lxs):
            # If player one called snyd, we win if previous call was wrong.
            if h[-1] is SNYD:
                res = not is_correct_call(d1+d2, h[-2])
            # Otherwise player two called snyd after this state, in which
            # case we win if our call was correct.
            else:
                res = is_correct_call(d1+d2, h[-1])
            res = int(res)*2-1
            U[i, ids[h]] = res
            # lhs >= rhs is equiv to lhs - rhs >= 0
            U[i, ids['l']] = -1
            if sum(U[i]) == -1:
                print('wat', ids[h], res)
        Us[-1].append(U)
        print('Bounds', d1, d2)
        print(U)
U = np.vstack(np.hstack(Uc for Uc in Ur) for Ur in Us)
#np.set_printoptions(threshold=10**4)
#print(U)

# Hard bounds. Lambdas can go in both directions
# At least if I let losses count as -1 rather than 0. Does that make a difference?
bounds = [(0,1) for _ in range(var_cnt)]
for i in range(0, var_cnt, 1+x_cnt):
    bounds[i] = (-1, 1)


# p
# Lower bounds: Ensures positive probabilities
# Total utility

# Lower bounds: Ensures positive probabilities
# With scipy we can do global bounds, which we set to 0,1.
# That should be fine even for the lambda variables.

print('Shapes')
print('c', c.shape)
print('U', U.shape)
print('E', E.shape)
print('ev', ev.shape)

res = scipy.optimize.linprog(
        c = -c,
        A_ub = -U,
        b_ub = np.zeros(U.shape[0]),
        A_eq = E,
        b_eq = ev,
        bounds = bounds,
        options = {'disp': True, 'maxiter': 10**8})

def scall(call):
    if call is None:
        return "snyd"
    return '{}x{}'.format(*call)

print('Trees:')
for i, d1 in enumerate(ROLLS):
    print('Roll: {}, Value: {}'.format(d1, res.x[i*(1+x_cnt) + ids['l']]))
    prune = set()
    for hist in histories:
       prob = 0
       if hist in xs:
            xhis = res.x[i*(1+x_cnt) + ids[hist]]
            xpar = res.x[i*(1+x_cnt) + ids[hist[:-2]]]
            prob = np.round(xhis/xpar, decimals=3) if xpar else 0
            if prob == 0:
                prune.add(hist)
       if any(hist[:j] in prune for j in range(len(hist)+1)):
            continue
       print('|  '*len(hist), scall(hist[-1]) if hist else 'root', end=' ')
       if prob:
            print('p={}'.format(prob), end=' ')
       #if hist:
       #   if hist[-1] is SNYD:
       #      if not is_correct_call((0,0), hist[-2]):
       #         print('+', end='')
       #   elif is_correct_call((0,0), hist[-1]):
       #      print('+', end='')
       print()

res.x[np.abs(res.x) < 1e-5] = 0
res.x[res.x > 1-1e-5] = 1
res.x[res.x < -1+1e-5] = -1
print(res)
print([res.x[i] for i in range(0, var_cnt, 1+x_cnt)])
print('Cx', c.dot(res.x))
print('Ux', U.dot(res.x))
print('Ex', E.dot(res.x))
