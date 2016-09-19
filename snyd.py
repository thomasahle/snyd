from collections import Counter
from collections import defaultdict
import itertools
import numpy as np
import scipy.optimize
import scipy.linalg
import scipy.sparse as sp
import sys

if len(sys.argv) != 3:
    print('Run {} [dice] [sides]'.format(sys.argv[0]))
    sys.exit()
else:
    DICE = int(sys.argv[1])
    SIDES = int(sys.argv[2])

################################################################
# Game definition
################################################################

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


################################################################
# Write as matrix
################################################################

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
            if np.sum(U[i]) == -1:
                print('wat', ids[h], res)
        Us[-1].append(U)
        print('Bounds', d1, d2)
        print(U)
U = np.bmat(Us)
# Add requirement that we win more with higher rolls.
# This makes strategies more believable, if possible.
#for i in range(len(ROLLS)-1):
#    lhs = np.zeros(var_cnt)
#    lhs[(i+1)*(1+x_cnt)] = 1
#    lhs[(i)*(1+x_cnt)] = -1
#    U = np.vstack((U, lhs))


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

def findStateValue(d1, d2, hist):
    ''' value and d1, d2 are relative to the current player '''
    if hist and hist[-1] is SNYD:
        return int(not is_correct_call(d1+d2, hist[-2]))*2-1
    res = 0
    #for call in possible_calls(hist):
    #    res += findCallProb(d1, d2, hist+(call,)) \
    #         *-findStateValue(d2, d1, hist+(call,))
    return res

def findCallProb(d, hist):
    if not hist:
        return 1
    if len(hist) % 2 == 1:
        i = ROLLS.index(d)
        xhis = res.x[i*(1+x_cnt) + ids[hist]]
        xpar = res.x[i*(1+x_cnt) + ids[hist[:-2]]]
        return xhis/xpar if xpar > 1e-10 else 0
    else:
        bestv = -1
        best = []
        for call in possible_calls(hist[:-1]):
            # Nej, man er n'dt til at beregne hvad sandsynligheden for spiller 1s terninger er p[ dette sted.
            #P[d2 = D | pos] = P[d2=D]/P[pos] * P[pos | d2 = D]
            v = sum(findStateValue(d, d2, hist[:-1]+(call,)) for d2 in ROLLS)/len(ROLLS)
            if v > bestv:
                best = []
            if v >= bestv:
                best.append(call)
                bestv = v
        if hist[-1] not in best:
            return 0
        return 1/len(best)

print('Trees:')
for i, d1 in enumerate(ROLLS):
    prune = set()
    for hist in histories:
        if not hist:
            print('Roll: {}, Value: {}'.format(d1, res.x[i*(1+x_cnt) + ids['l']]))
            continue
        if any(hist[:j] in prune for j in range(len(hist)+1)):
            continue
        s = '|  '*len(hist) + (scall(hist[-1]) if hist else 'root')
        if hist in xs:
            prob = findCallProb(d1, hist)
            prob = np.round(prob, decimals=4)
            if hist in xs and prob == 0:
                prune.add(hist)
                continue
            print('{} p={}'.format(s, prob))
        else:
            #ps = [findCallProb(roll, d1, hist) for roll in ROLLS]
            #ps = [str(np.round(prob, decimals=4)) for prob in ps]
            #print('{} p=({})'.format(s, ' '.join(ps)))
            print(s)

#res.x[np.abs(res.x) < 1e-5] = 0
#res.x[res.x > 1-1e-5] = 1
#res.x[res.x < -1+1e-5] = -1
#print(res)
#print([res.x[i] for i in range(0, var_cnt, 1+x_cnt)])
print('Cx', c.dot(res.x))
#print('Ux', U.dot(res.x))
#print('Ex', E.dot(res.x))

#totalValue = 0
#for d1 in ROLLS:
#    for d2 in ROLLS:
#        v = findStateValue(d1, d2, ())
#        print('Value of rolls {}, {}: {}'.format( d1, d2, v))
#        totalValue += v
#print('total value:', totalValue)

