from collections import Counter
from collections import defaultdict
import itertools
import sys
from ortools.linear_solver import pywraplp
import fractions

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

solver = pywraplp.Solver('', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
lvs = {d: solver.NumVar(-solver.infinity(), solver.infinity(), 'lambda {}'.format(d)) for d in ROLLS}
xvs = {(d,x): solver.NumVar(0, 1, 'x {},{}'.format(d,x)) for x in xs for d in ROLLS}

# The thing to maximize/minimize: all the lambdas
objective = solver.Objective()
for lv in lvs.values():
    objective.SetCoefficient(lv, 1)
objective.SetMaximization()

# Equalities
# Making sure probabilities sum to 1.
# Variables are associated with the history they lead to.
# Top variables are set to sum to state(the empty history),
# which we should make sure is 1.
for d in ROLLS:
    # The root sums to 1
    constraint = solver.Constraint(1, 1)
    constraint.SetCoefficient(xvs[d,()], 1)
    # Make siblings sum to their parent
    for hist in non_leafs:
        if len(hist) % 2 == 0:
            constraint = solver.Constraint(0, 0)
            constraint.SetCoefficient(xvs[d,hist[:-1]], -1)
            for call in possible_calls(hist):
                constraint.SetCoefficient(xvs[d,hist+(call,)], 1)

def score(ds, hist):
    ''' Get the score in {-1,1} relative to player 1 '''
    assert hist and hist[-1] is SNYD
    # Player 2 called snyd
    if len(hist) % 2 == 0:
        res = is_correct_call(ds, hist[-2])
    # Player 1 called snyd
    else:
        res = not is_correct_call(ds, hist[-2])
    return int(res)*2 - 1

# The bounds
# Go through all final states, and consider the expected result:
#   x_1*y_1*score_1 + x_2*y_2*score_2 + ...
# We need to average over d1, since it is unknown to us.
for d2 in ROLLS:
    for hist in lxs:
        constraint = solver.Constraint(0, solver.infinity())
        #constraint.SetCoefficient(lvs[d2], -len(ROLLS)**2)
        constraint.SetCoefficient(lvs[d2], -1)
        #print('Constraint')
        #print('-1', d2)
        for d1 in ROLLS:
            leaf_hist = hist if hist[-1] is SNYD else hist + (SNYD,)
            val = score(d1 + d2, leaf_hist)
            constraint.SetCoefficient(xvs[d1, hist], val)
            #print(score(d1+d2,leaf_hist), d1, leaf_hist)

# Formatting of solution
def scall(call):
    if call is None:
        return "snyd"
    return '{}x{}'.format(*call)

def findCallProb(d1, hist):
    ''' Return the probability that player 1 did the last move of hist '''
    assert len(hist) % 2 == 1
    xhis = xvs[d1, hist].solution_value()
    xpar = xvs[d1, hist[:-2]].solution_value()
    return xhis/xpar if xpar > 1e-10 else 0

def findP2Call(d2, hist):
    ''' Find the best call for p2, choosing the optimal deterministic counter strategy '''
    assert len(hist) % 2 == 1
    if sum(findCallProb(d1,hist) for d1 in ROLLS) < 1e-6:
        #if d2 == (0,) and hist == ((1,1),):
        #    print('findP2Call called on impossible history')
        return possible_calls(hist)[0]
    pd1s = estimateP1Rolls(hist)
    if d2 == (0,) and hist == ((1,1),):
        pass
        #print('pd1s', pd1s)
        #print('scores', [sum(p*stateValue(d1, d2, hist+(call,)) for p, d1 in zip(pd1s,ROLLS)) for p,d1 in zip(pd1s,ROLLS)])
    return min(possible_calls(hist), key=lambda call:
            sum(p*stateValue(d1, d2, hist+(call,)) for p, d1 in zip(pd1s,ROLLS)))


def stateValue(d1, d2, hist):
    ''' Return expected payoff for player 1 '''
    if hist and hist[-1] is SNYD:
        res = score(d1+d2, hist)
    # Player 1
    elif len(hist) % 2 == 0:
        res = sum(stateValue(d1, d2, hist+(call,))
                * findCallProb(d1, hist+(call,))
                for call in possible_calls(hist))
    # Player 2
    elif len(hist) % 2 == 1:
        p2call = findP2Call(d2, hist)
        res = stateValue(d1, d2, hist+(p2call,))
    #print('stateValue({}, {}, {}) = {}'.format(d1, d2, hist, res))
    return res

def estimateP1Rolls(hist):
    assert len(hist) % 2 == 1
    # Simple bayes
    prob_hist_given_d = [findCallProb(d1, hist) for d1 in ROLLS]
    if sum(prob_hist_given_d) < 1e-10:
        return [1/len(ROLLS) for _ in ROLLS]
    return [p/sum(prob_hist_given_d) for p in prob_hist_given_d]

def main():
    # Solve the system
    status = solver.Solve()
    if status != solver.OPTIMAL:
        print('Status:', status)
        return

    print('Trees:')
    for d1 in ROLLS:
        for hist in histories:
            # At root, print the roll value
            if not hist:
                avgValue = sum(stateValue(d1, d2, ()) for d2 in ROLLS)/len(ROLLS)
                values = [round(stateValue(d1, d2, ()), ndigits=4) for d2 in ROLLS]
                print('Roll: {}, Expected: {}, Values: {}'.format(d1, avgValue, values))
                continue
            # If a parent has zero probability, don't go there
            if any(findCallProb(d1, hist[:j]) < 1e-8 for j in range(1,len(hist)+1,2)):
                continue
            s = '|  '*len(hist) + (scall(hist[-1]) if hist else 'root')
            if hist in xs:
                prob = round(findCallProb(d1, hist), ndigits=4)
                print('{} p={}'.format(s, prob))
            else:
                tag = ''.join('_*'[hist[-1] == findP2Call(d2,hist[:-1])]
                        for d2 in ROLLS)
                print(s, tag)

    # Test feasability
    print('Score by d2')
    for d2 in ROLLS:
        expected_score = sum(stateValue(d1, d2, ()) for d1 in ROLLS)/len(ROLLS)
        print('{} strat {} >= lambda {}'.format(d2, expected_score, lvs[d2].solution_value()))

    #print('Lambdas', [(d2,lv.solution_value()) for d2,lv in lvs.items()])
    print('Lambdas', ', '.join(sorted('{}: {}'.format(d2, lv.solution_value()) for d2, lv in lvs.items())))

    res = sum(lv.solution_value() for lv in lvs.values())
    print('Score:', fractions.Fraction.from_float(res).limit_denominator())

    res2 = sum(stateValue(d1, d2, ()) for d1 in ROLLS for d2 in ROLLS)/len(ROLLS)**2
    print('Score:', fractions.Fraction.from_float(res2).limit_denominator())

if __name__ == '__main__':
    main()

