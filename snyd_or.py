from collections import Counter
from collections import defaultdict
import itertools
import sys
from ortools.linear_solver import pywraplp
import fractions
from functools import lru_cache


if len(sys.argv) < 4:
    print('Run {} [dice1] [dice2] [sides] mode'.format(sys.argv[0]))
    sys.exit()
else:
    DICE1 = int(sys.argv[1])
    DICE2 = int(sys.argv[2])
    SIDES = int(sys.argv[3])

NORMAL, JOKER, STAIRS = range(3)
if len(sys.argv) >= 5:
    mode = {'normal': NORMAL, 'joker': JOKER, 'stairs': STAIRS}[sys.argv[4]]
else:
    mode = NORMAL

################################################################
# Game definition
################################################################

if mode == NORMAL:
    CALLS = [(count, side)
            for count in range(1, DICE1+DICE2+1)
            for side in range(1, SIDES+1)]
if mode == JOKER:
    # With jokers we can't call 1
    CALLS = [(count, side)
        for count in range(1, DICE1+DICE2+1)
        for side in range(2, SIDES+1)]
if mode == STAIRS:
    # With stairs we can call up to four sixes...
    CALLS = [(count, side)
        for count in range(1, 2*(DICE1+DICE2)+1)
        for side in range(2, SIDES+1)]

ROLLS1 = list(itertools.product(range(1,SIDES+1), repeat=DICE1))
ROLLS2 = list(itertools.product(range(1,SIDES+1), repeat=DICE2))

SNYD = None

def possible_calls(hist):
    if not hist:
        return CALLS
    if hist[-1] is SNYD:
        return []
    return [call for call in CALLS if call > hist[-1]] + [SNYD]

def is_correct_call(d1, d2, call):
    count, side = call
    if mode == JOKER:
        d1 = tuple(side if d == 1 else d for d in d1)
        d2 = tuple(side if d == 1 else d for d in d2)
    if mode == STAIRS:
        if d1 == tuple(range(1,len(d1)+1)):
            d1 = (side,)*(len(d1)+1)
        if d2 == tuple(range(1,len(d2)+1)):
            d2 = (side,)*(len(d2)+1)
    return not bool(Counter({side: count}) - Counter(d1 + d2))

def is_leaf(hist):
    assert not hist or hist[0] is not SNYD, "SNYD can't be first call"
    return hist and hist[-1] is SNYD

def histories(hist=()):
    yield hist
    if not is_leaf(hist):
        for call in possible_calls(hist):
            yield from histories(hist+(call,))
dfs = list(histories())
histories = list(histories())
histories.sort(key = len)

leafs = [hist for hist in histories if is_leaf(hist)]
non_leafs = [hist for hist in histories if not is_leaf(hist)]

# xs (ys) are the states after a move by player + the root.
# Each of these are given a variable, since they are either leafs or parents to leafs.
# This is of course game specific, so maybe it's a bad way to do it...
xs = [h for h in histories if not h or len(h)%2==1]
ys = [h for h in histories if not h or len(h)%2==0]
# The inners are those with children. Note the root is present in both.
inner_xs = [h for h in xs if not is_leaf(h)]
inner_ys = [h for h in ys if not is_leaf(h)]

def score(d1, d2, hist):
    ''' Get the score in {-1,1} relative to player 1 '''
    assert hist and hist[-1] is SNYD
    # Player 2 called snyd
    if len(hist) % 2 == 0:
        res = is_correct_call(d1, d2, hist[-2])
    # Player 1 called snyd
    else:
        res = not is_correct_call(d1, d2, hist[-2])
    return int(res)*2 - 1

################################################################
# Write as matrix
################################################################

def initSolver():
    solver = pywraplp.Solver('', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
    zvs = {(d2,x): solver.NumVar(-solver.infinity(), solver.infinity(),
        'z{}h{}'.format(d2,shist(x))) for x in inner_xs for d2 in ROLLS2}
    xvs = {(d1,x): solver.NumVar(0, 1,
        'x{}h{}'.format(d1,shist(x))) for x in xs for d1 in ROLLS1}

    # The thing to maximize: f.T@z
    objective = solver.Objective()
    for d2 in ROLLS2:
        objective.SetCoefficient(zvs[d2,()], 1)
    objective.SetMaximization()

    # E: (d1,x) -> (d1,y_inner)
    # F^T: (d2,x_inner) -> (d2,y)
    # A: (d2,y) -> (d1,x)
    # A^T: (d1,x) -> (d2,y)
    # |z| : d2*x_inner

    # A simple strategy
    #simp = {(d1,x): 0 for d1 in ROLLS for x in xs}
    #for d1 in ROLLS:
    #    simp[d1,()] = 1
    #    for y in inner_ys:
    #        call = possible_calls(y)[0]
    #        simp[d1, y+(call,)] = simp[d1, y[:-1]]
    #print(simp)



    # Equalities: Ex = e
    #print('Equalities')
    for d1 in ROLLS1:
        #print('Roll', d1)
        # The root sums to 1
        constraint = solver.Constraint(1, 1)
        constraint.SetCoefficient(xvs[d1,()], 1)
        #print('1 =', xvs[d1,()].name())
        #print('1 =', simp[d1,()])
        # Make siblings sum to their parent
        for hist in inner_ys:
            constraint = solver.Constraint(0, 0)
            #print('0 =', end=' ')
            constraint.SetCoefficient(xvs[d1,hist[:-1]], 1)
            #print(xvs[d1,hist[:-1]].name(), end=' ')
            #print(simp[d1,hist[:-1]], end=' ')
            for call in possible_calls(hist):
                #print('-', xvs[d1,hist+(call,)].name(), end=' ')
                #print('-', simp[d1,hist+(call,)], end=' ')
                constraint.SetCoefficient(xvs[d1,hist+(call,)], -1)
            #print()

    #F = np.zeros((len(inner_xs)+1, len(ys)))
    #for i, x in enumerate(inner_xs):
    #    if x == ():
    #        F[i, ys.index(())] = 1
    #    else:
    #        F[i, ys.index(x[:-1])] = 1
    #        for call in possible_calls(x):
    #            F[i, ys.index(x+(call,))] = -1
    #print(F)
    #print(F.T)

    # F and A must have equal number of collumns
    # This is true, they both have ROLLS x ys

    # Bound zT@F - xT@A <= 0
    # Bound F.T@z - A.Tx <= 0
    # z@F0 - x@A0 >= 0, ...
    #print('Bounds')
    for d2 in ROLLS2:
        #print('Roll', d2)
        # Now the leafs
        for hist in ys:
            constraint = solver.Constraint(-solver.infinity(), 0)
            #print('0 >= ', end='')
            if hist == ():
                # We have to take care of the root as well
                # z@F:0
                #print(zvs[d2,()].name(), end=' ')
                constraint.SetCoefficient(zvs[d2,()], 1)
                for call in possible_calls(hist):
                    #print('+', zvs[d2,hist+(call,)].name(), end=' ')
                    constraint.SetCoefficient(zvs[d2,hist+(call,)], 1)
                #print()
                # A:0 is simply empty
                pass
                continue
            # z@F:i
            # I'm a y. To which internals am I a child, to which a parent?
            # We may not have any children, that's fine. If we are a leaf,
            # F will only have +1 entries for us.
            #print('-', zvs[d2,hist[:-1]].name(), end=' ')
            constraint.SetCoefficient(zvs[d2,hist[:-1]], -1)
            for call in possible_calls(hist):
                child = hist+(call,)
                if not is_leaf(child):
                    #print('+', zvs[d2,child].name(), end=' ')
                    constraint.SetCoefficient(zvs[d2,child], 1)
            # -x@A:i
            lhist = hist+(SNYD,) if hist[-1] is not SNYD else hist
            xhist = hist+(SNYD,) if hist[-1] is not SNYD else hist[:-1]
            for d1 in ROLLS1:
                sign = '-' if -score(d1, d2, lhist) < 0 else '+'
                #print(sign, xvs[d1,xhist].name(), end=' ')
                #print(sign, simp[d1,xhist], end=' ')
                constraint.SetCoefficient(xvs[d1,xhist], -score(d1, d2, lhist))
            #print()

    return solver, xvs, zvs

# Formatting of solution
def scall(call):
    if call is None:
        return "snyd"
    return '{}{}'.format(*call)

def shist(hist):
    return ','.join(map(scall,hist))

def sfrac(val):
    return str(fractions.Fraction.from_float(val).limit_denominator())

class CounterStrategy:
    def __init__(self, xvs):
        self.xvs = xvs

    @lru_cache(maxsize=10**5)
    def findCallProb(self, d1, hist):
        ''' Return the probability that player 1 did the last move of hist '''
        assert len(hist) % 2 == 1
        xhis = self.xvs[d1, hist].solution_value()
        xpar = self.xvs[d1, hist[:-2]].solution_value()
        return xhis/xpar if xpar > 1e-10 else 0

    @lru_cache(maxsize=10**5)
    def findP2Call(self, d2, hist):
        ''' Find the best call for p2, choosing the optimal deterministic counter strategy '''
        assert len(hist) % 2 == 1
        if sum(self.findCallProb(d1,hist) for d1 in ROLLS1) < 1e-6:
            #if d2 == (0,) and hist == ((1,1),):
            #    print('findP2Call called on impossible history')
            return possible_calls(hist)[0]
        pd1s = self.estimateP1Rolls(hist)
        if d2 == (0,) and hist == ((1,1),):
            pass
            #print('pd1s', pd1s)
            #print('scores', [sum(p*stateValue(d1, d2, hist+(call,)) for p, d1 in zip(pd1s,ROLLS)) for p,d1 in zip(pd1s,ROLLS)])
        return min(possible_calls(hist), key=lambda call:
                sum(p*self.stateValue(d1, d2, hist+(call,)) for p, d1 in zip(pd1s,ROLLS1)))

    @lru_cache(maxsize=10**5)
    def stateValue(self, d1, d2, hist):
        ''' Return expected payoff for player 1 '''
        if hist and hist[-1] is SNYD:
            res = score(d1, d2, hist)
        # Player 1
        elif len(hist) % 2 == 0:
            res = sum(self.stateValue(d1, d2, hist+(call,))
                    * self.findCallProb(d1, hist+(call,))
                    for call in possible_calls(hist))
        # Player 2
        elif len(hist) % 2 == 1:
            p2call = self.findP2Call(d2, hist)
            res = self.stateValue(d1, d2, hist+(p2call,))
        #print('stateValue({}, {}, {}) = {}'.format(d1, d2, hist, res))
        return res

    @lru_cache(maxsize=10**5)
    def estimateP1Rolls(self, hist):
        assert len(hist) % 2 == 1
        # Simple bayes
        prob_hist_given_d = [self.findCallProb(d1, hist) for d1 in ROLLS1]
        if sum(prob_hist_given_d) < 1e-10:
            return [1/len(ROLLS1) for _ in ROLLS1]
        return [p/sum(prob_hist_given_d) for p in prob_hist_given_d]

def printTrees(cs):
    print('Trees:')
    for d1 in ROLLS1:
        for hist in dfs:
            # At root, print the roll value
            if not hist:
                avgValue = sfrac(sum(cs.stateValue(d1, d2, ()) for d2 in ROLLS2)/len(ROLLS2))
                values = ', '.join(sfrac(cs.stateValue(d1, d2, ())) for d2 in ROLLS2)
                print('Roll: {}, Expected: {}, Values: {}'.format(d1, avgValue, values))
                continue
            # If a parent has zero probability, don't go there
            if any(cs.findCallProb(d1, hist[:j]) < 1e-8 for j in range(1,len(hist)+1,2)):
                continue
            s = '|  '*len(hist) + (scall(hist[-1]) if hist else 'root')
            if hist in xs:
                prob = sfrac(cs.findCallProb(d1, hist))
                print('{} p={}'.format(s, prob))
            else:
                tag = ''.join('_*'[hist[-1] == cs.findP2Call(d2,hist[:-1])]
                        for d2 in ROLLS2)
                print(s, tag)
def main():
    print('Setting up linear program')
    solver, xvs, zvs = initSolver()

    print('Solving')
    status = solver.Solve()
    if status != solver.OPTIMAL:
        print('Status:', status)
        print(zvs[(0,),()].solution_value())
        return

    cs = CounterStrategy(xvs)
    printTrees(cs)

    # Test feasability
    #print('Score by d2')
    #for d2 in ROLLS:
    #    expected_score = sum(stateValue(d1, d2, ()) for d1 in ROLLS)/len(ROLLS)
    #    print('{} strat {} >= lambda {}'.format(d2, expected_score, lvs[d2].solution_value()))

    #print('Zs', ', '.join(str(zv.solution_value()) for zv in zvs.values()))
    #print('Zs', ', '.join(sorted('{}x{}: {}'.format(d2, xinner, zv.solution_value()) for (d2, xinner), zv in zvs.items())))
    #for (d1, x), xv in sorted(xvs.items(), key=
    #        lambda a:(a[0][0], len(a[0][1]), str(a))):
    #    print(d1, x, xv.solution_value())
    #print('Xs', ', '.join(str(xv.solution_value()) for xv in xvs.values()))

    res = sum(zv.solution_value() for (_, hist), zv in zvs.items() if hist == ())
    res /= len(ROLLS1)*len(ROLLS2)
    print('Value:', sfrac(res))

    res2 = sum(cs.stateValue(d1, d2, ()) for d1 in ROLLS1 for d2 in ROLLS2)/len(ROLLS1)/len(ROLLS2)
    print('Score:', sfrac(res2))

if __name__ == '__main__':
    main()

