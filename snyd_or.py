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

# xs (ys) are the states after a move by player + the root.
# Each of these are given a variable, since they are either leafs or parents to leafs.
# This is of course game specific, so maybe it's a bad way to do it...
xs = [h for h in histories if not h or len(h)%2==1]
ys = [h for h in histories if not h or len(h)%2==0]
# The leafs are those with no children.
leaf_xs = [h for h in xs if is_leaf(h)]
leaf_ys = [h for h in ys if is_leaf(h)]
# The inners are those with children. Note the root is present in both.
inner_xs = [h for h in xs if not is_leaf(h)]
inner_ys = [h for h in ys if not is_leaf(h)]

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

################################################################
# Write as matrix
################################################################

def initSolver():
    solver = pywraplp.Solver('', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
    zvs = {(d,x): solver.NumVar(-solver.infinity(), solver.infinity(),
        'z{}h{}'.format(d[0],shist(x))) for x in inner_xs for d in ROLLS}
    xvs = {(d,x): solver.NumVar(0, 1,
        'x{}h{}'.format(d[0],shist(x))) for x in xs for d in ROLLS}

    # The thing to maximize: f.T@z
    objective = solver.Objective()
    for d2 in ROLLS:
        objective.SetCoefficient(zvs[d2,()], 1)
    objective.SetMaximization()

    # E: (d1,x) -> (d1,y_inner)
    # F^T: (d2,x_inner) -> (d2,y)
    # A: (d2,y) -> (d1,x)
    # A^T: (d1,x) -> (d2,y)
    # |z| : d2*x_inner

    # Equalities: Ex = e
    print('Equalities')
    for d1 in ROLLS:
        print('Roll', d1)
        # The root sums to 1
        constraint = solver.Constraint(1, 1)
        constraint.SetCoefficient(xvs[d1,()], 1)
        print('1 =', xvs[d1,()].name())
        # Make siblings sum to their parent
        for hist in inner_ys:
            constraint = solver.Constraint(0, 0)
            print('0 =', end=' ')
            constraint.SetCoefficient(xvs[d1,hist[:-1]], -1)
            print('-', xvs[d1,hist[:-1]].name(), end=' ')
            for call in possible_calls(hist):
                print('+', xvs[d1,hist+(call,)].name(), end=' ')
                constraint.SetCoefficient(xvs[d1,hist+(call,)], 1)
            print()

    # F and A must have equal number of collumns
    # This is true, they both have ROLLS x ys

    # Bound F.T@z - A.Tx >= 0
    # z@F0 - x@A0 >= 0, ...
    print('Bounds')
    for d2 in ROLLS:
        print('Roll', d2)
        # Now the leafs
        for hist in ys:
            constraint = solver.Constraint(-solver.infinity(), 0)
            print('0 >= ', end='')
            if hist == ():
                # We have to take care of the root as well
                # z@F:0
                print(zvs[d2,()].name())
                constraint.SetCoefficient(zvs[d2,()], 1)
                # A:0 is simply empty
                pass
                continue
            # z@F:i
            # I'm a y. To which internals am I a child, to which a parent?
            # We may not have any children, that's fine. If we are a leaf,
            # F will only have +1 entries for us.
            print(zvs[d2,hist[:-1]].name(), end=' ')
            constraint.SetCoefficient(zvs[d2,hist[:-1]], 1)
            for call in possible_calls(hist):
                child = hist+(call,)
                if not is_leaf(child):
                    print('-', zvs[d2,child].name(), end=' ')
                    constraint.SetCoefficient(zvs[d2,child], -1)
            # -x@A:i
            lhist = hist+(SNYD,) if hist[-1] is not SNYD else hist
            xhist = hist+(SNYD,) if hist[-1] is not SNYD else hist[:-1]
            for d1 in ROLLS:
                sign = '-' if -score(d1+d2, lhist) < 0 else '+'
                print(sign, xvs[d1,xhist].name(), end=' ')
                constraint.SetCoefficient(xvs[d1,xhist], -score(d1+d2, lhist))
            print()

    return solver, xvs, zvs

# Formatting of solution
def scall(call):
    if call is None:
        return "snyd"
    return '{}{}'.format(*call)

def shist(hist):
    return ','.join(map(scall,hist))

class CounterStrategy:
    def __init__(self, xvs):
        self.xvs = xvs

    def findCallProb(self, d1, hist):
        ''' Return the probability that player 1 did the last move of hist '''
        assert len(hist) % 2 == 1
        xhis = self.xvs[d1, hist].solution_value()
        xpar = self.xvs[d1, hist[:-2]].solution_value()
        return xhis/xpar if xpar > 1e-10 else 0

    def findP2Call(self, d2, hist):
        ''' Find the best call for p2, choosing the optimal deterministic counter strategy '''
        assert len(hist) % 2 == 1
        if sum(self.findCallProb(d1,hist) for d1 in ROLLS) < 1e-6:
            #if d2 == (0,) and hist == ((1,1),):
            #    print('findP2Call called on impossible history')
            return possible_calls(hist)[0]
        pd1s = self.estimateP1Rolls(hist)
        if d2 == (0,) and hist == ((1,1),):
            pass
            #print('pd1s', pd1s)
            #print('scores', [sum(p*stateValue(d1, d2, hist+(call,)) for p, d1 in zip(pd1s,ROLLS)) for p,d1 in zip(pd1s,ROLLS)])
        return min(possible_calls(hist), key=lambda call:
                sum(p*self.stateValue(d1, d2, hist+(call,)) for p, d1 in zip(pd1s,ROLLS)))

    def stateValue(self, d1, d2, hist):
        ''' Return expected payoff for player 1 '''
        if hist and hist[-1] is SNYD:
            res = score(d1+d2, hist)
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

    def estimateP1Rolls(self, hist):
        assert len(hist) % 2 == 1
        # Simple bayes
        prob_hist_given_d = [self.findCallProb(d1, hist) for d1 in ROLLS]
        if sum(prob_hist_given_d) < 1e-10:
            return [1/len(ROLLS) for _ in ROLLS]
        return [p/sum(prob_hist_given_d) for p in prob_hist_given_d]

def main():
    # Solve the system
    solver, xvs, zvs = initSolver()
    status = solver.Solve()
    if status != solver.OPTIMAL:
        print('Status:', status)
        print(zvs[(0,),()].solution_value())
        return

    print('Trees:')
    cs = CounterStrategy(xvs)
    for d1 in ROLLS:
        for hist in histories:
            # At root, print the roll value
            if not hist:
                avgValue = sum(cs.stateValue(d1, d2, ()) for d2 in ROLLS)/len(ROLLS)
                values = [round(cs.stateValue(d1, d2, ()), ndigits=4) for d2 in ROLLS]
                print('Roll: {}, Expected: {}, Values: {}'.format(d1, avgValue, values))
                continue
            # If a parent has zero probability, don't go there
            if any(cs.findCallProb(d1, hist[:j]) < 1e-8 for j in range(1,len(hist)+1,2)):
                continue
            s = '|  '*len(hist) + (scall(hist[-1]) if hist else 'root')
            if hist in xs:
                prob = round(cs.findCallProb(d1, hist), ndigits=4)
                print('{} p={}'.format(s, prob))
            else:
                tag = ''.join('_*'[hist[-1] == cs.findP2Call(d2,hist[:-1])]
                        for d2 in ROLLS)
                print(s, tag)

    # Test feasability
    #print('Score by d2')
    #for d2 in ROLLS:
    #    expected_score = sum(stateValue(d1, d2, ()) for d1 in ROLLS)/len(ROLLS)
    #    print('{} strat {} >= lambda {}'.format(d2, expected_score, lvs[d2].solution_value()))

    print('Zs', ', '.join(str(zv.solution_value()) for zv in zvs.values()))
    #print('Zs', ', '.join(sorted('{}x{}: {}'.format(d2, xinner, zv.solution_value()) for (d2, xinner), zv in zvs.items())))
    print('Xs', ', '.join(str(xv.solution_value()) for xv in xvs.values()))

    res = sum(zv.solution_value() for (_, hist), zv in zvs.items() if hist == ())
    print('Value:', fractions.Fraction.from_float(res).limit_denominator())

    res2 = sum(cs.stateValue(d1, d2, ()) for d1 in ROLLS for d2 in ROLLS)/len(ROLLS)**2
    print('Score:', fractions.Fraction.from_float(res2).limit_denominator())

if __name__ == '__main__':
    main()

