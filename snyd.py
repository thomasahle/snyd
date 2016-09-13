from collections import Counter
from collections import defaultdict
import itertools
import numpy as np
import scipy.optimize.linprog

DICE = 1
SIDES = 2

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

lxs = [h if len(h)%2==1 else h[:-1] for h in leafs]
xs = [h for h in histories if not h or len(h)%2==1]

# Because we can nearly always call snyd,
# the x leafs should be all the xs, but the root
assert set(xs) - set(lxs) == set([()])

ids = defaultdict(lambda: len(ids))
ids['l'] = 0

print('Tree')
#for hist in histories:
#   print(' '*2*len(hist), ids[hist], end='')
#   if hist:
#      if hist[-1] is SNYD:
#         if not is_correct_call((0,0), hist[-2]):
#            print('+', end='')
#      elif is_correct_call((0,0), hist[-1]):
#         print('+', end='')
#   print()

eqs = []

# Making sure probabilities sum to 1.
# Variables are associated with the history they lead to.
# Top variables are set to sum to state(the empty history),
# which we should make sure is 1.
for hist in non_leafs:
   if len(hist) % 2 == 0:
      lhs = [hist+(call,) for call in possible_calls(hist)]
      eqs.append((lhs, [hist[:-1]]))


print('Probability equations')
for (lsh, rhs) in eqs:
   print([ids[h] for h in lsh], '=', [ids[h] for h in rhs])

# Now for the linear programming
# Minimize: c^T * x
# Subject to: A_ub * x <= b_ub
# A_eq * x == b_eq

x_cnt = len(xs)
var_cnt = (1 + x_cnt) * len(ROLLS)

# The thing to maximize/minimize
c = np.zeros(var_cnt)
for i in range(0, var_cnt, 1 + x_cnt):
   c[i] = 1
#print(c)

# Equalities

Es = []
for d1 in ROLLS:
   E = np.zeros((len(eqs), 1+x_cnt))
   for i, (lhs, rhs) in enumerate(eqs):
      for h in lhs:
         E[i, ids[h]] = 1
      for h in rhs:
         E[i, ids[h]] = -1
   Es.append(E)
E = np.hstack(Es)

# Equality vector (make x0 = 1)
lhs = np.zeros((1, var_cnt))
ev = np.zeros(E.shape[1])
for i in range(ids[()], var_cnt, 1+x_cnt):
   ev[i] = 1
   lhs[0, ids[()]] = 1
E = np.vstack((E, lhs))

print(ev)

# The bounds
Us = []
for d1 in ROLLS:
   Us.append([])
   for d2 in ROLLS:
      U = np.zeros((len(lxs), 1+x_cnt))
      print((1+x_cnt, len(lxs)))
      for i, h in enumerate(lxs):
         if h[-1] is SNYD:
            res = 1 - is_correct_call(d1+d2, h[-2])
         else: res = is_correct_call(d1+d2, h[-1])
         U[i, ids[h]] = res
         U[i, ids['l']] = -1
      Us[-1].append(U)
U = np.vstack(np.hstack(Uc for Uc in Ur) for Ur in Us)
#np.set_printoptions(threshold=10**4)
#print(U)



#A_ub = -U




# p
# Lower bounds: Ensures positive probabilities
# Total utility

# Lower bounds: Ensures positive probabilities
# With scipy we can do global bounds, which we set to 0,1.
# That should be fine even for the lambda variables.

print('c', c.shape)
print('U', U.shape)
print('E', E.shape)
print('ev', ev.shape)

scipy.optimize.linprog(
      c = -c,
      A_ub = -U,
      A_eq = E,
      b_eq = ev,
      bounds = (0,1),
      options = {disp: True})


