import itertools
from collections import Counter

P1, P2, CHANCE = range(3)
class Game:
    def children(self):
        ''' [(prob, node)] pairs of possible moves.
            If the probabilities are unknown (players who haven't
            decided a strategy), they can be None. '''
        # Maybe we can use some kind of variable-id system for the unknown probs
    def owner(self):
        ''' The next player to move. One of P1, P2 or CHANCE. '''
    def value(self):
        ''' The value of the node relative to player1.
            Must be defined for leafs. '''

class Snyd:
    def __init__(self, dice1, dice2, sides):
        self.dice1 = dice1
        self.dice2 = dice2
        self.calls = [(count, side)
            for count in range(1, dice1+dice2+1)
            for side in range(sides)]
        self.rolls1 = list(itertools.product(range(sides), repeat=dice1))
        self.rolls2 = list(itertools.product(range(sides), repeat=dice2))
    def owner(self):
        return CHANCE
    def value(self):
        return None
    def children(self):
        return [(1/(len(self.rolls1)*len(self.rolls2)),
                 RolledSnyd(d1, d2, self.calls))
                for d1 in self.rolls1 for d2 in self.rolls2]

class RolledSnyd:
    def __init__(self, d1, d2, calls, hist=()):
        self.d1 = d1
        self.d2 = d2
        self.calls = calls
        self.hist = hist
    def owner(self):
        return len(self.hist) % 2
    def is_correct_call(ds, call):
        count, side = call
        return not bool(Counter({side: count}) - Counter(ds))
    
    def value(self):
        if self.hist and 
    def children(self):
        return [(None, RolledSnyd(d1, d2,  

        

