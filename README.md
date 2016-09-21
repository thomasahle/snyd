# An analysis of the endgame of Liar's Dice.

This project analyses the two player end game of [Liar's Dice](https://en.wikipedia.org/wiki/Liar%27s_dice)
using the theory of games with [Imperfect information](https://en.wikipedia.org/wiki/Perfect_information).
The Nash equilibrium is found by the method of [Daphne Koller and Nimrod Megiddo](http://www.sciencedirect.com/science/article/pii/089982569290035Q)
which reduces the game to a linear program of size polynomial in the number of states or "information sets".

|      	| Normal 	| Joker  	| Stairs 	|
|------	|--------	|--------	|--------	|
| 1, 1 	| 1      	|        	|        	|
| 2, 2 	| 0      	| 1      	| 0      	|
| 3, 3 	| -1/9   	| 1/3    	| 1/9    	|
| 4, 4 	| -1/8   	| 0      	| 0      	|
| 5, 5 	| -3/35  	| 0      	| 1/25   	|
| 6, 6 	| -1/9   	| -7/327 	|        	|

In the table above, the rows corespond to the number of sides on the dice of the players.
Scores are set as 1 when the first player to moves wins and -1 if player 2 wins.
For the 'normal' version of Liar's Dice with one six-sided die for each player, we get the expected score -1/9.
The 'joker' version, in which a 1 participates towards any call, the game turns out to be better balanced.

# Running the code

```
$ pip install py3-ortools
$ python3 snyd_or.py 1 6 normal
Setting up linear program
Solving
Trees:
Roll: (1,), Expected: -1, Values: -1, -1, -1, -1, -1, -1
|  15 p=1
|  |  16 ******
|  |  |  26 p=1

...

|  |  26 ______
|  |  |  snyd p=1
|  |  snyd ______
Value: -1/9
Score: -1/9
```
