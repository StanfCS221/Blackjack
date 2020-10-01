import util, math, random
from collections import defaultdict
from util import ValueIteration

############################################################
# Problem 2a
file = open('states.txt', 'w')
# If you decide 2a is true, prove it in blackjack.pdf and put "return None" for
# the code blocks below.  If you decide that 2a is false, construct a counterexample.
class CounterexampleMDP(util.MDP):
    # Return a value of any type capturing the start state of the MDP.
    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return 1
        # END_YOUR_CODE

    # Return a list of strings representing actions possible from |state|.
    def actions(self, state):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return [1]
        # END_YOUR_CODE

    # Given a |state| and |action|, return a list of (newState, prob, reward) s
    # corresponding to the states reachable from |state| when taking |action|.
    # Remember that if |state| is an end state, you should return an empty list [].
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return []
        # END_YOUR_CODE

    # Set the discount factor (float or integer) for your counterexample MDP.
    def discount(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return 1
        # END_YOUR_CODE

############################################################
# Problem 3a

class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: list of integers (face values for each card included in the deck)
        multiplicity: single integer representing the number of cards with each face value
        threshold: maximum number of points (i.e. sum of card values in hand) before going bust
        peekCost: how much it costs to peek at the next card
        """
        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look closely at this function to see an example of state representation for our Blackjack game.
    # Each state is a  with 3 elements:
    #   -- The first element of the  is the sum of the cards in the player's hand.
    #   -- If the player's last action was to peek, the second element is the index
    #      (not the face value) of the next card that will be drawn; otherwise, the
    #      second element is None.
    #   -- The third element is a  giving counts for each of the cards remaining
    #      in the deck, or None if the deck is empty or the game is over (e.g. when
    #      the user quits or goes bust).
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be placed into the succAndProbReward function below.
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Given a |state| and |action|, return a list of (newState, prob, reward) s
    # corresponding to the states reachable from |state| when taking |action|.
    # A few reminders:
    # * Indicate a terminal state (after quitting, busting, or running out of cards)
    #   by setting the deck to None.
    # * If |state| is an end state, you should return an empty list [].
    # * When the probability is 0 for a transition to a particular new state,
    #   don't include that state in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (our solution is 38 lines of code, but don't worry if you deviate from this)
        def deckIsEmpty(deck):
            for x in deck:
                if x > 0:
                    return False
            return True

        # state = (sumOfCardsInHand, NextCardIfPeeked, OfCardsLeft)
        # action = 'Take' / 'Peek' / 'Quit'

        # we return list of:
            # (newState, probability of that state, reward for that state

        sumOfCards, nextCard, deck = state

        if deck == None:
            return []

        result = []

        if action == 'Quit':
            # New state:
                # Sum of cards is same
                # Next Card is None
                # Deck is None
            # return new State, Prob = 1, reward is Sum of Cards
            newState = tuple([sumOfCards, None, None])
            return [(newState, 1, sumOfCards)]

        numberOfCards = sum(x for x in deck)

        if action == 'Peek':
            # Sum of Cards stays the same
            # Deck stays the same
            # Next card takes value from 0 to len(deckCardCounts)
            # Probability is equal to numberOfCardX/numberOfCards
            if nextCard != None: #already peeked last move
                return []
            for i, numberOfCardX in enumerate(deck):
                if numberOfCardX != 0:
                    newState = tuple([sumOfCards, i, deck])
                    result.append((newState, numberOfCardX/numberOfCards, -self.peekCost))
            return result

        if action == 'Take':
            # Sum of cards increases for the value of the card, Next Card is None, Deck: decrement the number of cards with value which was taken
            # probability numberOfCardX / numberOfCards
            # Cost = 0

            #State = sum, next, deck
            #return newState, prob, reward

            # If we peeked and know next card only one state is possible:

            if nextCard != None:
                newSum = sumOfCards + self.cardValues[nextCard]
                newDeck = list(deck)
                newDeck[nextCard] -= 1
                newDeck = tuple(newDeck)
                
                if(newSum > self.threshold):
                    newState = (newSum, None, None)
                    return [(newState, 1, 0)]
                if deckIsEmpty(newDeck):
                    newState = (newSum, None, None)
                    return [(newState, 1, newSum)]
                newState = (newSum, None, newDeck)
                return [(newState, 1, 0)]


            for i, numberOfCardX in enumerate(deck):
                if numberOfCardX != 0:
                    newSum = sumOfCards + self.cardValues[i]
                    newDeck = list(deck)
                    newDeck[i] -= 1
                    newDeck = tuple(newDeck)
                    if newSum > self.threshold:
                        newState = tuple([newSum, None, None]) #Deck is None = Game Over -> reward is 0
                        result.append((newState, numberOfCardX/numberOfCards, 0))
                    else: # sum is below the threshold
                        if deckIsEmpty(newDeck):
                            newState = tuple([newSum, None, None]) # Game is over but reward is newSum
                            result.append((newState, numberOfCardX / numberOfCards, newSum))
                            return result # If deck is empty after this move we can stop
                        # Deck is not empty and we are below the threshold
                        newState = tuple([newSum, None, newDeck])
                        result.append((newState, numberOfCardX / numberOfCards, 0))
            return result
        # END_YOUR_CODE

    def discount(self):
        return 1

############################################################
# Problem 3b

def peekingMDP():
    """
    Return an instance of BlackjackMDP where peeking is the
    optimal action at least 10% of the time.
    """
    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    peekMDP = BlackjackMDP([5,15],10,20,1)
    return peekMDP
    # END_YOUR_CODE

############################################################
# Problem 4a: Q learning

# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearningAlgorithm(util.RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState):
        # BEGIN_YOUR_CODE (our solution is 9 lines of code, but don't worry if you deviate from this)

        # w = w - eta*(Qopt(s,a,w) - (r+Gama*Vopt(s'))*Fi(s,a)
        if newState == None:
            return

        features = self.featureExtractor(state, action)
        Vopt = max(self.getQ(newState, newAction) for newAction in self.actions(newState))

        gradient = self.getStepSize()*(self.getQ(state, action)-(reward + self.discount*Vopt))
        for featureName, featureValue in features:
            if featureName in self.weights:
                self.weights[featureName] -= gradient * featureValue
            else:
                self.weights[featureName] = -gradient * featureValue

        # END_YOUR_CODE

# Return a single-element list containing a binary (indicator) feature
# for the existence of the (state, action) pair.  Provides no generalization.
def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]

############################################################
# Problem 4b: convergence of Q-learning
# Small test case
smallMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# Large test case
largeMDP = BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3, threshold=40, peekCost=1)

def simulate_QL_over_MDP(mdp, featureExtractor):
    # NOTE: adding more code to this function is totally optional, but it will probably be useful
    # to you as you work to answer question 4b (a written question on this assignment).  We suggest
    # that you add a few lines of code here to run value iteration, simulate Q-learning on the MDP,
    # and then print some stats comparing the policies learned by these two approaches.
    # BEGIN_YOUR_CODE
    rl = QLearningAlgorithm(mdp.actions, mdp.discount(), featureExtractor) #actions discount feature extractor
    util.simulate(mdp, rl, numTrials=30000)
    rl.explorationProb = 0
    valueIter = util.ValueIteration()
    valueIter.solve(mdp)


    numberOfStates = 0
    numberOfDifferentStates = 0
    for state in mdp.states:
        if state not in valueIter.pi:
            file.write('Pi does not contain state {}\n'.format(state))
        else:
            if valueIter.pi[state] != rl.getAction(state) and state[2] != None:
                numberOfDifferentStates += 1
                file.write('In state {} Pi gives action {}, but RL gives action {}\n'.format(state, valueIter.pi[state], rl.getAction(state)))
        numberOfStates += 1
    file.write('\n % of different actions = {}%\n'.format(numberOfDifferentStates/numberOfStates*100))
    for weight in rl.weights:
        file.write('weight ({}) =  {} \n'.format(weight, rl.weights[weight]))
    # END_YOUR_CODE


############################################################
# Problem 4c: features for Q-learning.

# You should return a list of (feature key, feature value) pairs.
# (See identityFeatureExtractor() above for a simple example.)
# Include the following features in the list you return:
# -- Indicator for the action and the current total (1 feature).
# -- Indicator for the action and the presence/absence of each face value in the deck.
#       Example: if the deck is (3, 4, 0, 2), then your indicator on the presence of each card is (1, 1, 0, 1)
#       Note: only add this feature if the deck is not None.
# -- Indicators for the action and the number of cards remaining with each face value (len(counts) features).
#       Note: only add these features if the deck is not None.
def blackjackFeatureExtractor(state, action):
    total, nextCard, counts = state

    # BEGIN_YOUR_CODE (our solution is 7 lines of code, but don't worry if you deviate from this)
    features = []
    features.append(('action ' + str(action) + ' with total ' + str(total), 1))
    # features.append(('total = ' + str(total),1))
    if counts != None:
        for i, card in enumerate(counts):
            features.append(('action ' + str(action) + 'while ' + str(card) + 'cards with index ' + str(i) + ' remain', 1))
        for i, card in enumerate(counts):
            cardExists = 1*(counts[i] != 0)
            features.append(('action ' + str(action) + ' while cards with index ' + str(i) + ' remain', cardExists))

    '''
    features = []
    if counts == None:
        return features
    perc = '<75%'
    if total > 40 * 7.5/10:
        perc = '>=75%'

    values = [1,3,5,8,10]
    # Best % of same results with value Iteration
    if nextCard != None:
        if total + values[nextCard] < 40:
            rew = 5
        else:
            rew = -5
        features.append((str(action) + ' ' + perc + ' ' + str(nextCard),rew))

    for position, card in enumerate(counts):
        if(position >= len(counts)*2/3):
            features.append((str(action) + ' ' + perc + ' ' + str(position) + ' ' + str(card), 1))
    '''
    return features

    # END_YOUR_CODE

############################################################
# Problem 4d: What happens when the MDP changes underneath you?!

# Original mdp
originalMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# New threshold
newThresholdMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=15, peekCost=1)

def compare_changed_MDP(original_mdp, modified_mdp, featureExtractor):
    # NOTE: as in 4b above, adding more code to this function is completely optional, but we've added
    # this partial function here to help you figure out the answer to 4d (a written question).
    # Consider adding some code here to simulate two different policies over the modified MDP
    # and compare the rewards generated by each.
    # BEGIN_YOUR_CODE
    valueIterOriginal = util.ValueIteration()
    valueIterOriginal.solve(original_mdp)
    fixedRL = util.FixedRLAlgorithm(valueIterOriginal.pi)
    rewards = util.simulate(modified_mdp, fixedRL)
    print("Fixed RL")
    for reward in rewards:
        print(reward)
    rewardsFromQ = util.simulate(modified_mdp, QLearningAlgorithm(modified_mdp.actions, modified_mdp.discount(), featureExtractor))
    print('QLearn')
    for reward in rewardsFromQ:
        print(reward)
    # END_YOUR_CODE

