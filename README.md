# Blackjack

Tests and solutions for Stanford's CS221 class. 
This specific repo is focused on Markov Decision Processes, modeling a sligthly modified game of Blackjack as a MDP, defining successor states, probabilities and rewards as well as using the rewards for improving the optimal strategy.
Optimal strategy is found using the Q-Learning algorithm. As the game of Blackjack (with a standard deck) can have a very high set if possible states and successor states we can use the fact that simillar states require simillar actions. This fact is used in implementing a feature extractor function which takes a specific state and returns its features. Recieved results are used in improving the optimal strategy by updating the weights of a given extracted feature.  (Example: Having a King and Queen is not much different from having a King and Jack in Blackjack).
