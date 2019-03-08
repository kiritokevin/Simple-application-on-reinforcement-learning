Blackjack with Reinforcement Learning
=========

The Game
-----
The game more or less follows the standard Blackjack rules. Read the game engine code to see minor simplification (note that the learning algorithms do not need to understand the rules). 

The Hit and Stand buttons are for playing the game manually. Clicking the MC, TD, and QL bottons starts/pauses the corresponding learning processes. On the screen it shows the values corresponding to the current state of the game; so if you click the Hit and Stand buttons you can see how different states get evaluated. Right now a dummy MC learning code is given, so when you click MC you can see the number keeps growing for the state you are in. After implementing the right methods, you'll see that these values will stablize (i.e. converge). 

The Play button at the end will automatically play the game with the learned Q values. You can check how many times you win or lose given the current Q values (so after learning for a while, you can check whether the policy is behaving well). 

We have three different evaluations here:
1. Monte Carlo Policy Evaluation 

2. Temporal-Difference Policy Evaluation

3. Q-Learning

=========
