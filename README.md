Assignment 4: Blackjack with Reinforcement Learning
=========
Your task is to implement Monte Carlo evaluation, Temporal Difference evaluation, and Q-Learning for Blackjack. The base game engine is from [here](https://github.com/ServePeak/Blackjack-Python/blob/master/blackjack.py). 

Due date
-----
March 3 (Sunday) 11:59pm Pacific Time. 

The Game
-----
The game more or less follows the standard Blackjack rules. Read the game engine code to see minor simplification (note that the learning algorithms do not need to understand the rules). 

The Hit and Stand buttons are for playing the game manually. Clicking the MC, TD, and QL bottons starts/pauses the corresponding learning processes. On the screen it shows the values corresponding to the current state of the game; so if you click the Hit and Stand buttons you can see how different states get evaluated. Right now a dummy MC learning code is given, so when you click MC you can see the number keeps growing for the state you are in. After implementing the right methods, you'll see that these values will stablize (i.e. converge). 

The Play button at the end will automatically play the game with the learned Q values. You can check how many times you win or lose given the current Q values (so after learning for a while, you can check whether the policy is behaving well). 

Grading
-----
Documentation (1 point): Comment your code generously. 

Functionality (13 points): Your task is to implement the following algorithms. In all of them, use 0.9 for the discount factor gamma. When the player wins, give reward +1, and when loses, give -1. Currently there is a "draw" case, which you can either give 0 or count it as the player losing in that case. 

Check comments in the code for hints. 

- (3 points) Monte Carlo Policy Evaluation 

Evaluate the policy "Hit (ask for a new card) if sum of cards is below 17, and Stand (switch to dealer) otherwise" using the Monte Carlo method -- namely, learn the utilities for each state under the policy. One should be able to click the "MC" white button to start or pause the learning process. When the user manually plays the game, the learned utility will be shown for the current state. 

- (4 points) Temporal-Difference Policy Evaluation

Evaluate the policy "Hit (ask for a new card) if sum of cards is below 17, and Stand (switch to dealer) otherwise" using the Temporal-Difference method. One should be able to click the "TD" white button to start or pause the learning process. When the user manually plays the game, the learned utility will be shown for the current state. 

- (6 points) Q-Learning

Implement the Q-learning algorithm. After learning, when the user plays manually, the Q values will be displayed for each action (two choices) to guide the user. 
