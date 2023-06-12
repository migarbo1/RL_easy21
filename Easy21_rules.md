Easy21 is a modification of `BlaskJack` proposed by David Silver at his RL course at the University College of London. The particulariries of this variants are:

- The deck is infinite (there can be repeated cards).
- Each time you draw a card, it can be a number between 1 and 10. And 2 out of three times the card will be black, red otherwise.
- There are no aces nor pictures (J, Q, K)
- At the start both dealer and player draw one black card.
- Black cards add value, red cards discount value.
- Player cannot exceed 21 or go under 1. In that case, loses (R = -1)
- When the player stiks, the dealer takes turns. The dealer always stiks with a sum of 17 or greater. 
- If the dealer goes bust, the player wins (R = +1)
- Otherwise wins the one with the largest sum. R = 0 in draw cases.
