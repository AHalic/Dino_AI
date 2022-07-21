"""
Dino Game by Sophie Dilhon
"""

from scipy import stats
import numpy as np

from play import *
from GAClassifier import *
import constants



if __name__ == '__main__':
    # initial state (distance, object altitude, speed)
    initial_state = [250, 325, 15, 
                     350, 260, 18, 
                     450, 300, 20]
    initial_state_label = ["K_UP",
                           "K_DOWN", 
                           "K_UP",
                           "K_DOWN",
                           "K_UP",
                           "K_DOWN"]

    print('Genetic Algorithm IA')
    # constants.aiPlayer = GAKeyClassifier(initial_state, initial_state_label)
    best_state, best_value = genetic_algorithm(initial_state, 1000, initial_state_label) 
    constants.aiPlayer = GAKeyClassifier(best_state, initial_state_label)
    res, value = manyPlaysResults(30)
    npRes = np.asarray(res)
    print('results:', res, '\nmean results:', npRes.mean(), '\nstd results:', npRes.std(), '\nmean - std', value)

# Parameters of the best solution : [378 345  67 421 300  25 133 260 411 173 260  11 541 300 119 213 325 352]
# Fitness value of the best solution = 430.1421395608432
# [391.5, 70.5, 215.25, 176.5, 66.0, 260.5, 50.75, 104.5, 417.0, 396.5, 217.0, 63.75, 223.0, 104.0, 287.5, 422.25, 279.25, 326.0, 456.25, 323.0, 131.75, 306.0, 420.5, 202.5, 347.75, 463.5, 244.25, 195.0, 132.5, 325.25]
# 254.0 
# 125.44180257527127 
# 128.55819742472875