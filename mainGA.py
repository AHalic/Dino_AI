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

    constants.initial_state_size = 7
    constants.coord_size = 3

    constants.label_state = ["K_UP", "K_DOWN", "K_NO"] * constants.initial_state_size

    print('Genetic Algorithm IA')
    best_state, best_value = genetic_algorithm(constants.initial_state_size, 43200) 
    constants.aiPlayer = GAKeyClassifier(best_state)
    res, value = manyPlaysResults(30)
    npRes = np.asarray(res)
    print('results:', res, '\nmean results:', npRes.mean(), '\nstd results:', npRes.std(), '\nmean - std', value)
