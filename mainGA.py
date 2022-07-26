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
    constants.AI_MODE = 'GA'
    constants.initial_state_size = 7
    constants.coord_size = 3

    constants.label_state = ["K_DOWN", "K_UP", "K_NO"] * constants.initial_state_size
    
    print('Genetic Algorithm IA')
    # best_state, best_value = genetic_algorithm(constants.initial_state_size, 3600) 
    best_state = [537, 325, 20, 117, 300, 14, 562, 300, 9, 11, 260, 13, 73, 345, 15, 520, 325, 2, 254, 345, 4]
    constants.aiPlayer = GAKeyClassifier(best_state)

    print('Solução encontrada, indo para a etapa de desempenho')
    f = open("solutions.txt", "a")

    res, value = manyPlaysResults(30)
    npRes = np.asarray(res)
    print('results:', res, '\nmean results:', npRes.mean(), '\nstd results:', npRes.std(), '\nmean - std', value)

    f.write(f'Resultados: {res}\nmean: {npRes.mean()} and std: {npRes.std()} and mean - std: {value}\n\n')
    f.close()

