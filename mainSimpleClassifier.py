"""
Dino Game by Sophie Dilhon
"""

from scipy import stats
import numpy as np

from play import *
from SimpleClassifier import *
import constants



if __name__ == '__main__':
    initial_state = [(15, 250), (18, 350), (20, 450), (1000, 550)]

    print('Simple Classifier IA')
    constants.aiPlayer = KeySimplestClassifier(initial_state)
    best_state, best_value = gradient_ascent(initial_state, 5000) 
    constants.aiPlayer = KeySimplestClassifier(best_state)
    res, value = manyPlaysResults(30)
    npRes = np.asarray(res)
    print(res, npRes.mean(), npRes.std(), value)