from scipy import stats
import numpy as np

from play import *
from SimpleClassifier import *



if __name__ == '__main__':
    initial_state = [(15, 250), (18, 350), (20, 450), (1000, 550)]
    aiPlayer = KeySimplestClassifier(initial_state)
    print(type(aiPlayer))
    best_state, best_value = gradient_ascent(initial_state, 5000) 
    aiPlayer = KeySimplestClassifier(best_state)
    res, value = manyPlaysResults(30)
    npRes = np.asarray(res)
    print(res, npRes.mean(), npRes.std(), value)