import numpy as np

from play import *
from RuleBasedClassifier import *
import constants



if __name__ == '__main__':
    constants.AI_MODE = 'RB'
    
    print('Rule Based Classifier professor')
    constants.aiPlayer = RuleBasedKeyClassifier()

    f = open("solutions_rule.txt", "a")
    
    res, value = manyPlaysResults(30)
    npRes = np.asarray(res)
    print('results:', res, '\nmean results:', npRes.mean(), '\nstd results:', npRes.std(), '\nmean - std', value)

    f.write(f'Resultados: {res}\nmean: {npRes.mean()} and std: {npRes.std()} and mean - std: {value}\n\n')
    f.close()

