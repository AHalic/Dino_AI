import time
import pygad
import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import Counter

from KeyClassifier import *
from Bird import *
import constants
from play import manyPlaysResults, playGame


class GAKeyClassifier(KeyClassifier):
    """
    Classe que implementa o classificador KNN

    parametros:
    - state: Vetor de coordenada dos pontos classificados
    """
    def __init__(self, state):
        self.state = state

        # Normaliza o estado
        self.scaler = StandardScaler()
        self.state = self.scaler.fit_transform(np.asarray(self.state).reshape(constants.initial_state_size,constants.coord_size))
        

    def find_closest(self, frame, k):
        """
        Função que busca o ponto mais próximo dos dados atuais

        parametros:
        - frame: Vetor contendo os dados atuais
            (distancia, altura do obstáculo, velocidade)
        - k: Número de pontos mais próximos aos dados atuais que serão considerados

        retorno:
        - index: Índice do ponto mais próximo
        """
        dist = [np.linalg.norm(self.state[i]- frame) for i in range(0, len(self.state))]

        index_min = dist.index(min(dist))
        dist = np.asarray(dist)
        idx = np.argpartition(dist, k)[:k]

        keys = [constants.label_state[val] for val in idx]

        count_keys = Counter(keys)
        most_keys = count_keys.most_common()

        # Caso empate, retorna o mais próximo
        if len(most_keys) > 1 and most_keys[0][1] > most_keys[1][1]:
            return most_keys[0][0]
        else:
            return constants.label_state[index_min]


    def norm_state(self, params):
        """
        Função que normaliza o vetor de coordenadas dos pontos classificados, e os dados atuais

        parametros:
        - params: Vetor de coordenadas dos dados atuais

        retorno:
        - params: Vetor normalizado
        """

        params = self.scaler.transform(np.asarray(params).reshape(1,constants.coord_size))
        params = params.reshape(constants.coord_size)

        return params
        


    def keySelector(self, params):
        """
        Função que retorna a tecla a ser pressionada para o jogador

        parametros:
        - distance: Distância do jogador até o obstáculo
        - obaltitude: Altura do obstáculo
        - speed: Velocidade do jogador

        retorno:
        - key: Tecla a ser pressionada
        """

        # caso em que não há obstáculo
        if params[1] == 0:
            return "K_NO"
        
        params = self.norm_state(params)

        closest = self.find_closest(params, 3)

        return closest

    def updateState(self, state):
        """
        Função que atualiza o vetor de coordenadas dos pontos classificados

        parametros:
        - state: Vetor de coordenadas dos pontos classificados
        """
        self.state = state

def fitness_func(solution, solution_idx):
    """
    Função que calcula o fitness de uma solução
    
    parametros:
    - solution: Solução a ser avaliada
    - solution_idx: Índice da solução na população
    """ 

    constants.aiPlayer = GAKeyClassifier(solution)
    res, value = manyPlaysResults(3)

    return value

def check_generation(instance):
    """
    Função que verifica se o tempo de execução do algoritmo está atingindo o limite

    parametros:
    - instance: Instância do algoritmo

    retorno:
    - "stop": Se o tempo de execução tiver atingido o limite
    - "continue": Se o tempo de execução não tiver atingido o limite
    """

    actual_time = time.time() - start_time 
    print('finished a generation in', actual_time, 'seconds')

    solution, solution_fitness, solution_idx = instance.best_solution()
    
    f = open("solutions.txt", "a")
    f.write(f'best solution of the {instance.generations_completed} generation: {solution}, fitness: {solution_fitness}\n')
    f.close()

    if actual_time > time_max:
        return "stop"
    return "continue"

def genetic_algorithm(state_size, max_time):
    """
    Função que implementa o algoritmo genético

    parametros:
    - state: Quantidade de pontos que serão buscados
    - max_time: Tempo máximo de execução do algoritmo

    retorno:
    - best_solution: Melhor solução encontrada
    - best_fitness: Fitness da melhor solução encontrada
    """
    global time_max, start_time
    time_max = max_time

    fitness_function = fitness_func

    num_generations = 800
    num_parents_mating = 2

    sol_per_pop = 20
    gene_type = int

    # Define o espaço para os genes referentes a altitude do obstáculo
    gene_space = [None, [260, 300, 325, 345], range(10,30)] * state_size
    initial_population = [[20, 300, 10, 108, 325, 11, 660, 325, 10, 20, 300, 10, 207, 325, 19, 12, 260, 16, 130, 345, 17],
                          [581, 260, 10, 151, 300, 46, 109, 260, 144, 571, 325, 23, 176, 345, 21, 711, 260, 53, 67, 300, 13],
                          [142, 260, 15, 63, 345, 19, 71, 260, 22, 163, 300, 20, 228, 345, 21, 650, 300, 10, 11, 300, 10],
                          [581, 260, 15, 133, 325, 18, 477, 325, 15, 58, 300, 11, 151, 345, 24, 248, 260, 14, 67, 300, 20],
                          [537, 325, 20, 117, 300, 14, 562, 300, 9, 11, 260, 13, 73, 345, 15, 520, 325, 2, 254, 345, 4],
                          [57, 300, 20, 117, 345, 14, 562, 300, 11, 11, 260, 13, 73, 325, 15, 520, 325, 2, 254, 300, 4]]

    num_genes = state_size * 3

    init_range_low = 0
    init_range_high = 600

    parent_selection_type = "tournament"
    keep_parents = 1

    crossover_type = "single_point"
    crossover_probability = 0.8
    K_tournament=4

    mutation_type = "random"
    mutation_probability = 0.4

    stop_criteria= "saturate_100"


    ga_instance = pygad.GA(num_generations=num_generations,
                            num_parents_mating=num_parents_mating,
                            fitness_func=fitness_function,
                            sol_per_pop=sol_per_pop,
                            gene_type=gene_type,
                            num_genes=num_genes,
                            gene_space=gene_space,
                            init_range_low=init_range_low,
                            init_range_high=init_range_high,
                            parent_selection_type=parent_selection_type,
                            keep_parents=keep_parents,
                            crossover_type=crossover_type,
                            K_tournament=K_tournament,
                            crossover_probability=crossover_probability,
                            mutation_type=mutation_type,
                            mutation_probability=mutation_probability,
                            stop_criteria=stop_criteria,
                            on_generation=check_generation)

    start_time = time.time()
    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    return solution, solution_fitness

