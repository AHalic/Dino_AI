import pygame
import os
import random
import time
from sys import exit

import pygad
import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import Counter

initial_state_size = 7
coord_size = 3
label_state = ["K_DOWN", "K_UP", "K_NO"] * initial_state_size

# global aiPlayer
GAME_MODE = "AI_MODE"

# Global Constants
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1100

RUNNING = [pygame.image.load(os.path.join("Assets/Dino", "DinoRun1.png")),
		   pygame.image.load(os.path.join("Assets/Dino", "DinoRun2.png"))]
JUMPING = pygame.image.load(os.path.join("Assets/Dino", "DinoJump.png"))
DUCKING = [pygame.image.load(os.path.join("Assets/Dino", "DinoDuck1.png")),
		   pygame.image.load(os.path.join("Assets/Dino", "DinoDuck2.png"))]

SMALL_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus1.png")),
				pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus2.png")),
				pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus3.png"))]
LARGE_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus1.png")),
				pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus2.png")),
				pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus3.png")),
				pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus4.png"))]

BIRD = [pygame.image.load(os.path.join("Assets/Bird", "Bird1.png")),
		pygame.image.load(os.path.join("Assets/Bird", "Bird2.png"))]


class Dinosaur:
	X_POS = 90
	Y_POS = 330
	Y_POS_DUCK = 355
	JUMP_VEL = 17
	JUMP_GRAV = 1.1

	def __init__(self):
		self.duck_img = DUCKING
		self.run_img = RUNNING
		self.jump_img = JUMPING

		self.dino_duck = False
		self.dino_run = True
		self.dino_jump = False

		self.step_index = 0
		self.jump_vel = 0
		self.jump_grav = self.JUMP_VEL
		self.image = self.run_img[0]
		self.dino_rect = self.image.get_rect()
		self.dino_rect.x = self.X_POS
		self.dino_rect.y = self.Y_POS

	def update(self, userInput):
		if self.dino_duck and not self.dino_jump:
			self.duck()
		if self.dino_run:
			self.run()
		if self.dino_jump:
			self.jump()

		if self.step_index >= 20:
			self.step_index = 0

		if userInput == "K_UP" and not self.dino_jump:
			self.dino_duck = False
			self.dino_run = False
			self.dino_jump = True
		elif userInput == "K_DOWN" and not self.dino_jump:
			self.dino_duck = True
			self.dino_run = False
			self.dino_jump = False
		elif userInput == "K_DOWN":
			self.dino_duck = True
			self.dino_run = False
			self.dino_jump = True
		elif not (self.dino_jump or userInput == "K_DOWN"):
			self.dino_duck = False
			self.dino_run = True
			self.dino_jump = False

	def duck(self):
		self.image = self.duck_img[self.step_index // 10]
		self.dino_rect = self.image.get_rect()
		self.dino_rect.x = self.X_POS
		self.dino_rect.y = self.Y_POS_DUCK
		self.step_index += 1

	def run(self):
		self.image = self.run_img[self.step_index // 10]
		self.dino_rect = self.image.get_rect()
		self.dino_rect.x = self.X_POS
		self.dino_rect.y = self.Y_POS
		self.step_index += 1

	def jump(self):
		self.image = self.jump_img
		if self.dino_duck:
			self.jump_grav = self.JUMP_GRAV * 4
		if self.dino_jump:
			self.dino_rect.y -= self.jump_vel
			self.jump_vel -= self.jump_grav
		if self.dino_rect.y > self.Y_POS + 10:
			self.dino_jump = False
			self.jump_vel = self.JUMP_VEL
			self.jump_grav = self.JUMP_GRAV
			self.dino_rect.y = self.Y_POS

	def draw(self, SCREEN):
		pass

	def getXY(self):
		return (self.dino_rect.x, self.dino_rect.y)


class Obstacle():
	def __init__(self, image, type):
		super().__init__()
		self.image = image
		self.type = type
		self.rect = self.image[self.type].get_rect()

		self.rect.x = SCREEN_WIDTH

	def update(self):
		self.rect.x -= game_speed
		if self.rect.x < - self.rect.width:
			obstacles.pop(0)

	def draw(self, SCREEN):
		pass

	def getXY(self):
		return (self.rect.x, self.rect.y)

	def getHeight(self):
		return y_pos_bg - self.rect.y

	def getType(self):
		return (self.type)


class SmallCactus(Obstacle):
	def __init__(self, image):
		self.type = random.randint(0, 2)
		super().__init__(image, self.type)
		self.rect.y = 345


class LargeCactus(Obstacle):
	def __init__(self, image):
		self.type = random.randint(0, 2)
		super().__init__(image, self.type)
		self.rect.y = 325


class Bird(Obstacle):
	def __init__(self, image):
		self.type = 0
		super().__init__(image, self.type)

		# High, middle or ground
		if random.randint(0, 3) == 0:
			self.rect.y = 345
		elif random.randint(0, 2) == 0:
			self.rect.y = 260
		else:
			self.rect.y = 300
		self.index = 0

	def draw(self, SCREEN):
		pass


class KeyClassifier:
	def __init__(self, state):
		pass

	def keySelector(self, distance, obHeight, speed, obType):
		pass

	def updateState(self, state):
		pass


def first(x):
	return x[0]


class KeySimplestClassifier(KeyClassifier):
	def __init__(self, state):
		self.state = state

	def keySelector(self, distance, obHeight, speed, obType):
		self.state = sorted(self.state, key=first)
		for s, d in self.state:
			if speed < s:
				limDist = d
				break
		if distance <= limDist:
			if isinstance(obType, Bird) and obHeight > 50:
				return "K_DOWN"
			else:
				return "K_UP"
		return "K_NO"

	def updateState(self, state):
		self.state = state


def playerKeySelector():
	userInputArray = pygame.key.get_pressed()

	if userInputArray[pygame.K_UP]:
		return "K_UP"
	elif userInputArray[pygame.K_DOWN]:
		return "K_DOWN"
	else:
		return "K_NO"


def playGame(aiPlayer, seed):
	global game_speed, x_pos_bg, y_pos_bg, points, obstacles
	random.seed (seed)

	run = True
	clock = pygame.time.Clock()
	player = Dinosaur()
	game_speed = 10
	x_pos_bg = 0
	y_pos_bg = 383
	points = 0
	obstacles = []
	death_count = 0
	spawn_dist = 0

	def score():
		global points, game_speed
		points += 0.25
		if points % 100 == 0:
			game_speed += 1


	while run:

		distance = 1500
		altitude = 0
		obHeight = 0
		obType = 2
		if len(obstacles) != 0:
			xy = obstacles[0].getXY()
			distance = xy[0]
			altitude = xy[1]
			obHeight = obstacles[0].getHeight()
			obType = obstacles[0]

		if GAME_MODE == "HUMAN_MODE":
			userInput = playerKeySelector()
		else:
			# userInput = aiPlayer.keySelector(game_speed, player, obType)
			# userInput = aiPlayer.keySelector(game_speed, obstacles, player)
			userInput = aiPlayer.keySelector([distance, altitude, game_speed])
			

		if len(obstacles) == 0 or obstacles[-1].getXY()[0] < spawn_dist:
			spawn_dist = random.randint(0, 670)
			if random.randint(0, 2) == 0:
				obstacles.append(SmallCactus(SMALL_CACTUS))
			elif random.randint(0, 2) == 1:
				obstacles.append(LargeCactus(LARGE_CACTUS))
			elif random.randint(0, 5) == 5:
				obstacles.append(Bird(BIRD))

		player.update(userInput)

		for obstacle in list(obstacles):
			obstacle.update()

		score()

		for obstacle in obstacles:
			if player.dino_rect.colliderect(obstacle.rect):
				death_count += 1
				return points


# Change State Operator

def change_state(state, position, vs, vd):
	aux = state.copy()
	s, d = state[position]
	ns = s + vs
	nd = d + vd
	if ns < 15 or nd > 1000:
		return []
	return aux[:position] + [(ns, nd)] + aux[position + 1:]


# Neighborhood

def generate_neighborhood(state):
	neighborhood = []
	state_size = len(state)
	for i in range(state_size):
		ds = random.randint(1, 10) 
		dd = random.randint(1, 100) 
		new_states = [change_state(state, i, ds, 0), change_state(state, i, (-ds), 0), change_state(state, i, 0, dd),
					  change_state(state, i, 0, (-dd))]
		for s in new_states:
			if s != []:
				neighborhood.append(s)
	return neighborhood


# Gradiente Ascent

def gradient_ascent(state, max_time):
	start = time.process_time()
	res, max_value = manyPlaysResults(KeySimplestClassifier(state), 3)
	better = True
	end = 0
	while better and end - start <= max_time:
		neighborhood = generate_neighborhood(state)
		better = False
		for s in neighborhood:
			aiPlayer = KeySimplestClassifier(s)
			res, value = manyPlaysResults(aiPlayer, 3)
			if value > max_value:
				state = s
				max_value = value
				better = True
		end = time.process_time()
	return state, max_value


from multiprocessing import Pool
from scipy import stats
import pandas as pd
import numpy as np
import shutil
import glob


def manyPlaysResults(aiPlayer, rounds):
	results = []
	with Pool (os.cpu_count ()-2) as p:
		results = p.starmap (playGame, zip ([aiPlayer]*rounds, range (rounds)))
	npResults = np.asarray(results)
	return_value = npResults.mean()
	if npResults.shape[0]>1:
		return_value -= npResults.std()
	return (results, return_value)


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
		self.state = self.scaler.fit_transform(np.asarray(self.state).reshape(initial_state_size,coord_size))
		# self.state = self.state.reshape(initial_state_size)

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

		# print(min(dist))
		index = dist.index(min(dist))
		dist = np.asarray(dist)
		idx = np.argpartition(dist, k)[:k]

		keys = [label_state[val] for val in idx]

		count_keys = Counter(keys)
		# print(count_keys)
		most_keys = count_keys.most_common()

		if len(most_keys) > 1 and most_keys[0][1] > most_keys[1][1]:
			return most_keys[0][0]
		else:
			return label_state[index]
		# if min(dist) < 0.2:
		#     return dist.index(min(dist))
		# else: 
		#     return -1

	def norm_state(self, params):
		"""
		Função que normaliza o vetor de coordenadas dos pontos classificados, e os dados atuais

		parametros:
		- params: Vetor de coordenadas dos dados atuais

		retorno:
		- params: Vetor normalizado
		"""

		params = self.scaler.transform(np.asarray(params).reshape(1,coord_size))
		params = params.reshape(coord_size)

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

		# if closest == -1:
		#     return "K_NO"
		# elif closest % 2 == 0:
		#     return "K_DOWN"
		# return "K_UP"


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

	aiPlayer = GAKeyClassifier(solution)
	# res, value = manyPlaysResults(3)
	value = playGame(aiPlayer, 10)

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
	# print(instance.pop_size)
	solution, solution_fitness, solution_idx = instance.best_solution()
	# print('best solution of this generation:', solution, 'fitness:', solution_fitness)
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
	gene_space = [None, [260, 300, 325, 345], range(0, 25)] * state_size

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
							# initial_population=initial_population,
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
	# print("Parameters of the best solution : {solution}".format(solution=solution))
	# print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

	return solution, solution_fitness


if __name__ == '__main__':
	# global initial_state_size, coord_size, label_state
	# initial_state_size = 7
	# coord_size = 3

	# label_state = ["K_UP", "K_DOWN", "K_NO"] * initial_state_size

	print('Genetic Algorithm IA')
	f = open("solutions.txt", "a")
	f.write(f'Start search\n')
	f.close()

	best_state, best_value = genetic_algorithm(initial_state_size, 360) 
	aiPlayer = GAKeyClassifier(best_state)

	print('Solução encontrada, indo para a etapa de desempenho')
	f = open("solutions.txt", "a")
	f.write(f'Final solution: {best_state}, fitness: {best_value}\n\n')
	f.close()

	res, value = manyPlaysResults(aiPlayer, 30)
	npRes = np.asarray(res)
	print('results:', res, '\nmean results:', npRes.mean(), '\nstd results:', npRes.std(), '\nmean - std', value)



