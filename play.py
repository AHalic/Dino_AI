import random
import pygame
from sys import exit
import numpy as np

# import classes
from Cloud import *
from Dinosaur import *
from SmallCactus import *
from LargeCactus import *
from Bird import *

# import all constants
import constants


def playerKeySelector():
    userInputArray = pygame.key.get_pressed()

    if userInputArray[pygame.K_UP]:
        return "K_UP"
    elif userInputArray[pygame.K_DOWN]:
        return "K_DOWN"
    else:
        return "K_NO"


def playGame():
    run = True
    clock = pygame.time.Clock()
    player = Dinosaur()
    cloud = Cloud()
    constants.game_speed = 10
    constants.x_pos_bg = 0
    constants.y_pos_bg = 383
    constants.points = 0
    font = pygame.font.Font('freesansbold.ttf', 20)
    constants.obstacles = []
    death_count = 0
    spawn_dist = 0

    def score():
        constants.points += 0.25
        if constants.points % 100 == 0:
            constants.game_speed += 1

        text = font.render("Points: " + str(int(constants.points)), True, (0, 0, 0))
        textRect = text.get_rect()
        textRect.center = (1000, 40)
        constants.SCREEN.blit(text, textRect)

    def background():
        image_width = constants.BG.get_width()
        constants.SCREEN.blit(constants.BG, (constants.x_pos_bg, constants.y_pos_bg))
        constants.SCREEN.blit(constants.BG, (image_width + constants.x_pos_bg, constants.y_pos_bg))
        if constants.x_pos_bg <= -image_width:
            constants.SCREEN.blit(constants.BG, (image_width + constants.x_pos_bg, constants.y_pos_bg))
            constants.x_pos_bg = 0
        constants.x_pos_bg -= constants.game_speed

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                exit()

        constants.SCREEN.fill((255, 255, 255))

        distance = 1500
        altitude = 0
        obHeight = 0
        obType = 2
        if len(constants.obstacles) != 0:
            xy = constants.obstacles[0].getXY()
            distance = xy[0]
            altitude = xy[1]
            obHeight = constants.obstacles[0].getHeight()
            obType = constants.obstacles[0]

        if constants.GAME_MODE == "HUMAN_MODE":
            userInput = playerKeySelector()
            if userInput == 'K_UP':
                print('------------')
                print('distance: ', distance, 'speed: ', constants.game_speed)
        else:
            # GS MODE
            if constants.AI_MODE == "GS" or constants.AI_MODE == "RB":
                userInput = constants.aiPlayer.keySelector(distance, obHeight, constants.game_speed, obType)
                # if distance > 0 and constants.game_speed > 15:
                #     print('distance: ', distance, 'speed: ', constants.game_speed)

            # GA MODE
            if constants.AI_MODE == 'GA':
                userInput = constants.aiPlayer.keySelector([distance, altitude, constants.game_speed])

        if len(constants.obstacles) == 0 or constants.obstacles[-1].getXY()[0] < spawn_dist:
            spawn_dist = random.randint(0, 670)
            if random.randint(0, 2) == 0:
                constants.obstacles.append(SmallCactus(constants.SMALL_CACTUS))
            elif random.randint(0, 2) == 1:
                constants.obstacles.append(LargeCactus(constants.LARGE_CACTUS))
            elif random.randint(0, 5) == 5:
                constants.obstacles.append(Bird(constants.BIRD))

        player.update(userInput)
        player.draw(constants.SCREEN)

        for obstacle in list(constants.obstacles):
            obstacle.update()
            obstacle.draw(constants.SCREEN)

        background()

        cloud.draw(constants.SCREEN)
        cloud.update()

        score()

        clock.tick(60)
        pygame.display.update()

        for obstacle in constants.obstacles:
            if player.dino_rect.colliderect(obstacle.rect):
                pygame.time.delay(2000)
                death_count += 1
                return constants.points


def manyPlaysResults(rounds):
    results = []
    for round in range(rounds):
        results += [playGame()]
    npResults = np.asarray(results)
    return (results, npResults.mean() - npResults.std())