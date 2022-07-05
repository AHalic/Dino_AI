import constants

class Obstacle():
    def __init__(self, image, type):
        super().__init__()
        self.image = image
        self.type = type
        self.rect = self.image[self.type].get_rect()

        self.rect.x = constants.SCREEN_WIDTH

    def update(self):
        self.rect.x -= constants.game_speed
        if self.rect.x < - self.rect.width:
            constants.obstacles.pop(0)

    def draw(self, SCREEN):
        SCREEN.blit(self.image[self.type], self.rect)

    def getXY(self):
        return (self.rect.x, self.rect.y)

    def getHeight(self):
        return constants.y_pos_bg - self.rect.y

    def getType(self):
        return (self.type)
