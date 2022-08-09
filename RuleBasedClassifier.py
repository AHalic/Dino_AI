from experta import *
from KeyClassifier import *
from Bird import *

class RuleBasedPlayer(KnowledgeEngine):

    def setAction(self, action):
        self.action = action

    def getAction(self):
        return self.action

    @Rule(AND(Fact(speed=P(lambda x: x < 15)),
              Fact(distance=P(lambda x: x > 200 and x < 300)),
              NOT(Fact(action='K_DOWN'))))   
    def jumpSlow(self):
           self.retract(1)
           self.declare(Fact(action='K_UP'))
 
    @Rule(AND(Fact(speed=P(lambda x: x >= 15 and x < 17)),
              Fact(distance=P(lambda x: x > 250 and x < 400)),
              NOT(Fact(action='K_DOWN'))))   
    def jumpFast(self):
           self.retract(1)
           self.declare(Fact(action='K_UP'))

    @Rule(AND(Fact(speed=P(lambda x: x >= 17)),
              Fact(distance=P(lambda x: x > 170 and x < 500)),
              NOT(Fact(action='K_DOWN'))))   
    def jumpVeryFast(self):
           self.retract(1)
           self.declare(Fact(action='K_UP'))

    @Rule(AND(Fact(speed=P(lambda x: x >= 20)),
              Fact(distance=P(lambda x: x > 100)),
              NOT(Fact(action='K_DOWN'))))   
    def jumpUltraFast(self):
           self.declare(Fact(action='K_UP'))           

    @Rule(AND(Fact(obType=P(lambda x: isinstance(x, Bird))),
              Fact(obHeight=P(lambda x: x > 50))))   
    def getDown(self): 
        self.retract(1)
        self.declare(Fact(action='K_DOWN'))

    @Rule(AND(Fact(distance=P(lambda x: x <= 0)),
              Fact(action='K_UP')))   
    def stopJump(self):
        self.declare(Fact(action='K_DOWN'))

    @Rule(Fact(action=MATCH.action))
    def selectAction(self, action):
        self.setAction(action)

class RuleBasedKeyClassifier(KeyClassifier):
    def __init__(self):
        self.engine = RuleBasedPlayer()

    def keySelector(self, dist, obH, sp, obT):    
        self.engine.reset()
        self.engine.declare(Fact(action='K_NO'))
        self.engine.declare(Fact(distance=dist))
        self.engine.declare(Fact(obHeight=obH))
        self.engine.declare(Fact(speed=sp))
        self.engine.declare(Fact(obType=obT))
        self.engine.run()
        return self.engine.getAction()
