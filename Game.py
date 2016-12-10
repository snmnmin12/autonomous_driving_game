import util
import math
from display import Display
from model import Model
from car import Car
from layout import Layout
from vector import Vec2d
import graphicsUtils
from const import Directions, Actions, Const
import optparse
import random
import copy
import time
from collections import Counter
from belief import MarginalInference


hostActions =  ['normal', 'acc', 'dec', 'stop', 'left', 'right']
otherActions = ['acc','dec','normal','stop']
actionReward = {'normal':3, 'acc':0, 'dec':-1, 'stop':-1, 'left': 0,'right': 0}

beliefstates = ['STOP','HESITATING','NORMAL','AGGRESSIVE']


def setToOne(vector):
    '''
    This fucntion is to ensure the car move at unit distance 
    for example, is the direction is a unit vector with angle 45,
    then the output for it should be (1, 1)
    '''
    length = vector.get_length()
    if length == 0:
        return length
    else:
        if vector[0]!= 0:
            return vector/vector[0]
        else:
            return vector/vector[1]

####if the car apply the action then it will 
# arrive at a new state
def ApplyAction(state, agentIndex, action):
    marigin = 1
    car = state.data.getCars()[agentIndex]
    #if checking: marigin = 5
    if action == 'normal':
        car.accelerate(car.friction*marigin)
        car.setWheelAngle(0)
    elif action == 'acc':
        car.accelerate(car.ACCELERATION*marigin)
        car.setWheelAngle(0)
    elif action == 'dec':
        car.accelerate(0)
        car.setWheelAngle(0)
    elif action == 'stop':
        car.velocity = Vec2d(0,0)
        car.setWheelAngle(0)
    elif action == 'left':
        car.setWheelAngle(-45)
        car.accelerate(car.ACCELERATION)
    elif action == 'right':
        car.setWheelAngle(45)
        car.accelerate(car.ACCELERATION)
    else:
        print "Illega Action"
    car.update()
## check if the car has reach the next states
def ReachNext(goal,state):
    pos = state.data.getJunior().pos
    pos = util.corToCenter(pos)
    pos = Vec2d(pos[0], pos[1])
    return pos.get_distance(goal) < Const.BELIEF_TILE_SIZE*0.5



class GameState(object):

    '''
    get legal action for the host car, all action in bounds
    '''
    def getLegalActions(self, agentIndex = 0):
        '''
        Here I will generate the leage actoin for all the cars, first I will consider all the actions 
        as above, but I will check the state after action taken to make sure the car will not collide with the 
        fixed obstacles 

        for the special action like left or right, I have to estimate the final destimation after action taken to make sure
        the car will stay at center of the road

        output is the dictionay for action and destination or state
        '''
        safemargin = 1.0
        legalActions = dict()
        actionlist = hostActions
        if agentIndex != 0: actionlist = otherActions
  
        for action in actionlist:

            model = Model(self.data)
            state = GameState(model)
            car = model.getCars()[agentIndex]
            #car = model.getJunior()
            ### apply the action to get new state 
            ApplyAction(state, agentIndex, action)

            if action == 'stop':
                legalActions[action] = car.pos
                continue

            ### check the new new state not collide with the fixed obstacles
            bounds = car.getBounds()
            isinBound = True 
            if model.checkCollision(car):
                isinBound = False

            if isinBound:
                ####this is for the left turn and right turn only, after taken this actoin : 
                if (action  == 'left' or action == 'right'): 
                    ndir = setToOne(car.dir)
                    pos = Vec2d(util.corToCenter(car.pos + ndir * Const.BELIEF_TILE_SIZE))
                    legalActions[action] = pos
                else: legalActions[action] = car.pos


        return legalActions
                

    def generateSuccessor(self, agentIndex, actions):
        '''
        get the successor state for the actions for further predictions
        '''
        model = Model(self.data)
        state = GameState(model)
        car = state.data.getCars()[agentIndex]
        action = actions[0]
        ApplyAction(state, agentIndex, action)
        return state


    def getNumAgents(self):
        return len(self.data.getCars())
    
    #####################################
    #####################################
    def __init__( self, model):
		self.data = model
      

class MultiAgentSearchAgent():
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the  ExpectimaxCarAgent.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.depth = int(depth)

    def evaluationFunction(self, currentGameState):
        '''
        Evaluation function need to check many features

        1. crash with the surroundings including othercars 
        2. distance to goal 
        3. distance to the neareast other cars, if it is two close, the score is less 
        '''
        mycar = currentGameState.data.getJunior()
        model = currentGameState.data
        if model.checkCollision(mycar):
            score = -float('inf')
            return score 
     
        goal = currentGameState.data.goal
        score = 0
        score += 100*(2-abs(goal[0]-mycar.pos[0])/960)
        score += 100*(2-abs(mycar.pos[1] - goal[1])/100)
        #score += 100*(5-util.manhattanDistance(goal, mycar.pos)/100)
        ### the angle between car direction with the goal center is

        ###ge neareast other cars 
        distance = float('inf')
        obstaclecar = None 
        cars = currentGameState.data.getOtherCars()
        if not cars: return score
        
        # make sure it stay away from other cars
        for car in cars:
            cardis = util.manhattanDistance(car.pos, mycar.pos)
            if cardis < distance:
                distance = cardis
                obstaclecar = car
        if obstaclecar.pos[0]> mycar.pos[0] and \
               (obstaclecar.pos[0] - mycar.pos[0]) < Const.BELIEF_TILE_SIZE*2\
                and abs(obstaclecar.pos[1] - mycar.pos[1]) < Car.WIDTH/2:
                  score -= 50

        # # # do not make turns when cars are very close to each other get too close to my car
        if  abs(obstaclecar.pos[0] - mycar.pos[0]) < Car.WIDTH*4:
            #print "I am here inside the special case: "
            score -=  5*abs(mycar.dir.get_angle_between(obstaclecar.dir))/45

        # we have to check when it reach it's final destination, it will collide with others
        # this is especailly important when make right or left turns
        pos = mycar.pos
        #mdir = mycar.dir
        mdir = setToOne(mycar.dir)
        '''
         I did this because, sometimes the destination is safe, but when the car moves toward to destination
         it will collide with many other cars, so i want to gradually move the car to the next state, to make sure 
         it will not crash with other cars
        '''

        steps  = 3
        for step in range(steps): 
          mycar.pos += mdir * Const.BELIEF_TILE_SIZE/steps
          if step == steps - 1: mycar.pos = Vec2d(util.corToCenter(mycar.pos))
          if model.checkCollision(mycar):  return -float('inf')
        # mycar.pos += mdir * Const.BELIEF_TILE_SIZE
        # mycar.pos = Vec2d(util.corToCenter(mycar.pos))
        # if model.checkCollision(mycar):  return -float('inf')
        # # now we have to check the other car moves, so to know what happend next
          velocity = mycar.velocity.get_length()
          if velocity != 0:
               time = ((mycar.pos - pos).get_length())/velocity
               for car in cars:
                  car.pos += car.velocity*time
                  if model.checkCollision(mycar):
                    return -float('inf')

        return score

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
        evaluation function for depth 2      
    """
    def value(self, gameState, agentindex, depth):
        if depth == self.depth or gameState.data.checkCollision(gameState.data.getJunior()):
            return self.evaluationFunction(gameState)
        agentindex =  agentindex % gameState.getNumAgents()
        if agentindex == 0:
            depth += 1
            return self.maxvalue(gameState, agentindex, depth)
        else:
            return self.expecvalue(gameState, agentindex, depth)       
        return 0

    def maxvalue(self, gameState, agentindex, depth):
        if depth == self.depth:
           return self.evaluationFunction(gameState)
        score = -float("inf")
        actions = gameState.getLegalActions(agentindex)
        for action in actions.items():
            newState = gameState.generateSuccessor(agentindex, action)
            score  = max(score, self.value(newState,agentindex+1, depth) + actionReward[action[0]])
            # print 'I am inside the maxvalue: ', action, score
        #return score
        return score
    
    def expecvalue(self, gameState, agentindex, depth):
        '''
        In this function, I can not use the expectation because the computation is quite slow 
        so, i just assume the other car only have the action normal
        '''
        score = 0.0
        actions = gameState.getLegalActions(agentindex)
        prob = 1/float(len(actions))
        act = 'normal'
        if act in actions.keys():
           action = (act, actions[act])
        else: 
            act = random.choice(actions.keys())
            action = (act,actions[act])
        newState = gameState.generateSuccessor(agentindex, action)
        score = self.value(newState, agentindex+1, depth) + actionReward[action[0]]
        return score

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        bestAction = 'stop'
        numAgents = gameState.getNumAgents()-1
        actions = gameState.getLegalActions(0)
        score =  -float('inf')
        if len(actions) == 1:
            return actions.keys()[0], actions.values()[0] 
        for action in actions.items():
             newgamestates = gameState.generateSuccessor(0, action)
             newscore = self.value(newgamestates, 1, 0) + actionReward[action[0]]
             if score < newscore:
                score = newscore
                bestAction = action[0]
        return bestAction, actions[bestAction]




class Control(object): 

    def __init__(self):
        self.layout = Layout(Const.WORLD)
        Display.initGraphics(self.layout)
        self.model = Model(self.layout)
        self.master = self.model.getJunior()
        self.gamestate = GameState(self.model)
        self.agents = self.model.getCars()
        self.agenNum = len(self.agents)
        self.quit = False 
        self.victory = False 
        self.collision = False 

    def run(self):

        self.render()
        self.iteration = 0
        remove = True
        goal = self.master.pos
        agent = ExpectimaxAgent(1)
        goal = None 
        test = ['left', 'acc','acc','right']
        remove = True
        i = 0
        while not self.isGameOver():
            keys = Display.getKeys()
            if 'q' in keys:
                self.quit = True
            probflag = False
            for index in xrange(self.agenNum):
                car = self.model.getCars()[index]
                oldDir = Vec2d(car.dir.x, car.dir.y)
                oldPos = Vec2d(car.pos.x, car.pos.y)
                actions = self.gamestate.getLegalActions(index)
                if car == self.master:
                    if car.reachgoal(goal):
                       action = car.decisionMaking(agent, self.gamestate, actions)
                       goal = action[1]
                    else:
                        action = ('acc',goal)
                else:
                    if (not probflag): 
                        self.infer(self.model)
                        probflag = True
                    act = 'acc'
                    if act in actions.keys():
                        action  =  (act, actions[act])
                    else:
                        act = random.choice(actions.keys())
                        action = (act, actions[act]) 

                car.applyAction(self.gamestate, action)
                i += 1
                

                newPos = car.getPos()
                newDir = car.getDir()

                deltaPos = newPos - oldPos
                deltaAngle = oldDir.get_angle_between(newDir)
                
                self.victory = self.model.checkVictory()
                self.collision = self.checkCollision(self.master)
                #display the cars now
                Display.move(car, deltaPos)
                Display.rotate(car, deltaAngle)     

            Display.graphicsSleep(0.05)
            self.iteration += 1
        if not self.quit:
            self.outputGameResult()
        return self.quit
        
    def freezeFrame(self):
        while True:
            keys = Display.getKeys()
            if 'q' in keys: return
            Display.graphicsSleep(0.1)
        
    def outputGameResult(self):
        collided = self.checkCollision(self.master)
        for car in self.model.getOtherCars():
            Display.drawCar(car)
        print '*********************************'
        print '* GAME OVER                     *'
        if collided:
            print '* CAR CRASH!!!!!'
        else:
            print '* You Win!'
        print '*********************************'    

    def checkCollision(self, car):
        return self.model.checkCollision(car)
            
    def isGameOver(self):
        if self.quit or self.victory:
            return True
        return self.collision

    def observe(self, state):

        self.master.makeObse(state)
        cars = self.model.getOtherCars()
        for index in xrange(len(cars)):
            car = cars[index]
            inference = car.getInference(index+1, state)
            inference.observe(state)
            # inference.observe(car.history)

    def infer(self, state):
        #try:
        self.observe(state)
        # except: pass
             # input("Press ENTER to continue.")
        beliefs = []
        colors = ['gray','green','yellow','red']
        for k, car in enumerate(self.model.getOtherCars()):
            belief = car.getInference(k+1, state).getBelief()
            # color = car.getColor()
            # Display.updateBelief(color, belief)
            beliefs.append(belief)
            m = max(belief)
            indexes = [i for i,j in enumerate(belief) if j == m]
            # print 'Car %s'%(k),indexes
            Display.colorchange(car,colors[random.choice(indexes)])

        # self.model.setProbCar(beliefs)
        
    def render(self):
        Display.drawBelief(self.model)
        Display.drawBlocks(self.model.getBlocks())
        Display.drawLine(self.model.getLine())
        for car in self.model.getOtherCars():
            Display.drawCar(car)
        Display.drawCar(self.model.junior)
        Display.drawFinish(self.model.getFinish())
        graphicsUtils.refresh()


if __name__ == '__main__':
    
    parser = optparse.OptionParser()
    parser.add_option('-l', '--layout', dest='layout', default='road')
    (options, _) = parser.parse_args()
    Const.WORLD = options.layout
    controller = Control()
    quit = controller.run()
    if not quit:
        controller.freezeFrame()
    print 'closing...'
    Display.endGraphics()