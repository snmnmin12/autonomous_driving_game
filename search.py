import util
import math
from display import Display
from model import Model
from car import*
from layout import Layout
from vector import Vec2d
import graphicsUtils
from const import Directions, Actions, Const
import optparse
import random
import copy
import time
from collections import Counter
''' below is the configuration for the search probelm used in the current problem 
#################################################################################
'''
def genericSearch(problem, container, heuristic = None):
    
    ###this stime i have added the previous direction to the current state
    open = container
    closed = set()
    problemstate = problem.getStartState()
    position = problemstate[0]
    cost = 0
    h = 0
    state = (position,[],cost)
    #check the generic case for the different data structures
    if isinstance(container, util.Stack) or isinstance(container, util.Queue):
         open.push(state)
    else:
       if heuristic:
         h = heuristic(position, problem)
       open. push(state, h + cost)

    while not open.isEmpty():
        state = open.pop()
        position = state[0]
        direction = state[1]
        cost = state[2]
        action = None 
        if direction:
            action = direction[-1]
        #if the current state is the goal state    
        if problem.isGoalState((position, action)): 
            return state[1]

        if position in closed: continue
        closed.add(position)
        #ge the successor
        newproblemstate = (position, action)
        successor = problem.getSuccessors(newproblemstate)

        if not successor: continue

        #check all the successors 
        for item in successor:
           if not item[0][0] in closed:
              chcost = cost + item[2]
              state = (item[0][0],direction+[item[1]],chcost)
              if isinstance(container, util.Stack) or isinstance(container, util.Queue):
                  open.push(state)  
              else:
                #add heuristic functions to the existing function
                  if heuristic:
                    h = heuristic(item[0][0], problem)
                  open. push(state, chcost + h)
                           
    return None


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """  
    return genericSearch(problem, util.Stack())
   
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    return genericSearch(problem, util.Queue())
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    return genericSearch(problem, util.PriorityQueue())
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    return genericSearch(problem, util.PriorityQueue(), heuristic)
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

###################################################################################
##################################################################################
'''serach configuration has just completed and we can do something else '''


class SearchProblem():
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """
    def __init__(self, model, costFn = lambda x: 1):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        #self.walls = gameState.getWalls()
        self.model = model
        pos = copy.copy(model.getJunior().getPos())
        position = (pos[0], pos[1])
        direction = 'E'
        self.startState = (position, direction)
        self.goal = model.getFinish().getCenter()
        self.costFn = costFn

        # For display purposes

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        #isGoal = state == self.goal
        newcar = Car(Vec2d(state[0][0], state[0][1]), 
            'east', 
            Vec2d(0, 0))
        #bounds = newcar.getBounds()
        bounds = newcar.pos
        isGoal = False 
        x,y = self.model.finish.getCenter()
        if abs(x-newcar.pos[0]) < 12 and abs(y-newcar.pos[1]) < 12:
            isGoal = True
        # for point in bounds:
        #     if self.model.finish.containsPoint(point[0], point[1]):
        #         isGoal = True
        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """
        successors = []
        safemargin = 1.5
        distanceToObject = 9999
        def choosenearest(pos, othercars):
              distance = 9999999
              nearest = None 
              for car in othercars:
                  cardistance = util.manhattanDistance(pos, car.pos)
                  if cardistance < distance:
                     distance = cardistance
                     nearest = car 
              return nearest 

        carProb = self.model.carProb
        cellProbability = 0
        x,y  = state[0]
        olddir = state[1] or 'E'
        # if state[1]: 
        #      oldvec = Actions.directionToVector(state[1])
        #      oldvec = Vec2d(oldvec)
        # else: oldvec = self.model.junior.getDir()

        for action in Directions.directions:
            # check the angle difference
            xdir, ydir = Actions.directionToVector(olddir)
            dcol, drow = Actions.directionToVector(action)
            oldvec = Vec2d(xdir, ydir)
            newvec = Vec2d(dcol,drow)
            angdiff = abs(oldvec.get_angle_between(newvec))
            if angdiff > 90: continue


            nextx, nexty = int(x + dcol*Const.BELIEF_TILE_SIZE*Const.ScaleRatio), \
            int(y + drow*Const.BELIEF_TILE_SIZE*Const.ScaleRatio)
            
            newcar = Car(Vec2d(nextx, nexty), 
            'east',
            Vec2d(0, 0) )
            #we have to keep some safety margin to be safe
            bounds = Car.getBoundsforAllCar(newcar, Car.LENGTH*safemargin, Car.WIDTH*safemargin)
            isinBound = True 
            for point in bounds:
               if not self.model.inBounds(point[0],point[1]):
                    isinBound = False 
                    break

            if not isinBound: continue

            newPos = (nextx, nexty)
            nextState = (newPos, action)
            direction = (dcol, drow)
            distancetocar = util.manhattanDistance(newPos, self.model.junior.pos)
           
            if not self.model.obstaclesPosition is None:
                 #nearestcar = choosenearest((nextx, nexty), self.model.getOtherCars())
                 #newcar.dir = Vec2d(direction)
                 neareastcar = choosenearest(newcar.pos, self.model.getOtherCars())
                 collided = neareastcar.collides(newPos, newcar.getBounds())
                 #collided = self.model.checkCollision(newcar)
                 if collided: continue
                 newcol = util.xToCol(nextx)
                 newrow = util.yToRow(nexty)
                 if self.model.obstaclesPosition[0] == newrow and self.model.obstaclesPosition[1] == newcol:
                    continue
                 cellProbability = carProb.getProb(newrow, newcol)

            # id do not need inference now
            neareastcar = choosenearest(newcar.pos, self.model.getOtherCars())
            collided = neareastcar.collides(newPos, newcar.getBounds())
             #collided = self.model.checkCollision(newcar)
            if collided: continue
                 
           # print evaluationfunction((state, direction), self.model)
            # angle difference is also to be considered as penalty
            # distance = evaluationfunction((state[0], direction), self.model)
            # if distance is None: distance = 1000
            cost = self.costFn(nextState) + 0.5*angdiff + 50*cellProbability/distanceToObject
            #above is the evaluastion functions used
            successors.append( (nextState, action, cost) )


        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

def evaluationfunction(carstate, model):
    pos = Vec2d(carstate[0])
    direction = Vec2d(carstate[1])
    mindirection = 90
    minblock = None 
    # find the neareast block in the moving direction
    for block in model.getBlocks():
        center = block.getCenter()
        blockvec = Vec2d(center[0]-pos[0], center[1]-pos[1])
        angdiff = abs(blockvec.get_angle_between(direction))
        if angdiff < mindirection:
            minblock = block
            mindirection = angdiff
    #check the distance between to the neareast block
    distance = None 
    if not minblock is None:
        initial = 5
        newvec = copy.copy(direction)
        for i in range(1,101):
           newvec.set_length(initial*i)
           if minblock.containsPoint(pos[0]+newvec[0], pos[1] + newvec[1]):
             break
        distance = newvec.get_length() 
    return distance 


def manhattanHeuristic(position, problem):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])
############################################################
'''Now we can go the control part for the car running'''
############################################################
class  Master(Car):
    
    ACCELERATION = 1.4
    FRICTION = 1
    WHEEL_TURN = 2.0
    WHEEL_TURN_HUMAN = 1.0
    MAX_WHEEL_ANGLE = 40.0
    MAX_SPEED = 5.0
    MIN_PROB = 0.02

    def __init__(self, pos, direction, velocity, model = None):
        Car.__init__(self, pos, direction, velocity)
        self.maxSpeed = Master.MAX_SPEED
        self.friction = Master.FRICTION
        self.maxWheelAngle = Master.MAX_WHEEL_ANGLE
        self.maxWheelAngle = Master.MAX_WHEEL_ANGLE *2
        self.pre = -1
        self.nodeId = 0
        self.burnInIterations = 30
        ''' Used later to check the reason why the car does not move due to the obstacles'''
        self.driveForward = True
        ''' we need to track the obstacles's position, so we can planned later'''
        self.obstaclesPosition = None 
        self.prepath = None
        self.model = model
        self.stopflag = False
        # for control of stop


    def isJunior(self):
        return True
   # def autonomousAction(self, beliefs, agentGraph):
    def autonomousAction(self, path, carProb = None):
        if path is None: return
        oldPos = Vec2d(self.pos.x, self.pos.y)
        oldDir = Vec2d(self.dir.x, self.dir.y)
        oldVel = Vec2d(self.velocity.x, self.velocity.y)
        actions = self.getAutonomousActions(path, carProb)
        assert self.pos == oldPos
        assert self.dir == oldDir
        assert self.velocity == oldVel
        if Car.DRIVE_FORWARD in actions:
            percent = actions[Car.DRIVE_FORWARD]
            sign = 1
            if percent < 0: sign = -1
            percent = abs(percent)
            percent = max(percent, 0.0)
            percent = min(percent, 1.0)
            percent *= sign
            self.accelerate(Master.ACCELERATION * percent)
        # else: self.velocity  = Vec2d(0, 0)
        if Car.TURN_WHEEL in actions:
            turnAngle = actions[Car.TURN_WHEEL]
            self.setWheelAngle(turnAngle)
        
    # def getAutonomousActions(beliefs, agentGraph, actions):
    #@staticmethod
    def getAutonomousActions(self, path, carProb = None):
        if self.burnInIterations > 0:
            self.burnInIterations -= 1
            return[]
        if not path: return []
        ## if the path has changed, so we have to start relabel again
        replanflag = False
        if not self.prepath == path:
            self.reset()
            replanflag = True 
            self.prepath = path
        # Choose a next node to drive towards. Note that you can ask
        # a if its a terminal using node.isTerminal
        ''' find the next location on the map'''
        nextId = self.nodeId + 1
        if self.nodeId >= len(path):
            self.nodeId = self.pre
        if nextId >= len(path):
            nextId  = self.nodeId
        nextpos = Vec2d(path[nextId])
        if nextpos.get_distance(self.pos) < Const.BELIEF_TILE_SIZE*0.5:
            self.pre = self.nodeId
            self.nodeId = nextId
            nextId = self.nodeId + 1

        if nextId >= len(path):
            nextId  = self.nodeId
        ''' i have found the next position on the map to go '''
        # given a next node, drive towards that node. Stop if you
        # are too close to another car

        goalPos = Vec2d(path[nextId])
        vectorToGoal = goalPos - self.pos
        wheelAngle = -vectorToGoal.get_angle_between(self.dir)
        sign = 1
        if wheelAngle < 0: sign = -1
        wheelAngle = min(abs(wheelAngle), self.maxWheelAngle)*sign
        #############
        '''check the drive forward conditions satisfied or not'''
        # if carProb and not replanflag:  
        #     self.driveForward = not self.isCloseToOtherCar(carProb)
        
        actions = {
            Car.TURN_WHEEL: wheelAngle
        }
        # if in intersection
        # set timer timer 
        if self.stopStatus and self.model.inIntersection(goalPos[0], goalPos[1]):
            if self.stopStatus and not self.stopflag:
                import time
                start_time = time.time()
                for i in range(30000):
                    pass
                self.stopflag = True
            actions[Car.DRIVE_FORWARD] = 0.6
            return actions
        ID = nextId 
        i = 0
        while ID < len(path)-1 and i < 3:
           ID += 1
           i += 1
        fpos = Vec2d(path[ID])
        if not self.stopflag:
            if  not self.model.inIntersection(self.pos[0], self.pos[1])\
                and self.model.inIntersection(fpos[0], fpos[1]):
               actions[Car.DRIVE_FORWARD] = 0.4
               return actions
            elif not self.model.inIntersection(self.pos[0], self.pos[1]) and \
                 self.model.inIntersection(goalPos[0], goalPos[1]) :
               actions[Car.DRIVE_FORWARD] = 0.0
               return actions
        #end check the intersection case 
        if self.driveForward:
            actions[Car.DRIVE_FORWARD] = 1.0
            if abs(wheelAngle) < 20:
               actions[Car.DRIVE_FORWARD] = 1.0
            elif abs(wheelAngle) < 45:
               actions[Car.DRIVE_FORWARD] = 0.5
            else:
               actions[Car.DRIVE_FORWARD] = 0.2
        else:
             actions[Car.DRIVE_FORWARD] = -0.01
            # self.velocity = Vec2d(0, 0)
        if replanflag:
            actions[Car.DRIVE_FORWARD] = 1.0
            # print actions

        # if self.model.inIntersection(self.pos[0], self.pos[1]):
        #     actions[Car.DRIVE_FORWARD] *= 0.5

        return actions

    def applyActions(self, actions):
        moveForward = Car.DRIVE_FORWARD in actions
        turnLeft = Car.TURN_LEFT in actions
        turnRight = Car.TURN_RIGHT in actions
        
        assert not (turnLeft and turnRight)
        
        if moveForward:
            self.accelerate(Master.ACCELERATION)
        if turnLeft:
            self.turnLeft(Master.WHEEL_TURN)
        if turnRight:
            self.turnRight(Master.WHEEL_TURN)

    def isCloseToOtherCar(self, beliefOfOtherCars):
        newBounds = []
        offset = self.dir.normalized() * 1.5 * Car.LENGTH
        newPos = self.pos + offset
        row = util.yToRow(newPos.y)
        col = util.xToCol(newPos.x)
        # row = min(row, beliefOfOtherCars.getNumRows()-1)
        # col = min(col, beliefOfOtherCars.getNumCols()-1)
        # print row, col, beliefOfOtherCars.getNumRows(),beliefOfOtherCars.getNumCols()
        p = beliefOfOtherCars.getProb(row, col)
        if p > Master.MIN_PROB:
            self.obstaclesPosition = (row, col)
            return True 
        else: return False
        # return p > Master.MIN_PROB

    def reset(self):
        self.pre    = -1
        self.nodeId = 0
    # # Funciton: Chose Next Id
    # # ---------------------
#############################################################
class MyModel(Model):

    def __init__(self, layout):
        super(MyModel, self).__init__(layout)
        startX = layout.getStartX()
        startY = layout.getStartY()
        startDirName = layout.getJuniorDir()
        startDirName = layout.getJuniorDir()
        mycar = Master(
            Vec2d(startX, startY), 
            startDirName, 
            Vec2d(0, 0),
            self
            )
        index = self.cars.index(self.junior)
        self.cars[index] = mycar
        self.junior = mycar
        self.obstaclesPosition = None
        self.carProb = None

####################################################
class SearchAgent(object):

    def __init__(self, model, fn = 'aStarSearch', heuristic = 'manhattanHeuristic'):
        self.problem = SearchProblem(model)
        func = globals()[fn]
        heur = globals()[heuristic]
        searchFunction = lambda x: func(x, heuristic=heur)
        self.actions = searchFunction(self.problem)
        # self.path = self.actionsToPath(actions)
    def actionsToPath(self, actions):
        if not actions: return None
            # actions = ['E']
        landmarks = {}
        startState = self.problem.getStartState()
        pos = startState[0]
        for i in range(len(actions)):
            x, y =Actions.directionToVector(actions[i])
            value = (Const.BELIEF_TILE_SIZE*x*Const.ScaleRatio + pos[0], \
                  Const.BELIEF_TILE_SIZE*y*Const.ScaleRatio + pos[1])
            pos = value
            landmarks[i] = value
        return landmarks
    def getActions(self):
        return self.actions
    def getPath(self):
        return self.actionsToPath(self.actions)

######################################
## Control object for the current file 
######################################

class SearchControl(object): 
    def __init__(self):
        self.layout = Layout(Const.WORLD)
        Display.initGraphics(self.layout)
        self.model = MyModel(self.layout)
        self.master = self.model.getJunior()
        self.plan = SearchAgent(self.model)
        self.quit = False 
        self.victory = False 
        self.collision = False 

    def run(self):
        self.render()
        self.iteration = 0
        path = self.plan.getPath()
        remove = True
        while not self.isGameOver():
            self.printPath(path)
            keys = Display.getKeys()
            if 'q' in keys:
                self.quit = True
            oldDir = Vec2d(self.master.dir.x, self.master.dir.y)
            oldPos = Vec2d(self.master.pos.x, self.master.pos.y)
            self.master.autonomousAction(path)
            self.master.update()

            self.victory = self.model.checkVictory()
            self.collision = self.checkCollision(self.master)

            newPos = self.master.getPos()
            newDir = self.master.getDir()

            deltaPos = newPos - oldPos
            deltaAngle = oldDir.get_angle_between(newDir)

            #display the cars now
            Display.move(self.master, deltaPos)
            Display.rotate(self.master, deltaAngle)             
            Display.graphicsSleep(0.05)
            self.iteration += 1

        return self.quit
    def printPath(self, path):
        Display.drawPath(path);
        
    def freezeFrame(self):
        while True:
            keys = Display.getKeys()
            if 'q' in keys: return
            Display.graphicsSleep(0.1)
        

    def checkCollision(self, car):
        bounds = car.getBounds()
        # check for collision with fixed obstacles
        for point in bounds:
            if not self.model.inBounds(point.x, point.y): return True
        return False
            
    def isGameOver(self):
        if self.quit or self.victory:
            return True
        return self.collision

    def round(self, num):
        return round(num * 1000) / 1000.0


    def juniorUpdate(self):
        junior = self.model.junior
        junior.action()
        self.move([junior])
    
        
    def act(self):
        start = time.time()
        for car in self.model.getOtherCars():
            car.action()
        self.actionTime += time.time() - start

    def move(self, cars):
        for car in cars:
            start = time.time()
            oldDir = Vec2d(car.dir.x, car.dir.y)
            oldPos = Vec2d(car.pos.x, car.pos.y)
            car.update()
            newPos = car.getPos()
            newDir = car.getDir()
            deltaPos = newPos - oldPos
            deltaAngle = oldDir.get_angle_between(newDir)
            self.updateTime += time.time() - start
            if Const.SHOW_CARS or car.isJunior():
                self.moveCarDisplay(car, deltaPos, deltaAngle)
            
            if self.isLearning:
                self.learner.noteCarMove(oldPos, newPos)
    

    def moveCarDisplay(self, car, deltaPos, deltaAngle):
        start = time.time()
        Display.move(car, deltaPos)
        Display.rotate(car, deltaAngle)
        self.drawTime += time.time() - start
        
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
    parser.add_option('-d', '--display', dest='display', default=False, action='store_false')
    parser.add_option('-l', '--layout', dest='layout', default='small')
    (options, _) = parser.parse_args()
    Const.WORLD = options.layout
    

    controller = SearchControl()
    quit = controller.run()
    if not quit:
        controller.freezeFrame()
    
    # print 'closing...'
    # Display.endGraphics()




