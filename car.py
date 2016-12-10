from vector import Vec2d
import util
from const import Const
import math
import random
from inference import *

hostActions =  ['normal', 'acc', 'dec', 'stop', 'left', 'right']
otherActions = ['acc','dec','normal','stop']
actionReward = {'normal':3, 'acc':0, 'dec':-1, 'stop':-1, 'left': 0,'right': 0}

beliefstates = ['STOP','HESITATING','NORMAL','AGGRESSIVE']

class Car(object):
    
    REVERSE = 'Reverse'
    DRIVE_FORWARD = 'Forward'
    TURN_LEFT = 'Left'
    TURN_RIGHT = 'Right'
    TURN_WHEEL = 'Wheel'
    
    MAX_WHEEL_ANGLE = 130.0
    MAX_SPEED = 16.0
    MAX_ACCELERATION = 10.0
    FRICTION = 2.0
    LENGTH = 25.0
    WIDTH = 12.5 
    RADIUS = math.sqrt(LENGTH ** 2 + WIDTH ** 2)
    
    def __init__(self, pos, dirName, velocity):
        self.initialPos = Vec2d(pos.x, pos.y)
        self.pos = pos
        self.velocity = velocity
        direction = self.dirFromName(dirName)
        self.dir = direction
        self.wheelAngle = 0
        self.maxSpeed = Car.MAX_SPEED
        self.friction = Car.FRICTION
        self.maxWheelAngle = Car.MAX_WHEEL_ANGLE
    # def copy()
    #     from copy import deepcopy
    #     return deepcopy(self)
    def copyAgent(self):
        from copy import copy, deepcopy
        #agent = Agent(self.pos, self.agentGraph, self.model)
        agent = Car(self.pos, 'east', self.velocity)
        # agent.pos = deepcopy(self.pos)
        agent.dir = deepcopy(self.dir)
        return agent
    def getPos(self):
        return self.pos
    
    def getDir(self):
        return self.dir

    def getVel(self):
         return self.velocity
        
    def getObservation(self, junior):
        #Sonar
        dist = (junior.pos - self.pos).get_length()
        std = Const.SONAR_STD
        return SonarObservation(random.gauss(dist, std))
        
        #Radar
        '''errorForwards = Const.RADAR_NOISE_STD
        errorSide = Const.RADAR_NOISE_STD
        noiseForwards = random.gauss(0, errorForwards)
        noiseSide = random.gauss(0, errorSide)
        
        dirVec = self.dir.normalized()
        sideVec = dirVec.perpendicular()
        point = Vec2d(self.pos.x, self.pos.y)
        point += dirVec * noiseForwards + sideVec * noiseSide
        return Observation(point)'''
        

    def turnCarTowardsWheels(self):
        if self.velocity.get_length() > 0.0:
            self.velocity.rotate(self.wheelAngle)
            self.dir = Vec2d(self.velocity.x, self.velocity.y)

    def update(self):
        self.turnCarTowardsWheels()
        #self.applyFriction()
        self.pos += self.velocity
        self.turnWheelsTowardsStraight()
        self.applyFriction()
    
    def turnWheelsTowardsStraight(self):
        # if self.wheelAngle < 0:
        #     self.wheelAngle += 0.7
        #     if self.wheelAngle > 0:
        #         self.wheelAngle = 0.0
        # if self.wheelAngle > 0:
        #     self.wheelAngle -= 0.7
        #     if self.wheelAngle < 0:
        #         self.wheelAngle = 0.0
        self.wheelAngle = 0.0
        
    def decellerate(self, amount):
        speed = self.velocity.get_length()
        if speed == 0: return
        frictionVec = self.velocity.get_reflection().normalized()
        frictionVec *= amount
        self.velocity += frictionVec
        angle = self.velocity.get_angle_between(frictionVec)
        if abs(angle) < 180:
            self.velocity = Vec2d(0, 0)
    
    def applyFriction(self):
        self.decellerate(self.friction)
        
    def setWheelAngle(self, angle):
        self.wheelAngle = angle
        if self.wheelAngle <= -self.maxWheelAngle:
            self.wheelAngle= -self.maxWheelAngle
        if self.wheelAngle >= self.maxWheelAngle:
            self.wheelAngle = self.maxWheelAngle
        
    def turnLeft(self, amount):
        self.wheelAngle -= amount
        if self.wheelAngle <= -self.maxWheelAngle:
            self.wheelAngle= -self.maxWheelAngle
        
    def turnRight(self, amount):
        self.wheelAngle += amount
        if self.wheelAngle >= self.maxWheelAngle:
            self.wheelAngle = self.maxWheelAngle
    
    def accelerate(self, amount):
        amount = min(amount, Car.MAX_ACCELERATION)
        if amount < 0:
            self.decellerate(amount)
        if amount == 0: return
        if amount > 0:
            acceleration = Vec2d(self.dir.x, self.dir.y).normalized()
            acceleration *= amount
            self.velocity += acceleration
            self.velocity = max(0, self.velocity)
            if (self.velocity.get_length() >= self.maxSpeed):
                self.velocity.set_length(self.maxSpeed)
           
    # http://www.gamedev.net/page/resources/_/technical/game-programming/2d-rotated-rectangle-collision-r2604 
    def collides(self, otherPos, otherBounds):
        diff = otherPos - self.pos
        dist = diff.get_length()
        if dist > Car.RADIUS * 2: return False
        
        bounds = self.getBounds()
        vec1 = bounds[0] - bounds[1]
        vec2 = otherBounds[0] - otherBounds[1]
        axis = [
            vec1,
            vec1.perpendicular(),
            vec2,
            vec2.perpendicular()
        ]
        for vec in axis:
            (minA, maxA) = Vec2d.projectPoints(bounds, vec)
            (minB, maxB) = Vec2d.projectPoints(otherBounds, vec)
            leftmostA = minA <= minB
            overlap = False
            if leftmostA and maxA >= minB: overlap = True
            if not leftmostA and maxB >= minA: overlap = True
            if not overlap: return False
        return True
            
    def getBounds(self):
        normalDir = self.dir.normalized()
        perpDir = normalDir.perpendicular()
        bounds = [
            self.pos + normalDir * Car.LENGTH / 2 + perpDir * Car.WIDTH / 2,
            self.pos + normalDir * Car.LENGTH / 2 - perpDir * Car.WIDTH / 2,
            self.pos - normalDir * Car.LENGTH / 2 + perpDir * Car.WIDTH / 2,
            self.pos - normalDir * Car.LENGTH / 2 - perpDir * Car.WIDTH / 2
        ]
        return bounds
    @staticmethod
    def getBoundsforAllCar(car, LENGTH, WIDTH):
        normalDir = car.dir.normalized()
        perpDir = normalDir.perpendicular()
        bounds = [
            car.pos + normalDir * LENGTH / 2 + perpDir * WIDTH / 2,
            car.pos + normalDir * LENGTH / 2 - perpDir * WIDTH / 2,
            car.pos - normalDir * LENGTH / 2 + perpDir * WIDTH / 2,
            car.pos - normalDir * LENGTH / 2 - perpDir * WIDTH / 2
        ]
        return bounds
        
    def dirFromName(self, dirName):
        if dirName == 'north': return Vec2d(0, -1)
        if dirName == 'west': return Vec2d(-1, 0)
        if dirName == 'south': return Vec2d(0, 1)
        if dirName == 'east': return Vec2d(1, 0)
        if dirName == 'ne': return Vec2d(1, -1).normalized()
        if dirName == 'se': return Vec2d(-1, 1).normalized()
        raise Exception(str(dirName) + ' is not a recognized dir.')
    def stopStatus(self):
        return self.velocity.get_length() == 0
class Master(Car):

    ACCELERATION = 1.4
    FRICTION = 1
    MAX_WHEEL_ANGLE = 45.0
    MAX_SPEED = 5.0

    def __init__(self, pos, direction, velocity):
        Car.__init__(self, pos, direction, velocity)
        self.maxSpeed = Master.MAX_SPEED
        self.friction = Master.FRICTION
        self.maxWheelAngle = Master.MAX_WHEEL_ANGLE
        ''' Used later to check the reason why the car does not move due to the obstacles'''
        self.driveForward = True
        ''' we need to track the obstacles's position, so we can planned later'''
        # self.model = model
        self.stopflag = False
        self.timer = 0
        # for control of stop
        #we only store 10 observation numbers
        self.obserNum = 10

    def isJunior(self):
        return True

    def applyAction(self, state, actiongoal):
        def carInintersection(car, state):
            bounds = car.getBounds()
            for point in bounds:
                if state.data.inIntersection(point[0], point[1]):
                    return True

        action = actiongoal[0]
        goalPos = actiongoal[1] 
        # print goalPos
        self.timer += 1
        if self.timer < 30 and self.stopflag:
            action = 'stop'
        # print self.timer
        actions = {'normal':[self.friction, 0], 'acc':[self.ACCELERATION, 0],\
         'dec':[self.ACCELERATION*0.0, 0],'left':[self.ACCELERATION,-45],\
         'right':[self.ACCELERATION, 45]}
    
        #
        if carInintersection(self, state) and not self.stopflag:
           #state.data.inIntersection(goalPos[0], goalPos[1]) and not self.stopflag :
           self.accelerate(self.ACCELERATION*0)
           self.setWheelAngle(0)
           self.stopflag = True 
           self.timer = 0
           self.update()

        else:
            if action == 'stop':
                self.velocity = Vec2d(0,0)
                self.setWheelAngle(0)
            elif action in ['normal','acc','dec','left','right']:
                self.accelerate(actions[action][0])
                self.setWheelAngle(actions[action][1])
            else:
                print "Illega Action"
            self.update()

    def isCloseToOtherCar(self, model):
        ### check the master car is close to others
        cars = model.getCars()
        if not cars: return False
        obstaclecar = None 
        distance = float('inf')
        for car in cars:
            if self == car: continue
            cardis = util.manhattanDistance(car.pos, self.pos)
            if cardis < distance:
                distance = cardis
                obstaclecar = car
        if abs(obstaclecar.pos[0] - self.pos[0]) < Const.BELIEF_TILE_SIZE*1.5 and \
                 abs(obstaclecar.pos[1] - self.pos[1]) < Car.WIDTH/2:
                 return True


    def decisionMaking(self, agent, gamestate, actions):
          act = random.choice(['acc','normal'])
          if self.isCloseToOtherCar(gamestate.data) or act not in actions.keys():
              action = agent.getAction(gamestate)
            # print "decision making", action
          else: 
            probchoice = 0.5
            prob = random.random()
            if prob < probchoice:
              action = (act, actions[act])
            else:
              action = agent.getAction(gamestate)
          return action

    def reachgoal(self, goal):
        if goal is None:
            return True
        return abs(self.pos[1] - goal[1]) < Const.BELIEF_TILE_SIZE*0.2

    def makeObse(self, state):
        cars = state.getOtherCars()
        for car in cars:
            observation = car.getObserv()
            obsv = observation
            if hasattr(observation, 'getDist'):
               obsv = observation.getDist()
            if (type(obsv) == Vec2d): obsv = obsv.get_length()
            obsv = max(obsv, 0)
            if (len(car.history) == 11):
                car.history.pop()
            car.history.push(obsv)
class Other(Master):
    ACCELERATION = 0.5
    FRICTION = 0.2
    MAX_WHEEL_ANGLE = 45.0
    MAX_SPEED = 2.0

    def __init__(self, pos, direction, velocity):

        Master.__init__(self, pos, direction, velocity)
        self.maxSpeed = Other.MAX_SPEED
        self.friction = Other.FRICTION
        self.maxWheelAngle = Other.MAX_WHEEL_ANGLE
        self.hasinference = False
        self.history = util.Queue()


    def getInference(self, index, state):
        if (not self.hasinference):
            self.inference = MarginalInference(len(beliefstates), index, state)
            self.hasinference = True
            return self.inference
        else:
            return self.inference

    def getObserv(self):
        std = 0.000000001
        #return util.SonarObservation(random.gauss(self.velocity, std)) 
        return self.velocity 
   
    def deepcopy(self):
        from copy import deepcopy
        agent = deepcopy(self)
        return agent

    def isJunior(self):
        return False