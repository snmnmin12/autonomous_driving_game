from car import *
from vector import Vec2d
from const import Const

import threading
import copy
from copy import deepcopy
import util

class Line(object):
    
    def __init__(self, row):
        self.x1 = row[0] * Const.BLOCK_TILE_SIZE
        self.y1 = row[1] * Const.BLOCK_TILE_SIZE
        self.x2 = row[2] * Const.BLOCK_TILE_SIZE
        self.y2 = row[3] * Const.BLOCK_TILE_SIZE
        
    def getstart(self):
        return (self.x1, self.y1)
    
    def getend(self):
        return (self.x2, self.y2)
    

class Block(object):
    
    def __init__(self, row):
        self.x1 = row[0] * Const.BLOCK_TILE_SIZE
        self.y1 = row[1] * Const.BLOCK_TILE_SIZE
        self.x2 = row[2] * Const.BLOCK_TILE_SIZE
        self.y2 = row[3] * Const.BLOCK_TILE_SIZE
        self.centerX = (self.x1 + self.x2) / 2.0
        self.centerY = (self.y1 + self.y2) / 2.0
        
    def getCenter(self):
        return Vec2d(self.centerX, self.centerY)
    
    def getWidth(self):
        return abs(self.x2 - self.x1)
    
    def getHeight(self):
        return abs(self.y2 - self.y1)
    
    def containsPoint(self,x, y):
        if x < self.x1: return False
        if y < self.y1: return False
        if x > self.x2: return False
        if y > self.y2: return False
        return True


class Model(object):

    def __init__(self, layout):
        if isinstance(layout, Model):
            oldmodel = layout
            self.layout = oldmodel.layout
            self.finish = oldmodel.finish
            self.junior = deepcopy(oldmodel.junior)
            self.finish = oldmodel.finish
            self.blocks = oldmodel.blocks
            self.intersections = oldmodel.intersections
            self.cars =[self.junior]
            # self.agentComm = AgentCommunication()
            self.otherCars = []
            for i in range(len(oldmodel.otherCars)):
                othercar = oldmodel.otherCars[i].deepcopy()
                self.otherCars.append(othercar)
                self.cars.append(othercar)
        else:
            self._initBlocks(layout)
            self._initOtherCars(layout)
            self._initLines(layout)
            self._initIntersections(layout)
            self.layout = layout
            startX = layout.getStartX()
            startY = layout.getStartY()
            startDirName = layout.getJuniorDir()
            self.junior = Master(
                Vec2d(startX, startY), 
                startDirName, 
                Vec2d(0, 0)
                )

            self.cars = [self.junior]
            self.otherCars = []
            self.finish = Block(layout.getFinish())
           
            for i in range(len(self.othercars)):
               startNode = self.othercars[i]
               other = Other(Vec2d(startNode[0], startNode[1]), startDirName, Vec2d(0, 0))
               self.cars.append(other)
               self.otherCars.append(other)

            self.observations = []
            self.modelLock = threading.Lock()
            self.probCarSet = False
        self.goal = self.getFinish().getCenter()
    
    def _initLines(self, layout):
        self.lines = []
        for lineData in layout.getLineData():
             line = Line(lineData)
             self.lines.append(line)
    def _initBlocks(self, layout):
        self.blocks = []
        for blockData in layout.getBlockData():
            block = Block(blockData)
            self.blocks.append(block)
    def _initOtherCars(self, layout):
        self.othercars = []
        if not layout.getOtherData() is None:
            for othercar in layout.getOtherData():
               self.othercars.append(othercar)     
    def _initIntersections(self, layout):
        self.intersections = []
        for blockData in layout.getIntersectionNodes():
            block = Block(blockData)
            self.intersections.append(block)
            
    def _getStartNode(self, agentGraph):
        while True:
            node = agentGraph.getRandomNode()
            pos = node.getPos()
            alreadyChosen = False
            for car in self.otherCars:
                if car.getPos() == pos:
                    alreadyChosen = True
                    break
            if not alreadyChosen: 
                return node
            
    def checkVictory(self):
        bounds = self.junior.getBounds()
        for point in bounds:
            if self.finish.containsPoint(point.x, point.y): return True
        return False
            
    def checkCollision(self, car):
        bounds = car.getBounds()
        # check for collision with fixed obstacles
        for point in bounds:
            if not self.inBounds(point.x, point.y): return True

        # i have changed the othercars 
        for other in self.cars:
            if other == car: continue
            if other.collides(car.getPos(), bounds): return True
        return False
        
    def getIntersection(self, x, y):
        for intersection in self.intersections:
            if intersection.containsPoint(x, y): return intersection
        return None

    def getIntersectionCenter(self):
        IntersectionCenter = []
        for intersection in self.intersections:
            IntersectionCenter.append(intersection.getCenter())
        return IntersectionCenter
        
    def inIntersection(self, x, y):
        return self.getIntersection(x, y) != None
    

    def inBounds(self, x, y):
        if x < 0 or x >= self.getWidth(): return False
        if y < 0 or y >= self.getHeight(): return False
        for block in self.blocks:
            if block.containsPoint(x, y): return False
        return True
    
    def getWidth(self):
        return self.layout.getWidth()
    
    def getHeight(self):
        return self.layout.getHeight()
    
    def getBeliefRows(self):
        return self.layout.getBeliefRows()
    
    def getBeliefCols(self):
        return self.layout.getBeliefCols()
            
    def getBlocks(self):
        return self.blocks
    def getLine(self):
        return self.lines
    def getFinish(self):
        return self.finish
        
    def getCars(self):
        return self.cars
    
    def getOtherCars(self):
        return self.otherCars
    
    def getJunior(self):
        return self.junior
    
    def getAgentGraph(self):
        return self.layout.getAgentGraph()
    
    def getJuniorGraph(self):
        return self.layout.getJuniorGraph()