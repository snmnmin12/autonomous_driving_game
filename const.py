

class Const(object):
    
    TITLE = "Driverless Car Simulator"
    
    LAYOUT_DIR = 'layouts'
    ScaleRatio = 0.5
    
    BLOCK_TILE_SIZE = 30
    BELIEF_TILE_SIZE = 30

    SONAR_STD = 15.0
    ScaleRatio = 0.5

class Directions(object):
    NORTH = 'North'
    SOUTH = 'South'
    EAST = 'East'
    WEST = 'West'
    STOP = 'Stop'
    NORTHEAST = 'NE'
    SOUTHEAST = 'SE'
    directions = ['N','S','E','W','SE','NE','SW','NW']

    LEFT =  {NORTH: WEST,
             SOUTH: EAST,
             EAST:  NORTH,
             WEST:  SOUTH,
             STOP:  STOP}

    RIGHT =  dict([(y,x) for x, y in LEFT.items()])

    REVERSE = {NORTH: SOUTH,
               SOUTH: NORTH,
               EAST: WEST,
               WEST: EAST,
               STOP: STOP
               } 
class Actions(object):
    @staticmethod
    def directionToVector(direction):
        if direction == 'N':
           return (0, -1)
        if direction == 'S':
           return (0,  1)
        if direction == 'E':
            return (1, 0)
        if direction == 'W':
            return (-1, 0)
        if direction == 'NE':
            return (1, -1)
        if direction == 'SE':
            return (1,  1)
        if direction == 'SW':
            return (-1, 1)
        if direction == 'NW':
            return (-1, -1)
    
