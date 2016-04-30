


class Action():
    NONE = -1
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

    LEFT = {NORTH: WEST, SOUTH: EAST, WEST: SOUTH, EAST: NORTH}
    RIGHT = {NORTH: EAST, SOUTH: WEST, WEST: NORTH, EAST: SOUTH}

    @staticmethod
    def isVertical(direction):
        return direction == Action.NORTH or direction == Action.SOUTH

    @staticmethod
    def isHorizontal(direction):
        return direction == Action.EAST or direction == Action.WEST

    @staticmethod
    def arePerpendicular(d1, d2):
        return ((Action.isVertical(d1) and Action.isHorizontal(d2))
                or (Action.isHorizontal(d1) and Action.isVertical(d2)))

    @staticmethod
    def areParallel(d1, d2):
        return not self.arePerpendicular(d1, d2)


class DumbPolicy():

    def __init__(self):
        return

    def get_next(self, state):
        return Action.NORTH
    



class ClassicPolicy():
    
    def __init__(self, width, height):
        self.arr = np.zeros((width, height))


    def get_next(self
