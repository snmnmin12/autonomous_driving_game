import util
import random

class Belief(object):
    # Function: Init
    def __init__(self, numElems, value = None):
        self.numElems = numElems
        if value == None:
            value = (1.0 / numElems)
        self.grid = [ value for _ in range(numElems)]

    def __iter__(self):
        return iter(self.grid)

    def next(self):
        return next(self.grid)
    
    # Sets the probability of a given row, col to be p
    def setProb(self, row, p):
        self.grid[row] = p
        
    def addProb(self, row, delta):
        self.grid[row] += delta
        assert self.grid[row] >= 0.0
        
    # Returns the belief for tile row, col.
    def getProb(self, row):
        return self.grid[row]
    
    # Function: Normalize
    def normalize(self):
        total = self.getSum()
        for r in range(self.numElems):
                self.grid[r] /= total
    
    # Function: Get Num Rows
    def getNumElems(self):
        return self.numElems
    
    # Function: Get Sum
    def getSum(self):
        total = 0.0
        for r in range(self.numElems):
                total += self.getProb(r)
        return total

class Inference(object):
# class ParticleFilter(object):
    BELIEF_TO_INDEX = {'STOP':0,'HESITATING':1,'NORMAL':2,'AGGRESSIVE':3}
    BELIEF_STATES = ['STOP','HESITATING','NORMAL','AGGRESSIVE']
    
    # Function: Init
    # --------------
    # Constructer that initializes an ExactInference object which has
    def __init__(self, numElems, index, state):
        ''' initialize any variables you will need later '''
        self.legalIntentions = self.BELIEF_STATES
        self.index = index
        self.initializeUniformly(state)
    
    def initializeUniformly(self):
         pass

    def observe(self, history):
        ''' your code here'''
        pass

    def elapseTime(self):
        pass


    def getMeanStandard(self, history, intention):

        vref =  sum(history)/len(history)
        if vref == 0: vref = 0.001
        sigma = 0.5*vref
        index = self.BELIEF_TO_INDEX[intention] 
        if  index == 0:
            return (0, sigma)
        elif index == 1:
            return (0.5*vref, sigma)
        elif index == 2:
            return (vref,sigma)
        elif index == 3:
            return (1.5*vref, sigma)
        else:
            raise('Undefined intention')
    # Function: Get Belief
    # ---------------------
    # Returns your belief of the probability that the car is in each tile. Your
    # belief probabilities should sum to 1.    
    def getBelief(self):
        return self.belief



class MarginalInference(Inference):
    "A wrapper around the JointInference module that returns marginal beliefs about ghosts."

    def __init__(self, numElems, index, state):
        super(MarginalInference, self).__init__(numElems, index, state)

    def initializeUniformly(self, gameState):
        "Set the belief state to an initial, prior value."
        if self.index == 1: jointInference.initializeUniformly(gameState, self.legalIntentions)
        # jointInference.addGhostAgent(self.ghostAgent)

    def observe(self, gameState):
        "Update beliefs based on the given distance observation and gameState."
        if self.index == 1: jointInference.observe(gameState, self.getMeanStandard)

    def elapseTime(self, gameState):
        "Update beliefs for a time step elapsing from a gameState."
        pass
        # if self.index == 1: jointInference.elapseTime(gameState)

    def getBelief(self):
        "Returns the marginal belief over a particular ghost by summing out the others."
        jointDistribution = jointInference.getBelief()
        # print jointDistribution
        dist = util.Counter()
        for t, prob in jointDistribution.items():
            index = self.BELIEF_TO_INDEX[t[self.index-1]]
            dist[index] += prob
        print dist
        #return the result as list    
        result = [0]*len(self.BELIEF_STATES)
        for t, prob in dist.items():
           result[t] = prob
        return result

class JointParticleFilter:
    "JointParticleFilter tracks a joint distribution over tuples of all ghost positions."
    # BELIEF_TO_INDEX = {'STOP':0,'HESITATING':1,'NORMAL':2,'AGGRESSIVE':3}
    # BELIEF_STATES = ['STOP','HESITATING','NORMAL','AGGRESSIVE']

    def __init__(self, numParticles=600):
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles

    def initializeUniformly(self, state, legalIntentions):
        "Stores information about the game, then initializes particles."
        self.numAgents = len(state.getOtherCars())
        self.Agents = []
        self.legalIntentions = legalIntentions
        self.beliefs = util.Counter()
        self.initializeParticles()

    def initializeParticles(self):
        """
        Initialize particles to be consistent with a uniform prior.  
        Each particle is a tuple of ghost positions. Use self.numParticles for
        """
        import itertools
        import random
        #create a list of possible ghost permutations, where each of three ghosts can be on any of the legal positions in the boards.
        permutations = list(itertools.product(self.legalIntentions, repeat=self.numAgents))
        
        random.shuffle(permutations)
        p = len(permutations)
        n = self.numParticles
        self.particles = []
        #create the particles
        while n >= p:
            self.particles += permutations
            n -= p
        #add the remainder
        self.particles += permutations[0: n - 1]

    def observe(self, gameState, meanfunc):
        """
        Resamples the set of particles using the likelihood of the noisy observations.
        To loop over the ghosts, use:
          for i in range(self.numGhosts):
        """
        if len(self.beliefs.keys()) == 1:
            self.initializeParticles()
        othercars = gameState.getOtherCars()    
        # history = othercars[index].history
        tempCounter = util.Counter()
        #weight every particle
        for intentions in self.particles:
            #weight a specific particle
            prob = 1
            # check the truedistance of a particle ghost with its respective
            for i in range(self.numAgents):
                history = othercars[i].history
                observedV = history[0]
                intention = intentions[i]
                (mean, sigma) = meanfunc(history, intention)
                prob *= util.pdf(mean, sigma, observedV) 
            # add this probability to the overall particle table
            tempCounter[intentions] += prob    
        
        # assign beliefs to this counter
        self.beliefs = tempCounter

        # resample
        if len(tempCounter.keys()) == 0:
            self.initializeParticles()
        else:
            self.beliefs.normalize()
            # print self.beliefs
            for i in xrange(len(self.particles)):
                newPos = util.sample(self.beliefs)
                self.particles[i] = newPos


    def elapseTime(self, gameState):
        """
        no need
        """
        pass

    def getBelief(self):
        beliefDist = util.Counter()
        for index in xrange(len(self.particles)):
            beliefDist[self.particles[index]] += 1
        
        beliefDist.normalize()
        return beliefDist

# One JointInference module is shared globally across instances of MarginalInference
jointInference = JointParticleFilter()

