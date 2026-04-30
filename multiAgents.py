# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # To implement this function, I referenced the Search slides + VSCode Copilot
        # The reflex agent's heuristic score increases or decreases depending on the action taken.
        # Since you want to collect the food closest first, you take inverse of that distance
        # Doing so makes it have a higher score.
        # This function rewards getting closer and closer to scared ghosts, while penalizing the agent
        # For approaching regular ghosts or not choosing the closest food.

        foodList = newFood.asList()
        score = successorGameState.getScore()

        #takes diagonal distance to new food and increments score by that 1/distance. If there are still too many food remaining then reduce score
        if foodList:
            closestFoodDistance = min(manhattanDistance(newPos, foodPos) for foodPos in foodList)
            score += 1.0 / (closestFoodDistance)
            score -= 2.0 * len(foodList)

        # checks distance from all ghosts and if they are scared. Rewards for being near scared ghosts
        # penalizes for being close to scared ghosts.
        for ghostState, scaredTime in zip(newGhostStates, newScaredTimes):
            ghostPos = ghostState.getPosition()
            if ghostPos is None:
                continue
            ghostDistance = manhattanDistance(newPos, ghostPos)
            if scaredTime > 0:
                score += 2.0 / (ghostDistance)
            else:
                if ghostDistance <= 1:
                    score -= 500
                else:
                    score -= 2.0 / ghostDistance

        if action == Directions.STOP:
            score -= 3

        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # To implement this function, I referenced the Search slides + VSCode Copilot
        #recursive value function that either takes the Max or Min value of the subtrees
        # if the current agent is pacman it takes max, otherwise it takes min.
        # repeats until the specificed depth is reached.
        def value(agentIndex, depth, state):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            if agentIndex == 0:
                return maxValue(agentIndex, depth, state)
            else:
                return minValue(agentIndex, depth, state)

        def maxValue(agentIndex, depth, state):
            v = float('-inf')
            nextAgent = (agentIndex + 1) % state.getNumAgents()
            nextDepth = depth + 1 if nextAgent == 0 else depth
            for action in state.getLegalActions(agentIndex):
                v = max(v, value(nextAgent, nextDepth, state.generateSuccessor(agentIndex, action)))
            return v

        def minValue(agentIndex, depth, state):
            v = float('inf')
            nextAgent = (agentIndex + 1) % state.getNumAgents()
            nextDepth = depth + 1 if nextAgent == 0 else depth
            for action in state.getLegalActions(agentIndex):
                v = min(v, value(nextAgent, nextDepth, state.generateSuccessor(agentIndex, action)))
            return v

        #Once it's pacman's turn it gets the actions pacman can take and evaluates al possible states from that action
        legalActions = gameState.getLegalActions(0)
        bestAction = None
        bestScore = float('-inf')
        for action in legalActions:
            score = value(1, 0, gameState.generateSuccessor(0, action))
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # To implement this function, I referenced the Search slides + VSCode Copilot
        #recursive value function that either takes the Max or Min value of the subtrees
        # if the current agent is pacman it takes max, otherwise it takes min.
        # repeats until the specificed depth is reached.
        def value(agentIndex, depth, state, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            if agentIndex == 0:
                return maxValue(agentIndex, depth, state, alpha, beta)
            else:
                return minValue(agentIndex, depth, state, alpha, beta)

        def maxValue(agentIndex, depth, state, alpha, beta):
            v = float("-inf")
            nextAgent = (agentIndex + 1) % state.getNumAgents()
            nextDepth = depth + 1 if nextAgent == 0 else depth
            #for each action of the current state, checks if 
            for action in state.getLegalActions(agentIndex):
                v = max(v, value(nextAgent, nextDepth, state.generateSuccessor(agentIndex, action), alpha, beta))
                #prune remaining branches
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v

        def minValue(agentIndex, depth, state, alpha, beta):
            v = float("inf")
            nextAgent = (agentIndex + 1) % state.getNumAgents()
            nextDepth = depth + 1 if nextAgent == 0 else depth

            for action in state.getLegalActions(agentIndex):
                v = min(v, value(nextAgent, nextDepth, state.generateSuccessor(agentIndex, action), alpha, beta))
                #prune remaining branches
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        bestAction = None
        bestScore = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        legalActions = gameState.getLegalActions(0)
        
        # When it is pacman's turn find and return the best action.
        for action in legalActions:
            score = value(1, 0, gameState.generateSuccessor(0, action), alpha, beta)
            if score > bestScore:
                bestScore = score
                bestAction = action
            
            # prune branches if needed
            if score > beta:
                return bestAction
            
            # Update alpha
            alpha = max(alpha, bestScore)
            
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # To implement this function, I referenced the Search with Other Agents II slides + VSCode Copilot
        # recursive value function that either takes the Max or Min value of the subtrees
        # if the current agent is pacman it takes max, otherwise it takes min.
        # repeats until the specificed depth is reached.

        # Why ExpectimaxAgent wins half the time while AlphaBetaAgent always loses is because
        # AlphaBeta assumes the ghost agents make the optimal choice. If it looks ahead and see that it's going to die, it will rush the closest ghost
        # but the agents we run the test against were random. Expectimax choose the best option based on the probabilities rather than
        # assuming that the adversaries have taken away the optimal choice.
        def value(agentIndex, depth, state):
            # Terminal state or max depth reached
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            
            if agentIndex == 0:
                return maxValue(agentIndex, depth, state)
            else:
                return expValue(agentIndex, depth, state)

        def maxValue(agentIndex, depth, state):
            v = float('-inf')
            nextAgent = (agentIndex + 1) % state.getNumAgents()
            
            if nextAgent == 0:
                nextDepth = depth + 1
            else:
                nextDepth = depth
            
            for action in state.getLegalActions(agentIndex):
                v = max(v, value(nextAgent, nextDepth, state.generateSuccessor(agentIndex, action)))
            return v

        #computes the probability of each successor 
        def expValue(agentIndex, depth, state):
            v = 0.0
            legalActions = state.getLegalActions(agentIndex)
            
            #if no actions return the state's utility
            if not legalActions:
                return self.evaluationFunction(state)
                
            #probability of the action
            probability = 1.0 / len(legalActions) 
            nextAgent = (agentIndex + 1) % state.getNumAgents()
            
            if nextAgent == 0:
                nextDepth = depth + 1
            else:
                nextDepth = depth
            
            for action in legalActions:
                successorValue = value(nextAgent, nextDepth, state.generateSuccessor(agentIndex, action))
                v += probability * successorValue
            return v

        # When it's pacman's turn find and return the best action.
        legalActions = gameState.getLegalActions(0)
        bestAction = None
        bestScore = float('-inf')
        for action in legalActions:
            # Pacman is 0, so the next agent is 1 and depth is still 0
            score = value(1, 0, gameState.generateSuccessor(0, action))
            if score > bestScore:
                bestScore = score
                bestAction = action
                
        return bestAction

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: 
    This evaluation function works by calculating a linear combination of features.
    It takes the base game score and adds/subtracts points based on:
    1. Distance to the closest food: Higher score for being closer to food (using reciprocal). 
    Amount of food left: Large penalty for having a lot of food left on the board.
    2. Distance to ghosts: If a ghost is scared, it strongly rewards getting closer to eat it.
    If a ghost is NOT scared, it slightly penalizes being close, and heavily penalizes 
    being immediately next to it (distance <= 1) to avoid death.
    3. Distance and Amount of Power pellets left: Reward being closer to power pellets. Penalty for leaving power pellets uneaten.
    """
    "*** YOUR CODE HERE ***"
    # To implement this function, I referenced the Search slides + VSCode Copilot
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    power_pellets = currentGameState.getCapsules()

    foodList = food.asList()
    score = currentGameState.getScore()

    # 1. Reward being close to food, penalty for remaining food
    if foodList:
        closestFoodDistance = min(manhattanDistance(pos, foodPos) for foodPos in foodList)
        score += 1.0 / closestFoodDistance
        score -= 2.0 * len(foodList)

    # 2. Penalizes being close to regular ghosts, rewards for hunting scared ghosts
    for ghostState, scaredTime in zip(ghostStates, scaredTimes):
        ghostPos = ghostState.getPosition()
        if ghostPos is None:
            continue
        ghostDistance = manhattanDistance(pos, ghostPos)
        
        if scaredTime > 0:
            # Reward chasing scared ghosts.
            if ghostDistance > 0:
                score += 2.0 / ghostDistance
            else:
                # Reward for eating a scared ghost
                score += 100 
        else:
            # Penalize being close to regular ghosts
            if ghostDistance <= 1:
                # Big Penalty for being too close to regular ghost
                score -= 500 
            elif ghostDistance > 1:
                score -= 2.0 / ghostDistance

    # 3. Power Pellets
    score -= 20.0 * len(power_pellets)
    if power_pellets:
        closestPelletDistance = min(manhattanDistance(pos, pelletPos) for pelletPos in power_pellets)
        score += 2.0 / closestPelletDistance

    return score

# Abbreviation
better = betterEvaluationFunction
