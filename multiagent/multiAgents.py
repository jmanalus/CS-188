# multiAgents.py
# 
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
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
    some Directions.X for some X in the set {North, South, West, East, Stop}
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

    food=[]
    for x in newFood.asList():
      food.append(manhattanDistance(newPos,x))
    score=0
    if len(food)>0 :
      score=1/sum(food)+5/min(food)+successorGameState.getScore()
    for x in newGhostStates:
      if manhattanDistance(newPos,x.getPosition())<2:
        return -10
    if score > 0:
      return score
    return successorGameState.getScore()

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
      This class provides some common xents to all of your
      multiagent searchers.  Any methods defined here will be available
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
        """
        "*** YOUR CODE HERE ***"
        return self.minMax(gameState, 1, 0 )

    def minMax(self, gameState, currentDepth, agentIndex):
      if gameState.isWin() or gameState.isLose():
            return gameState.getScore()
          
      if currentDepth > self.depth:
            return self.evaluationFunction(gameState)
      
      nextBestMove = [action for action in gameState.getLegalActions(agentIndex) if action!='Stop']
      nextIndex, nextDepth = agentIndex + 1, currentDepth
      if nextIndex >= gameState.getNumAgents():
          nextIndex, nextDepth = 0, nextDepth + 1
      
      payOff = [self.minMax( gameState.generateSuccessor(agentIndex, action) ,\
                                    nextDepth, nextIndex) for action in nextBestMove]
      if agentIndex == 0 and currentDepth == 1: 
          bestMove = max(payOff)
          bestIndexMove = random.choice([index for index in range(len(payOff)) if payOff[index] == bestMove])
          return nextBestMove[bestIndexMove]
      
      if agentIndex == 0:
          bestMove = max(payOff)
          return bestMove
      else:
          bestMove = min(payOff)
          return bestMove

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """
  def minAgent(self, depth, gameState, ghostIndex, alpha, beta):
    
    if(self.depth == depth or gameState.isLose() or gameState.isWin()):
        return self.evaluationFunction(gameState)
    
    score = float('infinity')
    
    for action in gameState.getLegalActions(ghostIndex):
      gameState.generateSuccessor(ghostIndex, action)
      
      if ghostIndex < gameState.getNumAgents() - 1:
        score = min(score, self.minAgent(depth, gameState.generateSuccessor(ghostIndex, action), ghostIndex + 1, alpha, beta))
      elif ghostIndex == gameState.getNumAgents() - 1:
        score = min(score, self.maxAgent(depth + 1, gameState.generateSuccessor(ghostIndex, action), alpha, beta))
      if(score < alpha):
        return score
      beta = min(score, beta)
    return score

  def maxAgent(self, depth, gameState, alpha, beta):
    
    if (self.depth == depth or gameState.isLose() or gameState.isWin()):
      return self.evaluationFunction(gameState)
    
    actions = gameState.getLegalActions(0)
    if Directions.STOP in actions:
        actions.remove(Directions.STOP)
    score = float('-infinity')
    for action in actions:
      score = max(score, self.minAgent(depth, gameState.generateSuccessor(0, action), 1, alpha, beta))
      if(score > beta):
        return score
      alpha = max(alpha, score)
    return score

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"

    alpha = float('-infinity')
    beta = float('infinity')
    score = float('-infinity')
    actions = gameState.getLegalActions(0)
    if Directions.STOP in actions:
        actions.remove(Directions.STOP)
    for action in actions:
      if(self.minAgent(0, gameState.generateSuccessor(0, action), 1, alpha, beta) > score):
        score, actions[0] = self.minAgent(0, gameState.generateSuccessor(0, action), 1, alpha, beta), action
      if(self.minAgent(0, gameState.generateSuccessor(0, action), 1, alpha, beta) > beta):
        return actions[0]
      alpha = max(alpha, self.minAgent(0, gameState.generateSuccessor(0, action), 1, alpha, beta))
    return actions[0]  

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
    return self.expectimaxValue(gameState, 0, 0)[1]
        
  def expectimaxValue(self, gameState, depth, agentIndex):
    if agentIndex >= gameState.getNumAgents():
        depth, agentIndex = 1+depth, 0
        
    if Directions.STOP in gameState.getLegalActions(agentIndex):
        gameState.getLegalActions(agentIndex).remove(Directions.STOP)
    
    if depth >= self.depth or len(gameState.getLegalActions(agentIndex)) == 0:
        return (self.evaluationFunction(gameState), None)
    
    random.shuffle(gameState.getLegalActions(agentIndex))
    score = []
    for action in gameState.getLegalActions(agentIndex):
        score.append(self.expectimaxValue(gameState.generateSuccessor(agentIndex, action), depth, agentIndex+1)[0])
    
    if agentIndex == 0:
        finalScore = max(score)
    else:
        finalScore = sum(score) / len(score)

    for index in range(len(score)):
        if finalScore == score[index]:
            return (finalScore, gameState.getLegalActions(agentIndex)[index])

    return (finalScore, None)

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"
  maxDist = currentGameState.getFood().height + currentGameState.getFood().width
  if len(currentGameState.getFood().asList()) == 0:
    foodNotEaten = 9999
  else:
    foodNotEaten = 1/(maxDist * len(currentGameState.getFood().asList()))
  score = 0
  minDist = maxDist
  for food in currentGameState.getFood().asList():
    distance = eclidDis(currentGameState.getPacmanPosition(), food)
    if distance < minDist:
      minDist = distance
  score = 1/(minDist)
  return currentGameState.getScore() + 2*foodNotEaten + 2*score

def eclidDis(start, end):
  return ( (start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2 ) ** 0.5
  
# Abbreviation
better = betterEvaluationFunction


