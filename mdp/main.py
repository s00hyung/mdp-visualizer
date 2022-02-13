import mdp
import gui

################################### PARAMETERS ##########################################

# Environment Grid - None is wall, otherwise its reward value for each state
GRID = [
    [-0.04, -0.001, -0.001, -0.001],
    [-0.04, -0.001, -0.001, +1],
    [-0.04, None, -0.001, -1],
    [-0.04, -0.001, -0.001, -0.001],
]

# Terminal state
TERMINALS = [(3,2), (3,1)]

# Discount factor
GAMMA = 0.9
EPSILON = 0.01

# STRAIGHT, RIGHT, LEFT
TRANSITION_MODEL = {
    "S": 0.8,
    "R": 0.1,
    "L": 0.1,
}

##########################################################################################

# Create new mdp environment
env = mdp.environment(grid=GRID, terminals=TERMINALS, transition_model=TRANSITION_MODEL, gamma=GAMMA)

# Run value iteration on mdp environment
U, data = mdp.value_iteration(env, epsilon=EPSILON)

# Get best policy of mdp environment with some utility U 
pi = mdp.best_policy(env, U)

# Run policy iteration on mdp environment
pi2 = mdp.policy_iteration(env)

# Draw GUI
gui.draw(env, pi, U=U, data=data)
#gui.draw(env, pi2)

##########################################################################################

pi = mdp.best_policy(env, U)
