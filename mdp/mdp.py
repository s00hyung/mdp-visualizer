import random
from utils import turn_right, turn_left, try_move
import pprint


class Mdp:
    def __init__(
        self,
        grid = [[-0.04, -0.001, -0.001, +1],[-0.04, None, -0.001, -1],[-0.04, -0.001, -0.001, -0.001]],
        terminals = [(3,2), (3,1)],
        transition_model = {"S": 0.8,"R": 0.1,"L": 0.1},
        gamma=0.9,
    ):
        # Define 4 possible actions
        self.four_actions = [
            (1, 0),  # RIGHT
            (0, 1),  # UP
            (-1, 0),  # LEFT
            (0, -1),  # DOWN
        ]

        # Environment information (grid, initial position, terminal position)
        grid.reverse()  # row 0 on bottom, not on top
        self.grid = grid  # grid
        self.rows = len(grid)  # number of elements in row
        self.cols = len(grid[0])  # number of elements in col
        self.terminals = terminals  # terminal states

        # Gamma = discount factor
        if not (0 < gamma <= 1):
            raise ValueError("An MDP must have 0 < gamma <= 1")
        self.gamma = gamma

        # State information where states = {s0,s1,s2,...}
        states = set(
            (x, y) for x in range(self.cols) for y in range(self.rows) if grid[y][x]
        )
        self.states = states

        # Reward information where reward = {s0: R(s0), s1: R(s1),...}
        reward = {
            (x, y): grid[y][x]
            for x in range(self.cols)
            for y in range(self.rows)
            if grid[y][x]
        }
        self.reward = reward

        # Transition Model
        if sum([transition_model[d] for d in transition_model]) != 1:
            raise ValueError("Probability in the transition model should sum up to 1")
        """
        transitions = {
            s0: {
                LEFT:    [P(s'|s0, LEFT)],
                DOWN:    [P(s'|s0, DOWN)],
                UP:      [P(s'|s0, UP)],
                RIGHT:   [P(s'|s0, RIGHT)]
            },
            ...
        }
        """
        transitions = {}
        for s in states:
            transitions[s] = {}
            for a in self.four_actions:
                transitions[s][a] = [
                    # calculate P(s'|s, a). This is possible since global
                    # environment is fully observable by the agent
                    (transition_model["S"], try_move(states, s, a)),
                    (transition_model["R"], try_move(states, s, turn_right(a))),
                    (transition_model["L"], try_move(states, s, turn_left(a))),
                ]
        self.transitions = transitions

    def R(self, s):
        # R(s) - Return reward at state s
        return self.reward[s]

    def P(self, s, a):
        # P(s,a) - Return a list of tuple of (P(s'|s,a), s')
        if a:
            return self.transitions[s][a]
        else:
            return [(0.0, s)]

    def actions(self, s):
        # Return a list of possible action from the state s
        if s in self.terminals:
            return [None]
        else:
            return self.four_actions

    def to_arrows(self, policy):
        arrows = {(1, 0): "▶", (0, 1): "▲", (-1, 0): "◀", (0, -1): "▼", None: "□"}
        return {s: arrows[a] for (s, a) in policy.items()}

def value_iteration(mdp, epsilon=0.001):

    # FOR GUI PURPOSE
    data = {s: [] for s in mdp.states}

    # Start with arbitrary initial values for the utilities (U0)
    U0 = {s: 0 for s in mdp.states}

    # R = reward, P = P(s'|s,a), gamma = discount factor
    R, P, gamma = mdp.R, mdp.P, mdp.gamma

    while True:

        U = U0.copy()
        delta = 0

        # For every state s:
        for s in mdp.states:

            sum_of_P_U_i = []

            # For every possible action from state s, (NONE if its in the terminal state)
            for a in mdp.actions(s):
                # p = P(s_|s,a) for some state s and its successor state s_
                # U[s_] = U_i(s_) for some state s and its successor state s_
                P_times_U = []
                for (p, s_) in P(s, a):
                    P_times_U.append(p * U[s_])
                sum_of_P_U_i.append(sum(P_times_U))

            # U_i+1(S) = R(S) + SUM[P(s'|s,a) * U_i(s')]
            U0[s] = R(s) + gamma * max(sum_of_P_U_i)

            # FOR GUI PURPOSE
            data[s].append(U0[s])

            delta = max(delta, abs(U0[s] - U[s]))

        # termination condition
        if delta <= epsilon * (1 - gamma) / gamma:
            return U, data

def policy_iteration(mdp):
    """Solve an MDP by policy iteration [Figure 17.7]"""

    U = {s: 0 for s in mdp.states}
    pi = {s: random.choice(mdp.actions(s)) for s in mdp.states}
    while True:
        U = policy_evaluation(pi, U, mdp)
        unchanged = True
        for s in mdp.states:
            a = max(mdp.actions(s), key=lambda a: expected_utility(a, s, U, mdp))
            if a != pi[s]:
                pi[s] = a
                unchanged = False
        if unchanged:
            return pi

def best_policy(mdp, U):
    """Given an MDP and a utility function U, determine the best policy,
    as a mapping from state to action. [Equation 17.4]"""

    pi = {}
    for s in mdp.states:
        pi[s] = max(mdp.actions(s), key=lambda a: expected_utility(a, s, U, mdp))
    return pi


def expected_utility(a, s, U, mdp):
    """The expected utility of doing a in state s, according to the MDP and U."""

    return sum(p * U[s1] for (p, s1) in mdp.P(s, a))


def policy_evaluation(pi, U, mdp, k=20):
    """Return an updated utility mapping U from each state in the MDP to its
    utility, using an approximation (modified policy iteration)."""

    R, P, gamma = mdp.R, mdp.P, mdp.gamma
    for _ in range(k):
        for s in mdp.states:
            U[s] = R(s) + gamma * sum(p * U[s1] for (p, s1) in P(s, pi[s]))
    return U

orientations = EAST, NORTH, WEST, SOUTH = [(1, 0), (0, 1), (-1, 0), (0, -1)]
turns = LEFT, RIGHT = (+1, -1)

def turn_heading(heading, inc, headings=orientations):
    return headings[(headings.index(heading) + inc) % len(headings)]

def turn_right(heading):
    return turn_heading(heading, RIGHT)

def turn_left(heading):
    return turn_heading(heading, LEFT)

def try_move(states, state, direction):
    """
    Return the next state if you move by the direction
    e.g.
    state = (0,0)
    direction = (1,0)
    next_state = (1,0) only if there is no wall on (1,0)
    """
    next_state = tuple(map(sum, zip(state, direction)))
    return next_state if next_state in states else state