"""Provides some utilities widely used by other modules"""

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
