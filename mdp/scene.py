from manim import *
import mdp

class MdpScene(Scene):

    def setup(self):
        GRID = [
            [-0.04, -0.001, -0.001, -0.001],
            [-0.04, -0.001, -0.001, +1],
            [-0.04, None, -0.001, -1],
            [-0.04, -0.001, -0.001, -0.001],
        ]
        TERMINALS = [(3,2), (3,1)]
        GAMMA = 0.9
        EPSILON = 0.01
        TRANSITION_MODEL = {
            "S": 0.8,
            "R": 0.1,
            "L": 0.1,
        }
        self.mdp = mdp.Mdp()

    def construct(self):
        self.setup()
        self.row = 4
        self.col = 3
        self.add_grid()

    def add_grid(self):
        base = self.get_rect().move_to(LEFT * self.row)
        grid = []
        for i in range(self.col):
            row_base = base
            self.add(Text(f"{self.col-i}").move_to(row_base))
            for j in range(self.row):
                rect = self.get_rect().next_to(row_base, RIGHT, buff=.0)
                row_base = rect
                grid.append(rect)
                if i == self.col - 1:
                    self.add(Text(f"{j+1}").next_to(row_base, DOWN))
            base = self.get_rect().next_to(base, DOWN, buff=.0)
        self.add(*grid)

    def get_rect(self):
        rect = Rectangle(width=1, height=1)
        rect.set_stroke(width=1)
        return rect
