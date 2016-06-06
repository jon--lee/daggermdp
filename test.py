"""from analysis import Analysis
a = Analysis(15, 15, 10)
a.save('analysis_test.p')
a.display_train_test(20, 30, 1)
b = Analysis.load('analysis_test.p')
print b.h, b.w"""

from plot_class import Plotter
from gridworld import BasicGrid
from policy import ClassicPolicy
from mdp import ClassicMDP

grid = BasicGrid(15, 15)
plotter = Plotter(15, 15)
mdp = ClassicMDP(ClassicPolicy(grid), grid)
mdp.load_policy()

plotter.plot_state_actions(mdp.pi)
