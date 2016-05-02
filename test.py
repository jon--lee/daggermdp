import pickle
import policy
import gridworld

grid = gridworld.BasicGrid(10, 10)
p = policy.ClassicPolicy(grid)
print p.arr

f = open('test.p', 'w')
pickle.dump(p, f)
f.close()

f = open('test.p', 'r')
a = pickle.load(f)
print a.arr

