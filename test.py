import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)


def animate(i):
    pullData = open("sampleText.txt","r").read()
    dataArray = pullData.split('\n')
    xar = []
    yar = []
    print i
    for j, eachLine in enumerate(dataArray):
        if len(eachLine)>1 and j == i:
            x,y = eachLine.split(',')
            xar.append(int(x))
            yar.append(int(y))
    ax1.clear()
    ax1.scatter(xar,yar)
ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()
