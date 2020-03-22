import random
import math
import matplotlib.pyplot as plt

def fact(x):
    f = 1
    for k in range(1,x+1):
        f = f * k
    return f

def gaussian(r, a, b):
    sig = 1; mu = 0
    X, Px = [], []

    for i in range(r):
        x = a + ((b-a) * random.random())
        prob_x = (1/((math.sqrt(2*math.pi)*sig))) * math.exp(-1*((x-mu)**2)/(2*sig*sig))
        X.append(x)
        Px.append(prob_x)

    return (X,Px)


def poisson(a, b, l = 10):
    X, Px = [], []

    for i in range(a, b):
        px = ((l ** i) * math.exp(-1 * l)) / fact(i)
        Px.append(px)
        X.append(i)

    return (X,Px)


def uniform(r, a, b):
    X, Px = [],[]

    for i in range(r):
        x = a + ((b-a) * random.random())
        X.append(x)
        px = 1/(b-a)
        Px.append(px)

    return (X, Px)


def exponential(r, a, b, l= 1.3):
    X, Px = [], []

    for i in range(r):
        x = a + ((b-a) * random.random())
        X.append(x)

        if x < 0:
            px = 0
        else:
            px = l * math.exp(-1 * l * x)

        Px.append(px)

    return (X, Px)


if __name__ == "__main__":
    plot_list = []
    plot_list.append(gaussian(500,-3,3))
    plot_list.append(uniform(500,2,6))
    plot_list.append(poisson(3,100))
    plot_list.append(exponential(500,3,6))

    title = ["Gaussian", "Unifrom", "Possion", "Exponential"]
    
    for i in range(0,len(plot_list)):
        plt.subplot(2,2,i+1)
        plt.scatter(plot_list[i][0],plot_list[i][1])
        plt.title(title[i])
        plt.xlabel("X")
        plt.ylabel("prob dis of x")

    plt.show()
