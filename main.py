import numpy as np
import pandas as pd

# Main
dataset = pd.read_csv('train.csv')
dataset = np.array(dataset)

vector = dataset[0][1:].T

subset = dataset[:10].T
dataset = dataset.T

weights = np.random.rand(10, len(vector))

output = np.dot(weights, subset[1:])


# Each row printed here is a weight of a
for i in range(len(output[0])):
    a = ""
    for j in range(len(output)):
        a += "{:<20} ".format(str(output[j][i]))
    print(a)