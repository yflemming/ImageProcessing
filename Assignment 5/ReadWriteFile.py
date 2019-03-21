'''
Created on Apr 4, 2014

@author: Yuon
'''

import numpy as np

def WriteNodes(filename, matrix):
    f = open(filename, 'w')
    f.write('%d %d' % (matrix.shape[0], matrix.shape[1]))
    for i in range(0, matrix.shape[0]):
        f.write('\n')
        for j in range(0, matrix.shape[1]):
            f.write('%+f ' % (matrix[i][j]))
    f.close()
            
def WriteElements(filename, matrix):
    f = open(filename, 'w')
    f.write('%d %d' % (matrix.shape[0], matrix.shape[1]))
    for i in range(0, matrix.shape[0]):
        f.write('\n')
        for j in range(0, matrix.shape[1]):
            f.write('%+d ' % (matrix[i][j]))
    f.close()
            
def ReadNodes(filename):
    f = open(filename, 'r')
    line = f.readline().split()
    x = int(line[0])
    y = int(line[1])
    nodes = []
    for i in range(1,x+1):
        newline = f.readline().split()
        values = []
        for j in range(0, y):
            values.append(float(newline[j]))
        nodes.append(values)
    return nodes

def ReadElements(filename):
    f = open(filename, 'r')
    line = f.readline().split()
    x = int(line[0])
    y = int(line[1])
    elements = []
    for i in range(1,x+1):
        newline = f.readline().split()
        values = []
        for j in range(0, y):
            values.append(int(newline[j]))
        elements.append(values)
    return elements


    

coords = np.random.uniform(-20, 20, size=(20,2))
elts = np.random.randint(-20, 20, size=(10,3))

WriteNodes('nodes.txt', coords)
WriteElements('elements.txt', elts)

nodes = ReadNodes('nodes.txt')
elements = ReadElements('elements.txt')

normNodes = np.linalg.norm(nodes - coords)
normElts = np.linalg.norm(elements - elts)

print normNodes
print normElts










