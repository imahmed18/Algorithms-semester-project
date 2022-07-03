from os import sep
import sys
import numpy as np
import copy


fNum = 10
try:
    a_file = open(rf"input{fNum}.txt")
    print("File Found!")
except:
    print("File Not Found!")
    exit()
lines = a_file.readlines()
num = int(lines[2])
end = int(lines[6+2*num])
listX = []
listY = []
listMat = [[0]*num for _ in range(num)]
listMatD = [[0]*num for _ in range(num)]
print(num)
print(end)
for y in range(4, 4+num):
    strList = list(map(float, lines[y].split("\t")))
    listX.append(strList[1])
    listY.append(strList[2])
    # print(strList)
print(listX)
print(listY)
print(f"HERE: {listMat}")
for y in range(5+num, 5+2*num):
    strList = lines[y]
    strList = strList[0:len(strList)-2]
    strList = list(map(float, strList.split("\t")))
    print(strList)
    print(int(strList[0]))
    for index in range(int(strList[0])):
        print(
            f"{int(strList[0])} - {int(strList[index*4+1])} - {strList[int(index*4+3)]}")
        if listMat[int(strList[0])][int(strList[index*4+1])] == 0 or (listMat[int(strList[0])][int(strList[index*4+1])] != 0 and listMat[int(strList[0])][int(strList[index*4+1])] > strList[int(index*4+3)]):
            listMatD[int(strList[0])][int(strList[index*4+1])
                                      ] = strList[int(index*4+3)]
            listMat[int(strList[0])][int(strList[index*4+1])
                                     ] = strList[int(index*4+3)]
            listMat[int(strList[index*4+1])][int(strList[0])
                                             ] = strList[int(index*4+3)]

print("start")
print(*listMatD, sep="\n")
print("end")

print("Matrix is:")
for i in range(fNum):
    print(listMat[i])

matrix = copy.deepcopy(listMat)
for i in range(fNum):
    print(matrix[i])

start_index = end

# class Graph():

#     def _init_(self, vertices):
#         self.V = vertices
#         self.graph = [[0 for column in range(vertices)]
#                     for row in range(vertices)]

#     # A utility function to print the constructed MST stored in parent[]
#     def printMST(self, parent):
#         print("Edge \tWeight")
#         for i in range(1, self.V):
#             print(parent[i], "-", i, "\t", self.graph[i][ parent[i] ])

#     # A utility function to find the vertex with
#     # minimum distance value, from the set of vertices
#     # not yet included in shortest path tree
#     def minKey(self, key, mstSet):
#         min_index = 0
#         # Initialize min value
#         min = sys.maxsize
#         for v in range(self.V):
#             if key[v] < min and mstSet[v] == False:
#                 min = key[v]
#                 min_index = v

#         return min_index

#     # Function to construct and print MST for a graph
#     # represented using adjacency matrix representation
#     def primMST(self):

#         # Key values used to pick minimum weight edge in cut
#         key = [sys.maxsize] * self.V
#         parent = [None] * self.V # Array to store constructed MST
#         # Make key 0 so that this vertex is picked as first vertex
#         key[0] = 0
#         mstSet = [False] * self.V

#         parent[0] = -1 # First node is always the root of

#         for cout in range(self.V):

#             # Pick the minimum distance vertex from
#             # the set of vertices not yet processed.
#             # u is always equal to src in first iteration
#             u = self.minKey(key, mstSet)

#             # Put the minimum distance vertex in
#             # the shortest path tree
#             mstSet[u] = True

#             # Update dist value of the adjacent vertices
#             # of the picked vertex only if the current
#             # distance is greater than new distance and
#             # the vertex in not in the shortest path tree
#             for v in range(self.V):

#                 # graph[u][v] is non zero only for adjacent vertices of m
#                 # mstSet[v] is false for vertices not yet included in MST
#                 # Update the key only if graph[u][v] is smaller than key[v]
#                 if self.graph[u][v] > 0 and mstSet[v] == False and key[v] > self.graph[u][v]:
#                         key[v] = self.graph[u][v]
#                         parent[v] = u

#         self.printMST(parent)

# g = Graph(fNum)
# g.graph = matrix


# g.primMST()


# Python program for Dijkstra's single
# source shortest path algorithm. The program is
# for adjacency matrix representation of the graph

# Library for INT_MAX


# class Graph():

#     def __init__(self, vertices):
#         self.V = vertices
#         self.graph = [[0 for column in range(vertices)]
#                       for row in range(vertices)]

#     def printSolution(self, dist):
#         print("Vertex \tDistance from Source")
#         for node in range(self.V):
#             print(node, "\t", dist[node])

#     # A utility function to find the vertex with
#     # minimum distance value, from the set of vertices
#     # not yet included in shortest path tree
#     def minDistance(self, dist, sptSet):

#         # Initialize minimum distance for next node
#         min = 99999999

#         # Search not nearest vertex not in the
#         # shortest path tree
#         for u in range(self.V):
#             if dist[u] < min and sptSet[u] == False:
#                 min = dist[u]
#                 min_index = u

#         return min_index

#     # Function that implements Dijkstra's single source
#     # shortest path algorithm for a graph represented
#     # using adjacency matrix representation
#     def dijkstra(self, src):

#         dist = [99999999] * self.V
#         dist[src] = 0
#         sptSet = [False] * self.V

#         for cout in range(self.V):

#             # Pick the minimum distance vertex from
#             # the set of vertices not yet processed.
#             # x is always equal to src in first iteration
#             x = self.minDistance(dist, sptSet)

#             # Put the minimum distance vertex in the
#             # shortest path tree
#             sptSet[x] = True

#             # Update dist value of the adjacent vertices
#             # of the picked vertex only if the current
#             # distance is greater than new distance and
#             # the vertex in not in the shortest path tree
#             for y in range(self.V):
#                 if self.graph[x][y] > 0 and sptSet[y] == False and \
#                         dist[y] > dist[x] + self.graph[x][y]:
#                     dist[y] = dist[x] + self.graph[x][y]

#         self.printSolution(dist)


# # Driver program
# g = Graph(fNum)
# g.graph = matrix

# g.dijkstra(1)

# This code is contributed by Divyanshu Mehta
# class Graph:

#     def __init__(self, vertices):
#         self.V = vertices  # No. of vertices
#         self.graph = []

#     # function to add an edge to graph
#     def addEdge(self, u, v, w):
#         self.graph.append([u, v, w])

#     # utility function used to print the solution
#     def printArr(self, dist):
#         print("Vertex Distance from Source")
#         for i in range(self.V):
#             print("{0}\t\t{1}".format(i, dist[i]))

#     # The main function that finds shortest distances from src to
#     # all other vertices using Bellman-Ford algorithm. The function
#     # also detects negative weight cycle
#     def BellmanFord(self, src):

#         # Step 1: Initialize distances from src to all other vertices
#         # as INFINITE
#         dist = [float("Inf")] * self.V
#         dist[src] = 0

#         # Step 2: Relax all edges |V| - 1 times. A simple shortest
#         # path from src to any other vertex can have at-most |V| - 1
#         # edges
#         for _ in range(self.V - 1):
#             # Update dist value and parent index of the adjacent vertices of
#             # the picked vertex. Consider only those vertices which are still in
#             # queue
#             for u, v, w in self.graph:
#                 if dist[u] != float("Inf") and dist[u] + w < dist[v]:
#                     dist[v] = dist[u] + w

#         # Step 3: check for negative-weight cycles. The above step
#         # guarantees shortest distances if graph doesn't contain
#         # negative weight cycle. If we get a shorter path, then there
#         # is a cycle.

#         for u, v, w in self.graph:
#             if dist[u] != float("Inf") and dist[u] + w < dist[v]:
#                 print("Graph contains negative weight cycle")
#                 return

#         # print all distance
#         self.printArr(dist)


# g = Graph(fNum)
# for i in range(0, fNum):
#     for j in range(0, fNum):
#         if listMat[i][j] != 0:
#             g.addEdge(i, j, listMat[i][j])


# # Print the solution
# g.BellmanFord(1)

# Initially, Contributed by Neelam Yadav
# Later On, Edited by Himanshu Garg
# Python Program for Floyd Warshall Algorithm

# Number of vertices in the graph
V = 4

# Define infinity as the large
# enough value. This value will be
# used for vertices not connected to each other
INF = 99999

# Solves all pair shortest path
# via Floyd Warshall Algorithm


def floydWarshall(graph):
    """ dist[][] will be the output
    matrix that will finally
            have the shortest distances
            between every pair of vertices """
    """ initializing the solution matrix
	same as input graph matrix
	OR we can say that the initial
	values of shortest distances
	are based on shortest paths considering no
	intermediate vertices """

    dist = list(map(lambda i: list(map(lambda j: j, i)), graph))

    """ Add all vertices one by one
	to the set of intermediate
	vertices.
	---> Before start of an iteration,
	we have shortest distances
	between all pairs of vertices
	such that the shortest
	distances consider only the
	vertices in the set
	{0, 1, 2, .. k-1} as intermediate vertices.
	----> After the end of a
	iteration, vertex no. k is
	added to the set of intermediate
	vertices and the
	set becomes {0, 1, 2, .. k}
	"""
    for k in range(fNum):

        # pick all vertices as source one by one
        for i in range(fNum):

            # Pick all vertices as destination for the
            # above picked source
            for j in range(fNum):

                # If vertex k is on the shortest path from
                # i to j, then update the value of dist[i][j]
                dist[i][j] = min(dist[i][j],
                                 dist[i][k] + dist[k][j]
                                 )
    printSolution(dist)


# A utility function to print the solution
def printSolution(dist):
    print("Following matrix shows the shortest distances\
between every pair of vertices")
    for i in range(V):
        for j in range(V):
            if(dist[i][j] == INF):
                print("%7s" % ("INF")),
            else:
                print("%7d\t" % (dist[i][j])),
            if j == V-1:
                print("")


# Driver program to test the above program
# Let us create the following weighted graph
"""
			10
	(0)------->(3)
		|		 /|\
	5 |		 |
		|		 | 1
	\|/		 |
	(1)------->(2)
			3		 """

graph = listMatD
for i in range(0, fNum):
    for j in range(0, fNum):
        if graph[i][j] == 0:
            graph[i][j] = INF
for i in range(0, fNum):
    for j in range(0, fNum):
        if i == j:
            graph[i][j] = 0
# Print the solution
floydWarshall(graph)
print(*listMatD, sep="\n")
# This code is contributed by Mythri J L
