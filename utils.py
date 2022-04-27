import numpy as np
from skimage import io, util
from mincutpatch import MinCutPatch

''' Python program for finding min-cut in the given graph
    Complexity : (E*(V^3))
    Total augmenting path = VE and BFS
    with adj matrix takes :V^2 times '''


class GraphCut:

    def __init__(self, graph):
        self.graph = graph  # residual graph
        self.org_graph = [i[:] for i in graph]
        self.ROW = len(graph)
        self.COL = len(graph[0])

    '''Returns true if there is a path from 
    source 's' to sink 't' in 
    residual graph. Also fills 
    parent[] to store the path '''

    def BFS(self, s, t, parent):

        # Mark all the vertices as not visited
        visited = [False] * self.ROW

        # Create a queue for BFS
        queue = [s]

        # Mark the source node as visited and enqueue it
        visited[s] = True

        # Standard BFS Loop
        while queue:

            # Dequeue a vertex from queue and print it
            u = queue.pop(0)

            # Get all adjacent vertices of
            # the dequeued vertex u
            # If a adjacent has not been
            # visited, then mark it
            # visited and enqueue it
            for ind, val in enumerate(self.graph[u]):
                if visited[ind] == False and val > 0:
                    queue.append(ind)
                    visited[ind] = True
                    parent[ind] = u

                    # If we reached sink in BFS starting
        # from source, then return
        # true, else false
        return True if visited[t] else False

    # Function for Depth first search
    # Traversal of the graph
    def dfs(self, graph, s, visited):
        visited[s] = True
        for i in range(len(graph)):
            if graph[s][i] > 0 and not visited[i]:
                self.dfs(graph, i, visited)

    # Returns the min-cut of the given graph
    def minCut(self, source, sink):

        # This array is filled by BFS and to store path
        global s
        parent = [-1] * self.ROW

        max_flow = 0  # There is no flow initially

        # Augment the flow while there is path from source to sink
        while self.BFS(source, sink, parent):

            # Find minimum residual capacity of the edges along the
            # path filled by BFS. Or we can say find the maximum flow
            # through the path found.
            path_flow = float("Inf")
            s = sink
            while s != source:
                path_flow = min(path_flow, self.graph[parent[s]][s])
                s = parent[s]

            max_flow += path_flow

            v = sink
            while v != source:
                u = parent[v]
                self.graph[u][v] -= path_flow
                self.graph[v][u] += path_flow
                v = parent[v]

        visited = len(self.graph) * [False]
        self.dfs(self.graph, s, visited)

        left_image = []
        right_image = []

        for i in range(self.ROW):
            for j in range(self.COL):
                if self.graph[i][j] == 0 and \
                        self.org_graph[i][j] > 0 and visited[i]:
                    left_image.append(i)
                    right_image.append(j)

        return left_image, right_image


def L2OverlapDiff(patch, patchLength, overlap, res, y, x):
    error = 0

    if x > 0:
        left = patch[:, :overlap] - res[y:y + patchLength, x:x + overlap]
        error += np.sum(left ** 2)

    if y > 0:
        up = patch[:overlap, :] - res[y:y + overlap, x:x + patchLength]
        error += np.sum(up ** 2)

    if x > 0 and y > 0:
        corner = patch[:overlap, :overlap] - res[y:y + overlap, x:x + overlap]
        error -= np.sum(corner ** 2)

    return error


def FindRandomPatch(texture, patchLength):
    h, w, _ = texture.shape
    i = np.random.randint(h - patchLength)
    j = np.random.randint(w - patchLength)

    return texture[i:i + patchLength, j:j + patchLength]


def FindRandomBestPatch(texture, patchLength, overlap, res, y, x):
    h, w, _ = texture.shape
    errors = np.zeros((h - patchLength, w - patchLength))

    for i in range(h - patchLength):
        for j in range(w - patchLength):
            patch = texture[i:i + patchLength, j:j + patchLength]
            e = L2OverlapDiff(patch, patchLength, overlap, res, y, x)
            errors[i, j] = e

    i, j = np.unravel_index(np.argmin(errors), errors.shape)
    return texture[i:i + patchLength, j:j + patchLength]


successfulPatch = []


def quilt_image(texture, patchLength, numPatches, mode="cut", sequence=True):
    min_cut = MinCutPatch()
    texture = util.img_as_float(texture)

    overlap = patchLength // 6
    numPatchesHigh, numPatchesWide = numPatches

    h = (numPatchesHigh * patchLength) - (numPatchesHigh - 1) * overlap
    w = (numPatchesWide * patchLength) - (numPatchesWide - 1) * overlap

    res = np.zeros((h, w, texture.shape[2]))

    counter = 0

    for i in range(numPatchesHigh):
        for j in range(numPatchesWide):
            x = i * (patchLength - overlap)
            y = j * (patchLength - overlap)
            print("X", x)
            print("Y", y)

            if mode == "best":
                patch = FindRandomBestPatch(texture, patchLength, overlap, res, y, x)
            elif mode == "cut":
                counter += 1
                patch = FindRandomBestPatch(texture, patchLength, overlap, res, y, x)
                patch = min_cut.minCutPatch(patch, patchLength, overlap, res, y, x, counter)

            else:
                patch = FindRandomPatch(texture, patchLength)
                successfulPatch = patch

            res[y:y + patchLength, x:x + patchLength] = patch

            if sequence:
                io.imshow(res)
                io.show()

    return res


def getAvg(pixelArray):
    return sum(pixelArray) / 3

