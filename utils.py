import numpy as np
from skimage import io, util

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


class MinCutPatch:

    @classmethod
    def minCutPatch(cls, patch, patchLength, overlap, res, y, x, texture, counter):
        patch_copy = patch.copy()
        res_copy = res.copy()

        if x > 0:
            patch_right = res_copy[x:x + patchLength, y:y + overlap]
            patch_left = patch_copy[:, :overlap]

            totalNodes = (len(patch_right) * len(patch_right[0])) + 2

            '''Creating Adjacency matrix'''
            matrix = []
            for i in range(0, (len(patch_right) * len(patch_right[0])) + 2):
                rows = []
                for j in range(0, (len(patch_right) * len(patch_right[0])) + 2):
                    rows.append(0)
                matrix.append(rows)

            for i in range(1, len(patch_right) + 1):
                # print("Assigning inf from 0",i)
                matrix[0][i] = float('inf')
            for i in range(25 * (overlap - 1) + 1, totalNodes - 1):
                # print("Assigning inf from ",i,totalNodes-1)
                matrix[i][totalNodes - 1] = float('inf')

            cols = len(patch_right)
            for i in range(1, patchLength + 1):
                prev = i
                for j in range(0, overlap):
                    if j + 1 < overlap:

                        matrix[prev][prev + cols] = getAvg(abs(patch_left[i - 1][j] - patch_right[i - 1][j]) + abs(
                            patch_left[i - 1][j + 1] - patch_right[i - 1][j + 1]))
                        if matrix[prev][prev + cols] == 0:
                            matrix[prev][prev + cols] = float('inf')

                    if i < patchLength:

                        matrix[prev][prev + 1] = getAvg(
                            abs(patch_left[i - 1][j] - patch_right[i - 1][j]) + abs(
                                patch_left[i][j] - patch_right[i][j]))
                        if matrix[prev][prev + 1] == 0:
                            matrix[prev][prev + 1] = float('inf')
                    prev = prev + cols

            # saving adjacency matrix to a file

            matrix_fileStr = 'Adjacency Matrices/Adjacency_matrix_' + str(counter) + '.txt'
            with open(matrix_fileStr, 'w') as f:
                for i in range(0, len(matrix)):
                    for j in range(0, len(matrix[0])):
                        f.write(str(matrix[i][j]))
                        f.write(" ")
                    f.write("\n")

            graph = matrix

            g = GraphCut(graph)

            source = 0
            sink = 101

            left_pixels, right_pixels = g.minCut(source, sink)

            # saving pixels to a file
            pixels_fileStr = 'Pixels/Pixel_' + str(counter) + '.txt'
            with open(pixels_fileStr, 'w') as f:
                for i in range(0, len(left_pixels)):
                    f.write(str(left_pixels[i]))
                    f.write("-")
                    f.write(str(right_pixels[i]))
                    f.write("\n")

            pixelsFromLeft = set()
            pixelsFromRight = set()

            for i in range(0, len(left_pixels)):
                # vertical cut
                if right_pixels[i] != left_pixels[i] + 1:

                    row = left_pixels[i] % 25
                    if row == 0:
                        row = 25
                    ctr = 0

                    # all nodes to the left of the cut in the row will be from left hand side

                    for j in range(row, left_pixels[i] + 1):
                        pixelsFromLeft.add(j)
                        ctr += 1

                    # all nodes to the right of the cut in the row will be from right hand side

                    j = right_pixels[i]
                    while ctr < overlap:
                        pixelsFromRight.add(j)
                        j = j + 25
                        ctr += 1

        if y > 0:
            # lets define two sets
            matrix = []
            patch_up = res_copy[x:(x + overlap), y:y + patchLength]
            patch_bottom = patch_copy[:overlap, :]

            for i in range(0, (len(patch_up) * len(patch_up[0])) + 2):
                rows = []
                for j in range(0, (len(patch_up) * len(patch_up[0])) + 2):
                    rows.append(0)
                matrix.append(rows)

            for i in range(1, len(patch_up) + 1):
                matrix[0][i] = float('inf')
            for i in range(97, 101):
                matrix[i][101] = float('inf')

            cols = len(patch_up)
            for i in range(1, overlap + 1):
                prev = i
                for j in range(0, patchLength):
                    if j + 1 < patchLength:
                        matrix[prev][prev + cols] = getAvg(abs(patch_up[i - 1][j] - patch_bottom[i - 1][j]) + abs(
                            patch_up[i - 1][j + 1] - patch_bottom[i - 1][j + 1]))

                    if i < overlap:
                        matrix[prev][prev + 1] = getAvg(
                            abs(patch_up[i - 1][j] - patch_bottom[i - 1][j]) + abs(patch_up[i][j] - patch_bottom[i][j]))
                    prev = prev + cols

            matrix_fileStr = 'Adjacency Matrices/Adjacency_matrix_' + str(counter) + '.txt'
            with open(matrix_fileStr, 'w') as f:
                for i in range(0, len(matrix)):
                    for j in range(0, len(matrix[0])):
                        f.write(str(matrix[i][j]))
                        f.write(" ")
                    f.write("\n")

            graph = matrix

            g = GraphCut(graph)

            source = 0
            sink = 101

            left_pixels, right_pixels = g.minCut(source, sink)

            pixels_fileStr = 'Pixels/Pixel_' + str(counter) + '.txt'
            with open(pixels_fileStr, 'w') as f:
                for i in range(0, len(left_pixels)):
                    f.write(str(left_pixels[i]))
                    f.write("-")
                    f.write(str(right_pixels[i]))
                    f.write("\n")

            pixelsFromUp = set()
            pixelsFromBottom = set()

            for i in range(0, len(left_pixels)):
                # horizontal cut
                if right_pixels[i] == left_pixels[i] + 1:

                    row = left_pixels[i] % 4
                    if row == 0:
                        row = 4
                    column = int(min(left_pixels[i] / 4, right_pixels[i] / 4))
                    ctr = 0
                    for j in range((column * 4) + 1, left_pixels[i] + 1):
                        pixelsFromUp.add(j)
                        ctr += 1
                    j = right_pixels[i]
                    while ctr < overlap:
                        pixelsFromBottom.add(j)
                        j = j + 1
                        ctr += 1

        return patch


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
                patch = min_cut.minCutPatch(patch, patchLength, overlap, res, y, x, texture, counter)

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

