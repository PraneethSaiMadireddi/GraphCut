class MinCutPatch:

    @classmethod
    def minCutPatch(cls, patch, patchLength, overlap, res, y, x, counter):
        patch_copy = patch.copy()
        res_copy = res.copy()

        if x > 0:
            from utils import getAvg
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

                matrix[0][i] = float('inf')
            for i in range(25 * (overlap - 1) + 1, totalNodes - 1):

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
            from utils import GraphCut
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
            from utils import getAvg
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
            from utils import GraphCut
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
