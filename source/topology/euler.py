# Determination of Euler

def get_number_holes(image):
    matches_Q1, matches_Q3, matches_QD = compute_matches_Q(image)
    # invert Q3 and Q1 since in this case black is object instead of white
    euler = matches_Q3 - matches_Q1 - 2 * matches_QD
    # C is always 1 since we assume only 1 object in the sketch
    return 1 - int(euler / 4)


def compute_matches_Q(image):
    shape_x, shape_y = image.shape
    Q1, Q3, QD = 0, 0, 0
    for x in range(shape_x - 1):
        for y in range(shape_y - 1):
            current_pixel = image[x][y]
            neighbors = [(x + 1, y), (x, y + 1), (x + 1, y + 1)]
            neighbor_sum = 0
            for i in neighbors:
                neighbor_sum += image[i[0], i[1]]
            sum = neighbor_sum + current_pixel

            if sum == 1:
                Q1 += 1
            elif sum == 3:
                Q3 += 1
            elif sum == 2:
                if current_pixel == 1 and image[neighbors[2][0], neighbors[2][1]] == 1:
                    QD += 1
                if current_pixel == 0 and image[neighbors[2][0], neighbors[2][1]] == 0:
                    QD += 1
    return Q1, Q3, QD
