import cv2
from imutils.perspective import four_point_transform
from skimage import color, filters


def find_puzzle(image, debug=True):
    # Convert to grayscale and blur image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur image
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)

    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    if debug:
        cv2.imshow("Puzzle Thresh", thresh)
        #cv2.waitKey(0)

    # Find contours
    # CHAIN_APPROX_SIMPLE: It removes all redundant points and compresses the contour
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort by contour area descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Contour approximation, find square, 4 points
    # "it approximates a contour shape to another shape with less number of vertices depending upon the precision we specify"
    print(f"Points before approximation: {len(contours[0])}")
    puzzle_contour = None
    for c in contours:
        perimeter = cv2.arcLength(c, True)  # Closed shape
        epsilon = 0.01 * perimeter
        approx_countour = cv2.approxPolyDP(c, epsilon, True)

        # if the approximation has 4 points we assume its the puzzle square
        if len(approx_countour) == 4:  # Square
            puzzle_contour = approx_countour
            break

    if puzzle_contour is None:
        raise Exception(("Could not find Sudoku puzzle outline. "
                         "Try debugging your thresholding and contour steps."))

    print(f"Points after approximation: {len(puzzle_contour)}, {puzzle_contour}")
    if debug:
        # draw the contour of the puzzle on the image and then display
        # it to our screen for visualization/debugging purposes
        output = image.copy()
        cv2.drawContours(output, [puzzle_contour], -1, (0, 255, 0), 2)
        cv2.imshow("Puzzle Outline", output)
        #cv2.waitKey(0)

    # Apply 4 point perspective transform to obtain
    # a top-down bird's eye view of the puzzle
    print(puzzle_contour.reshape(4, 2))
    puzzle = four_point_transform(image, puzzle_contour.reshape(4, 2))
    warped = four_point_transform(gray, puzzle_contour.reshape(4,2))

    if debug:
        cv2.imshow("Puzzle Transform", puzzle)
        #cv2.waitKey(0)

    # Returns tuple o puzzle RGB and grayscale transformed
    return puzzle, warped

if __name__ == "__main__":
    image = cv2.imread("../sudoku_puzzle.jpg")
    cv2.imshow("Initial Puzzle", image)
    find_puzzle(image)
    cv2.waitKey(0)
