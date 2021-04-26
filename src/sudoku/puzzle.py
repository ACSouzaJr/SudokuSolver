import cv2
import numpy as np
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import matplotlib.pyplot as plt


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
        cv2.waitKey(0)

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
        cv2.waitKey(0)

    # Apply 4 point perspective transform to obtain
    # a top-down bird's eye view of the puzzle
    print(puzzle_contour.reshape(4, 2))
    puzzle = four_point_transform(image, puzzle_contour.reshape(4, 2))
    warped = four_point_transform(gray, puzzle_contour.reshape(4, 2))

    if debug:
        cv2.imshow("Puzzle Transform", puzzle)
        cv2.waitKey(0)

    # Returns tuple o puzzle RGB and grayscale transformed
    return puzzle, warped


def extract_digit(digit, debug=True):
    # Apply threshold to binary digit and clear border
    _, thresh = cv2.threshold(digit, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    thresh = clear_border(thresh)

    # check to see if we are visualizing the cell thresholding step
    if debug:
        cv2.imshow("Cell Thresh", thresh)
        cv2.waitKey(0)

    # Look or contour (number)
    # CHAIN_APPROX_SIMPLE: It removes all redundant points and compresses the contour
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Returns None if no contour (number) - empty space
    if len(contours) == 0:
        return None

    # otherwise, mask the largest contour - apply mask to remove background, noise from number
    c = max(contours, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)  # thickness -1 draw on interior

    # Check digit mask
    if debug:
        cv2.imshow("Cell Mask", mask)
        cv2.waitKey(0)

    # Compute the percentage of masked pixels - (detect noisy only image)
    (h, w) = thresh.shape
    percentage_filled = cv2.countNonZero(mask) / (h * w)

    # If percentage is less than 3% we assume its only noise - (empty cell)
    if percentage_filled < 0.03:
        return None

    # Apply the mask to the thresholded cell
    digit = cv2.bitwise_and(thresh, thresh, mask=mask)

    # check to see if we should visualize the masking step
    if debug:
        cv2.imshow("Digit", digit)
        cv2.waitKey(0)

    return digit

if __name__ == "__main__":
    image = cv2.imread("../../sudoku_puzzle.jpg")
    cv2.imshow("Initial Puzzle", image)
    cv2.waitKey(0)
    find_puzzle(image)
