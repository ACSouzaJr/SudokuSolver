# USAGE
# python solve_sudoku_puzzle.py --model output/number_classifier.h5 --image sudoku_puzzle.jpg
import argparse
import cv2
import imutils
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from src.sudoku.puzzle import find_puzzle, extract_digit
from sudoku import Sudoku

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="path to trained number classifier")
    parser.add_argument("-i", "--image", required=True, help="path to Sudoku puzzle image")
    parser.add_argument("-d", "--debug", default=False, help="visualize each step of pipeline")

    args = parser.parse_args()

    # Load number classifier
    print("[INFO] loading digit classifier...")
    model = load_model(args.model)

    # Load input image
    print("[INFO] processing image...")
    image = cv2.imread(args.image)
    image = imutils.resize(image, width=600)  # Resize maintaining proportion

    # Find puzzle on image
    puzzle_image, warped = find_puzzle(image, args.debug)

    # initialize 9x9 board
    board = np.zeros((9, 9), dtype="int")

    # We can infer the position of each cell dividing
    # the warped image into a 9x9 grid
    step_X = warped.shape[1] // 9
    step_Y = warped.shape[0] // 9

    # Store cells coords
    cells_location = []

    # Generate cells location
    for y in range(0, 9):
        row = []
        # x - col
        # y - row
        for x in range(0, 9):
            # Compute start-end x-y location of current cell
            start_X = x * step_X
            start_Y = y * step_Y
            end_X = (x + 1) * step_X
            end_Y = (y + 1) * step_Y

            # Append coordinates to row
            row.append((start_X, start_Y, end_X, end_Y))

            # crop image and extract digit
            current_cell = warped[start_Y:end_Y, start_X:end_X]
            digit = extract_digit(current_cell, debug=args.debug)
            # cv2.imwrite(f"src/output/number{y}{x}.jpg", current_cell)

            if digit is not None:
                # resize the cell and prepare for classification
                roi = cv2.resize(digit, (28, 28))
                roi = roi.astype("float") / 255.0 * 0.99 + 0.01
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)  # Dimension (28, 28, 1)

                # Classify the digit and update sudoku board
                predict = model.predict(roi).argmax(axis=1)
                # print(f"Rest: {predict}")
                board[y, x] = predict

        # Add row to cells location
        cells_location.append(row)

    # construct a Sudoku puzzle from the board
    print("[INFO] OCR'd Sudoku board:")
    puzzle = Sudoku(3, 3, board=board.tolist())
    puzzle.show()

    # solve the Sudoku puzzle
    print("[INFO] solving Sudoku puzzle...")
    solution = puzzle.solve()
    solution.show_full()

    # Print solution on image
    for cell_row, board_row in zip(cells_location, solution.board):
        for cell, digit in zip(cell_row, board_row):
            start_X, start_Y, end_X, end_Y = cell
            # Compute digit location
            text_X = int((end_X - start_X) * 0.33)  # 1/3
            text_Y = int((end_Y - start_Y) * -0.2)
            text_X += start_X
            text_Y += end_Y
            print(f"Cell: {cell}; {(text_X, text_Y)}")

            # Draw number - use bottom-left corner
            cv2.putText(puzzle_image, str(digit), (text_X, text_Y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # show the output image
    cv2.imshow("Sudoku Result", puzzle_image)
    cv2.waitKey(0)
