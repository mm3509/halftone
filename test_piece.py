import cv2
import numpy

MAX_SPACING_PX = 10

BAND_WIDTH = 20 * 20
BAND_HEIGHT = 20
BAND_SPACING = 40

BLACK_BAND = numpy.zeros((BAND_HEIGHT, BAND_WIDTH), dtype = numpy.uint8)
WHITE_SPACE = 255 * numpy.ones((BAND_SPACING, BAND_WIDTH), dtype = numpy.uint8)

TEST_FILEPATH = "images/test.png"

def generate_test_piece():

    ## Odd numbers only, since the convolution filter takes only those
    for spacing in range(1, MAX_SPACING_PX + 1, 2):

        addition = numpy.vstack([BLACK_BAND,
                                 255 * numpy.ones((spacing, BAND_WIDTH), dtype = numpy.uint8),
                                 BLACK_BAND,
                                 WHITE_SPACE])
        if 1 == spacing:
            piece = addition
        else:
            piece = numpy.vstack([piece, addition])

    return piece

def save_test_piece(filepath):
    piece = generate_test_piece()
    cv2.imwrite(filepath, piece, [cv2.IMWRITE_PNG_BILEVEL, 1])

def main():
    save_test_piece(TEST_FILEPATH)

if "__main__" == __name__:
    main()
                         
    
