import cv2
import os

import numpy

MAX_SPACING_PX = 10

BAND_WIDTH = 20 * 20
BAND_HEIGHT = 20
BAND_SPACING = 40

BLACK_BAND = numpy.zeros((BAND_HEIGHT, BAND_WIDTH), dtype = numpy.uint8)
WHITE_SPACE = 255 * numpy.ones((BAND_SPACING, BAND_WIDTH), dtype = numpy.uint8)

OUTPUT_DIR = "images"

def generate_lines_piece():
    """Generate test piece with lines to gauge the amount of burn.
    """
    
    ## Use odd numbers only, since the convolution filter takes only those
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

def generate_squares_piece():
    """Generate test piece with squares to gauge the minimum area for burning.
    """

    for square_dim in range(1, MAX_SPACING_PX):

        center = MAX_SPACING_PX // 2
        coord_start = center - square_dim // 2
        coord_end = coord_start + square_dim + 1

        addition = 255 * numpy.ones((MAX_SPACING_PX + 2, MAX_SPACING_PX + 2), dtype = numpy.uint8)
        addition[numpy.ix_(range(coord_start, coord_end), range(coord_start, coord_end))] = 0

        multiple = numpy.hstack([addition, addition, addition])
        
        if 1 == square_dim:
            piece = multiple
        else:
            piece = numpy.vstack([piece, multiple])

    return piece

def save_test_pieces(dir):
    lines = generate_lines_piece()
    cv2.imwrite(os.path.join(dir, "lines.png"), lines, [cv2.IMWRITE_PNG_BILEVEL, 1])

    squares = generate_squares_piece()
    cv2.imwrite(os.path.join(dir, "squares.png"), squares, [cv2.IMWRITE_PNG_BILEVEL, 1])
                

def main():
    save_test_pieces(dir = OUTPUT_DIR)

if "__main__" == __name__:
    main()
                         
    
