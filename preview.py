import argparse
import cv2
import glob
import numpy
import os
import pathlib

IMAGE_EXTENSIONS = ("jpg", "jpeg", "png")
SUFFIX = "_preview"

# For the Betsy laser cutter at the Cambridge Makespace, and with laser plywood
# from Slec, use either spillovers of 3 and 3 and a burn of 0.38, or spillovers
# of 5 and 5 and a burn of 0.13

def get_args():
    """This function parses and return arguments passed in"""
    # Assign description to the help doc
    parser = argparse.ArgumentParser(
        description='Preview result of bitmaps engraved in laser-cutter')
    # Add arguments
    parser.add_argument("-f", "--filepath", required=True)
    parser.add_argument("-x", "--x_spillover", default=3)
    parser.add_argument("-y", "--y_spillover", default=3)
    parser.add_argument("-b", "--burn", default=0.38)

    args = parser.parse_args()
    return args

def generate_influence(x_pad, y_pad, burn):
    """Creates an image with the burn of the black pixels."""

    # Initialize output
    rows = 1 + 2 * y_pad
    cols = 1 + 2 * x_pad    
    influence = numpy.zeros((rows, cols), dtype = numpy.float32)

    # The semi-axes of the ellipsis equal an increment of x_pad and y_pad
    # which lie beyond the span of the filter and equal 0
    x_axis = x_pad + 1
    y_axis = y_pad + 1
    
    # Assign to each pixel the gradient of its distance to center
    for col in range(cols):
        for row in range(rows):
            distance = ((row - y_pad)**2 / y_axis**2 + 
                        (col - x_pad)**2 / x_axis**2)
            #value = max(0, 1 - distance)**3 / 9
            if distance == 0:
                influence[row, col] = 1
            elif distance < 1:
                influence[row, col] = burn * (1 - distance)

    return influence

def convolution(img, influence):

    influence_rows, influence_cols = influence.shape
    if 0 == influence_rows % 2:
        raise ValueError("The influence should have an odd number of rows")
    if 0 == influence_cols % 2:
        raise ValueError("The influence should have an odd number of columns")
    
    # Pad image with white
    pad_x = influence_cols // 2
    pad_y = influence_rows // 2
    padded = cv2.copyMakeBorder(img, pad_y, pad_y, pad_x, pad_x,
                                cv2.BORDER_CONSTANT, value = 255)

    # The ouptut contains the amount of char, or black
    img_rows, img_cols = img.shape
    output = numpy.zeros((img_rows, img_cols), dtype = numpy.uint8)
    
    for i in range(img_rows):
        for j in range(img_cols):

            # Extract region of interest
            roi = padded[i:i + 2 * pad_y + 1, j:j + 2 * pad_x + 1]

            # Convert to burn on percent scale
            area = 1 - roi / 255

            # Convolve, with maximum at 100% black
            burn_spread = min(1, (influence * area).sum())

            # Store output back to white
            output[i, j] = int(255 * (1 - burn_spread))

    return output


def process_image(filepath, x, y, burn):
    new_filepath = os.path.splitext(filepath)[0] + SUFFIX + ".png"
    if os.path.exists(new_filepath):
        #raise ValueError("The destination file %s already exists" % new_filepath)
        # TODO(mmorin): remove this
        pass

    # X and Y should be odd, otherwise increment
    x_odd = x if 0 != x % 2 else x + 1
    y_odd = y if 0 != y % 2 else y + 1
    x_pad = x_odd // 2
    y_pad = y_odd // 2
    
    influence = generate_influence(x_pad = x_pad, y_pad = y_pad, burn = burn)

    # Save filter to disk
    cv2.imwrite("images/filter.png", (255 * (1 - influence)).astype(numpy.uint8))

    # Read image as grayscale
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    # Convolution
    output = convolution(img, influence)
    
    # Save
    cv2.imwrite(new_filepath, output)
    
    
def main():
    args = get_args()

    filepath = args.filepath
    x_spillover = args.x_spillover
    y_spillover = args.y_spillover
    burn = float(args.burn)

    p = pathlib.Path(filepath)
    if not p.exists():
        raise ValueError("The source file or directory %s does not exist" %filepath)
    if not p.is_dir():
        process_image(filepath, x = x_spillover, y = y_spillover, burn = burn)
    else:
        for f in glob.glob(os.path.join(filepath, "*.*")):
            if (f.lower().endswith(IMAGE_EXTENSIONS) and
                not os.path.splitext(f)[0].endswith(SUFFIX)):
                
                print("Processing %s" % f)
                process_image(f, x = x_spillover, y = y_spillover, burn = burn)
    

if "__main__" == __name__:
    main()
