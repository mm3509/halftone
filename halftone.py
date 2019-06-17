import argparse
import math
import numpy
import os

import cv2

WHITE = [255, 255, 255]
QUANTILES = "quantiles"
BINS = "bins"

RED_STANDARD = 299
GREEN_STANDARD = 587
BLUE_STANDARD = 114
ALL_WEIGHTS = {"std": [RED_STANDARD, GREEN_STANDARD, BLUE_STANDARD],
               "red": [1, 0, 0],
               "green": [0, 1, 0],
               "blue": [0, 0, 1]}

# 0 means black
TONE_CHOICES = [1, 2, 3]
TONES1 = [[0], [1]]

TONES2 = [[0, 0,
           0, 0],
          [0, 1,
           0, 0],
          [1, 1,
           0, 0],
          [1, 1,
           0, 1],
          [1, 1,
           1, 1]]

TONES3 = [[0, 0, 0,
          0, 0, 0,
          0, 0, 0],
         [0, 1, 0,
          0, 0, 0,
          0, 0, 0],
         [0, 1, 0,
          0, 0, 0,
          0, 0, 1],
         [1, 1, 0,
          0, 0, 0,
          0, 0, 1],
         [1, 1, 0,
          0, 0, 0,
          1, 0, 1],
         [1, 1, 1,
          0, 0, 0,
          1, 0, 1],
         [1, 1, 1,
          0, 0, 1,
          1, 0, 1],
         [1, 1, 1,
          0, 0, 1,
          1, 1, 1],
         [1, 1, 1,
          1, 0, 1,
          1, 1, 1],
         [1, 1, 1,
          1, 1, 1,
          1, 1, 1]]

ALL_TONES = [TONES1, TONES2, TONES3]

def get_args():
    """This function parses and return arguments passed in"""
    # Assign description to the help doc
    parser = argparse.ArgumentParser(
        description='Convert images to laser-ready bitmaps using half-toning')
    # Add arguments
    parser.add_argument("-f", "--file", required=True)
    parser.add_argument("-s", "--size", default="75x105")
    parser.add_argument("-l", "--line_gap", default=0.05)
    parser.add_argument("-o", "--overflow", choices=["y", "n"], default="y")
    parser.add_argument("-a", "--all_options", choices=["y", "n"], default="n")
    parser.add_argument("-r", "--red",  default=RED_STANDARD)
    parser.add_argument("-g", "--green", default=GREEN_STANDARD)
    parser.add_argument("-b", "--blue", default=BLUE_STANDARD)
    parser.add_argument("-t", "--tones", choices=TONE_CHOICES, default=3)
		
    # Array for all arguments passed to script
    args = parser.parse_args()
    # Assign args to variables
    return args

def process_tones(tones, tone_num, tone_dim):
    """Converts the tones above to image ready."""
    tones_dict = dict()

    for t in tones:
        assert tone_num - 1 == len(t), "Found %s tones instead of %s " % (str(tone_num), str(len(t)))
        brightness = sum(t)
        bitmap_tone = numpy.reshape(t, (tone_dim, tone_dim)) * 255
        tones_dict[brightness] = bitmap_tone
    return(tones_dict)

def rescale(img, width, height, line_gap, tone_dim, overflow):
    
    """Rescales image to have the right size for half-toning and engraving."""
    width_lines = int(round(math.ceil(width / line_gap / tone_dim)))
    height_lines = int(round(math.ceil(height / line_gap / tone_dim)))

    img_height = img.shape[0]
    img_width = img.shape[1]
    
    # Find which is the biting dimension, then resize and pad with
    # centering. This image is centered but the half-toned image may not be
    # because 1 pixel difference between the paddings here means 3 pixels of
    # difference in the final image with 3x3 tones.
    width_correct = (float(width_lines) / img_width) < (float(height_lines) / img_height)

    # Flip it in case the image needs to overflow the size
    if "y" == overflow:
        width_correct = not width_correct
        
    if width_correct:
        target_lines = int(math.ceil(width_lines * img_height / img_width))
        
        pad = height_lines - target_lines
        pad_top = pad // 2
        pad_bottom = pad - pad_top
        pad_left = 0
        pad_right = 0
        
        height_lines = target_lines
    else:
        target_lines = int(math.ceil(height_lines * img_width / img_height))
        
        pad = width_lines - target_lines
        pad_left = pad // 2
        pad_right = pad - pad_left
        pad_top = 0
        pad_bottom = 0

        width_lines = target_lines
        
    # Resize: note that width comes first, unlike the shape!
    resized = cv2.resize(img, (width_lines, height_lines), interpolation = cv2.INTER_CUBIC)

    # Pad to reach requested size
    padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=WHITE)
    return padded

def convert_to_grayscale(img, red_weight, green_weight, blue_weight):
    """Converts RGB to grayscale using file-level constant weights."""
    coefficients = [blue_weight, green_weight, red_weight]
    m = numpy.array(coefficients).reshape((1,3)) / (blue_weight + green_weight + red_weight)

    # TODO(mmorin): change division to also adjust overall brightness?
    gray = cv2.transform(img, m)
    return gray

def halftone(gray, tones_dict, tone_num, tone_dim):

    #return gray
    
    num_rows = gray.shape[0]
    num_cols = gray.shape[1]

    output = numpy.zeros((num_rows * tone_dim, num_cols * tone_dim),
                         dtype = numpy.uint8)

    # Go through each pixel
    for i in range(num_rows):
        i_output = range(i * tone_dim, (i + 1)* tone_dim)

        for j in range(num_cols):
            j_output = range(j * tone_dim, (j + 1)* tone_dim)
            
            pixel = gray[i, j]
            brightness = int(round((tone_num - 1) * pixel / 255))

            output[numpy.ix_(i_output, j_output)] = tones_dict[brightness]
            
    return output

def process_image(filepath, width, height, line_gap, red_weight,
                  green_weight, blue_weight, tones, overflow, suffix = ""):
    """Wrapper function that calls the others."""

    new_filepath = os.path.splitext(filepath)[0] + suffix + ".bmp"
    assert not os.path.exists(new_filepath)

    img = cv2.imread(filepath)

    assert tones in TONE_CHOICES
    if 1 == tones:
        tones = TONES1
    elif 2 == tones:
        tones = TONES2
    elif 3 == tones:
        tones = TONES3
    else:
        assert False, "Unknown option %s" % tones

    tone_num = len(tones)
    tone_dim = int(math.sqrt(tone_num - 1))
    
    # Rescale
    img_scaled = rescale(img, width = width, height = height,
                         line_gap = line_gap, tone_dim = tone_dim,
                         overflow = overflow)

    # Convert to gray
    gray = convert_to_grayscale(img_scaled, red_weight = red_weight,
                                green_weight = green_weight,
                                blue_weight = blue_weight)
    
    # Half-tone
    tones_dict = process_tones(tones, tone_num = tone_num, tone_dim = tone_dim)
    bmp = halftone(gray, tones_dict = tones_dict,
                   tone_num = tone_num, tone_dim = tone_dim)

    # Save
    cv2.imwrite(new_filepath, bmp)

def process_all_options(filepath, width, height, line_gap, overflow):


    for tones in [1, 2, 3]:
        for weight_str in ALL_WEIGHTS:
            suffix = " %d %s" % (tones, weight_str)
            weights = ALL_WEIGHTS[weight_str]
            red_weight = weights[0]
            green_weight = weights[1]
            blue_weight = weights[2]

            process_image(filepath = filepath,
                          width = width,
                          height = height,
                          line_gap = line_gap,
                          red_weight = red_weight,
                          green_weight = green_weight,
                          blue_weight = blue_weight,
                          tones = tones,
                          suffix = suffix,
                          overflow = overflow)

    
def main():
    args = get_args()

    filepath = args.file
    size = args.size
    line_gap = float(args.line_gap)
    overflow = args.overflow
    all_options = args.all_options

    assert os.path.exists(filepath), "Input file missing!"
    
    delimiter_index = size.index("x")
    width = int(size[:delimiter_index])
    height = int(size[delimiter_index+1:])

    if "y" == all_options:
        process_all_options(filepath, width = width, height = height,
                            line_gap = line_gap, overflow = overflow)
    else:
        red_weight = int(args.red)
        green_weight = int(args.green)
        blue_weight = int(args.blue)
        tones = args.tones

        process_image(filepath, width = width, height = height,
                      line_gap = line_gap, overflow = overflow,
                      red_weight = red_weight, green_weight = green_weight,
                      blue_weight = blue_weight, tones = tones)
                      

if "__main__" == __name__:
    main()
