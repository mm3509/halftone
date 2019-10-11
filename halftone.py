import argparse
import math
import numpy
import os

import cv2

import preview

WHITE = [255, 255, 255]
QUANTILES = "quantiles"
BINS = "bins"

# This value is the default for brightness adjustment of an image and works well
# for the laser cutter in Cambridge.
MINIMUM_BRIGHTNESS = 0.66

RED_STANDARD = 299
GREEN_STANDARD = 587
BLUE_STANDARD = 114
ALL_WEIGHTS = {"std": [RED_STANDARD, GREEN_STANDARD, BLUE_STANDARD],
               "red": [1, 0, 0],
               "green": [0, 1, 0],
               "blue": [0, 0, 1]}

# 0 means black
TONE_CHOICES_STR = ['3'] # ['1', '2', '3']
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
    parser.add_argument("-t", "--tones", default='3')
    parser.add_argument("-m", "--minimum_brightness", default=MINIMUM_BRIGHTNESS)
		
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

def adjust_brightness_directly(gray_img, minimum_brightness):
    """Adjusts the brightness of a grayscale image with a direct calculation of bias
    and gain.
    """

    if 3 <= len(gray_img.shape):
        raise ValueError("Expected a grayscale image, color channels found")
    
    cols, rows = gray_img.shape
    brightness = numpy.sum(gray_img) / (255 * cols * rows)

    ratio = brightness / minimum_brightness
    if ratio >= 1:
        print("Image already bright enough")
        return img

    # Otherwise, adjust brightness to get the target brightness. Except for
    # saturation arithmetics, the new brightness should be the target brightness
    bright_img = cv2.convertScaleAbs(gray_img, alpha = 1 / ratio, beta = 0)

    print("Old: " + str(brightness))
    print("New: " + str(numpy.sum(bright_img) / (255 * cols * rows)))

    return bright_img

def percentile_to_bias_and_gain(gray_img, percentile):
    """Computs the bias and gain that corresponds to clipping the given percentile
    from the shadows and the highlights.
    """

    if 3 <= len(gray_img.shape):
        raise ValueError("Expected a grayscale image, color channels found")
    
    # Code from
    # https://stackoverflow.com/questions/57030125/adjusting-brightness-automatically-and-without-veil/57046925

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist = percentile * maximum / (2 * 100.0)

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = - minimum_gray * alpha

    return alpha, beta
    

def adjust_brightness_with_histogram(gray_img, minimum_brightness):
    """Adjusts brightness with histogram clipping by trial and error.
    """

    if 3 <= len(gray_img.shape):
        raise ValueError("Expected a grayscale image, color channels found")

    new_img = gray_img
    percentile = 0.1

    while True:
        cols, rows = new_img.shape
        brightness = numpy.sum(new_img) / (255 * cols * rows)

        if brightness >= minimum_brightness:
            break

        percentile += 0.1
        alpha, beta = percentile_to_bias_and_gain(new_img, percentile)
        new_img = cv2.convertScaleAbs(gray_img, alpha = alpha, beta = beta)

    print("New brightness: " + str(brightness))
        
    return new_img


def rescale(img, width, height, line_gap, tone_dim, overflow):
    """Rescales image to have the right size for half-toning and engraving."""
    width_lines_target = int(round(math.ceil(width / line_gap / tone_dim)))
    height_lines_target = int(round(math.ceil(height / line_gap / tone_dim)))

    img_height = img.shape[0]
    img_width = img.shape[1]
    
    # Find which is the biting dimension, then resize and pad with
    # centering. This image is centered but the half-toned image may not be
    # because 1 pixel difference between the paddings here means 3 pixels of
    # difference in the final image with 3x3 tones.
    width_correct = ((float(width_lines_target) / img_width) <
                     (float(height_lines_target) / img_height))

    # Flip it in case the image needs to overflow the size
    if "y" == overflow:
        width_correct = not width_correct
        
    if width_correct:
        width_lines = width_lines_target
        height_lines = int(math.ceil(width_lines * img_height / img_width))
    else:
        height_lines = height_lines_target
        width_lines = int(math.ceil(height_lines * img_width / img_height))

    print("New image size: %d x %d" % (height_lines, width_lines))

    # Resize: note that width comes first, unlike the shape!
    resized = cv2.resize(img, (width_lines, height_lines), interpolation = cv2.INTER_CUBIC)

    # If the image needs does not overflow, return now
    if "y" == overflow:
        return resized

    # Otherwise, pad it
    if width_correct:
        pad = height_lines_target - height_lines
        pad_top = pad // 2
        pad_bottom = pad - pad_top
        pad_left = 0
        pad_right = 0
    else:
        pad = width_lines_target - width_lines
        pad_left = pad // 2
        pad_right = pad - pad_left
        pad_top = 0
        pad_bottom = 0
        
    # Pad and return
    return cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=WHITE)

def convert_to_grayscale(img, red_weight, green_weight, blue_weight):
    """Converts RGB to grayscale using file-level constant weights."""
    coefficients = [blue_weight, green_weight, red_weight]
    m = numpy.array(coefficients).reshape((1,3)) / (blue_weight + green_weight + red_weight)

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

        if 0 == i % 10:
            print("At row %d of %d" % (i, num_rows))

        for j in range(num_cols):
            j_output = range(j * tone_dim, (j + 1)* tone_dim)
            
            pixel = gray[i, j]
            brightness = int(round((tone_num - 1) * pixel / 255))

            output[numpy.ix_(i_output, j_output)] = tones_dict[brightness]
            
    return output

def process_image(filepath, width, height, line_gap, red_weight,
                  green_weight, blue_weight, tones_str, overflow, minimum_brightness, suffix = "", do_halftoning = True, do_previewing = True):
    """Wrapper function that calls the others."""

    new_filepath = os.path.splitext(filepath)[0] + suffix + ".png"

    img = cv2.imread(filepath)

    assert tones_str in TONE_CHOICES_STR, "Unable to deal with '%s' tone(s), type: %s" % (tones_str, type(tones_str))
    if '1' == tones_str:
        tones = TONES1
    elif '2' == tones_str:
        tones = TONES2
    elif '3' == tones_str:
        tones = TONES3
    else:
        assert False, "Unknown option %s" % tones

    # Tone number is the number of tones. Tone dimension is the side length of
    # the square with the tones, i.e. 1 for 2 tones, 2 for 5 tones, 3 for 10
    # tones
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
    
    # Adjust brightness
    if False:
        brightener = adjust_brightness_directly
    else:
        brightener = adjust_brightness_with_histogram
    bright = brightener(gray, minimum_brightness = minimum_brightness)

    if not do_halftoning:
        cv2.imwrite(new_filepath, bright)
        return
    
    # Half-tone
    print("Starting half-toning")
    tones_dict = process_tones(tones, tone_num = tone_num, tone_dim = tone_dim)
    png = halftone(bright, tones_dict = tones_dict,
                   tone_num = tone_num, tone_dim = tone_dim)

    # Save
    cv2.imwrite(new_filepath, png, [cv2.IMWRITE_PNG_BILEVEL, 1])
    print("Saved %s" % new_filepath)

    if not do_previewing:
        return

    # Also save a preview of what the result will be? Not for now, because it takes long
    preview_fp = preview.process_image(new_filepath)
    print("Saved preview at %s" % preview_fp)

def process_all_options(filepath, width, height, line_gap, overflow, minimum_brightness):


    for tones_str in TONE_CHOICES_STR:
        for weight_str in ALL_WEIGHTS:
            suffix = " %s %s" % (tones_str, weight_str)
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
                          tones_str = tones_str,
                          suffix = suffix,
                          overflow = overflow,
                          minimum_brightness = minimum_brightness)

    
def main():
    args = get_args()

    filepath = args.file
    size = args.size
    line_gap = float(args.line_gap)
    overflow = args.overflow
    all_options = args.all_options
    minimum_brightness = float(args.minimum_brightness)

    assert os.path.exists(filepath), "Input file missing!"
    
    delimiter_index = size.index("x")
    width = int(size[:delimiter_index])
    height = int(size[delimiter_index+1:])

    if "y" == all_options:
        process_all_options(filepath, width = width, height = height,
                            line_gap = line_gap, overflow = overflow,
                            minimum_brightness = minimum_brightness)
    else:
        red_weight = int(args.red)
        green_weight = int(args.green)
        blue_weight = int(args.blue)
        tones_str = args.tones

        process_image(filepath, width = width, height = height,
                      line_gap = line_gap, overflow = overflow,
                      red_weight = red_weight, green_weight = green_weight,
                      blue_weight = blue_weight, tones_str = tones_str,
                      minimum_brightness = minimum_brightness)
                      

if "__main__" == __name__:
    main()
