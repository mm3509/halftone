import argparse
import math
import os

import numpy as np
import cv2

import preview

WHITE = [255, 255, 255]
QUANTILES = "quantiles"
BINS = "bins"

WIDTH_TARGET = "width target"
HEIGHT_TARGET = "height target"
WIDTH_TARGET_BEFORE_HALFTONE = "width target before halftone"
HEIGHT_TARGET_BEFORE_HALFTONE = "height target before halftone"

# If LaserCutPro receives a 100 pixel image with 5mm of size and engraves it at
# 0.05 scan gap, it will skip one pixel every 20 pixels or so, probably due to
# aliasing. So I increase the size by 1 pixel, or whatever is below, so the image
# has 101 pixels with 5mm of size, and every line to engraving has at least one
# pixel of information to control it.
LASER_CUT_PRO_ALIAS_CORRECTION = 1

# This value is the default for brightness adjustment of an image and works well
# for the laser cutter in Cambridge.
MINIMUM_BRIGHTNESS = 0.66
PERCENTILE_SATURATION = 1

RED_STANDARD = 299
GREEN_STANDARD = 587
BLUE_STANDARD = 114
ALL_WEIGHTS = {"std": [RED_STANDARD, GREEN_STANDARD, BLUE_STANDARD],
               "red": [1, 0, 0],
               "green": [0, 1, 0],
               "blue": [0, 0, 1]}

DIRECT_BRIGHTENER = "direct"
ALPHABETA_BRIGHTENER = "alpha-beta"
GAMMA_BRIGHTENER = "gamma"
ALPHABETAGAMMA_BRIGHTENER = "alpha-beta-gamma"
ALL_BRIGHTENERS = [DIRECT_BRIGHTENER,
                   ALPHABETA_BRIGHTENER,
                   GAMMA_BRIGHTENER,
                   ALPHABETAGAMMA_BRIGHTENER,
]
PERCENTILE_STEP = 1
GAMMA_STEP = 0.01

# 0 means black
TONE_CHOICES_STR = ['1', '2', '3']
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
    parser.add_argument("-s", "--size", default="112x147") 
    parser.add_argument("-l", "--line_gap", default=0.05)
    parser.add_argument("-o", "--overflow", choices=["y", "n"], default="y")
    parser.add_argument("-a", "--all_options", choices=["y", "n"], default="n")
    parser.add_argument("-r", "--red",  default=RED_STANDARD)
    parser.add_argument("-g", "--green", default=GREEN_STANDARD)
    parser.add_argument("-b", "--blue", default=BLUE_STANDARD)
    parser.add_argument("-t", "--tones", default='3')
    parser.add_argument("-m", "--minimum_brightness", default=MINIMUM_BRIGHTNESS)
    parser.add_argument("-p", "--do_previewing", default="n")
    parser.add_argument("-c", "--color_brightener", choices=ALL_BRIGHTENERS, default=ALPHABETAGAMMA_BRIGHTENER)
    parser.add_argument("-q", "--percentile_saturation", default=PERCENTILE_SATURATION)
		
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
        bitmap_tone = np.reshape(t, (tone_dim, tone_dim)) * 255
        tones_dict[brightness] = bitmap_tone
    return(tones_dict)

def convertScale(img, alpha, beta):
    """Add bias and gain to an image with saturation arithmetics. Unlike
    cv2.convertScaleAbs, it does not take an absolute value, which would lead to
    nonsensical results (e.g., a pixel at 44 with alpha = 3 and beta = -210
    becomes 78 with OpenCV, when in fact it should become 0).
    """

    new_img = img * alpha + beta
    new_img[new_img < 0] = 0
    new_img[new_img > 255] = 255
    return new_img.astype(np.uint8)
            
def adjust_brightness_directly(gray_img, minimum_brightness):
    """Adjusts the brightness of a grayscale image with a direct calculation of bias
    and gain.
    """

    if 3 <= len(gray_img.shape):
        raise ValueError("Expected a grayscale image, color channels found")
    
    cols, rows = gray_img.shape
    brightness = np.sum(gray_img) / (255 * cols * rows)

    ratio = brightness / minimum_brightness
    if ratio >= 1:
        print("Image already bright enough")
        return img

    # Otherwise, adjust brightness to get the target brightness. Except for
    # saturation arithmetics, the new brightness should be the target brightness
    bright_img = convertScale(gray_img, alpha = 1 / ratio, beta = 0)

    print("Old: " + str(brightness))
    print("New: " + str(np.sum(bright_img) / (255 * cols * rows)))

    return bright_img

def percentile_to_bias_and_gain(gray_img, percentile):
    """Computes the bias and gain that corresponds to clipping the given percentile
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
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = - minimum_gray * alpha

    return alpha, beta
    

def adjust_brightness_with_histogram(gray_img, minimum_brightness, percentile_step = PERCENTILE_STEP):
    """Adjusts brightness with histogram clipping by trial and error.
    """

    if 3 <= len(gray_img.shape):
        raise ValueError("Expected a grayscale image, color channels found")

    new_img = gray_img
    percentile = percentile_step
    brightness_changed = False

    while True:
        cols, rows = new_img.shape
        brightness = np.sum(new_img) / (255 * cols * rows)

        if not brightness_changed:
            old_brightness = brightness

        if brightness >= minimum_brightness:
            break

        percentile += percentile_step
        alpha, beta = percentile_to_bias_and_gain(new_img, percentile)
        new_img = convertScale(gray_img, alpha = alpha, beta = beta)
        brightness_changed = True

    if brightness_changed:
        print("Old brightness: %3.3f, new brightness: %3.3f " %(old_brightness, brightness))
    else:
        print("Maintaining brightness at %3.3f" % old_brightness)
        
    return new_img

def saturate(img, percentile):
    """Changes the scale of the image so that half of percentile at the low range
    becomes 0, half of percentile at the top range becomes 255.
    """

    if 2 != len(img.shape):
        raise ValueError("Expected an image with only one channel")

    # copy values
    channel = img[:, :].copy()
    flat = channel.ravel()

    # copy values and sort them
    sorted_values = np.sort(flat)

    # find points to clip
    max_index = len(sorted_values) - 1
    half_percent = percentile / 200
    low_value = sorted_values[math.floor(max_index * half_percent)]
    high_value = sorted_values[math.ceil(max_index * (1 - half_percent))]

    # saturate
    channel[channel < low_value] = low_value
    channel[channel > high_value] = high_value

    # scale the channel
    channel_norm = channel.copy()
    cv2.normalize(channel, channel_norm, 0, 255, cv2.NORM_MINMAX)

    return channel_norm

def adjust_gamma(img, gamma):
    """Build a lookup table mapping the pixel values [0, 255] to
	their adjusted gamma values.
    """

    # code from
    # https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
    
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return cv2.LUT(img, table)


def adjust_brightness_with_gamma(gray_img, minimum_brightness, gamma_step = GAMMA_STEP):

    """Adjusts the brightness of an image by saturating the bottom and top
    percentiles, and changing the gamma until reaching the required brightness.
    """
    if 3 <= len(gray_img.shape):
        raise ValueError("Expected a grayscale image, color channels found")

    cols, rows = gray_img.shape
    changed = False
    old_brightness = np.sum(gray_img) / (255 * cols * rows)
    new_img = gray_img
    gamma = 1

    while True:
        brightness = np.sum(new_img) / (255 * cols * rows)
        if brightness >= minimum_brightness:
            break

        gamma += gamma_step
        new_img = adjust_gamma(gray_img, gamma = gamma)
        changed = True

    if changed:
        print("Old brightness: %3.3f, new brightness: %3.3f " %(old_brightness, brightness))
    else:
        print("Maintaining brightness at %3.3f" % old_brightness)
        
    return new_img

def adjust_brightness_alpha_beta_gamma(gray_img, minimum_brightness, percentile_step = PERCENTILE_STEP, gamma_step = GAMMA_STEP):
    """Adjusts brightness with histogram clipping by trial and error.
    """

    if 3 <= len(gray_img.shape):
        raise ValueError("Expected a grayscale image, color channels found")

    new_img = gray_img
    percentile = percentile_step
    gamma = 1
    brightness_changed = False

    while True:
        cols, rows = new_img.shape
        brightness = np.sum(new_img) / (255 * cols * rows)

        if not brightness_changed:
            old_brightness = brightness

        if brightness >= minimum_brightness:
            break
        
        # adjust alpha and beta
        percentile += percentile_step
        alpha, beta = percentile_to_bias_and_gain(new_img, percentile)
        new_img = convertScale(gray_img, alpha = alpha, beta = beta)
        brightness_changed = True

        # adjust gamma
        gamma += gamma_step
        new_img = adjust_gamma(new_img, gamma = gamma)

    if brightness_changed:
        print("Old brightness: %3.3f, new brightness: %3.3f " %(old_brightness, brightness))
    else:
        print("Maintaining brightness at %3.3f" % old_brightness)
        
    return new_img

def get_right_dimensions(img, width, height, overflow, line_gap, tone_dim):

    """Compute the right dimensions with these settings. This makes a difference in
    the case where the requested size is not divisible by the dimension of the
    half-tone, e.g. 85 mm with 0.05 scan gap is 1700 pixels, but the rescaling
    above will produce a 1701 pixels image for 3x3 half-tones.

    """

    width_target = width / line_gap + LASER_CUT_PRO_ALIAS_CORRECTION
    assert width_target.is_integer()
    height_target = height / line_gap + LASER_CUT_PRO_ALIAS_CORRECTION
    assert height_target.is_integer()

    width_target_before_tones = int(round(math.ceil(width_target / tone_dim)))
    height_target_before_tones = int(round(math.ceil(height_target / tone_dim)))

    img_height = img.shape[0]
    img_width = img.shape[1]
    
    # Find which is the biting dimension, then resize and pad with
    # centering. This image is centered but the half-toned image may not be
    # because 1 pixel difference between the paddings here means 3 pixels of
    # difference in the final image with 3x3 tones.
    width_correct = ((float(width_target) / img_width) <
                     (float(height_target) / img_height))

    # Flip it in case the image needs to overflow the size
    if "y" == overflow:
        width_correct = not width_correct
        
    if width_correct:
        width_lines = width_target_before_tones
        height_lines = int(math.ceil(width_lines * img_height / img_width))
    else:
        height_lines = height_target_before_tones
        width_lines = int(math.ceil(height_lines * img_width / img_height))

    return {WIDTH_TARGET: int(width_target),
            HEIGHT_TARGET: int(height_target),
            WIDTH_TARGET_BEFORE_HALFTONE: width_lines,
            HEIGHT_TARGET_BEFORE_HALFTONE: height_lines}


def rescale(img, width, height, overflow, line_gap, tone_dim):
    """Rescales image to have the right size for half-toning and engraving."""

    # Resize: note that width comes first, unlike the shape!
    resized = cv2.resize(img, (width, height), interpolation = cv2.INTER_CUBIC)

    # If the image needs does not overflow, return now
    if "y" == overflow:
        return resized

    # TODO: refactor this

    width_target = width / line_gap + LASER_CUT_PRO_ALIAS_CORRECTION
    assert width_target.is_integer()
    height_target = height / line_gap + LASER_CUT_PRO_ALIAS_CORRECTION
    assert height_target.is_integer()

    width_target_before_tones = int(round(math.ceil(width_target / tone_dim)))
    height_target_before_tones = int(round(math.ceil(height_target / tone_dim)))

    img_height = img.shape[0]
    img_width = img.shape[1]
    
    # Find which is the biting dimension, then resize and pad with
    # centering. This image is centered but the half-toned image may not be
    # because 1 pixel difference between the paddings here means 3 pixels of
    # difference in the final image with 3x3 tones.
    width_correct = ((float(width_target) / img_width) <
                     (float(height_target) / img_height))

    
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
    m = np.array(coefficients).reshape((1,3)) / (blue_weight + green_weight + red_weight)

    gray = cv2.transform(img, m)
    return gray

def halftone(gray, tones_dict, tone_num, tone_dim):
    """Generate a new image where each pixel is replaced by one of the previously calculated tones. For 3x3 half-toning, tone_dim is 3 and tone_num is 10.
    """

    num_rows = gray.shape[0]
    num_cols = gray.shape[1]

    output = np.zeros((num_rows * tone_dim, num_cols * tone_dim),
                         dtype = np.uint8)

    # Go through each pixel
    for i in range(num_rows):
        i_output = range(i * tone_dim, (i + 1)* tone_dim)

        if 0 == i % 10:
            print("At row %d of %d" % (i, num_rows))

        for j in range(num_cols):
            j_output = range(j * tone_dim, (j + 1)* tone_dim)
            
            pixel = gray[i, j]
            brightness = int(round((tone_num - 1) * pixel / 255))

            output[np.ix_(i_output, j_output)] = tones_dict[brightness]
            
    return output

def halftone_alexander(gray, tones_dict):
    """Generate a new image where each pixel is replaced by one with the values in tones_dict.
    """

    num_tones = len(tones_dict)
    tone_width = int(math.sqrt(num_tones - 1))

    mapped_brightness = np.array(
        [list(map(tones_dict.__getitem__, row)) 
         for row in (gray * ((num_tones - 1) / 255)).round()],
        dtype=np.uint8)

    output = np.array(
        [np.concatenate(row, axis=1) for row in mapped_brightness]
    ).reshape(*(n * tone_width for n in gray.shape))

    return output

def process_image(filepath, width, height, line_gap, red_weight,
                  green_weight, blue_weight, tones_str, overflow, minimum_brightness,
                  brightener, percentile_saturation,
                  suffix = "", do_halftoning = True, do_previewing = True):

    """Wrapper function that calls the others."""

    new_filepath = os.path.splitext(filepath)[0] + suffix + ".png"
    if new_filepath == filepath:
        new_filepath = os.path.splitext(filepath)[0] + "-halftoned" + ".png"

    img = cv2.imread(filepath)

    # Flip if the image is portrait and the dimensions are landscape
    rows, cols = img.shape[:2]
    if (cols > rows and width < height) or (cols < rows and width > height):
        width_ok = height
        height_ok = width
    else:
        width_ok = width
        height_ok = height

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
    
    # Get the riht dimensions
    dimensions = get_right_dimensions(img, width = width_ok, height = height_ok,
                                      overflow = overflow, line_gap = line_gap,
                                      tone_dim = tone_dim)

    # Rescale
    img_scaled = rescale(img, width = dimensions[WIDTH_TARGET_BEFORE_HALFTONE],
                         height = dimensions[HEIGHT_TARGET_BEFORE_HALFTONE],
                         overflow = overflow, line_gap = line_gap, tone_dim = tone_dim)

    # Convert to gray
    gray = convert_to_grayscale(img_scaled, red_weight = red_weight,
                                green_weight = green_weight,
                                blue_weight = blue_weight)

    # Saturate
    if percentile_saturation > 0:
        gray = saturate(gray, percentile_saturation)

    # Brightening method
    if DIRECT_BRIGHTENER == brightener:
        brightening_fn = adjust_brightness_directly
    elif ALPHABETA_BRIGHTENER == brightener:
        brightener_fn = adjust_brightness_with_histogram
    elif GAMMA_BRIGHTENER == brightener:
        brightener_fn = adjust_brightness_with_gamma
    elif ALPHABETAGAMMA_BRIGHTENER == brightener:
        brightener_fn = adjust_brightness_alpha_beta_gamma
    else:
        assert False, "Unknown brightener: " + brightener
    bright = brightener_fn(gray, minimum_brightness = minimum_brightness)

    if not do_halftoning:
        cv2.imwrite(new_filepath, bright)
        return
    
    # Half-tone
    print("Starting half-toning")
    tones_dict = process_tones(tones, tone_num = tone_num, tone_dim = tone_dim)
    #    png = halftone_alexander(bright, tones_dict = tones_dict,
    #               tone_num = tone_num, tone_dim = tone_dim)
    png = halftone_alexander(bright, tones_dict = tones_dict)

    # Crop the image to the exact size for the ouptut, losing maybe one or two
    # thirds of the rightmost or bottom-most pixel of the original image in case
    # of 3x3 tones; and also the amount of overflow
    rows, cols = png.shape
    col_start = max(0, int((cols - dimensions[WIDTH_TARGET])/2))
    col_end = col_start + dimensions[WIDTH_TARGET]
    row_start = max(0, int((rows - dimensions[HEIGHT_TARGET])/2))
    row_end = row_start + dimensions[HEIGHT_TARGET]
    png_cropped = png[row_start:row_end, col_start:col_end]

    # Save
    cv2.imwrite(new_filepath, png_cropped, [cv2.IMWRITE_PNG_BILEVEL, 1])
    print("Saved %s" % new_filepath)

    if not do_previewing:
        return

    # Also save a preview of what the result will be? Not for now, because it takes long
    preview_fp = preview.process_image(new_filepath)
    print("Saved preview at %s" % preview_fp)

def process_all_options(filepath, width, height, line_gap, overflow, minimum_brightness,
                        percentile_saturation, do_previewing):


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
                          minimum_brightness = minimum_brightness,
                          do_previewing = do_previewing)

def debug(filepath):
    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    i1 = cv2.convertScaleAbs(gray, alpha = 1.3076, beta = -77.153)
    i11 = convertScale(gray, alpha = 1.3076, beta = -77.153)
    i2 = cv2.convertScaleAbs(gray, alpha = 3, beta = -210)
    print(gray)
    print(i2)
    i21 = convertScale(gray, alpha = 3, beta = -210)
    print(i21)
    cv2.imwrite(os.path.join(DEBUG_DIR, "i11.png"), i11)
    cv2.imwrite(os.path.join(DEBUG_DIR, "i21.png"), i21)
    
    
def main():
    args = get_args()

    filepath = args.file
    size = args.size
    line_gap = float(args.line_gap)
    overflow = args.overflow
    all_options = args.all_options
    minimum_brightness = float(args.minimum_brightness)
    percentile_saturation = float(args.percentile_saturation)
    brightener = args.color_brightener
    do_previewing = args.do_previewing == "y"

    assert os.path.exists(filepath), "Input file not found!"

    delimiter_index = size.index("x")
    width = float(size[:delimiter_index])
    height = float(size[delimiter_index+1:])

    if "y" == all_options:
        process_all_options(filepath, width = width, height = height,
                            line_gap = line_gap, overflow = overflow,
                            minimum_brightness = minimum_brightness,
                            percentile_saturation = percentile_saturation,
                            do_previewing = do_previewing)
    else:
        red_weight = int(args.red)
        green_weight = int(args.green)
        blue_weight = int(args.blue)
        tones_str = args.tones

        process_image(filepath, width = width, height = height,
                      line_gap = line_gap, overflow = overflow,
                      red_weight = red_weight, green_weight = green_weight,
                      blue_weight = blue_weight, tones_str = tones_str,
                      minimum_brightness = minimum_brightness,
                      percentile_saturation = percentile_saturation,
                      brightener = brightener,
                      do_previewing = do_previewing)
                      

if "__main__" == __name__:
    main()
