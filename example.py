import math
import time

import numpy as np

TONES = [[0, 0,
          0, 0],
         [0, 1,
          0, 0],
         [1, 1,
          0, 0],
         [1, 1,
          0, 1],
         [1, 1,
          1, 1]]

def process_tones():
    """Converts the tones above to the right shape."""
    tones_dict = dict()

    for t in TONES:
        brightness = sum(t)
        bitmap_tone = np.reshape(t, (2, 2)) * 255
        tones_dict[brightness] = bitmap_tone
    return(tones_dict)

def halftone(gray, tones_dict):
    """Generate a new image where each pixel is replaced by one with the values in tones_dict.
    """

    num_rows = gray.shape[0]
    num_cols = gray.shape[1]
    num_tones = len(tones_dict)
    tone_width = int(math.sqrt(num_tones - 1))

    output = np.zeros((num_rows * tone_width, num_cols * tone_width),
                         dtype = np.uint8)

    # Go through each pixel
    for i in range(num_rows):
        i_output = range(i * tone_width, (i + 1)* tone_width)

        for j in range(num_cols):
            j_output = range(j * tone_width, (j + 1)* tone_width)
            
            pixel = gray[i, j]
            brightness = int(round((num_tones - 1) * pixel / 255))

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

def generate_gray_image(width = 100, height = 100):
    """Generates a random grayscale image.
    """

    return (np.random.rand(width, height) * 256).astype(np.uint8)

gray = generate_gray_image()
tones_dict = process_tones()

start = time.time()
for i in range(10):
    binary = halftone_alexander(gray, tones_dict = tones_dict)
duration = time.time() - start
print("Average loop time: " + str(duration))
