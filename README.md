# Halftone

I have been playing with the laser cutter to engrave images for a while. This
repo contains the code for the best results on my machine, which started as a
thread on
[StackExchange](https://graphicdesign.stackexchange.com/questions/124529/filter-to-preview-result-of-laser-engraving-of-photo). I
used [this method of half-toning](https://github.com/timfeirg/Basic-image-manipulation-in-OpenCV-under-C--).

## Installation on macOS

At the command-line, install [Open Computer Vision (OpenCV)](https://docs.opencv.org/):

```bash
xcode-select --install  # command-line tools
brew tap homebrew/science
brew install opencv
```

Clone this repo and `cd` into it:

```bash
git clone git@github.com:miguelmorin/halftone.git
cd halftone
```

## Help

Run this command at the command-line to get the list of all the options:

```bash
$ python3 halftone.py -h
usage: halftone.py [-h] -f FILE [-s SIZE] [-l LINE_GAP] [-o {y,n}] [-a {y,n}]
                   [-r RED] [-g GREEN] [-b BLUE] [-t {1,2,3}]

Convert images to laser-ready bitmaps using half-toning

optional arguments:
  -h, --help            show this help message and exit
  -f FILE, --file FILE
  -s SIZE, --size SIZE
  -l LINE_GAP, --line_gap LINE_GAP
  -o {y,n}, --overflow {y,n}
  -a {y,n}, --all_options {y,n}
  -r RED, --red RED
  -g GREEN, --green GREEN
  -b BLUE, --blue BLUE
  -t {1,2,3}, --tones {1,2,3}
```

## Usage

Start wih an image such as this one (it may take some time to load):

![Original
poster](https://github.com/miguelmorin/halftone/blob/master/images/Attack%20of%20the%2050%20ft%20improvisers.jpg){ height=200px }

For this image, the best results occured with the red channel, 3x3 tones, and a
scan gap of 0.05. Reproduce the results with:

```bash
python3 halftone.py -l 0.05 -s 75x105 -f 'images/Attack of the 50 ft improvisers.jpg' -g 0 -b 0 -t 3 -o n
```

See below for an explanation of these options. The result is:

![Bitmap of
poster](https://github.com/miguelmorin/halftone/blob/master/images/Attack%20of%20the%2050%20ft%20improvisers.bmp){ height=200px }


For best results on LaserCut Pro 5, you should give it a 1-bit depth bitmap that
does not require additional processing, simply telling the laser where to turn
on and off. Otherwise, you will need to process it with `Tools > Half Bmitmap`
for example. I have been unable to save bitmaps with depth of 1 bit in OpenCV,
so I open it in Photoshop, change the image mode to Grayscale, then to Bitmap
with 50% threshold, and I save as a bitmap with depth of 1-bit. I also need to
rescale it to the desired size because the default resolution of 72 ppi changes
the size of the image and the line gap of 0.05mm no longer corresponds to one
pixel.

I used this image at the Cambridge Makespace on pine wood with the following
nice result:

![Engraving on pine wood](https://github.com/miguelmorin/halftone/blob/master/images/result.JPG)


## Options

The code contains three types of tones:

- 10 tones of size 3x3 (3 pixels x 3 pixels)
- 5 tones of size 2x2 (2 pixels x 2 pixels)
- 2 tones of size 1x2 (1 pixel x 1 pixel)

The last one is the same as 50% thresholding. You can choose the tone with flag
`-t` or `--tones` and argument 1, 2, or 3.

You can specify a line gap for the laser cutter, which is the distance between
two passes of the laser with flag `-l` or `-line_gap`. The default is 0.05mm.

You can choose the size of the final engraving with flag `-s` or `--size` in
millimeters. Together with the line gap, these determine the number of pixels in
the final image. With a line gap of 0.05, the number of pixels is 20 times the
size in millimeters.

You can choose whether the final image should overflow this size (the default)
or whether it should fit strictly into it (with `-o n` or `--overflow n`).

You can choose different weights for each color channel. The default weights are
the standard in grayscale conversion: `0.299` for red, `0.587` for green, and
`0.114` for blue. If you want only the red channel, use `-g 0 -b 0` or `--green
0 --blue 0`.


## All options

Each image is different. To avoid trial and error, I often do the Cartesian
product of all possible options and choose the one that has better contrast on
the screen, then engrave that one. To produce all options, run:

```bash
python3 halftone.py -l 0.05 -s 75x105 -f 'images/Attack of the 50 ft improvisers.jpg' --all y
```

## Speed

The code is slow at the moment because of assigning a tone to each pixel. I may
improve it in the future.

## Issues and bug reports

Please use the Issues on the Github page to file issues with problems or suggestions.

## Commissions

Please file an issue if you would like a commission of laser-engraved pictures
on wood.
