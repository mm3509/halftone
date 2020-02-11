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

Start wih an image such as this one, reprinted with permission from the author (it may take some time to load):

<img src="https://github.com/miguelmorin/halftone/blob/master/images/Attack%20of%20the%2050%20ft%20improvisers.jpg" width="250" alt="Original poster">

For this image, the best results occured with the red channel, 3x3 tones, and a
scan gap of 0.05. Reproduce the results with:

```bash
python3 halftone.py -l 0.05 -s 75x105 -f 'images/Attack of the 50 ft improvisers.jpg' -g 0 -b 0 -t 3 -o n
```

See below for an explanation of these options. The result is here (it may take
some time to load):

<img alt="Bitmap of poster" src="https://github.com/miguelmorin/halftone/blob/master/images/Attack%20of%20the%2050%20ft%20improvisers.png" width=250>

## Cutting

For best results on LaserCut Pro 5, you should give it a 1-bit depth bitmap that
does not require additional processing, simply telling the laser where to turn
on and off. Otherwise, you will need to process it with `Tools > Half Bmitmap`
for example. The code saves files as 1-bit PNG files. You should also rescale
the image to the desired size because the default resolution of 72 ppi changes
the size of the image and the line gap of 0.05mm no longer corresponds to one
pixel.

If you need to cut around the image too, create the outline with the desired size,
and centre this outline and the raster image by clicking the icon that looks
like "centre-justify this piece of text" on both of them. This will align both
objects to the centre of the board and they will be aligned.

I used this image at the Cambridge Makespace on pine wood with the nice result
that follows:

![Engraving on pine wood](https://github.com/miguelmorin/halftone/blob/master/images/result.JPG)

## Material

The results depend on the material. When the laser engraves plywood, it sets it
on a tiny fire that spreads locally. So the same image engraved in plywood will
look darker than engraved in acrylic.

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

# problems and adjustments

This section uses the following picture of a bunny to demonstrate results:

<img
src="https://github.com/miguelmorin/halftone/blob/master/images/bunny.JPG" alt="Sample photo of a bunny">

## Laser Cut Pro skipping lines

LaserCut Pro some

Ideally each binary pixel in the the PNG file controls whether the laser turns
on or off. For example, when the result is 400 pixels by 400 pixels, resized to
20 mm x 20 mm in LaserCut Pro, and the scan gap is 0.05, the each pixels
corresponds to a square of 0.05 x 0.05, so a black pixel should turn the laser
on and a white pixel should turn the laser off.

But LaserCut Pro skips lines when this alignment is exact as in this photo with
scan gaps 0.2 and 0.1:

<img src="https://github.com/miguelmorin/halftone/blob/master/images/Laser%20Cut%20Pro%20skipping%20lines.JPG" width="250" alt="Laser cut pro skipping lines">

My solution at the moment consists of adding one extra pixel so each scan gap is
in between two lines of pixels in the original.

## dependence on material

The results heavily depend on the material. Different woods, different types of
plywood, and different batches of the same plywood. Even different sides of the
same plywood stock will give different results, as shown in these pictures with
the same halftoned image where the front of the material is notably darker:

<img
src="https://github.com/miguelmorin/halftone/blob/master/images/bunny_front.JPG" alt="bunny engraving at the front">

<img
src="https://github.com/miguelmorin/halftone/blob/master/images/bunny_back.JPG" alt="bunny engraving at the back">

In fact, this project started because of the dependence on material: I started
with a dither filter from ImageMagick, tried with one image, and obtained such
great results that I was hooked. Then with another image and a different type of
wood, the results were poor. So I decided to use a method that would work well
on any wood.
