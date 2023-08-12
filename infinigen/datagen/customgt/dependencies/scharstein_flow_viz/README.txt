Some utilities for reading, writing, and color-coding .flo images

Daniel Scharstein, 7/2/07
updated 2/9/08 to fix bug in color_flow.cpp
updated 6/9/09 to make robust to NaN or constant 0 flow (thanks Jan Bouecke)

See flowIO.cpp for sample code for reading and writing .flo files.
Here's an excerpt from this file describing the flow file format:

// ".flo" file format used for optical flow evaluation
//
// Stores 2-band float image for horizontal (u) and vertical (v) flow components.
// Floats are stored in little-endian order.
// A flow value is considered "unknown" if either |u| or |v| is greater than 1e9.
//
//  bytes  contents
//
//  0-3     tag: "PIEH" in ASCII, which in little endian happens to be the float 202021.25
//          (just a sanity check that floats are represented correctly)
//  4-7     width as an integer
//  8-11    height as an integer
//  12-end  data (width*height*2*4 bytes total)
//          the float values for u and v, interleaved, in row order, i.e.,
//          u[row0,col0], v[row0,col0], u[row0,col1], v[row0,col1], ...
//


Once you have a .flo file, you can create a color coding of it using
color_flow

Use colortest to visualize the encoding


To compile

cd imageLib
make
cd ..
make
./colortest 10 colors.png
