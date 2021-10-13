# npy2bdv
 A library for writing HDF5/XML and N5/XML datasets of 
 Fiji BigDataViewer/BigStitcher format as numpy arrays. 
 
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
 
 ## Documentation

 ## Supported HDF5/XML and N5/XML writing options
 * compression methods `None`, `gzip`, `lzf` (`None` by default).
 * downsampling options: 
    - any number of mipmap levels
    - computed via averaging or decimating, compatible with BigDataViewer/BigStitcher convention.
 * user-defined block sizes for H5 and N5 storage (default `4,256,256`)
 * any number of time points, illuminations, channels, tiles, angles.
 * arbitrary affine transformation for each individual view (e.g. translation, rotation, shear).
 * arbitrary voxel calibration for each view, to account for spatial anisotropy.
 * individual views can differ in dimensions, voxel size, voxel units, exposure time, and exposure units.
 * missing views are labeled in XML automatically.
 * appending data to stacks of arbitrary size, by plane or sub-stack. Handy when your stack is larger than your RAM or in separate files (e.g., MM TIFF files).
    - virtual stacks can be written with multiple subsampling levels and compression.
 
 ## Writing speed
Writing speeds up to 2300 MB/s can be achieved on a PC with SSD drive. 
The speed of writing for long time series (>100 stacks) is typically about 700-900 MB/s. 
This is in the range of full-speed camera acquisition 
of Hamamatsu Orca Flash4, e.g. 840 MB/s (2048x2048 px at 100 Hz).

 ## Acknowledgements
 This code was inspired by [Talley Lambert's](https://github.com/tlambert03/imarispy) code 
 with further modification from Adam Glaser, [VolkerH](https://github.com/VolkerH), Doug Shepherd and 
 [Peter H](https://github.com/abred).
 
 ## Citation
 If you find this library useful, please cite it. Thanks!
 
 [![DOI](https://zenodo.org/badge/203410946.svg)](https://zenodo.org/badge/latestdoi/203410946)
