import npy2bdv
import os
import time
import shutil
import unittest
import numpy as np
import tifffile
from tifffile import TiffFile

filepath = '/ispim2_data/H17_B4_S3_16x_0.5xPBS/ex1'

pixel_spacing = 0.406008 # um, pixel size on camera
frame_spacing = 1.4126000 # um, spacing between frames
tile_spacing = 700.0 # um, offset between image strips
theta = 45 # deg., light sheet angle

nchannels = 1 # number of channels
nblocks = 27 # number of TIFF files per strip
nilluminations = 1 # number of illumination directions
ntiles = 16 # number of tiles
nangles = 1 # number of collection angles

downsample_raw = (1,4,4) # int, downsample raw data factors xyz
nthreads = 8 # number of threads to use for N5 I/O

subsamp = ((1, 1, 1), (2, 2, 2), (4, 4, 4),) # pyramind subsampling factors xyz
blockdim = ((64, 64, 64), (64, 64, 64), (64, 64, 64),) # chunksize xyz

sx = pixel_spacing*downsample_raw[2] # effective voxel size in x direction
sy = pixel_spacing*np.cos(theta*np.pi/180.0)*downsample_raw[1] # effective voxel size in y direction
sz = frame_spacing*downsample_raw[0] # effective voxel size in z direction (scan direction)

scale_x = sx/sy # normalized scaling in x
scale_y = sy/sy # normalized scaling in y
scale_z = sz/sy # normalized scaning in z

shear = -np.tan(theta*np.pi/180.0)*sy/sz # shearing based on theta and y/z pixel sizes

shiftx = scale_x*(tile_spacing/sx) # shift tile in x, unit pixels
shifty = 0

# HARDCODED - GET STACK DIMS FROM FIRST FILE
iminfo = TiffFile(filepath + '/ex1_MMStack_Pos8.ome.tif')
stack_dims = (len(iminfo.pages), iminfo.pages[0].shape[0], iminfo.pages[0].shape[1])
print(stack_dims)

start = time.time()

bdv_writer = npy2bdv.BdvWriter('test',
                                nchannels = nchannels,
                                nilluminations = nilluminations,
                                ntiles = ntiles,
                                nangles = nangles,
                                subsamp_zyx = subsamp,
                                blockdim = blockdim,
                                compression = None,
                                format = 'n5',
                                overwrite = True)

for i_ch in range(nchannels):
    for i_illum in range(nilluminations):
        for i_tile in range(ntiles):
            for i_angle in range(nangles):

                bdv_writer.append_view(stack_dim = (nblocks*int(stack_dims[0]/downsample_raw[0]),int(stack_dims[1]/downsample_raw[1]),int(stack_dims[2]/downsample_raw[2])),
                                        channel = i_ch,
                                        illumination = i_illum,
                                        tile = i_tile,
                                        angle = i_angle,
                                        voxel_size_xyz = (sx, sy, sz),
                                        voxel_units = 'um',
                                        n_threads = nthreads)

                for i_block in range(0,nblocks):

                    print(i_block)

                    if i_block == 0:
                        iminfo = TiffFile(filepath + '/ex1_MMStack_Pos' + str(i_tile+7) + '.ome.tif')
                        num_imgs = len(iminfo.pages)
                        imgs = tifffile.imread(filepath + '/ex1_MMStack_Pos' + str(i_tile+7) + '.ome.tif', key = range(0, num_imgs))
                    else:
                        iminfo = TiffFile(filepath + '/ex1_MMStack_Pos' + str(i_tile+7) + '_' + str(i_block) + '.ome.tif')
                        num_imgs = len(iminfo.pages)
                        imgs = tifffile.imread(filepath + '/ex1_MMStack_Pos' + str(i_tile+7) + '_' + str(i_block) + '.ome.tif', key = range(0, num_imgs))                       
                    
                    imgs = imgs[::downsample_raw[0],::downsample_raw[1],::downsample_raw[2]]
                    imgs = np.transpose(imgs,(0,2,1))

                    bdv_writer.append_planes(planes = imgs,
                                            channel = i_ch,
                                            illumination = i_illum,
                                            tile = i_tile,
                                            angle = i_angle,
                                            pyramid_method = 'decimate')

bdv_writer.write_xml()
bdv_writer.write_settings()

for i_ch in range(nchannels):
    for i_illum in range(nilluminations):
        for i_tile in range(ntiles):
            for i_angle in range(0,nangles):

                m_deskew = np.array(([1.0, 0.0, 0.0, 0.0],
                                     [0.0, 1.0, 0.0, 0.0],
                                     [0.0, shear, 1.0, 0.0]))

                m_scale = np.array(([scale_x, 0.0, 0.0, 0.0],
                                    [0.0, scale_y, 0.0, 0.0],
                                    [0.0, 0.0, scale_z, 0.0]))

                m_overlap = np.array(([1.0, 0.0, 0.0, i_tile*shiftx],
                                      [0.0, 1.0, 0.0, shifty],
                                      [0.0, 0.0, 1.0, 0.0]))

                bdv_writer.append_affine(m_affine = m_deskew,
                                        name_affine = 'deskew',
                                        channel=i_ch,
                                        illumination=i_illum,
                                        tile=i_tile,
                                        angle=i_angle)

                bdv_writer.append_affine(m_affine = m_scale,
                                        name_affine = 'scale',
                                        channel=i_ch,
                                        illumination=i_illum,
                                        tile=i_tile,
                                        angle=i_angle)

                bdv_writer.append_affine(m_affine = m_overlap,
                                        name_affine = 'overlap',
                                        channel=i_ch,
                                        illumination=i_illum,
                                        tile=i_tile,
                                        angle=i_angle)

bdv_writer.close()

end = time.time()

print(end-start)
