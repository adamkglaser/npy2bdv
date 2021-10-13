# Fast writing of numpy arrays to HDF5 and N5 format compatible with Fiji/BigDataViewer and BigStitcher
# Authors: Nikita Vladimirov and Adam Glaser
# License: GPL-3.0

import os
import h5py
import numpy as np
from xml.etree import ElementTree as ET
import skimage.transform
import shutil
import z5py
# import zarr

class BdvWriter():

    def __init__(self, filename, format,
                 subsamp_zyx=((1, 1, 1),),
                 blockdim=((4, 256, 256),),
                 compression=None,
                 nilluminations=1, nchannels=1, ntiles=1, nangles=1,
                 overwrite=False):

        """Class for writing multiple numpy 3d-arrays into BigDataViewer/BigStitcher compatible dataset.
        Currently implemented formats: H5, N5.

        Parameters:
        -----------
            filename: string
                Name of the N5 or H5 file to create (full path).
            format: str
                Data format to use: 'h5' (default), 'n5'.
            subsamp_zyx: tuple of tuples
                Subsampling levels in (z,y,x) order. Integers >= 1, default value ((1, 1, 1),) for no subsampling.
            blockdim: tuple of tuples
                Block size for h5 storage, in pixels, in (z,y,x) order. Default ((4,256,256),), see notes.
            compression: None or str
                H5 compression options: (None, 'gzip', 'lzf'), Default is None for high-speed writing.
                N5 compression: (None, 'gzip', 'xz'), default is None (raw).
            nilluminations: int
            nchannels: int
            ntiles: int
            nangles: int
                Number of view attributes, >=1.
            overwrite: boolean
                If True, overwrite existing file. Default False.

        .. note::
        ------
        Input stacks and output files are assumed uint16 type.

        The recommended block (chunk) size should be between 10 KB and 1 MB, larger for large arrays.
        For example, block dimensions (4,256,256)px gives ~0.5MB block size for type int16 (2 bytes) and writes very fast.
        Block size can be larger than stack dimension.
        """
        assert nilluminations >= 1, "Total number of illuminations must be at least 1."
        assert nchannels >= 1, "Total number of channels must be at least 1."
        assert ntiles >= 1, "Total number of tiles must be at least 1."
        assert nangles >= 1, "Total number of angles must be at least 1."
        assert '.' not in filename, "Filename does not include extension" 
        assert all([isinstance(element, int) for tupl in subsamp_zyx for element in
                    tupl]), 'subsamp values should be integers >= 1.'
        if len(blockdim) < len(subsamp_zyx):
            print(f"INFO: blockdim levels ({len(blockdim)}) < subsamp levels ({len(subsamp_zyx)}):"
                  f" First-level block size {blockdim[0]} will be used for all levels")
        self._fmt = 't{:05d}/s{:02d}/{}'
        self.nsetups = nilluminations * nchannels * ntiles * nangles
        self.nilluminations = nilluminations
        self.nchannels = nchannels
        self.ntiles = ntiles
        self.nangles = nangles
        self.subsamp_zyx = np.asarray(subsamp_zyx)
        self.nlevels = len(subsamp_zyx)
        self.chunks = self._compute_chunk_size(blockdim)
        self.max_intensity = 65535.0
        self.stack_shapes = {}
        self.affine_matrices = {}
        self.affine_names = {}
        self.calibrations = {}
        self.voxel_size_xyz = {}
        self.voxel_units = {}
        self.exposure_time = {}
        self.exposure_units = {}
        self.filename = filename
        self.filename_xml = filename + '.xml'
        self.file_format = format
        self.compression = None
        self._root = None

        if os.path.exists(self.filename + '.' + self.file_format):
            if overwrite:
                if self.file_format == 'h5':
                    os.remove(self.filename + '.' + self.file_format)
                    print("Warning: H5 file already exists, overwriting.")
                elif self.file_format == 'n5':
                    shutil.rmtree(self.filename + '.' + self.file_format)
                    print("Warning: N5 dataset already exists, overwriting.")
            else:
                raise FileExistsError(f"File {self.filename + '.' + self.file_format} already exists.")
        if self.file_format == 'h5':
            self.file_object = h5py.File(self.filename + '.' + self.file_format, 'a')
            assert compression in (None, 'gzip', 'lzf'), f'H5 compression unknown: {compression}'
            self.compression = compression
            self._write_H5_headers()
        elif self.file_format == 'n5':
            self.file_object = z5py.File(self.filename + '.' + self.file_format, 'a')
            # self.file_object = zarr.N5Store(filename[:-2] + 'n5')
            # self.file_object = zarr.DirectoryStore(filename[:-2] + 'n5')
            assert compression in (None, 'gzip', 'xz'), \
                f'N5 compression unknown: {compression}'
            self.compression = 'raw' if compression is None else compression
        else:
            raise ValueError("File format unknown")
        self.virtual_stacks = False
        self.setup_id_present = [[False] * self.nsetups]

    def append_planes(self, planes, pyramid_method = 'decimate', time=0, illumination=0, channel=0, tile=0, angle=0):
        """Append planes to a virtual stack.

        Parameters:
        -----------
            planes: array_like
                A 3d numpy array of (z,y,x) pixel values.       
            pyramid_method: string
                Downsampling method (mean or decimate)
            time: int
                Time index of the view, >=0.
            illumination: int
            channel: int
            tile: int
            angle: int
                Indices of the view attributes, >=0. 
        """

        isetup = self._determine_setup_id(illumination, channel, tile, angle)
        self._update_setup_id_present(isetup, time)
        assert pyramid_method == 'mean' or 'decimate', "Pyramid method must be mean or decimate"

        self.max_intensity = (self.max_intensity + np.max(planes[:])/4.0)/2.0

        for ilevel in range(self.nlevels):
            substack = self._subsample_stack(planes, self.subsamp_zyx[ilevel], pyramid_method).astype('int16')
            if self.file_format == 'h5':
                group_name = self._fmt.format(time, isetup, ilevel)
                dataset = self.file_object[group_name]["cells"]
            elif self.file_format == 'n5':
                dataset = self.file_object[f"setup{isetup}/timepoint{time}/s{ilevel}"]       
            else:
                raise ValueError("File format unknown")

            dataset[self.start_index[ilevel]:self.start_index[ilevel] + substack.shape[0]] = substack 

            self.start_index[ilevel] = self.start_index[ilevel] + substack.shape[0]

    def append_view(self, stack_dim=(0, 0, 0),
                    time=0, illumination=0, channel=0, tile=0, angle=0,
                    m_affine=None, name_affine='manually defined',
                    voxel_size_xyz=(1, 1, 1), voxel_units='px', calibration=(1, 1, 1),
                    exposure_time=0, exposure_units='s', n_threads = 1):
        """
        Write 3-dimensional numpy array (stack) to the dataset (H5 or N5) with specified attributes.

        Parameters:
        -----------
            stack_dim: Tuple of (z,y,x) dimensions.
                Dimensions to allocate a stack and fill it later.
            time: int
                Time index, >=0.
            illumination: int
            channel: int
            tile: int
            angle: int
                Indices of the view attributes, >= 0.
            m_affine: a numpy array of shape (3,4), optional.
                Coefficients of affine transformation matrix (m00, m01, ...)
            name_affine: str, optional
                Name of the affine transformation.
            voxel_size_xyz: tuple of size 3, optional
                The physical size of voxel, in voxel_units. Default (1, 1, 1).
            voxel_units: str, optional
                Spatial units, default is 'px'.
            calibration: tuple of size 3, optional
                The anisotropy factors for (x,y,z) voxel calibration. Default (1, 1, 1).
                Leave it default unless you know how it affects transformations.
            exposure_time: float, optional
                Camera exposure time for this view, default 0.
            exposure_units: str, optional
                Time units for this view, default "s".
            n_threads: int, optional
                N5 only, number of threads for parallel I/O, default 1.
        """
        isetup = self._determine_setup_id(illumination, channel, tile, angle)
        self._update_setup_id_present(isetup, time)
        assert len(calibration) == 3, "Calibration must be a tuple of 3 elements (x, y, z)."
        assert len(voxel_size_xyz) == 3, "Voxel size must be a tuple of 3 elements (x, y, z)."
        if stack_dim is not None:
            assert np.array(stack_dim).dtype == 'int32', "Stack dims should be integers"
            assert len(stack_dim) == 3, "Stack should be a 3-dimensional tuple (z,y,x)"
            self.stack_shapes[isetup] = stack_dim

        if m_affine is not None:
            self.affine_matrices[isetup] = m_affine.copy()
            self.affine_names[isetup] = name_affine
        self.calibrations[isetup] = calibration
        self.voxel_size_xyz[isetup] = voxel_size_xyz
        self.voxel_units[isetup] = voxel_units
        self.exposure_time[isetup] = exposure_time
        self.exposure_units[isetup] = exposure_units
        self.start_index = np.zeros(self.nlevels, dtype = int)

        if self.file_format == 'h5':
            self._initialize_h5_dataset(stack_dim, isetup, time)
        else:
            self._initialize_n5_dataset(stack_dim, isetup, time, n_threads)

    def _initialize_h5_dataset(self, stack_dim, isetup, time, dtype='int16'):
        """Initialize the view (stack) as H5 dataset in BigDataViewer format.
        Note that type must be int16, rather than uint16, for correct reading by Fiji (a bug?)"""
        if stack_dim is not None:
            for ilevel in range(self.nlevels):
                group_name = self._fmt.format(time, isetup, ilevel)
                if group_name in self.file_object:
                    del self.file_object[group_name]
                grp = self.file_object.create_group(group_name)
                grp.create_dataset('cells', chunks=self.chunks[ilevel], shape = np.asarray(stack_dim) // self.subsamp_zyx[ilevel],
                                       maxshape=(None, None, None), compression=self.compression, dtype=dtype)

    def _initialize_n5_dataset(self, stack_dim, isetup, time, n_threads, dtype='uint16'):
        """Initialize the view (stack) as N5 dataset in BigDataViewer format using z5py"""
        grp = self.file_object.create_group(f"setup{isetup}/timepoint{time}")
        self.file_object[f"setup{isetup}"].attrs['downsamplingFactors'] = np.flip(self.subsamp_zyx, 1).tolist()
        self.file_object[f"setup{isetup}"].attrs['dataType'] = dtype
        self.file_object[f"setup{isetup}/timepoint{time}"].attrs["resolution"] = list(self.voxel_size_xyz[isetup])
        self.file_object[f"setup{isetup}/timepoint{time}"].attrs["saved_completely"] = True
        self.file_object[f"setup{isetup}/timepoint{time}"].attrs["multiScale"] = True
        if stack_dim is not None:
            for ilevel in range(self.nlevels):
                grp.create_dataset(name = f"s{ilevel}", chunks=self.chunks[ilevel], shape = np.asarray(stack_dim) // self.subsamp_zyx[ilevel],
                                   compression=self.compression, dtype=dtype, n_threads = n_threads)
                self.file_object[f"setup{isetup}/timepoint{time}/s{ilevel}"].attrs['downsamplingFactors'] = \
                    np.flip(self.subsamp_zyx[ilevel]).tolist()

        """Initialize the view (stack) as N5 dataset in BigDataViewer format using zarr"""
        # setup_grp = zarr.group(store = self.file_object, path = f"setup{isetup}")
        # time_grp = setup_grp.create_group(name = f"timepoint{time}")
        # setup_grp.attrs['downsamplingFactors'] = np.flip(self.subsamp_zyx, 1).tolist()
        # # setup_grp.attrs['dataType'] = "dtype"
        # time_grp.attrs["resolution"] = list(self.voxel_size_xyz[isetup])
        # time_grp.attrs["saved_completely"] = True
        # time_grp.attrs["multiScale"] = True
        # if stack_dim is not None:
        #     for ilevel in range(self.nlevels):
        #         dset = time_grp.create_dataset(f"s{ilevel}", chunks=self.chunks[ilevel], shape = np.asarray(stack_dim) // self.subsamp_zyx[ilevel],
        #                            compression=self.compression, dtype=dtype)
        #         dset.attrs['downsamplingFactors'] = \
        #             np.flip(self.subsamp_zyx[ilevel]).tolist()

    def write_settings(self):
        """
        Write BDV XML visualization settings file.

        """
        root = ET.Element('Settings')
        vs = ET.SubElement(root, 'ViewerState')
        sources = ET.SubElement(vs, 'Sources')
        for iillumination in range(self.nilluminations):
            for ichannel in range(self.nchannels):
                for itile in range(self.ntiles):
                    for iangle in range(self.nangles):
                        isetup = self._determine_setup_id(iillumination, ichannel, itile, iangle)
                        if any([self.setup_id_present[t][isetup] for t in range(len(self.setup_id_present))]):
                            source = ET.SubElement(sources, 'Source')
                            ET.SubElement(source, 'active').text = 'true'

        sg = ET.SubElement(vs, 'SourceGroups')
        for i in range(10):
            group = ET.SubElement(sg, 'SourceGroup')
            ET.SubElement(group, 'active').text = 'true'
            ET.SubElement(group, 'name').text = 'group ' + str(i+1)
            if i <= len(self.setup_id_present):
                ET.SubElement(group, 'id').text = str(i)

        display = ET.SubElement(vs, 'DisplayMode').text = 'fs'
        interpolation = ET.SubElement(vs, 'Interpolation').text = 'nearestneighbor'
        cs = ET.SubElement(vs, 'CurrentSource').text = '0'
        cs = ET.SubElement(vs, 'CurrentGroup').text = '0'
        ct = ET.SubElement(vs, 'CurrentTimePoint').text = '0'

        sa = ET.SubElement(root, 'SetupAssignments')
        converter = ET.SubElement(sa, 'ConverterSetups')

        for iillumination in range(self.nilluminations):
            for ichannel in range(self.nchannels):
                for itile in range(self.ntiles):
                    for iangle in range(self.nangles):
                        isetup = self._determine_setup_id(iillumination, ichannel, itile, iangle)
                        if any([self.setup_id_present[t][isetup] for t in range(len(self.setup_id_present))]):
                            setup = ET.SubElement(converter, 'ConverterSetup')
                            ET.SubElement(setup, 'id').text = str(isetup)
                            ET.SubElement(setup, 'min').text = "{:.1f}".format(90)
                            ET.SubElement(setup, 'max').text = "{:.1f}".format(self.max_intensity)
                            ET.SubElement(setup, 'color').text = '-1'
                            ET.SubElement(setup, 'groupId').text = '0'

        mmgroups = ET.SubElement(sa, 'MinMaxGroups')
        mmgroups = ET.SubElement(mmgroups, 'MinMaxGroup') 
        ET.SubElement(mmgroups, 'id').text = '0'
        ET.SubElement(mmgroups, 'fullRangeMin').text = '-2.147483648E9'
        ET.SubElement(mmgroups, 'fullRangeMax').text = '2.147483647E9'
        ET.SubElement(mmgroups, 'rangeMin').text = "{:.1f}".format(0)
        ET.SubElement(mmgroups, 'rangeMax').text = "{:.1f}".format(65535)
        ET.SubElement(mmgroups, 'currentMin').text = "{:.1f}".format(90)
        ET.SubElement(mmgroups, 'currentMax').text = "{:.1f}".format(self.max_intensity)

        m_affine = np.array(([1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]))
        mx_string = np.array2string(m_affine.flatten(), separator=' ',
                                    precision=1, floatmode='fixed',
                                    max_line_width=(5*12+10))

        mst = ET.SubElement(root, 'ManualSourceTransforms')
        for iillumination in range(self.nilluminations):
            for ichannel in range(self.nchannels):
                for itile in range(self.ntiles):
                    for iangle in range(self.nangles):
                        isetup = self._determine_setup_id(iillumination, ichannel, itile, iangle)
                        if any([self.setup_id_present[t][isetup] for t in range(len(self.setup_id_present))]):
                            st = ET.SubElement(mst, 'SourceTransform')
                            st.set('type', 'affine')
                            ET.SubElement(st, 'affine').text = mx_string[1:-1].strip()

        bookmark = ET.SubElement(root, 'Bookmarks')

        self._xml_indent(root)
        tree = ET.ElementTree(root)
        tree.write(self.filename + '.settings.xml', xml_declaration=True, encoding='utf-8', method="xml")

    def write_xml(self, ntimes=1):
        """
        Write BDV XML header file for the HDF5 or N5 file.

        Parameters:
        -----------
            ntimes: int
                Number of time points
        """
        assert ntimes >= 1, "Total number of time points must be at least 1."
        root = ET.Element('SpimData')
        root.set('version', '0.2')
        bp = ET.SubElement(root, 'BasePath')
        bp.set('type', 'relative')
        bp.text = '.'
        seqdesc = ET.SubElement(root, 'SequenceDescription')
        imgload = ET.SubElement(seqdesc, 'ImageLoader')
        if self.file_format == 'h5':
            imgload.set('format', 'bdv.hdf5')
            el = ET.SubElement(imgload, 'hdf5')
        else:
            imgload.set('format', 'bdv.n5')
            imgload.set('version', '1.0')
            el = ET.SubElement(imgload, 'n5')
        el.set('type', 'relative')
        el.text = os.path.basename(self.filename + '.' + self.file_format)
        # write ViewSetups
        viewsets = ET.SubElement(seqdesc, 'ViewSetups')

        for iillumination in range(self.nilluminations):
            for ichannel in range(self.nchannels):
                for itile in range(self.ntiles):
                    for iangle in range(self.nangles):
                        isetup = self._determine_setup_id(iillumination, ichannel, itile, iangle)
                        if any([self.setup_id_present[t][isetup] for t in range(len(self.setup_id_present))]):
                            vs = ET.SubElement(viewsets, 'ViewSetup')
                            ET.SubElement(vs, 'id').text = str(isetup)
                            ET.SubElement(vs, 'name').text = 'setup ' + str(isetup)
                            nz, ny, nx = tuple(self.stack_shapes[isetup])
                            ET.SubElement(vs, 'size').text = '{} {} {}'.format(nx, ny, nz)
                            vox = ET.SubElement(vs, 'voxelSize')
                            ET.SubElement(vox, 'unit').text = self.voxel_units[isetup]
                            dx, dy, dz = self.voxel_size_xyz[isetup]
                            ET.SubElement(vox, 'size').text = '{} {} {}'.format(dx, dy, dz)
                            a = ET.SubElement(vs, 'attributes')
                            ET.SubElement(a, 'illumination').text = str(iillumination)
                            ET.SubElement(a, 'channel').text = str(ichannel)
                            ET.SubElement(a, 'tile').text = str(itile)
                            ET.SubElement(a, 'angle').text = str(iangle)

        # write Attributes (range of values)
        attrs_illum = ET.SubElement(viewsets, 'Attributes')
        attrs_illum.set('name', 'illumination')
        for iilumination in range(self.nilluminations):
            illum = ET.SubElement(attrs_illum, 'Illumination')
            ET.SubElement(illum, 'id').text = str(iilumination)
            ET.SubElement(illum, 'name').text = 'illumination ' + str(iilumination)

        attrs_chan = ET.SubElement(viewsets, 'Attributes')
        attrs_chan.set('name', 'channel')
        for ichannel in range(self.nchannels):
            chan = ET.SubElement(attrs_chan, 'Channel')
            ET.SubElement(chan, 'id').text = str(ichannel)
            ET.SubElement(chan, 'name').text = 'channel ' + str(ichannel)

        attrs_tile = ET.SubElement(viewsets, 'Attributes')
        attrs_tile.set('name', 'tile')
        for itile in range(self.ntiles):
            tile = ET.SubElement(attrs_tile, 'Tile')
            ET.SubElement(tile, 'id').text = str(itile)
            ET.SubElement(tile, 'name').text = 'tile ' + str(itile)

        attrs_ang = ET.SubElement(viewsets, 'Attributes')
        attrs_ang.set('name', 'angle')
        for iangle in range(self.nangles):
            ang = ET.SubElement(attrs_ang, 'Angle')
            ET.SubElement(ang, 'id').text = str(iangle)
            ET.SubElement(ang, 'name').text = 'angle ' + str(iangle)

        # Time points
        tpoints = ET.SubElement(seqdesc, 'Timepoints')
        tpoints.set('type', 'range')
        ET.SubElement(tpoints, 'first').text = str(0)
        ET.SubElement(tpoints, 'last').text = str(ntimes - 1)

        # missing views
        if any(True in l for l in self.setup_id_present):
            miss_views = ET.SubElement(seqdesc, 'MissingViews')
            for t in range(len(self.setup_id_present)):
                for i in range(len(self.setup_id_present[t])):
                    if not self.setup_id_present[t][i]:
                        miss_view = ET.SubElement(miss_views, 'MissingView')
                        miss_view.set('timepoint', str(t))
                        miss_view.set('setup', str(i))

        # Transformations of coordinate system
        vregs = ET.SubElement(root, 'ViewRegistrations')
        for itime in range(ntimes):
            for isetup in range(self.nsetups):
                if self.setup_id_present[itime][isetup]:
                    vreg = ET.SubElement(vregs, 'ViewRegistration')
                    vreg.set('timepoint', str(itime))
                    vreg.set('setup', str(isetup))
                    # write arbitrary affine transformation, specific for each view
                    if isetup in self.affine_matrices.keys():
                        vt = ET.SubElement(vreg, 'ViewTransform')
                        vt.set('type', 'affine')
                        ET.SubElement(vt, 'Name').text = self.affine_names[isetup]
                        n_prec = 6
                        mx_string = np.array2string(self.affine_matrices[isetup].flatten(), separator=' ',
                                                    precision=n_prec, floatmode='fixed',
                                                    max_line_width=(n_prec+6)*4)
                        ET.SubElement(vt, 'affine').text = mx_string[1:-1].strip()

        self._xml_indent(root)
        tree = ET.ElementTree(root)
        tree.write(self.filename_xml, xml_declaration=True, encoding='utf-8', method="xml")

    def append_affine(self, m_affine, name_affine="Appended affine transformation using npy2bdv.",
                      time=0, illumination=0, channel=0, tile=0, angle=0):
        """" Append affine matrix transformation to a view.
        If using in `BdvWriter`, call `BdvWriter.write_xml_file(...)` first, to create a valid XML tree.
        The transformation will be placed on top,  e.g. executed by the BigStitcher last.
        The transformation is defined as matrix of shape (3,4).
        Each column represents coordinate unit vectors after the transformation.
        The last column represents translation in (x,y,z).
        Parameters:
        -----------
            time: int
                Time index, >=0.
            illumination: int
            channel: int
            tile: int
            angle: int
                Indices of the view attributes, >= 0.
            m_affine: numpy array of shape (3,4)
                Coefficients of affine transformation matrix (m00, m01, ...)
            name_affine: str, optional
                Name of the affine transformation.
            """
        self._get_xml_root()
        isetup = self._determine_setup_id(illumination, channel, tile, angle)
        assert m_affine.shape == (3,4), "m_affine must be a numpy array of shape (3,4)"
        found = False
        for node in self._root.findall('./ViewRegistrations/ViewRegistration'):
            if int(node.attrib['setup']) == isetup and int(node.attrib['timepoint']) == time:
                found = True
                break
        assert found, f'Node not found: <ViewRegistration setup="{isetup}" timepoint="{time}">'
        vt = ET.Element('ViewTransform')
        node.insert(0, vt)
        vt.set('type', 'affine')
        ET.SubElement(vt, 'Name').text = name_affine
        n_prec = 6
        mx_string = np.array2string(m_affine.flatten(), separator=' ',
                                    precision=n_prec, floatmode='fixed',
                                    max_line_width=((n_prec+4)*12+10))
        ET.SubElement(vt, 'affine').text = mx_string[1:-1].strip()
        self._xml_indent(self._root)
        tree = ET.ElementTree(self._root)
        tree.write(self.filename_xml, xml_declaration=True, encoding='utf-8', method="xml")

    def close(self):
        """Save changes and close the data file if needed."""
        if self.file_format == "h5":
            self.file_object.close()

    def _update_setup_id_present(self, isetup, itime):
        """Update the lookup table (list of lists) for missing setups"""
        self.setup_id_present[itime][isetup] = True

    def _compute_chunk_size(self, blockdim):
        """Populate the size of h5 chunks.
        Use first-level chunk size if there are more subsampling levels than chunk size levels.
        """
        chunks = []
        base_level = blockdim[0]
        if len(blockdim) < len(self.subsamp_zyx):
            for ilevel in range(len(self.subsamp_zyx)):
                chunks.append(base_level)
            chunks_tuple = tuple(chunks)
        else:
            chunks_tuple = blockdim
        return chunks_tuple

    def _subsample_stack(self, stack, subsamp_level, pyramid_method):
        """Subsampling of a 3d stack.
        
        Parameters:
        -----------
            stack, numpy 3d array (z,y,x)
            subsamp_level, array-like with 3 elements, eg (2,4,4) for downsampling z(x2), x and y (x4).
            pyramid_method, string ('mean', 'decimate')
            
        Returns:
        --------
            down-scaled stack, unit16 type.
        """
        if all(subsamp_level[:] == 1):
            stack_sub = stack
        else:
            if pyramid_method == 'mean':
                stack_sub = skimage.transform.downscale_local_mean(stack, tuple(subsamp_level)).astype(np.uint16)
            elif pyramid_method == 'decimate':
                stack_sub = stack[int(np.floor(subsamp_level[0]/2))::subsamp_level[0], int(np.floor(subsamp_level[1]/2))::subsamp_level[1], int(np.floor(subsamp_level[2]/2))::subsamp_level[2]].astype(np.uint16)
            else:
                print('pyramid method must be mean or decimate')

            if (stack.shape[0] % 2) != 0:
                stack_sub = stack_sub[0:-1,:,:]
            if (stack.shape[1] % 2) != 0:
                stack_sub = stack_sub[:,0:-1,:]
            if (stack.shape[2] % 2) != 0:
                stack_sub = stack_sub[:,:,0:-1]

        return stack_sub

    def _subsample_plane(self, plane, subsamp_level):
        """Subsampling of a 2d plane.
        
        Parameters:
        -----------
            plane: numpy 2d array (y,x) of int16
            subsamp_level: array-like with 3 elements, eg (1,4,4) for downsampling x and y (x4).
            
        Returns:
        --------
            down-scaled plane, unit16 type.
        """
        if all(subsamp_level[:] == 1):
            plane_sub = plane
        else:
            plane_sub = skimage.transform.downscale_local_mean(plane, tuple(subsamp_level[1:])).astype(np.uint16)
        return plane_sub

    def _subsample_index(self, index, subsamp_level):
        """Subsampling of stack index.
        
        Parameters:
        -----------
            index: stack index
            subsamp_level: 1 element downsampling level.
            
        Returns:
        --------
            down-scaled index, int type.
        """    
        if index == 0:
            index_sub = index
        else:
            index_sub = round(index/subsamp_level)-1
        return int(index_sub)

    def _write_H5_headers(self):
        """Write resolutions and subdivisions for all setups into h5 file."""
        for isetup in range(self.nsetups):
            group_name = 's{:02d}'.format(isetup)
            if group_name in self.file_object:
                del self.file_object[group_name]
            grp = self.file_object.create_group(group_name)
            data_subsamp = np.flip(self.subsamp_zyx, 1).astype('float64')
            data_chunks = np.flip(self.chunks, 1).astype('int32')
            grp.create_dataset('resolutions', data=data_subsamp, dtype='<f8')
            grp.create_dataset('subdivisions', data=data_chunks, dtype='<i4')

    def _determine_setup_id(self, illumination=0, channel=0, tile=0, angle=0):
        """Takes the view attributes (illumination, channel, tile, angle) and converts them into unique setup_id.
        Parameters:
        -----------
            illumination: int
            channel: int
            tile: int
            angle: int
        Returns:
        --------
            setup_id: int, >=0 (first setup)
            """
        if self.nsetups is not None:
            setup_id_matrix = np.arange(self.nsetups)
            setup_id_matrix = setup_id_matrix.reshape((self.nilluminations, self.nchannels, self.ntiles, self.nangles))
            setup_id = setup_id_matrix[illumination, channel, tile, angle]
        else:
            setup_id = None
        return setup_id

    def _xml_indent(self, elem, level=0):
        """Pretty printing function"""
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self._xml_indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    def _get_xml_root(self):
        """Load the meta-information information from XML header file"""
        assert os.path.exists(self.filename_xml), f"Error: {self.filename_xml} file not found"
        if self._root is None:
            with open(self.filename_xml, 'r') as file:
                self._root = ET.parse(file).getroot()
        else:
            pass