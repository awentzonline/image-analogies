import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d


def _calc_patch_grid_dims(shape, patch_size, patch_stride):
    x_w, x_h, x_c = shape
    num_rows = 1 + (x_h - patch_size) // patch_stride
    num_cols = 1 + (x_w - patch_size) // patch_stride
    return num_rows, num_cols


def make_patch_grid(x, patch_size, patch_stride=1):
    '''x shape: (num_channels, rows, cols)'''
    x = x.transpose(2, 1, 0)
    patches = extract_patches_2d(x, (patch_size, patch_size))
    x_w, x_h, x_c  = x.shape
    num_rows, num_cols = _calc_patch_grid_dims(x.shape, patch_size, patch_stride)
    patches = patches.reshape((num_rows, num_cols, patch_size, patch_size, x_c))
    patches = patches.transpose((0, 1, 4, 2, 3))
    return patches


def combine_patches_grid(in_patches, out_shape):
    '''Reconstruct an image from these `patches`

    input shape: (rows, cols, channels, patch_row, patch_col)
    '''
    num_rows, num_cols = in_patches.shape[:2]
    num_channels = in_patches.shape[-3]
    patch_size = in_patches.shape[-1]
    num_patches = num_rows * num_cols
    in_patches = np.reshape(in_patches, (num_patches, num_channels, patch_size, patch_size))  # (patches, channels, pr, pc)
    in_patches = np.transpose(in_patches, (0, 2, 3, 1)) # (patches, p, p, channels)
    recon = reconstruct_from_patches_2d(in_patches, out_shape)
    return recon.transpose(2, 1, 0)


def normalize_patches(patches):
    norm = np.sqrt(np.sum(np.square(patches), axis=(2, 3, 4), keepdims=True))
    return patches / norm


def congrid(a, newdims, method='linear', centre=False, minusone=False):
    '''Arbitrary resampling of source array to new dimension sizes.
    Currently only supports maintaining the same number of dimensions.
    To use 1-D arrays, first promote them to shape (x,1).

    Uses the same parameters and creates the same co-ordinate lookup points
    as IDL''s congrid routine, which apparently originally came from a VAX/VMS
    routine of the same name.

    method:
    neighbour - closest value from original data
    nearest and linear - uses n x 1-D interpolations using
                         scipy.interpolate.interp1d
    (see Numerical Recipes for validity of use of n 1-D interpolations)
    spline - uses ndimage.map_coordinates

    centre:
    True - interpolation points are at the centres of the bins
    False - points are at the front edge of the bin

    minusone:
    For example- inarray.shape = (i,j) & new dimensions = (x,y)
    False - inarray is resampled by factors of (i/x) * (j/y)
    True - inarray is resampled by(i-1)/(x-1) * (j-1)/(y-1)
    This prevents extrapolation one element beyond bounds of input array.
    '''
    if not a.dtype in [np.float64, np.float32]:
        a = np.cast[float](a)

    m1 = np.cast[int](minusone)
    ofs = np.cast[int](centre) * 0.5
    old = np.array( a.shape )
    ndims = len( a.shape )
    if len( newdims ) != ndims:
        print "[congrid] dimensions error. " \
              "This routine currently only support " \
              "rebinning to the same number of dimensions."
        return None
    newdims = np.asarray( newdims, dtype=float )
    dimlist = []

    if method == 'neighbour':
        for i in range( ndims ):
            base = np.indices(newdims)[i]
            dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                            * (base + ofs) - ofs )
        cd = np.array( dimlist ).round().astype(int)
        newa = a[list( cd )]
        return newa

    elif method in ['nearest','linear']:
        # calculate new dims
        for i in range( ndims ):
            base = np.arange( newdims[i] )
            dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                            * (base + ofs) - ofs )
        # specify old dims
        olddims = [np.arange(i, dtype = np.float) for i in list( a.shape )]

        # first interpolation - for ndims = any
        mint = scipy.interpolate.interp1d( olddims[-1], a, kind=method )
        newa = mint( dimlist[-1] )

        trorder = [ndims - 1] + range( ndims - 1 )
        for i in range( ndims - 2, -1, -1 ):
            newa = newa.transpose( trorder )

            mint = scipy.interpolate.interp1d( olddims[i], newa, kind=method )
            newa = mint( dimlist[i] )

        if ndims > 1:
            # need one more transpose to return to original dimensions
            newa = newa.transpose( trorder )

        return newa
    elif method in ['spline']:
        oslices = [ slice(0,j) for j in old ]
        oldcoords = np.ogrid[oslices]
        nslices = [ slice(0,j) for j in list(newdims) ]
        newcoords = np.mgrid[nslices]

        newcoords_dims = range(np.rank(newcoords))
        #make first index last
        newcoords_dims.append(newcoords_dims.pop(0))
        newcoords_tr = newcoords.transpose(newcoords_dims)
        # makes a view that affects newcoords

        newcoords_tr += ofs

        deltas = (np.asarray(old) - m1) / (newdims - m1)
        newcoords_tr *= deltas

        newcoords_tr -= ofs

        newa = scipy.ndimage.map_coordinates(a, newcoords)
        return newa
    else:
        print "Congrid error: Unrecognized interpolation type.\n", \
              "Currently only \'neighbour\', \'nearest\',\'linear\',", \
              "and \'spline\' are supported."
        return None
