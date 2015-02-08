import numpy as np
np.random.seed(0)
import ipdb
import matplotlib.pylab as pl
import cPickle as pickle
nfit_default = 10000


def main():
    embed_and_save()
    make_big_image()


def make_big_image(nfit=nfit_default, force_grid=False, density=1.0, 
                   npix_horizontal=6000, npix_vertical=4000, 
                   npix_small=50):

    # load and rescale the 2d embedding.
    id, x, y = load_embedding(nfit=nfit)
    x -= x.min(); x /= x.max(); x *= 0.9; x += 0.05
    y -= y.min(); y /= y.max(); y *= 0.9; y += 0.05
    
    # stupid matplotlib axis convention...
    nx_big = npix_vertical
    ny_big = npix_horizontal

    # initialize the big image.
    big = np.zeros((nx_big, ny_big, 3))
    if not(force_grid):
        weight = np.zeros((nx_big, ny_big, 3))
    nx_small, ny_small = npix_small, npix_small
    n_obj = len(x)

    # loop over small images and add them to the big image.
    for counter, this_id, this_x, this_y in zip(range(n_obj), id, x, y):
        if (counter % 100)==0: print '*** %i/%i ***'%(counter, n_obj)

        # get the location of this image.
        a = np.ceil(this_x * (nx_big-nx_small)+1)
        b = np.ceil(this_y * (ny_big-ny_small)+1)
        if force_grid:
            a = a-np.mod(a-1,nx_small)+1
            b = b-np.mod(b-1,ny_small)+1
            # if there is already an image here, skip it.
            if big[a,b,1] != 0

        # randomly decide whether or not we keep this image.
        if np.random.random() > density: continue

        # make sure the new small image will fit in the big image.
        if ((a<0) | (b<0) | (a+nx_small>nx_big) | (b+ny_small>ny_big)):
            continue

        # load the new small image.
        this_filename = 'data/images_training_rev1/%i.jpg'%this_id
        this_small = preprocess_image(this_filename, nside_out=nx_small)

        # make sure this image doesn't have a wonky, non-black background.
        background = 0.25*(this_small[0,:,:].mean() + 
                           this_small[:,0,:].mean() +
                           this_small[-1,:,:].mean() +
                           this_small[:,-1,:].mean())
        if background>20: continue

        # put the new image into the big image.
        if force_grid:
            image[a:a+nx_small, b:b+ny_small, :] = this_image
        else:
            # this is a nice way of blending in overlapping images.
            this_weight = (this_image.mean(2))[:,:,np.newaxis]
            weight[a:a+nx_small, b:b+ny_small, :] += this_weight
            image[a:a+nx_small, b:b+ny_small, :] += (this_image*this_weight)

    # divide out the weight array from the big image.
    image[weight>0] /= weight[weight>0]

    # convert to 8-bit and save as jpg.
    image = image.astype('uint8')
    pl.imsave('test.jpg', image)


def load_embedding(nfit=nfit_default):
    savename = 'embed/embed%i.pkl'%nfit
    id, x, y = pickle.load(open(savename,'r'))
    return id, x, y


def embed_and_save(nfit=nfit_default):
    # load labels
    data = load_data()

    # shuffle the galaxies
    nsamples, ndim = data.shape
    ind = np.arange(nsamples)
    np.random.shuffle(ind)
    data = data[ind, :]

    # grab a subset of data (NFIT long)
    id = data[0:nfit, 0]
    pos = data[0:nfit, 1:]

    # renormalize the morphology vectors
    #pos /= pos.sum(0)
    #pos /= (pos**2).sum(0)

    # embed into the 2d space
    x, y = embed(pos).T

    # save
    import os
    if not os.path.exists('embed'):
        os.makedirs('embed')
    savename = 'embed/embed%i.pkl'%nfit
    pickle.dump((id, x, y), open(savename,'w'))


def embed(pos):
    from sklearn.manifold import TSNE
    model = TSNE(n_components=2, random_state=0, init='pca', metric='l2')
    nsamples, ndim = pos.shape
    print '...fitting %i samples...'%(nsamples)
    pos2d = model.fit_transform(pos)
    return pos2d

def load_data():
    print '...loading data...'
    filename='data/training_solutions_rev1.csv'
    return np.loadtxt(filename,skiprows=1,delimiter=',')


def preprocess_image(filename, simple_ds=4., mask_strength=None, nside_out=50):
    from PIL import Image
    from skimage.filter import gaussian_filter
    # open file
    x=np.array(Image.open(filename), dtype=np.float)
    if ((simple_ds>1) & (simple_ds!=None)):
        # Gaussian smooth with FWHM = "simple_ds" pixels.
        for i in range(3):
            x[:,:,i] = gaussian_filter(x[:,:,i], 1.*simple_ds/2.355)
        # subsample by simple_ds.
        x = x[0::int(simple_ds), 0::int(simple_ds), :]
        # take inner nside_out x nside_out.
        ntmp = x.shape[0]-nside_out
        if (ntmp % 2)==0:
            nside=ntmp/2
            x = x[nside:-nside, nside:-nside, :]
        else:
            nside=(ntmp-1)/2
            x = x[nside+1:-nside, nside+1:-nside, :]
    # If desired, apply mask.
    if (mask_strength==None) | (mask_strength=='none'):
        return x
    else:
        if (mask_strength=='weak'): mask_thresh = 15.
        if (mask_strength=='strong'): mask_thresh = 30.
        avg = np.mean(x,axis=-1)
        mask = get_mask(avg, thresh=mask_thresh)
        x *= mask[:,:,np.newaxis]
        return x

def get_mask(small, thresh=25):
    # add color cut in here?
    from skimage.filter import gaussian_filter
    from scipy import ndimage
    (nx,ny)=small.shape
    sm = gaussian_filter(small, 4.0)
    #sm = gaussian_filter(small, 3.0)    
    notdark = sm>thresh
    label_objects, nb_labels = ndimage.label(notdark)
    mask = label_objects == label_objects[np.round(nx/2.), np.round(ny/2.)]
    return mask

def cimshow(img):
    pl.ion()
    tmp=img.copy()
    tmp -= tmp.min()
    tmp /= tmp.max()
    pl.imshow(np.array(tmp*255., dtype=np.uint8))
    pl.draw()
    return


if __name__ == "__main__":
    main()

# Data is from:
# http://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/download/training_solutions_rev1.zip
# https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/download/images_training_rev1.zip

    
