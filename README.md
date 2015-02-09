# tsne_galzoo
t-SNE visualization of SDSS galaxies.  Clustering is based on morphologies collected by [Galaxy Zoo](http://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge).

Running 
`python tsne_galzoo.py`
will 
- download and untar 800 MB of image data (~20 minutes), 
- embed 10k randomly-selected galaxies into a 2d space based on their GalaxyZoo morphologies, using the `scikit-learn` t-SNE implementation (~30-60 minutes), and
- make a big jpg image showing those galaxies in the 2d space (~5 minutes).

To explore that image in the browser, modify the `index.html` and `my.js` files accordingly.

See [my example here](http://stanford.edu/~rkeisler/tsne_galzoo/).
