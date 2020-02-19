# pca_nmf_faces.py

# Authors: Vlad Niculae, Alexandre Gramfort, Slim Essid
# License: BSD


##VERSION CORRIGE par rapport a l'enonce

##Pour la partie "Extraction de visage" du TP, le code fourni n'a pas l'air de marcher, en le modifiant comme ça ça a l'air de mieux fonctionner :

##il faut également 
##- supprimer le paramètre hold=True  dans la fonction plot_gallery
##- Remplacer init=0 par init=None  dans la définition de estimators 



from time import time
from numpy.random import RandomState
import pylab as pl
import numpy as np

from sklearn.datasets import fetch_olivetti_faces
from sklearn import decomposition

# -- Prepare data and define utility functions ---------------------------------

n_row, n_col = 2, 5
n_components = n_row * n_col
image_shape = (64, 64)
rng = RandomState(0)

# Load faces data
dataset = fetch_olivetti_faces(shuffle=True, random_state=rng)
faces = dataset.data

n_samples, n_features = faces.shape

# global centering

faces_centered = faces - faces.mean(axis=0, dtype=np.float64)

print("Dataset consists of %d faces" % n_samples)

def plot_gallery(title, images):
    pl.figure(figsize=(2. * n_col, 2.26 * n_row))
    pl.suptitle(title, size=16)
    for i, comp in enumerate(images):
        pl.subplot(n_row, n_col, i + 1)
        
        comp = comp.reshape(image_shape)
        vmax = comp.max()
        vmin = comp.min()
        dmy = np.nonzero(comp<0)
        if len(dmy[0])>0:
            yz, xz = dmy
        comp[comp<0] = 0
       
        pl.imshow(comp, cmap=pl.cm.gray, vmax=vmax, vmin=vmin)
        #print "vmax: %f, vmin: %f" % (vmax, vmin)
        #print comp
      
        if len(dmy[0])>0:
            pl.plot( xz, yz, 'r,')    #, hold=True)
            print(len(dmy[0]), "negative-valued pixels")
            
        pl.xticks(())
        pl.yticks(())
        
    pl.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
    
# Plot a sample of the input data
plot_gallery("First centered Olivetti faces", faces_centered[:n_components])

# -- Decomposition methods -----------------------------------------------------

# List of the different estimators and whether to center the data

estimators = [
    ('pca', 'Eigenfaces - PCA',
     decomposition.PCA(n_components=n_components, whiten=True),
     True),
  
    ('nmf', 'Non-negative components - NMF',
     decomposition.NMF(n_components=n_components, init=None, tol=1e-6,
                       max_iter=2000),
     False)      
]

# -- Transform and classify ----------------------------------------------------

labels = dataset.target
X = faces
X_ = faces_centered

for shortname, name, estimator, center in estimators:
    #if shortname != 'nmf': continue
    print("Extracting the top %d %s..." % (n_components, name))
    t0 = time()

    data = X
    if center:
        data = X_
   
    data = estimator.fit_transform(data)
    
    train_time = (time() - t0)
    print("done in %0.3fs" % train_time)
    
    components_ = estimator.components_
    
    plot_gallery('%s - Train time %.1fs' % (name, train_time),
                     components_[:n_components])
