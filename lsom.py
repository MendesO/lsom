########################################################
### Light Self Organizing Maps
# 2017, LOCEAN, UPMC, Paris
# Mendes Oulamara
# Distributed with no guarantees.
# This file is in Public Domain.
# A simple but flexible class for Self Organizing Maps
# with batch learning
########################################################
import numpy as np
import matplotlib.pyplot as plt

####
# Example code:
# from lsom import *
# X = np.concatenate((np.random.rand(100,3) , np.random.rand(100,3)+np.asarray([1,1,1])))
# (vap, vepu, pc) = pca(X)
# koh = SOM((5,5), 3, hexagonal=True, init_fun=init_koh_pca(vap, 200, (5,5)) )  
# koh.train(pc, niter=30, lrate=.9, iradius=5)
# koh.quality(pc)
# draw_kohonen_hex(koh.K, pc, ax1=0, ax2=1)
####
class SOM:
####
# shape: the 2D shape of the grid of the SOM
# dim: the size of the vectors to classify
# hexagonal: whether the grid of the SOM is hexagonal or rectangilar
# init_fun: a function to initialize the referent of the SOM, of the form 
#    ds -> K
#    where ds is a dataset of shape (_, dim) and K is the grid of referents with shape (shape[0], shape[1], dim)
# layer_dist_metr: the distance between coordinates on the grid
# nbd_kernel: the function that, given a distance, returns the weight of a neigbour
# nbd_decay: given the iteration it and the total number of iterations itmax, returns the decay coefficient of the influence 
#    of the neigbourhood
# learn_decay: given the iteration it and the total number of iterations itmax, returns the decay learning coefficient
####
    def __init__(self, shape, dim, hexagonal=False, init_fun=None, layer_dist_metr=None, 
            nbd_kernel=None, nbd_decay=None, learn_decay=None):
        self._shape = shape
        self._dim = dim
        self._hexagonal = hexagonal
        self._init_fun = init_fun
        

        if layer_dist_metr != None:
            self.layer_dist_metric = layer_dist_metr
        elif hexagonal:
            A = np.asarray([[1,.5],[0,3**.5/2]])
            self.layer_dist_metric = lambda v1,v2: np.linalg.norm(np.dot(A,
                                    np.reshape(np.asarray(v1)-np.asarray(v2),(2,1))))
        else:
            self.layer_dist_metric = lambda v1, v2: np.linalg.norm(np.asarray(v1)-np.asarray(v2))


        self.K = None
        self.nbd_kernel=nbd_kernel
        self.nbd_decay=nbd_decay
        self.learn_decay=learn_decay


        if nbd_kernel is None:
            # We choose a gaussian neighborhood kernel
            self.nbd_kernel = lambda d: np.exp(-.5 * d**2) 
        if nbd_decay is None:
            #We choose an exponential radius decay
            self.nbd_decay = lambda it, itmax:  np.exp(-2.3 * it/np.sqrt(itmax))
            # 2.3 is set to have exactly one order of magnitude at the square root of the itmax 
            # We choose a linearly interpolated influence radius
            # self.nbd_decay = lambda it, itmax : (1-it/itmax)
        if learn_decay is None:
            # We choose an exponential learning decay
            # 2.3 is set to have exactly one order of magnitude of difference between 
            # the first and the last rate
            self.learn_decay = lambda it, itmax:  np.exp(-2.3 * it/itmax)

####
# Initializes the refeents of the SOM, given a dataset of samples ds. 
####
    def init_som(self, ds):
        if self._init_fun is None:
            self.K = np.zeros((self._shape[0], self._shape[1], self._dim))
            for i in range(self._shape[0]):
                for j in range(self._shape[1]):
                    self.K[i][j] = ds[np.random.choice(ds.shape[0])].copy()
        else:
            self.K = self._init_fun(ds)
####
# Tests if two coordinates on the grid are neigbour.
####
    def is_neighbour(self, v1,v2):
        d = self.layer_dist_metric(v1,v2)
        return d>.9 and d<1.1

####
# Given a sample, returns the coordinates on the grid of its closest referent 
# (its 'best matching unit').
####     
    def bmu(self, sample):
        loc = np.argmin(np.linalg.norm(self.K-sample, axis=2))
        return (int(loc/self.K.shape[1]), int(loc%self.K.shape[1]))

####
# Applies bmu to every sample of a dataset ds.
####
    def bmus(self, ds):
        return np.asarray([self.bmu(sample) for sample in ds])

####
# Given a dataset ds, returns for every sample its bmu, and second closest referent
# ('next best matching unit').
####
    def n2bmus(self,ds):
        bmus=self.bmus(ds)
        nbmus=np.zeros(bmus.shape)
        for i in range(ds.shape[0]):
            distmin=-1
            for j in range(self.K.shape[0]):
                for k in range(self.K.shape[1]):
                    dist = np.linalg.norm(self.K[j][k]-ds[i])
                    if(distmin == -1 or dist<distmin) and ((j,k) != bmus[i]).any():
                        distmin=dist
                        nbmus[i]=(j,k)
        return (bmus,nbmus)

####
# Trains the SOM on a dataset ds, for niter iteration, with initial learning rate lrate and initial
# influence radius iradius.
####
    def train(self, ds, niter=20, lrate=.7, iradius=None):
        if iradius is None:
            iradius = max(self._shape)
        if self.K is None:
            self.init_som(ds)
        distance = np.zeros((self.K.shape[0], self.K.shape[1], self.K.shape[0], self.K.shape[1]))
        for i1 in range(self.K.shape[0]):
            for i2 in range(self.K.shape[0]):
                for j1 in range(self.K.shape[1]):
                    for j2 in range(self.K.shape[1]):
                        distance[i1,j1,i2,j2] = self.layer_dist_metric((i1,j1), (i2,j2))
  
        for it in range(niter):
            coeffK = np.zeros((self.K.shape[0], self.K.shape[1]))
            deltaK = np.zeros(self.K.shape)
            altdK = np.zeros(self.K.shape)
            altcK = np.zeros((self.K.shape[0], self.K.shape[1]))
            for sample in ds:
                (xr,yr) = self.bmu(sample)
                deltaK +=np.transpose(
                    self.nbd_kernel(distance[xr][yr]/(self.nbd_decay(it,niter) * iradius))*
                                np.transpose((sample-self.K), [2,0,1]) , [1,2,0])
                coeffK += self.nbd_kernel(distance[xr][yr]/(self.nbd_decay(it,niter) * iradius))
                altdK[xr][yr] += sample
                altcK[xr][yr] += 1
            deltaK = np.transpose(np.transpose(deltaK, [2,0,1]) / (coeffK+(coeffK==0)*1 ), [1,2,0])
            altdK = np.transpose(np.transpose(altdK, [2,0,1]) / (altcK+(altcK==0)*1 ), [1,2,0])
            self.K += lrate*self.learn_decay(it,niter) * deltaK

####
# Returns the the distorsion and the topographical error of the SOM, computed on a dataset ds.
####
    def quality(self, ds):
        bmus,nbmus = self.n2bmus(ds)
     
        qerr = np.linalg.norm(ds - self.K[bmus[:,0],bmus[:,1]])
        qerr/= np.sqrt(ds.shape[0])

        terr=1.*np.sum([not self.is_neighbour(bmus[i], nbmus[i]) for i in range(ds.shape[0])])
        terr /= ds.shape[0]

        return (qerr, terr)


####
# Computes the U-matrix of a classification of a dataset X by the SOM.
####
    def Umatrix(self, X):
        um = np.zeros((self.K.shape[0]*2-1 , self.K.shape[1]*2-1))
     
        occurences = np.zeros((self.K.shape[0], self.K.shape[1]))
        dispersion = np.zeros((self.K.shape[0], self.K.shape[1]))
        bmus = self.bmus(X)
        for i in range(X.shape[0]):
            occurences[bmus[i][0]][bmus[i][1]] += 1
            dispersion[bmus[i][0]][bmus[i][1]] += np.sum((X[i]-self.K[bmus[i,0] , bmus[i,1]])**2)
            
        dispersion /= np.max((occurences , .1+np.zeros(occurences.shape)), axis=0)
        dispersion = np.sqrt(dispersion)
    
        for i in range(self.K.shape[0]):
            for j in range(self.K.shape[1]):
                um[2*i,2*j] = dispersion[i,j]
                if (i+1 < self.K.shape[0]):
                    um[2*i+1,2*j] = np.linalg.norm(self.K[i,j]-self.K[i+1,j]) 
                if (j+1 < self.K.shape[1]):
                    um[2*i,2*j+1] = np.linalg.norm(self.K[i,j]-self.K[i,j+1]) 
                if (i+1 < self.K.shape[0] and j+1 < self.K.shape[1]):
                    um[2*i+1,2*j+1] = np.linalg.norm(self.K[i,j]-self.K[i+1,j+1]) 
        return um


#################################
# PCA

####
# Minimalist PCA and un PCA functions: takes a centred dataset X, and returns the
# (eigenvalues, unitary eigenvectors (in column), and (non normalized) principal components)
#### 
def pca(X):
    (vap,vepu) = np.linalg.eig(np.dot(np.transpose(X),X))
    pc = np.dot(X, vepu)
    return (vap,vepu,pc)

def unpca(pc, vepu):
    return np.dot(pc , np.transpose(vepu[:,:pc.shape[1]])) 

####
# With the unitary eigenvector vepu, and eigen values vap given by a PCA, draws
# the circle of correlation of the coordinates (in the original space) with names
# names, in the plane of the principal components ax1 and ax2.
####
def corrcir(vepu, vap, names = [], ax1=0, ax2=1):
    if names == []:
        names = [str(i) for i in range(vepu.shape[0])]
    circle1=plt.Circle((0,0),radius=1, color='g', fill=False)
    fig = plt.gcf()
    fig.gca().add_artist(circle1)
    for i in range(vepu.shape[0]):
        x = np.sqrt(np.max((0,vap[ax1]))) * vepu[i, ax1] / np.sqrt(6000)
        y = np.sqrt(np.max((0,vap[ax2]))) * vepu[i, ax2] / np.sqrt(6000)
        plt.plot([0.0,x],[0.0,y],'k-')
        plt.plot(x, y, 'rx')
        plt.annotate(names[i], xy=(x,y))
    plt.xlim((-1,1))
    plt.ylim((-1,1))
    plt.title("Circle of Correlations on axes "+str(ax1)+" -- " +str(ax2) 
        + " Inertia : " + str(int(100*vap[ax1]/np.sum(vap))) +"% || " +str(int(100*vap[ax2]/np.sum(vap))) + "%" )

#################################
# SOM complementary functions

####
# We assume that the samples are given in their representation in principal
# components (pc), after a PCA.
# Given a list of eigenvalues from the PCA, the number of sample on which
# the PCA was computed, and the shape of the SOM grid, it returns a function
# that, given a dataset ds, returns an initial grid that is spread in the plane of
# the two largest eigenvelues (in PCA coordinates), with ratio of the grid linked to the ratio
# of the eigenvalues.
####
def init_koh_pca(vap, nbsample, shape):
    std = np.sqrt(vap/nbsample)
    def _init_koh(ds):
        carte=np.zeros((shape[0],shape[1],ds.shape[1]))
        for i in range(shape[0]):
            for j in range(shape[1]):
                carte[i][j][0] = 4*i*std[0]/shape[0]-2*std[0]
                carte[i][j][1] = 4*j*std[1]/shape[1]-2*std[1]
        return carte
    return _init_koh

####
# Given a SOM koh, a dataset X, and an array of quantities associated to the sample of X
# (i.e. an array with shape (X.shape[0], q) such that quand[i,j] is the value of the j-th quantity
# (say, the temperature) for sample i of X), it returns for each referent of the SOM grid, the mean
# of the quantities among samples associated to that referent, and the number of samples associated to
# each referent (the occurences).
####
def quant_on_koh(koh , X, quant):
    quantk = np.zeros((quant.shape[1] , koh.K.shape[0] , koh.K.shape[1]) )
    bmus = koh.bmus(X)
    occurences = np.zeros((koh.K.shape[0], koh.K.shape[1]))
    for i in range(X.shape[0]):
        occurences[bmus[i][0]][bmus[i][1]] += 1
        quantk[: , bmus[i][0] , bmus[i][1] ] += quant[i]
    for i in range(quant.shape[1]):
        quantk[i] /= np.max((occurences , .1+np.zeros(occurences.shape)), axis=0)

    return quantk , occurences

####
# Given a SOM koh, a dataset X, and an array of quantities associated to the sample of X
# (i.e. an array with shape (X.shape[0], q) such that quand[i,j] is the value of the j-th quantity
# (say, the temperature) for sample i of X), it returns for each referent of the SOM grid, the 
# standard deviation of the quantities among samples associated to that referent.
####
def std_on_koh(koh , X, quant):
    quantk, occ = quant_on_koh(koh, X, quant)

    std = np.zeros((quant.shape[1] , koh.K.shape[0] , koh.K.shape[1]) )
    bmus = koh.bmus(X)

    for i in range(X.shape[0]):
        std[: , bmus[i][0] , bmus[i][1] ] += (quant[i]-quantk[:, bmus[i,0], bmus[i,1]])**2

    for i in range(quant.shape[1]):
        std[i] /= np.max((occ , .1+np.zeros(occ.shape)), axis=0)

    return np.sqrt(std)

####
# We suppose that the data and the referent are in PCA coordinates. Given a SOM koh, and the
# unitary eigenvectors from the PCA, computes the value of the original coordinates of the referents.
####
def quantref_on_koh(koh , vepu):
    quantk = np.zeros((koh.K.shape[2], koh.K.shape[0], koh.K.shape[1]))
    for i in range(koh.K.shape[0]):
        for j in range(koh.K.shape[1]):
            quantk[:,i,j] = np.dot(vepu, koh.K[i,j].reshape((koh.K.shape[2],1)) ).squeeze()
    return quantk

#################################
# SOM drawing

####
# If the SOM is rectangular.
# We assume that the samples are given in their representation in principal
# components (pc), after a PCA.
# Draws the projection of the SOM (with grid of referents K), and a dataset
# (with PCA representation pc) in the plan of principal component ax1 and ax2.
# dcl is an array of integers of length pc.shape[0], the dots of the samples with the same
# integer will have the same color.
####
def draw_kohonen(K, pc, ax1=0, ax2=1, dcl=None ):
    plt.clf()
    if(dcl == None):
        dcl = np.zeros(pc.shape[0])
    dcl.astype(int)
    plt.scatter(pc[:,ax1], pc[:,ax2], c=dcl)
    plt.scatter(K[:,:,ax1].reshape(K.shape[0]*K.shape[1]), K[:,:,ax2].reshape(K.shape[0]*K.shape[1]))
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            if(i+1<K.shape[0]):
                plt.plot([K[i,j,ax1], K[i+1,j,ax1] ],[ K[i,j, ax2] , K[i+1,j,ax2]], color='b')
            if(j+1<K.shape[1]):
                plt.plot([K[i,j,ax1], K[i,j+1,ax1] ],[ K[i,j, ax2] , K[i,j+1,ax2]], color='b')
    plt.show()

####
# If the SOM is hexagonal.
# We assume that the samples are given in their representation in principal
# components (pc), after a PCA.
# Draws the projection of the SOM (with grid of referents K), and a dataset
# (with PCA representation pc) in the plan of principal component ax1 and ax2.
# dcl is an array of integers of length pc.shape[0], the dots of the samples with the same
# integer will have the same color.
####
def draw_kohonen_hex(K, pc, ax1=0, ax2=1, dcl=None):
    plt.clf()
    if(dcl == None):
        dcl = np.zeros(pc.shape[0])
    dcl.astype(int)
    plt.scatter(pc[:,ax1], pc[:,ax2], c=dcl)
    plt.scatter(K[:,:,ax1].reshape(K.shape[0]*K.shape[1]), K[:,:,ax2].reshape(K.shape[0]*K.shape[1]))
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            if(i+1<K.shape[0]):
                plt.plot([K[i,j,ax1], K[i+1,j,ax1] ],[ K[i,j, ax2] , K[i+1,j,ax2]], color='b')
            if(j+1<K.shape[1]):
                plt.plot([K[i,j,ax1], K[i,j+1,ax1] ],[ K[i,j, ax2] , K[i,j+1,ax2]], color='b')
            if(i+1<K.shape[0] and j>0):
                plt.plot([K[i,j,ax1], K[i+1,j-1,ax1] ],[ K[i,j, ax2] , K[i+1,j-1,ax2]], color='b')
    plt.show()


#################################
# Hierarchical Clustering (HCA)

####
# Shows the dendrogram leading to a hierarchical clustering of the referents of a SOM koh. 
####
def show_dendrogram(koh):
    link = clt.hierarchy.linkage(koh.K.reshape((koh.K.shape[0]*koh.K.shape[1],koh.K.shape[2])), 'ward')
    clt.hierarchy.dendrogram(link) 
    plt.show()

####
# Given a SOM koh, a dataset of samples ds, and a distance value dist, returns the classification in 
# clusters of the referents on the grid (kcluster) and the samples (dcluster). The clusters are computed
# via HCA with the Ward criterion cutting at distance dist.
####
def produce_cluster(koh, ds, dist):
    link = clt.hierarchy.linkage(koh.K.reshape((koh.K.shape[0]*koh.K.shape[1],koh.K.shape[2])), 'ward')
    kcluster = clt.hierarchy.fcluster(link, dist, criterion='distance')
    kcluster = kcluster.reshape((koh.K.shape[0], koh.K.shape[1]))
    
    bmus = koh.bmus(ds)
    dcluster = kcluster[bmus[:,0], bmus[:,1]]
    return (kcluster, dcluster)








