# lsom
Light Self Organizing Maps (SOM, a.k.a. Kohonen Maps) in Python

For a research project at LOCEAN, UPMC, Paris, I needed to use SOMs in Python. There is a huge toolbox available in Matlab (http://www.cis.hut.fi/somtoolbox/), but I struggled to find a package in Python suitable to my needs (a light one, but flexible, with multiple steps training, well commented, where it is possible to change the parameters, running batch learning). I decided to code what I needed, and I share it here, hoping that it will be useful to someone.

Here is an example code, comments in the file should be complete enough:

```
from lsom import *
X = np.concatenate((np.random.rand(100,3) , np.random.rand(100,3)+np.asarray([1,1,1])))  
(vap, vepu, pc) = pca(X)
koh = SOM((5,5), 3, hexagonal=True, init_fun=init_koh_pca(vap, 200, (5,5)) )  
koh.train(pc, niter=30, lrate=.9, iradius=5)
koh.quality(pc)
draw_kohonen_hex(koh.K, pc, ax1=0, ax2=1)
```

This code is in Public Domain.

References:

http://www.cis.hut.fi/projects/somtoolbox/theory/somalgorithm.shtml

Kohonen, T. (1995). Self-Organizing Maps. Series in Information Sciences, Vol. 30. Springer, Heidelberg. Second ed. 1997. 
