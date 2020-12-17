# Example


This code shows how to implement a simple physarum-simulation in python, using my `physarum`-package.  The package is heavily inspired by/derivative of work by [**Sage Jenson**](https://sagejenson.com/physarum) and [**Jason Rampe**](https://softologyblog.wordpress.com/2019/04/11/physarum-simulations/), and geared towards the production of digital art and playful experimentation. Have fun!

## Installation

You should be able to install the package with a simple:
`pip install physarum`


## Usage
We start by importing the `physarum`-package.


```python
%load_ext autoreload
%autoreload 2
import physarum


```


```python
# # standards:
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# # image utils:
# from PIL import Image as IMG
# from IPython.display import Image 
# from skimage import data, color
# from skimage.transform import rescale, resize, downscale_local_mean
# import cv2 # for rescaling

# # for the blurring:
# from scipy.ndimage.filters import gaussian_filter, uniform_filter

# # to monitor progress:
# import tqdm

# # to save movies:
# import imageio
# from datetime import datetime

# # for colormaps
# import palettable
# import cmocean 
# from colorsys import hls_to_rgb
# from matplotlib import cm
# from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# # to speed up computation
# import numba

# # noise for the starting-positions:
# import noise
# from scipy import interpolate
```

## Setting up the colormaps
Before we run our simulation, we have to set up the colormaps that we want to use for each of our physarum-populations. I like to use the little function below that allows me to produce colormaps from a list of `[hue,chroma,luminosity]`-triples, but in principle you can just use any matplotlib-colormap.

The first colormap can be fully opaque, but the other two will be overlayed, so the will need some transparancy given by the ` alpha-distribution`-parameter in the form `[alpha with which to start, percentage at which to start, alpha with which to end, percentage at which to end]`. The current examples leave the bottom 10% of the colormap fully transparent, the top 20% at 90% transparency, with a regular increase in between.

The function also automatically plots your colormap. With the basecolor-parameter, you can set the background of this plot.




```python
### # make_colormap([[0,90,50],[50,90,50],[100,90,50],[150,90,50],[200,90,50],[250,90,50],[300,90,50],[350,90,50]],alpha=False)
species_a_color = physarum.make_colormap([[30,56,72],[225,42,17]],alpha_distribution=False)
species_b_color = physarum.make_colormap([[27,94,58],[2,100,50]],alpha_distribution=[0.1,0.1,0.9,0.8], basecolor= species_a_color(0))


```


![png](output_4_0.png)



![png](output_4_1.png)


## Setting up basic parameters

We set up the basic parameters for our simulation: width, height (in px) and the amount of itererations for which we want it to run. 


```python
width = 700
height = 700
t = 200 
```

## Setting up the initial positions
Then we have to set up the initial positions of our particles. The package provides several functions for this, which all return a tuple of xy-coordinates. A list of these tuples will then be passed to the init-function when we construct a physarum-population.

In this example, we set up a population that is initiated with a large circle made of 100000 particles in the center and some perlin-noise (300000 particles) around it. To see how that looks like, we then plot the resulting coordinates with matplotlib.



```python
init=[physarum.get_filled_circle_init(n=100000, center=(350,350),radius=100),
        physarum.get_perlin_init(shape=(height,width), n=300000,scale=80)]


# plot the init with matplotlib (not necessary, only for illustration):
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5,5))
init_to_plot = np.vstack([i[0] for i in init]),np.vstack([i[1] for i in init])
init_to_plot = np.hstack(init_to_plot)

plt.scatter(init_to_plot[:,0],init_to_plot[:,1],s=0.1, c='black',alpha=0.2)
```




    <matplotlib.collections.PathCollection at 0x1af809a4a88>




![png](output_8_1.png)


## Setting up the populations

Now we set up our physarum-populations. There are several parameters to play with. First we need to inform each population about the environment it's living in by passing it the height, width and time-parameters. Then we need to set up the length of every step which each particle will do in every tick of the simulation (`horizon_walk`), and how far ahead the particle will look before doing that step (`horizon_sense`).



```python
species_a = physarum.physarum_population(height=height,width=width,t=t,
                                horizon_walk=1,horizon_sense=9,
                                theta_walk=15,theta_sense=10.,walk_range = [1.,2.],
                                colormap=species_a_color,
                                social_behaviour =0,trace_strength = 1,
                                initialization=init)


species_b = physarum.physarum_population(height=height,width=width,t=t,
                                horizon_walk=1,horizon_sense=9,
                                theta_walk=15,theta_sense=10.,walk_range = [0.9,1.2],
                                colormap=species_b_color,
                                social_behaviour = -16,trace_strength = 1,
                                initialization=[physarum.get_perlin_init(shape=(height,width), n=300000,scale=380)])


```

## Running the simulation



```python

species_list = [species_a,species_b]
images=[]
physarum.run_physarum_simulation(populations = species_list, image_list=images,show_image_every=50)
```


    HBox(children=(HTML(value=''), FloatProgress(value=0.0, max=200.0), HTML(value='')))


    Step No.: 0
    


![png](output_12_2.png)


    Step No.: 50
    


![png](output_12_4.png)


    Step No.: 100
    


![png](output_12_6.png)


    Step No.: 150
    


![png](output_12_8.png)


    
    




    [<PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85CDD448>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85CDD808>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85CDD308>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85CDD3C8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85CDDB48>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85CDD088>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85CDD488>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85CDDB08>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85CDD348>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85CDD388>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85772608>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85530BC8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85772DC8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85772808>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85772408>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85772708>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85772988>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85772388>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85772D88>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85772848>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85772D08>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85772D48>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85772208>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF857725C8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85772CC8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85772B48>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85772288>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85772FC8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF857723C8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85772E88>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85772148>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85772108>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85772648>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85772A48>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85CDD5C8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF8282C748>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82A91EC8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82A91F08>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82A91148>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82A91B08>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82A91F48>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82A91C08>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82A91E08>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82A91D08>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82A91F88>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82A91A48>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82A91CC8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82A91B48>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82A91988>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82A91E88>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82A910C8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82A919C8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82A91788>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85891148>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82A91888>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82A91908>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82A91948>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82A91DC8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82A91388>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82A914C8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82A91348>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82A91648>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82A91548>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82A915C8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82A91588>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82A91288>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85CB7F48>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85ED0208>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85ED0C48>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85ED0348>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85772488>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85772B08>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85CDD588>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85C8D208>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85C8D848>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85CDD608>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85957688>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF859577C8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85957248>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85957088>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85957908>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF859576C8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF853DCDC8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF853DCD08>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF853DC8C8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF853DCFC8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF853DC948>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82878C48>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF8289E1C8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82346FC8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85CDDFC8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82342408>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF853DCA88>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82342908>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82342A08>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85C8D248>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF853DC808>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF92A9E048>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF858919C8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF8581BDC8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF8581B8C8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF8581B608>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF8581B048>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85EA4A88>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF8581B508>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF8581BCC8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF852D7CC8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF8294FB48>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85749E88>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85749D48>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85749BC8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85749348>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85749048>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85749AC8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85749B48>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85749848>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85749A48>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85749088>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85749588>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85749408>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85749288>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85749DC8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85749248>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85749508>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85749908>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85749188>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85749748>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85749CC8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85749B08>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85749308>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85749B88>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85749108>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85749208>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85749C48>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85CDD248>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82B0CF08>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85749688>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82B0CAC8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82B0CC88>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82B0CD08>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82B0CB88>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82A91248>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AFF103C648>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AFF103C208>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AFF103CB08>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AFF103C388>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AFF103CEC8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AFF103C3C8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AFF103C788>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AFF103CC88>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AFF103CD88>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AFF103CA48>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AFF103C5C8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85CDDAC8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AFF103CE48>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AFF103C188>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AFF103C588>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AFF103C408>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AFF103C1C8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AFF103C748>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AFF103CE88>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AFF103C108>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AFF103CC08>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AFF103C848>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AFF103C148>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AFF103C448>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AFF103CF48>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AFF103CD48>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AFF103CFC8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AFF103CBC8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AFF103C248>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85CDD2C8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85749788>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF824BCF48>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF824BC888>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF824BC5C8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF824BCE88>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF824BCA88>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF824BCFC8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF824BCF88>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF824BC2C8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF824BCA08>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF824BCE08>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF824BCDC8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF824BC9C8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF85CDD408>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82342788>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82B1D9C8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82B1D188>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82B1D348>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82B1D2C8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82B1D308>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82B1D648>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82B1DE88>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82B1D448>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82B1D508>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82B1D4C8>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82B1DD08>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82B1D688>,
     <PIL.Image.Image image mode=RGBA size=700x700 at 0x1AF82B1D5C8>]



## Saving


```python
physarum.save_film(images,name="example_film.mp4")

physarum.save_film(images,name="example_film.gif", format="gif")
```

    IMAGEIO FFMPEG_WRITER WARNING: input image is not divisible by macro_block_size=16, resizing from (700, 700) to (704, 704) to ensure video compatibility with most codecs and players. To prevent resizing, make your input image divisible by the macro_block_size or set the macro_block_size to 1 (risking incompatibility).
    

![](example_film.gif)


```python
physarum.save_grid(images)

```


    HBox(children=(HTML(value=''), FloatProgress(value=0.0, max=5.0), HTML(value='')))


    
    


![png](output_16_2.png)



```python
physarum.save_single_image_grid(images,image_no=70,dimensions=(4,2))
```


![png](output_17_0.png)



```python

```
