# physarum package

## Module contents


### physarum.get_circle_init(n=100000, center=(500, 500), radius=200, width=100)
Returns tuple of x,y-coordinates sampled from a ring with the given center, radius and width.


### physarum.get_filled_circle_init(n=100000, center=(500, 500), radius=200)
Returns tuple of x,y-coordinates sampled from a circle with the given center and radius


### physarum.get_gaussian_gradient(n=100000, center=(500, 500), sigma=20)
Returns tuple of x,y-coordinates sampled from a 2-d gaussian around a given center with a given sigma


### physarum.get_image_init_array(image, shape=(1000, 1000))

### physarum.get_image_init_positions(image, shape=(1000, 1000), n=100000, flip=False)

### physarum.get_perlin_init(shape=(1000, 1000), n=100000, cutoff=None, repetition=(1000, 1000), scale=100, octaves=20.0, persistence=0.1, lacunarity=2.0)
Returns a tuple of x,y-coordinates sampled from Perlin noise.
This can be used to initialize the starting positions of a physarum-
population, as well as to generate a cloudy feeding-pattern that will
have a natural feel to it. This function wraps the one from the noise-
library from Casey Duncan, and is in parts borrowed from here (see also this for a good explanation of the noise-parameters):
[https://medium.com/@yvanscher/playing-with-perlin-noise-generating-realistic-archipelagos-b59f004d8401](https://medium.com/@yvanscher/playing-with-perlin-noise-generating-realistic-archipelagos-b59f004d8401)
The most relevant paramaters for our purposes are:


* **Parameters**

    
    * **shape** (*Tuple of integers with the form** (**width**, **height**)***) – The shape of the area in which the noise is to be generated. Defaults to (1000,1000)


    * **n** – Number of particles to sample. When used as a feeeding trace,


this translates to the relative strength of the pattern. defaults to 100000.


* **Parameters**

    
    * **cutoff** – value below which noise should be set to zero. Default is None. Will lead to probabilities ‘contains NaN-error, if to high’


    * **scale** – (python-noise parameter) The scale of the noise – larger or smaller patterns, defaults to 100.


    * **repetition** – (python-noise parameter) Tuple that denotes the size of the area in which the noise should repeat itself. Defaults to (1000,1000)



### physarum.get_uniform_init(n=100000, shape=(1000, 1000))
Returns tuple of x,y-coordinates uniformly distributed over an area of the given shape.


### physarum.leave_feeding_trace(x, y, shape, trace_strength=1.0, sigma=7, mode='wrap', wrap_around=True, truncate=4)
Turns x,y-coordinates returned by a list of calls to init-functions into a smooth feeding array.


### physarum.make_colormap(colorlist, alpha_distribution=False, basecolor=None)
returns a matplotlib colormap that goes through the values speciefied in hue, luminosity, saturation-format.
:param alpha: How transparancey should be distributed over the colormap.
List of the form [start alpha at this percentage, value of alpha at the start, end alpha at this percentage,
value of alpha at the end.]. If True is passed, a standard value is used, if flase, the colormap has no alpha overlayed.


* **Parameters**

    **basecolor** – Basis on which to plot the transparant colormaps. Usually the background-color.


Not relevant for the actual colormap produced.


### physarum.my_hls_to_rgb(color)
Turn HLS-triplet into RGB


### class physarum.physarum_population(t=400, timestep=0, height=1000, width=1000, horizon_walk=10.0, horizon_sense=40.0, theta_walk=20.0, theta_sense=20.0, walk_range=1.0, trace_strength=0.3, colormap=<matplotlib.colors.LinearSegmentedColormap object>, social_behaviour=-0.5, initialization=[[0], [0]], template='None', template_strength=1.0)
Bases: `object`

A class that contains the parameters that set up a physarum-population and keep track of its development.


#### add_organisms(initialization=[[0], [0]])

#### diffuse_gaussian(sigma=7, mode='wrap', truncate=4)
Pheromones get distributed using gaussian smoothing.


#### diffuse_median(size=3, mode='wrap')
Pheromones get distributed using uniform smoothing. This can lead to nice artefacts,
but also to diagonal drift at high values for size (?)


#### diffuse_uniform(size=3, mode='wrap')
Pheromones get distributed using uniform smoothing. This can lead to nice artefacts,
but also to diagonal drift at high values for size (?)


#### leave_trace(additive=False)
Each particle leaves it’s pheromone trace on the trace array.
If the trace is additive, it gets added, otherwise the trace array is set to the value of the trace.


#### static numba_update_positions(x, y, angle, theta_sense, horizon_sense, theta_walk, horizon_walk, trace_array)
Internal numba-function that returns the adapted physarum-positions, given initial coordinates and constants


#### update_constant(this_constant)

#### update_positions(other_populations)
Intermediate function, to get everythin in order for numba


### physarum.plot_init_trace(feeding_trace, cmap=<matplotlib.colors.ListedColormap object>, scale=0.3)
Displays a given trace_array, to check what it would look like


### physarum.run_physarum_simulation(populations, image_list, additive_trace=True, diffusion='uniform', mask=False, cmap_rescaling=['sqrt', 'normalize'], decay=0.9, show_image_every=20, img_rescale=0.2)

### physarum.save_film(images, name=False, fps=20, format='mp4', loop=False)

### physarum.save_grid(images, name=False, folder=False)

### physarum.save_single_image_grid(images, name=False, folder=False, image_no=False, dimensions=(5, 5))

### physarum.test_spline(value, t)
Plots the spline that willl be extrapolated for a list of changing constants.


* **Parameters**

    
    * **value** – list of values


    * **t** – number of timesteps to take
