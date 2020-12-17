# standards:
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# image utils:
from PIL import Image as IMG
from IPython.display import Image
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
import cv2  # for rescaling

# for the blurring:
from scipy.ndimage.filters import gaussian_filter, uniform_filter, median_filter


import scipy.spatial as spatial

# to monitor progress:
import tqdm

# to save movies:
import imageio
from datetime import datetime

# for colormaps
import palettable
import cmocean
from colorsys import hls_to_rgb
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# to speed up computation
import numba

# noise for the starting-positions:
import noise
from scipy import interpolate

# to distribute walking-speeds over range
from sklearn.preprocessing import minmax_scale


class physarum_population:
    """A class that contains the parameters that set up a physarum-population and keep track of its development."""

    def __init__(
        self,
        t=400,
        timestep=0,
        height=1000,
        width=1000,
        horizon_walk=10.0,
        horizon_sense=40.0,
        theta_walk=20.0,
        theta_sense=20.0,
        walk_range=1.0,
        trace_strength=0.3,
        colormap=palettable.cmocean.diverging.Curl_15.mpl_colormap,
        social_behaviour=-0.5,
        initialization=[[0], [0]],
        template="None",
        template_strength=1.0,
    ):
        """Initialize a physarum-population, with sensible defaults. The initialization should be set manually.

        :param t: Time for which the simulation will run, integer, defaults to 4000.
        :param timestep: Current timestep. starts at zero, gets updated in the simulation-loop.
        :param height: height of the simulation-field the population lives in. Defaults to 1000.
        :param width: width of the simulation-field the population lives in.  Defaults to 1000.
        :param horizon_walk: How far the cells walk every-step. Either float or list of floats that
                        will be distributed over the total time t, so that the value at every timestep is
                        approximated through a spline over the values given in the list.
        :param horizon_sense: How far each cell looks out before itself, to check where to go. Either float or a list of floats that
                        will be distributed over the total time t, so that the value at every timestep is
                        approximated through a spline over the values given in the list.
        :param theta_walk: Angle in degrees that the cell turns left or right on every step. Either float or list of floats that
                        will be distributed over the total time t, so that the value at every timestep is
                        approximated through a spline over the values given in the list.
        :param theta_sense: Angle in degrees that the cell checks on the left or right to decide where to go. Either float or a list of floats that
                        will be distributed over the total time t, so that the value at every timestep is
                        approximated through a spline over the values given in the list.
        :param walk_range: Range that will be multiplied with the current horizon_walk value, so that the values for all cells
                        are randomly distributed over it. This allows for faster and slower cells in the same population. Float or tuple (e.g (0.9, 2.1))
        :param trace_strength: Strength of the trace left by the population. This allows to weight different populations differently in their influence. Float, default .3
        :param colormap: matplotlib-colormap. Defaults to Curl_15 from palletable.
        :param
        """
        self.height = height
        self.width = width
        self.walk_range = walk_range

        self.horizon_sense = np.float64(horizon_sense)
        self.theta_walk = np.float64(theta_walk)
        self.theta_sense = np.float64(theta_sense)
        self.initialization = initialization
        self.colormap = colormap
        self.social_behaviour = social_behaviour
        self.trace_strength = trace_strength
        self.template_strength = template_strength

        self.x, self.y = (
            np.vstack([i[0] for i in self.initialization]),
            np.vstack([i[1] for i in self.initialization]).astype(np.float64),
        )
        self.n = np.int64(self.x.shape[0])  # here we should check xy dimensions

        self.horizon_walk = np.float64(horizon_walk)
        if isinstance(walk_range, (list, np.ndarray, tuple)):
            self.random_walking_base = np.float64(np.random.rand(self.n)).reshape(
                -1, 1
            )  # This can be used to update later.
            walking_constant = (
                horizon_walk[0]
                if isinstance(horizon_walk, (list, np.ndarray, tuple))
                else horizon_walk
            )
            self.walking_array = minmax_scale(
                self.random_walking_base,
                feature_range=(
                    walking_constant * walk_range[0],
                    walking_constant * walk_range[1],
                ),
            )

        self.angle = np.float64(np.random.rand(self.n).reshape(-1, 1) * (2 * np.pi))

        self.trace_array = np.float64(np.zeros((self.height, self.width)))

        self.t = t
        self.timestep = timestep

        if template == "None":  # hmm?
            self.template = np.zeros((height, width))
        else:
            self.template = template

    def add_organisms(self, initialization=[[0], [0]]):
        new_x, new_y = (
            np.vstack([i[0] for i in initialization]),
            np.vstack([i[1] for i in initialization]).astype(np.float64),
        )

        self.x, self.y = np.vstack([self.x, new_x]), np.vstack([self.y, new_y])

        new_angles = np.float64(
            np.random.rand(new_x.shape[0]).reshape(-1, 1) * (2 * np.pi)
        )
        self.angle = np.vstack([self.angle, new_angles])

        if isinstance(self.walk_range, (list, np.ndarray, tuple)):
            new_random_walking_base = np.float64(
                np.random.rand(new_x.shape[0])
            ).reshape(
                -1, 1
            )  # This can be used to update later.
            new_walking_array = minmax_scale(
                self.random_walking_base,
                feature_range=(
                    horizon_walk * walk_range[0],
                    horizon_walk * walk_range[1],
                ),
            )
            self.walking_array = np.vstack([self.walking_array, new_walking_array])
            return self.x, self.y, self.angle, self.walking_array

        # Add changing walking-array
        return self.x, self.y, self.angle

    # def get_point_density(self):
    # 	#not ideal, but faster than kde:
    # 	# build zero array, than add each point:
    # 	base = np.zeros(self.trace_array.shape)
    # 	self.trace_array[np.floor(self.x% (self.trace_array.shape[0])).astype(int),
    # 					np.floor(self.y% (self.trace_array.shape[1])).astype(int)] = trace_strength +  self.trace_array[np.floor(self.x% (self.trace_array.shape[0])).astype(int),
    # 																			np.floor(self.y% (self.trace_array.shape[1])).astype(int)]

    # def explode(self, top_n=5000, type='radial'):
    # 	""" Moves the organisms in dense areas."""

    def update_constant(self, this_constant):

        if isinstance(this_constant, (list, np.ndarray)):
            anchors = np.linspace(0, self.t, len(this_constant))
            spline_points = np.array([anchors, this_constant]).T
            tck = interpolate.splrep(
                spline_points[:, 0],
                spline_points[:, 1],
                s=0,
                k=np.min([len(spline_points[:, 1]) - 1, 2]),
            )
            return interpolate.splev(self.timestep, tck)
        else:
            return this_constant

    # 			 except Exception:
    # 				 print(Exception)
    # 				 pass

    def leave_trace(self, additive=False):
        """Each particle leaves it's pheromone trace on the trace array.
        If the trace is additive, it gets added, otherwise the trace array is set to the value of the trace."""

        trace_strength = self.update_constant(self.trace_strength)
        if additive == True:
            # self.trace_array[np.floor(self.x% (self.trace_array.shape[0])).astype(int),
            # 			np.floor(self.y% (self.trace_array.shape[1])).astype(int)] = trace_strength +  self.trace_array[np.floor(self.x% (self.trace_array.shape[0])).astype(int),
            # 																	np.floor(self.y% (self.trace_array.shape[1])).astype(int)]
            #
            # for i,j in zip(self.x,self.y):
            # 	self.trace_array[np.floor(i % (self.trace_array.shape[0])).astype(int),
            # 			np.floor(j % (self.trace_array.shape[1])).astype(int)] = trace_strength + self.trace_array[np.floor(i % (self.trace_array.shape[0])).astype(int),
            # 																	np.floor(j % (self.trace_array.shape[1])).astype(int)]
            my_x = np.floor(self.x % (self.trace_array.shape[0])).astype(int)
            my_y = np.floor(self.y % (self.trace_array.shape[1])).astype(int)

            vals, idx_start, count = np.unique(
                np.hstack([my_x, my_y]), return_counts=True, return_index=True, axis=0
            )
            self.trace_array[vals[:, 0], vals[:, 1]] = self.trace_array[
                vals[:, 0], vals[:, 1]
            ] + (count * trace_strength)
            # print(vals.shape, idx_start.shape, count.shape)

            # combined_coordinates = ['_'.join([str(a),str(b)]) for a,b in zip(my_x, my_y)]
            # print(combined_coordinates[0:100])

        else:
            self.trace_array[
                np.floor(self.x % (self.trace_array.shape[0])).astype(int),
                np.floor(self.y % (self.trace_array.shape[1])).astype(int),
            ] = trace_strength
        return self.trace_array

    def diffuse_gaussian(self, sigma=7, mode="wrap", truncate=4):
        """Pheromones get distributed using gaussian smoothing."""
        self.trace_array = gaussian_filter(
            self.trace_array, sigma=sigma, mode=mode, truncate=truncate
        )
        return self.trace_array

    def diffuse_uniform(self, size=3, mode="wrap"):
        """Pheromones get distributed using uniform smoothing. This can lead to nice artefacts,
        but also to diagonal drift at high values for size (?)"""
        self.trace_array = uniform_filter(self.trace_array, size=size, mode=mode)
        return self.trace_array

    def diffuse_median(self, size=3, mode="wrap"):
        """Pheromones get distributed using uniform smoothing. This can lead to nice artefacts,
        but also to diagonal drift at high values for size (?)"""
        self.trace_array = median_filter(self.trace_array, size=size, mode=mode)
        return self.trace_array

    def update_positions(self, other_populations):
        """Intermediate function, to get everythin in order for numba"""
        x = self.x
        y = self.y
        angle = self.angle
        theta_sense = self.update_constant(self.theta_sense)
        horizon_sense = self.update_constant(self.horizon_sense)
        theta_walk = self.update_constant(self.theta_walk)

        if isinstance(
            self.walk_range, (list, np.ndarray, tuple)
        ):  # If we specified a range for walking, we adapt the walking array to it, and pass it to the calculation of the new positions. Otherwise we pass the constant.
            walking_constant = self.update_constant(self.horizon_walk)
            horizon_walk = minmax_scale(
                self.random_walking_base,
                feature_range=(
                    walking_constant * self.walk_range[0],
                    walking_constant * self.walk_range[1],
                ),
            )
        else:
            horizon_walk = self.update_constant(self.horizon_walk)

        adapted_trace_array = self.trace_array
        other_populations = other_populations
        social_behaviour = self.update_constant(self.social_behaviour)
        template = self.template
        template_strength = self.update_constant(self.template_strength)

        # 		 density_cutoff = np.max(self.trace_array) - explosion_proneness
        # 		 print(density_cutoff)
        # 		 print(np.max(self.trace_array))
        adapted_trace_array = adapted_trace_array + (template * template_strength)

        for (
            this_pop
        ) in (
            other_populations
        ):  # include the other species-patterns in the present ones feeding-behaviour
            adapted_trace_array = adapted_trace_array + (this_pop * social_behaviour)

        self.x, self.y, self.angle = self.numba_update_positions(
            x,
            y,
            angle,
            theta_sense,
            horizon_sense,
            theta_walk,
            horizon_walk,
            trace_array=adapted_trace_array,
        )
        return self.x, self.y, self.angle

    @staticmethod
    @numba.njit(fastmath=True)
    def numba_update_positions(
        x, y, angle, theta_sense, horizon_sense, theta_walk, horizon_walk, trace_array
    ):
        """Internal numba-function that returns the adapted physarum-positions, given initial coordinates and constants"""

        # check for high densities, and breack them up, if required: (currently not implemented.)
        # 		 for i in range(len(x)):
        # 			 current_density = trace_array[np.int64(np.floor(x[i][0])) -1, np.int64(np.floor(y[i][0])) -1]

        # 			 if current_density > density_cutoff:
        # 				 angle[i] = np.random.rand()*(2*np.pi)
        # #				 angle[i] = np.random.rand()*(2*np.pi) # set angle to random
        # 				 x[i] = (x[i] + explosion_radius * np.cos(angle[i])) % (trace_array.shape[0])
        # 				 y[i] = (y[i] + explosion_radius * np.sin(angle[i])) % (trace_array.shape[1])
        # 		 else:
        # 			 pass

        # get values at potential new positions
        angles_to_test = np.hstack(
            (
                (angle - theta_sense) % (2 * np.pi),
                angle,
                (angle + theta_sense) % (2 * np.pi),
            )
        )
        x_to_test = (x + horizon_sense * np.cos(angles_to_test)) % (
            trace_array.shape[0]
        )  # get coordinates and wrap around
        y_to_test = (y + horizon_sense * np.sin(angles_to_test)) % (
            trace_array.shape[1]
        )

        angles_to_walk = np.hstack(
            (
                (angle - theta_walk) % (2 * np.pi),
                angle,
                (angle + theta_walk) % (2 * np.pi),
            )
        )
        x_to_walk = (x + horizon_walk * np.cos(angles_to_walk)) % (
            trace_array.shape[0]
        )  # get coordinates and wrap around
        y_to_walk = (y + horizon_walk * np.sin(angles_to_walk)) % (trace_array.shape[1])

        x_to_test_arr_coords = np.floor(x_to_test).astype(np.int64) - 1
        y_to_test_arr_coords = np.floor(y_to_test).astype(np.int64) - 1

        # get the trace-values at the lookup coordinates. Somewhat annoying in numba:

        trace_outlooks = np.empty(x_to_test_arr_coords.shape)
        for i in range(len(x_to_test_arr_coords)):
            for j in range(len(y_to_test_arr_coords[i])):
                trace_outlooks[i, j] = trace_array[
                    x_to_test_arr_coords[i, j], y_to_test_arr_coords[i, j]
                ]

        # Get the highest trace-value for each position (left spin due to failing argmax?)
        trace_max = np.empty(len(trace_outlooks))
        for i in range(len(trace_outlooks)):
            trace_max[i] = trace_outlooks[i].max()
        rows = np.where(trace_outlooks == trace_max.reshape(-1, 1))[0]
        rows_multiple_max = rows[:-1][rows[:-1] == rows[1:]]

        where_best = np.empty(len(trace_outlooks))
        for i in range(len(trace_outlooks)):
            where_best[i] = trace_outlooks[i].argmax()

        where_best[rows_multiple_max] = np.random.randint(1, trace_outlooks.shape[1])

        # select the coordinates with the highest trace-value, and return
        new_x = np.empty(len(trace_outlooks)).reshape(-1, 1)
        new_y = np.empty(len(trace_outlooks)).reshape(-1, 1)
        new_angle = np.empty(len(trace_outlooks)).reshape(-1, 1)

        for i in range(len(trace_outlooks)):
            best_index = np.int64(where_best[i])
            new_x[i] = x_to_walk[i, best_index]
            new_y[i] = y_to_walk[i, best_index]
            new_angle[i] = angles_to_walk[i, best_index]

        return new_x, new_y, new_angle


##############	MaIN SIMULATION	#################


def run_physarum_simulation(
    populations,
    image_list,
    additive_trace=True,
    diffusion="uniform",
    mask=False,
    cmap_rescaling=["sqrt", "normalize"],
    decay=0.9,
    show_image_every=20,
    img_rescale=0.2,
):

    width, height = populations[0].width, populations[0].height

    for this_step in tqdm.tqdm_notebook(range(0, populations[0].t)):

        # Adding stuff along the simulation (not fully implemented into the function yet.)
        # 		 if this_step in [int(x) for x in np.linspace(0,400,30)]:
        # 			 species_c.add_organisms(initialization=[physarum.get_filled_circle_init(n=10000, center=(np.random.randint(width),np.random.randint(height)),radius=np.random.randint(10,high=40)) for x in range(0,1)])

        species_images = []
        for ix, this_species in enumerate(populations):
            this_species.timestep = this_step

            this_species.leave_trace(additive=additive_trace)

            if diffusion == "uniform":
                this_species.diffuse_median(size=5)
            elif diffusion == "median":
                this_species.diffuse_median(size=5)
            elif diffusion == "gaussian":
                this_species.diffuse_gaussian(sigma=2, mode="wrap", truncate=5)

            # We can set a predifined mask, e.g. for labyrinths:
            if mask != False:
                this_species.trace_array[mask] = 0.0

            other_populations = [x for i, x in enumerate(populations) if i != ix]
            other_populations = [species.trace_array for species in other_populations]
            this_species.update_positions(other_populations=other_populations)
            im = this_species.trace_array

            if "sqrt" in cmap_rescaling:
                im = np.sqrt(im + 0.1) - np.sqrt(0.1)

            if "log" in cmap_rescaling:
                im = np.log(im + 1)

            if "normalize" in cmap_rescaling:
                im = (im - np.min(im)) / (np.max(im) - np.min(im))

            im = this_species.colormap(im)
            im = np.uint8(im * 255)
            im = IMG.fromarray(im).convert("RGBA")

            species_images.append(im)
            this_species.trace_array = this_species.trace_array * decay

        im = species_images[0]
        for image_to_concat in species_images[1:]:
            im = IMG.alpha_composite(im, image_to_concat)

        image_list.append(im)
        if this_step % show_image_every == 0:
            print(("Step No.: " + str(this_step)))
            display(im.resize((int(width * img_rescale), int(height * img_rescale))))
    return image_list


###############	INIT FUNCTIONS	##################
def leave_feeding_trace(
    x, y, shape, trace_strength=1.0, sigma=7, mode="wrap", wrap_around=True, truncate=4
):
    """Turns x,y-coordinates returned by a list of calls to init-functions into a smooth feeding array."""
    base = np.zeros(shape)
    if wrap_around == True:
        x = x % shape[0]
        y = y % shape[1]
    else:
        print("CutOff overly large x and y: Not yet implemented.")

    base[
        np.floor(x % (base.shape[0])).astype(int),
        np.floor(y % (base.shape[1])).astype(int),
    ] = trace_strength
    base = gaussian_filter(base, sigma=sigma, mode=mode, truncate=truncate)
    return base


def get_perlin_init(
    shape=(1000, 1000),
    n=100000,
    cutoff=None,
    repetition=(1000, 1000),
    scale=100,
    octaves=20.0,
    persistence=0.1,
    lacunarity=2.0,
):

    """Returns a tuple of x,y-coordinates sampled from Perlin noise.
    This can be used to initialize the starting positions of a physarum-
    population, as well as to generate a cloudy feeding-pattern that will
    have a natural feel to it. This function wraps the one from the noise-
    library from Casey Duncan, and is in parts borrowed from here (see also this for a good explanation of the noise-parameters):
    https://medium.com/@yvanscher/playing-with-perlin-noise-generating-realistic-archipelagos-b59f004d8401
    The most relevant paramaters for our purposes are:

    :param shape: The shape of the area in which the noise is to be generated. Defaults to (1000,1000)

    :type shape: Tuple of integers with the form (width, height).

    :param n: Number of particles to sample. When used as a feeeding trace,
    this translates to the relative strength of the pattern. defaults to 100000.

    :param cutoff: value below which noise should be set to zero. Default is None. Will lead to probabilities 'contains NaN-error, if to high'

    :param scale: (python-noise parameter) The scale of the noise -- larger or smaller patterns, defaults to 100.

    :param repetition: (python-noise parameter) Tuple that denotes the size of the area in which the noise should repeat itself. Defaults to (1000,1000)




    """
    shape = [i - 1 for i in shape]
    world = np.zeros(shape)

    # make coordinate grid on [0,1]^2
    x_idx = np.linspace(0, shape[0], shape[0])
    y_idx = np.linspace(0, shape[1], shape[1])
    world_x, world_y = np.meshgrid(x_idx, y_idx)

    # apply perlin noise, instead of np.vectorize, consider using itertools.starmap()
    world = np.vectorize(noise.pnoise2)(
        world_x / scale,
        world_y / scale,
        octaves=int(octaves),
        persistence=persistence,
        lacunarity=lacunarity,
        repeatx=repetition[0],
        repeaty=repetition[1],
        base=np.random.randint(0, 100),
    )
    # world = world * 3
    # 	 Sample particle init from map:
    world[world <= 0.0] = 0.0  # filter negative values
    if cutoff != None:
        world[world <= cutoff] = 0.0
    linear_idx = np.random.choice(
        world.size, size=n, p=world.ravel() / float(world.sum())
    )
    x, y = np.unravel_index(linear_idx, shape)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    return x, y


def get_circle_init(n=100000, center=(500, 500), radius=200, width=100):
    """Returns tuple of x,y-coordinates sampled from a ring with the given center, radius and width."""
    x = (center[0] + radius * np.cos(np.linspace(0, 2 * np.pi, n))).reshape(-1, 1)
    y = (center[1] + radius * np.sin(np.linspace(0, 2 * np.pi, n))).reshape(-1, 1)
    # perturb coordinates:

    x = x + np.random.normal(0.0, 0.333, size=(n, 1)) * width
    y = y + np.random.normal(0.0, 0.333, size=(n, 1)) * width
    return x, y


def get_filled_circle_init(n=100000, center=(500, 500), radius=200):
    """Returns tuple of x,y-coordinates sampled from a circle with the given center and radius"""

    t = 2 * np.pi * np.random.rand(n)
    r = np.random.rand(n) + np.random.rand(n)
    r[r > 1] = 2 - r[r > 1]
    x = center[0] + r * radius * np.cos(t)
    y = center[1] + r * radius * np.sin(t)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    return x, y


def get_gaussian_gradient(n=100000, center=(500, 500), sigma=20):
    """Returns tuple of x,y-coordinates sampled from a 2-d gaussian around a given center with a given sigma"""

    x = np.random.normal(center[0], sigma, size=(n, 1))
    y = np.random.normal(center[1], sigma, size=(n, 1))
    return x, y


def get_uniform_init(n=100000, shape=(1000, 1000)):
    """Returns tuple of x,y-coordinates uniformly distributed over an area of the given shape."""
    x = np.random.randint(shape[0], size=n).reshape(-1, 1)  # starting_position x
    y = np.random.randint(shape[1], size=n).reshape(-1, 1)  # starting_position y
    return x, y


def get_image_init_positions(
    image, shape=(1000, 1000), n=100000, flip=False
):  # shape should be set to something similar to the prop. of the image
    init_image = IMG.open(image).convert("L")
    init_image = init_image.resize(tuple(np.flip(shape)))
    init_image = np.array(init_image) / 255
    if flip == True:
        init_image = 1 - init_image
    linear_idx = np.random.choice(
        init_image.size, size=n, p=init_image.ravel() / float(init_image.sum())
    )
    x, y = np.unravel_index(linear_idx, shape)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    return x, y


def get_image_init_array(
    image, shape=(1000, 1000)
):  # shape should be set to something similar to the prop. of the image
    init_image = IMG.open(image).convert("L")
    init_image = init_image.resize(tuple(np.flip(shape)))
    init_image = np.array(init_image) / 255
    return init_image


################### COLOR FUNCTIONS ##################


def my_hls_to_rgb(color):
    """Turn HLS-triplet into RGB"""
    hue = color[0] / 360
    saturation = color[1] / 100
    luminosity = color[2] / 100
    rgb = hls_to_rgb(hue, luminosity, saturation)
    # 	 rgb = [x * 255 for x in rgb]
    return rgb


def make_colormap(colorlist, alpha_distribution=False, basecolor=None):
    """returns a matplotlib colormap that goes through the values speciefied in hue, luminosity, saturation-format.
    :param alpha: How transparancey should be distributed over the colormap.
    List of the form [start alpha at this percentage, value of alpha at the start, end alpha at this percentage,
    value of alpha at the end.]. If True is passed, a standard value is used, if flase, the colormap has no alpha overlayed.

    :param basecolor: Basis on which to plot the transparant colormaps. Usually the background-color.
    Not relevant for the actual colormap produced.

    """

    these_colors = LinearSegmentedColormap.from_list(
        "map", [my_hls_to_rgb(x) for x in colorlist], N=400
    )

    if alpha_distribution == True:
        these_colors = these_colors(np.linspace(0, 1, 400))
        # these_colors[0:100,-1] = np.linspace(0.0,0,100)
        these_colors[0:400, -1] = np.linspace(0.0, 0.8, 400)
        these_colors = ListedColormap(these_colors)
    elif isinstance(alpha_distribution, (list, np.ndarray, tuple)):
        these_colors = these_colors(np.linspace(0, 1, 400))

        starting_point = int(alpha_distribution[1] * 400)
        ending_point = int(alpha_distribution[3] * 400)
        these_colors[:starting_point, -1] = 0
        these_colors[starting_point:ending_point, -1] = np.linspace(
            alpha_distribution[0], alpha_distribution[2], ending_point - starting_point
        )
        these_colors[ending_point:, -1] = alpha_distribution[2]

        these_colors = ListedColormap(these_colors)

    fig, ax = plt.subplots(figsize=(8, 1))
    plt.pcolormesh(np.linspace(0, 400, 400).reshape(1, -1), cmap=these_colors)
    if basecolor != None:
        ax.set_facecolor(basecolor)
    return these_colors


################ SOME UTILS FOR INSPECTION ##############


def plot_init_trace(feeding_trace, cmap=cm.get_cmap("viridis"), scale=0.3):
    """Displays a given trace_array, to check what it would look like"""
    im = (feeding_trace - np.min(feeding_trace)) / (
        np.max(feeding_trace) - np.min(feeding_trace)
    )
    im = cmap(im)  # to take a look at our feeding trace..
    im = np.uint8(im * 255)
    display(
        IMG.fromarray(im).resize(
            (int(feeding_trace.shape[0] * scale), int(feeding_trace.shape[1] * scale))
        )
    )


def test_spline(value, t):
    """Plots the spline that willl be extrapolated for a list of changing constants.

    :param value: list of values
    :param t: number of timesteps to take
    """

    anchors = np.linspace(0, t, len(value))
    base = np.linspace(0, t, t)

    spline_points = np.array([anchors, value]).T

    tck = interpolate.splrep(
        spline_points[:, 0],
        spline_points[:, 1],
        s=0,
        k=np.min([len(spline_points[:, 1]) - 1, 2]),
    )

    plt.scatter(base, interpolate.splev(base, tck))
    plt.scatter(x=anchors, y=value)


################ Saving #####################


def save_film(images, name=False, fps=20, format="mp4", loop=False):

    if name == False:
        now = datetime.now()  # current date and time
        date_time = now.strftime("%m-%d-%Y--%H-%M-%S")
        name = date_time + ".mp4"

    if loop == True:
        a = images.copy()
        images.reverse()
        images = a + images

    imageio.mimsave(name, images, format=format, fps=fps)


def save_grid(images, name=False, folder=False):

    images_to_use = [np.floor(x) for x in np.linspace(5, len(images) - 1, 5 * 5)]

    a = np.zeros((5, 5))

    x = np.array([0])
    y = np.array([0])

    for k in range(1, 5):
        space = np.linspace(0, k, k + 1)
        if k % 2 == 0:
            x, y = np.hstack([x, space]), np.hstack([y, space[::-1]])
        else:
            x, y = np.hstack([x, space[::-1]]), np.hstack([y, space])

    for k in range(1, 5):
        space = np.linspace(4, k, 5 - k)
        if k % 2 == 0:
            x, y = np.hstack([x, space[::-1]]), np.hstack([y, space])
        else:
            x, y = np.hstack([x, space]), np.hstack([y, space[::-1]])

    for s in range(len(x)):
        a[int(x[s]), int(y[s])] = images_to_use[s]

    rows = []
    for i in tqdm.tqdm_notebook(range(len(a))):
        rows.append(
            np.hstack([np.array(pic) for pic in [images[int(j)] for j in a[i, :]]])
        )

    image_collection = IMG.fromarray(np.vstack(rows))
    display(image_collection)
    if name == False:
        now = datetime.now()  # current date and time
        date_time = now.strftime("%m-%d-%Y--%H-%M-%S")
        name = date_time + ".png"

    if folder != False:
        name = folder + name
    image_collection.save(name)


def save_single_image_grid(
    images, name=False, folder=False, image_no=False, dimensions=(5, 5)
):

    if image_no == False:
        tile_array = np.hstack([np.array(images[-1])] * dimensions[0])
    else:
        tile_array = np.hstack([np.array(images[image_no])] * dimensions[0])

    tile_array = np.vstack([tile_array] * dimensions[1])
    image_collection = IMG.fromarray(tile_array)

    display(image_collection)
    if name == False:
        now = datetime.now()  # current date and time
        date_time = now.strftime("%m-%d-%Y--%H-%M-%S")
        name = date_time + ".png"

    if folder != False:
        name = folder + name
    image_collection.save(name)

    image_collection.save("physarum_tiles/" + date_time + ".png")
