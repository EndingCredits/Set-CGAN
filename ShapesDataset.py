from __future__ import division

import numpy as np

class shapeGenerator():
    params = {
        'num_points': 15,
        'num_extra_points': 10,
        'uniform_point_distribution': False,
        'point_noise_scale': 0.1,
        'shape_noise_scale': 0.5,
        'scale_min': 0.1,
        'initial_seed': 1234,
        'dataset_size': 100000
    }

    def __init__(self, params = {}):
        self.params.update(params)

        self.num_points = self.params['num_points']
        self.num_extra_points = self.params['num_extra_points']
        self.point_noise_scale = self.params['point_noise_scale']
        self.shape_noise_scale = self.params['shape_noise_scale']
        self.scale_min = self.params['scale_min']
        self.point_dist = self.params['uniform_point_distribution']

        self.dataset_size = self.params['dataset_size']
        self.initial_seed = self.params['initial_seed']
        self.initial_seed_cv = self.initial_seed + self.dataset_size
        self.num_samples = 0         #Number of samples taken.
        self.num_samples_cv = 0      #Number of samples taken for validation.

    def elementSize(self):
        return 2
    
    def numClasses(self):
        return 3

    def getBatch(self, batch_size, validation=False):
        shapes = []
        labels = []
        metadata = []
        for i in range(batch_size):
            shape, label, data = self._get_shape(self._get_seed(validation))
            shapes.append(shape) ; labels.append(label) ; metadata.append(data)
        return shapes, labels, metadata


    def _get_shape(self, seed):

        np.random.seed(seed)        

        shape = []

        # Generate shape variables
        shape_type = np.random.randint(3)                              # type of shape
        if self.num_extra_points > 0:
            extra_points = np.random.randint(self.num_extra_points) 
        else:
            extra_points = 0
        N = self.num_points + extra_points                             # number of points
        scale = self.scale_min + np.random.random()*(1-self.scale_min) # scale factor
        rot = np.random.random()                                       # rotation factor
        x_shift = (2*np.random.random()-1)*self.shape_noise_scale      # x noise for shape
        y_shift = (2*np.random.random()-1)*self.shape_noise_scale      # y noise for shape

        # Generate variables for each point
        pos = np.random.rand(N)                                        # position of each point along the outline of the shape
        noise_x = (2*np.random.rand(N)-1)*self.point_noise_scale       # x noise for each point
        noise_y = (2*np.random.rand(N)-1)*self.point_noise_scale       # y noise for each point
         
        # Precalc useful values
        cos_rot = np.cos(rot * 2 * np.pi)
        sin_rot = np.sin(rot * 2 * np.pi)

        for i in range(N):
          # Get point from position and shape type
          if self.point_dist == True: 
              x, y = self._get_point(shape_type, pos[i])
          else:
              x, y = self._get_point(shape_type, float(i)/N)

          # Add noise
          x += noise_x[i] ; y += noise_y[i]

          # Scale
          x *= scale ; y *= scale

          # Rotate
          x_ = x*cos_rot - y*sin_rot
          y_ = x*sin_rot + y*cos_rot

          # Translate
          x_ += x_shift ; y_ += y_shift 

          # Add point to shape
          shape.append([x_,y_])

        label = np.zeros(3) ; label[shape_type] = 1
        data = np.array([scale, x_shift, y_shift])
        return shape, label, data

    def _get_point(self, shape_type, percent):
      if shape_type == 0: #"circle"
        x = np.cos(percent * 2 * np.pi)
        y = np.sin(percent * 2 * np.pi)
        return x,y

      if shape_type == 1: #"square"
        # Split into 4 lines, renormalise percentage, and return position along line      
        if percent < 0.25:
          per = percent*4
          # Bottom left to bottom right
          x = 2*per-1 ; y = -1
        elif percent < 0.5:
          per = (percent-0.25)*4
          # Bottom right to top right
          x = 1 ; y = 2*per-1
        elif percent < 0.75:
          per = (percent-0.5)*4
          # Top right to top left
          x = 1-2*per ; y = 1
        else:
          per = (percent-0.75)*4
          # Top left to bottom left
          x = -1 ; y = 1-2*per
        return x,y

      if shape_type == 2: #"triangle"
        # Split into 3 lines, renormalise percentage, and return position along line, then scale to a radius of 1
        if percent < 0.3333:
          per = percent*3
          # Bottom left to bottom right
          x = per-0.5 ; y = -0.2887
        elif percent < 0.6666:
          per = (percent-0.3333)*3
          # Bottom right to top
          x = 0.5-0.5*per ; y = -0.2887+0.866*per
        else:
          per = (percent-0.6666)*3
          # Top to bottom left
          x = -0.5*per ; y = 0.5774-0.866*per
        #x *= 1.73 ; y *= 1.73
        x *= 2 ; y *= 2
        return x,y

    def _get_seed(self, validation=False):
        if validation:
            offset = self.num_samples_cv
            self.num_samples_cv += 1
            return self.initial_seed_cv + offset
        else:
            offset = self.num_samples if self.dataset_size == 0 else self.num_samples % self.dataset_size
            self.num_samples += 1
            return self.initial_seed + offset

