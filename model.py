from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *

#import matplotlib for plotting
import matplotlib.pyplot as plt

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class DCGAN(object):
  def __init__(self, sess, input_height=108, input_width=108, crop=True,
         batch_size=64, sample_num = 64, output_height=64, output_width=64,
         y_dim=128, z_dim=100, gf_dim=64, df_dim=64,
         gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
         input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None,
         use_tags=False, compare_examples=True):
    """

    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
    self.sess = sess
    self.crop = crop
    
    self.use_tags = use_tags
    self.compare_examples = compare_examples
    
    self.use_regularization = True
    self.use_stacked_discrim = False

    self.batch_size = batch_size
    self.sample_num = sample_num
    self.example_num = self.batch_size//4

    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width

    self.y_dim = y_dim
    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim
    
    

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')
    self.d_bn3 = batch_norm(name='d_bn3')

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')
    self.g_bn3 = batch_norm(name='g_bn3')
    self.g_bn4 = batch_norm(name='g_bn4')

    self.dataset_name = dataset_name
    self.input_fname_pattern = input_fname_pattern
    self.checkpoint_dir = checkpoint_dir
    self.shapesdataset = (dataset_name=="shapes")
    if self.shapesdataset:
      print("Using shapes dataset")

      from ShapesDataset import shapeGenerator
      params = {
        'point_noise_scale': 0.01,
        'shape_noise_scale': 0.001,
        'scale_min': 0.5,
        'uniform_point_distribution': True,
        'num_points': self.batch_size*2,
        'num_extra_points': 0
      }
      self.shapes = shapeGenerator(params)
      self.use_regularization = False
      
    else:
      self.data = []
      self.names = []
      paths = glob(os.path.join("./data", self.dataset_name, "*/"))
      if not self.use_tags:
        # Group by folders
        for path in paths:
          data = glob(os.path.join(path, self.input_fname_pattern))
          if len(data) > self.batch_size:
            self.data.append(data)
            _, name = os.path.split(os.path.split(path)[0])
            #print name
            self.names.append(name)
      else:
        # Load and read .tags files
        from collections import defaultdict
        all_data = defaultdict(lambda: [])
        print("Building dataset...")
        for path in paths:
          data = glob(os.path.join(path, self.input_fname_pattern))
          for d in data:
            with open(d + '.tags', "r") as f:
              this_tags = f.readline().split()
            for t in this_tags:
              all_data[t].append(d)
        self.data = all_data.values()
        self.names = all_data.keys()

      print(len(self.data))
      
      imreadImg = imread(self.data[0][0])
      self.c_dim = 1
      if len(imreadImg.shape) >= 3:
        #check if image is a non-grayscale image by checking channel number
        self.c_dim = imreadImg.shape[-1]
      self.grayscale = (self.c_dim == 1)

    self.build_model()
    

  def build_model(self):


    if not self.shapesdataset:
      image_dims = [self.output_height, self.output_width, self.c_dim]
    else:
      image_dims = [2]

    self.inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='real_images')
      
    self.class_examples = tf.placeholder(
      tf.float32, [self.example_num] + image_dims, name='class_examples')
      
    self.fake_examples = tf.placeholder(
      tf.float32, [self.example_num] + image_dims, name='class_examples')

    inputs = self.inputs

    self.z = tf.placeholder(
      tf.float32, [self.batch_size, self.z_dim], name='z')
    self.z_sum = histogram_summary("z", self.z)
    
    if not self.shapesdataset:
      self.y                = self.gen_embeddings(self.class_examples, reuse=False)
    else:
      self.y                = self.shape_embeddings(self.class_examples, reuse=False)

    self.G                  = self.generator(self.z, self.y)
    self.sampler            = self.sampler(self.z, self.y)
    

    if self.shapesdataset:
      self.y_ = self.discrim_embeddings(self.class_examples, reuse=False)
      self.D, self.D_logits = self.shapes_discriminator(inputs, self.y_, reuse=False)
      self.D_, self.D_logits_ = self.shapes_discriminator(self.G, self.y_, reuse=True)

      if self.compare_examples:
        y_f = self.discrim_embeddings(self.fake_examples, reuse=True)
        self.Df, self.D_logitsf = self.shapes_discriminator(inputs, y_f, reuse=True)
        self.Df_, self.D_logitsf_ = self.shapes_discriminator(self.G, y_f, reuse=True)
        
    else:
      img_embs = self.discrim_embedder(inputs, reuse=False)
      g_embs = self.discrim_embedder(self.G, reuse=True)
      c_e_embs = self.discrim_embedder(self.class_examples, reuse=True)
      self.D, self.D_logits = self.discriminator(img_embs, c_e_embs, reuse=False)
      self.D_, self.D_logits_ = self.discriminator(g_embs, c_e_embs, reuse=True)
      
      if self.compare_examples:
        f_e_embs = self.discrim_embedder(self.fake_examples, reuse=True)
        self.Df, self.D_logitsf = self.discriminator(img_embs, f_e_embs, reuse=True)
        self.Df_, self.D_logitsf_ = self.discriminator(g_embs, f_e_embs, reuse=True)
        
    #if self.use_stacked_discrim:
      #full_inputs = tf.concat( [inputs, self.G], axis=0 )
      #D, D_logits = self.discriminator(full_inputs, self.class_examples, reuse=False)
      #self.D, self.D_ = tf.reshape(D, [2, self.batch_size, -1])
      #self.D_logits, self.D_logits_ = tf.reshape(D_logits, [2, self.batch_size, -1])    
    
    self.d_sum = histogram_summary("d", self.D)
    self.d__sum = histogram_summary("d_", self.D_)
    if not self.shapesdataset:
      self.G_sum = image_summary("G", self.G)
    else:
      self.G_sum = histogram_summary("G", self.G)

    def sigmoid_cross_entropy_with_logits(x, y):
      try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
      except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

    self.d_loss_real = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
    self.d_loss_fake = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
      
    self.d_loss = self.d_loss_real + self.d_loss_fake
     
    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
         
    self.g_loss = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
    if self.compare_examples:
      g_loss_comp = tf.reduce_mean(
        sigmoid_cross_entropy_with_logits(self.D_logitsf_, tf.zeros_like(self.Df_)))
      self.g_loss += g_loss_comp
      
      d_loss_fake_comp = tf.reduce_mean(
        sigmoid_cross_entropy_with_logits(self.D_logitsf, tf.zeros_like(self.Df)))
      self.d_loss += d_loss_fake_comp + g_loss_comp
    
    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)
    
    if self.use_regularization:
      self.d_reg = self.Discriminator_Regularizer(self.D, self.D_logits, self.inputs, self.D_, self.D_logits_, self.G)
      assert self.d_loss.shape == self.d_reg.shape
      self.d_loss += (self.gamma/2.0)*self.d_reg

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]
    self.e_vars = [var for var in t_vars if '_e_' in var.name]
    self.saver = tf.train.Saver()


  def train(self, config):
    # Create optimisers
    d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.g_loss, var_list=self.g_vars)
              
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    # Summaries
    self.g_sum = merge_summary([self.z_sum, self.d__sum, self.G_sum,
      self.d_loss_fake_sum, self.g_loss_sum])
    self.d_sum = merge_summary(
        [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
    self.writer = SummaryWriter("./logs", self.sess.graph)


    #TODO: Select data in an appropriate way
    self.sample_num=self.batch_size
    sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))
    
    if not self.shapesdataset:
      sample_inputs = []
      num_sample_classes = min(5, len(self.data))
      for i in range(num_sample_classes):
        sample_files = self.data[i][0:self.example_num]
        sample = [
            get_image(sample_file,
                      input_height=self.input_height,
                      input_width=self.input_width,
                      resize_height=self.output_height,
                      resize_width=self.output_width,
                      crop=self.crop,
                      grayscale=self.grayscale) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)
        save_images(sample_images, image_manifold_size(sample_images.shape[0]),
            './{}/class_examples_{}.png'.format(config.sample_dir, self.names[i]))
        sample_inputs.append(sample_images)

    elif self.shapesdataset:
      sample_inputs = []
      num_sample_classes = 5
      for i in range(num_sample_classes):
        sample_inputs.append(self.shapes._get_shape(i)[0][:self.example_num])
      
    # Set up networks
    counter = 1
    start_time = time.time()
    could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    for epoch in xrange(config.epoch):
    
      #TODO: Select data in an appropriate way
      batch_idxs = 1000
      #min(min([len(data) for data in self.data]), config.train_size) // config.batch_size

      for idx in xrange(batch_idxs):
        if not self.shapesdataset:
        
          # Get samples and fake sampes
          k, k_ = np.random.choice(len(self.data), [2])
          data = self.data[k]
          fake_data = self.data[k_]
          
          batch_files = np.random.choice(data, [config.batch_size])
          batch = [
              get_image(batch_file,
                        input_height=self.input_height,
                        input_width=self.input_width,
                        resize_height=self.output_height,
                        resize_width=self.output_width,
                        crop=self.crop,
                        grayscale=self.grayscale) for batch_file in batch_files]
          batch_images = np.array(batch).astype(np.float32)
          
          if config.compare_examples:
            #fake examples
            batch_files = np.random.choice(fake_data, [self.example_num])
            batch = [
                get_image(batch_file,
                          input_height=self.input_height,
                          input_width=self.input_width,
                          resize_height=self.output_height,
                          resize_width=self.output_width,
                          crop=self.crop,
                          grayscale=self.grayscale) for batch_file in batch_files]
            fake_examples = np.array(batch).astype(np.float32)
          else:
            fake_examples = batch_images
          
        else:
          batch_images = self.shapes._get_shape(idx)[0][:self.batch_size]
          class_examples = self.shapes._get_shape(idx)[0][self.batch_size:]
          fake_examples = self.shapes._get_shape(idx+1000)[0][self.batch_size:]

        batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
              .astype(np.float32)
              
        feed_dict = { 
                      self.inputs: batch_images,
                      self.class_examples: batch_images[:self.example_num],
                      self.fake_examples: fake_examples[:self.example_num],
                      self.z: batch_z,
                    }

        # Update D network
        _, summary_str = self.sess.run([d_optim, self.d_sum],
          feed_dict=feed_dict)
        self.writer.add_summary(summary_str, counter)

        # Update G network
        _, summary_str = self.sess.run([g_optim, self.g_sum],
          feed_dict=feed_dict )
        self.writer.add_summary(summary_str, counter)
        
        # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
        _, summary_str = self.sess.run([g_optim, self.g_sum],
          feed_dict=feed_dict )
        self.writer.add_summary(summary_str, counter)
        
        errD_fake = self.d_loss_fake.eval(feed_dict)
        errD_real = self.d_loss_real.eval(feed_dict)
        errG = self.g_loss.eval(feed_dict)

        # Step
        counter += 1
        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
          % (epoch, idx, batch_idxs,
            time.time() - start_time, errD_fake+errD_real, errG))
            
        if np.mod(counter, config.sample_step) == 1:
          if not self.shapesdataset:
          
            for i in range(num_sample_classes):
              samples = self.sess.run(
                self.sampler,
                feed_dict={
                    self.z: sample_z,
                    self.class_examples: sample_inputs[i],
                })
              
              save_images(samples, image_manifold_size(samples.shape[0]),
                    './{}/train_{:02d}_{:04d}_{}.png'.format(\
                    config.sample_dir, epoch, idx, self.names[i]))
            
          else:
            
            c = ['r','g','b','c']
            
            for i in range(num_sample_classes):
              shape = self.sess.run(
                  self.sampler,
                  feed_dict={
                      self.z: sample_z,
                      self.class_examples: sample_inputs[i],
                  })
              s = np.transpose(shape)
              plt.scatter(s[0], s[1], 40, c='r')
              
              s = np.transpose(sample_inputs[i])
              plt.scatter(s[0], s[1], 40, c='g')

              plt.savefig('./{}/train_{:02d}_{:04d}_{}.png'.format(\
                      config.sample_dir, epoch, idx, i))
              plt.close()
            

        if np.mod(counter, 500) == 2:
          self.save(config.checkpoint_dir, counter)
          
          
  def test(self, config):
    self.sample_num=self.batch_size
    sample_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
    
    for i, data in enumerate(self.data):
      print self.names[i]
      if len(data) > self.example_num:
        batch_files = data[:self.example_num]
        batch = [
            get_image(batch_file,
                      input_height=self.input_height,
                      input_width=self.input_width,
                      resize_height=self.output_height,
                      resize_width=self.output_width,
                      crop=self.crop,
                      grayscale=self.grayscale) for batch_file in batch_files]
        sample_inputs = np.array(batch).astype(np.float32)
        
        samples = self.sess.run(
          self.sampler,
          feed_dict={
              self.z: sample_z,
              self.class_examples: sample_inputs,
          },
        )
        
        save_images(sample_inputs, image_manifold_size(sample_inputs.shape[0]),
              './{}/{}_examples.png'.format(\
              "samples", self.names[i]))
              
        save_images(samples, image_manifold_size(samples.shape[0]),
              './{}/{}_samples.png'.format(\
              "all_samples", self.names[i]))
                

  def discrim_embeddings_(self, image, reuse=False):
    with tf.variable_scope("discrim_embedding") as scope:
      if reuse:
        scope.reuse_variables()
      h3_s = self.discrim_embedder(image, reuse=reuse)
      h3_p = tf.reduce_mean(h3_s, axis=0, keep_dims=True)
      h4 = lrelu(linear(h3_p, 128, 'd_e_h4_lin'))
      h5 = linear(h4, self.y_dim, 'd_e_h5_lin')
      return h5
      
  def discrim_embedder(self, image, reuse=False, name=""):
    with tf.variable_scope("discrim_embedder"+name) as scope:
      if reuse:
        scope.reuse_variables()
        
      #dim_1 = tf.shape(image)[0]
      s_h, s_w = self.output_height, self.output_width
      s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
      s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
      s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
      s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

      h0 = lrelu(conv2d(image, self.df_dim, name='d_e_h0_conv'))
      h1 = lrelu(conv2d(h0, self.df_dim*2, name='d_e_h1_conv'))
      h2 = lrelu(conv2d(h1, self.df_dim*4, name='d_e_h2_conv'))
      h3 = lrelu(conv2d(h2, self.df_dim*8, name='d_e_h3_conv'))
      h3_s = tf.reshape(h3, [-1,  s_h16*s_w16*512])
      return h3_s
      
  def discrim_embeddings(self, image, reuse=False):
    with tf.variable_scope("discrim_embedding") as scope:
      if reuse:
        scope.reuse_variables()

      h0 = tf.nn.relu(linear(image, 256, 'd_h0_lin'))
      h1 = tf.nn.relu(linear(h0, 256, 'd_h1_lin'))
      h1_ = tf.reduce_max(h1, axis=0, keep_dims=True)
      h2 = linear(h1_, 256, 'd_h2_lin')
      h3 = linear(h2, 256, 'd_h3_lin')
      return h2
      
      
  def discriminator(self, image_e, sample_image_e, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      #h3y_s = self.discrim_embedder(sample_image, reuse=reuse)
      h3y_s = sample_image_e
      h3y_p = tf.reduce_max(h3y_s, axis=0, keep_dims=True)
      h4y = lrelu(linear(h3y_p, 1024, 'd_h4y_lin'))
      
      
      #h3_s = self.discrim_embedder(image, reuse=True)
      h3_s = image_e
      h3_p = h3_s - tf.reduce_max(h3_s, axis=0, keep_dims=True)
      h4 = lrelu(linear(h3_p, 1024, 'd_h4_lin'))
      h4_ = h4# - tf.reduce_max(h4, axis=0, keep_dims=True)
      h5 = lrelu(linear(h4_-h4y, 1024, 'd_h5_lin'))
      h6 = linear(h5, 1, 'd_h7_lin')
      
      return tf.nn.sigmoid(h6), h6
      
  def shapes_discriminator(self, image, y=None, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      h0 = lrelu(linear(image, 1024, 'd_h0_lin')+linear(y, 1024, 'd_hy_lin'))
      h1 = lrelu(linear(h0, 1024, 'd_h1_lin'))
      h1_ = h1 - tf.reduce_max(h1, axis=0, keep_dims=True)
      h2 = lrelu(linear(h1_, 1024, 'd_h2_lin'))
      h2_ = h2# - tf.reduce_max(h2, axis=0, keep_dims=True)
      h3 = lrelu(linear(h2_, 1024, 'd_h3_lin'))
      h3_ = h3# - tf.reduce_max(h3, axis=0, keep_dims=True)
      h4 = linear(h3_, 1, 'd_h4_lin')
      
      return tf.nn.sigmoid(h4), h4
      
      
  def gen_embeddings(self, image, reuse=False):
    with tf.variable_scope("gen_embedding") as scope:
      if reuse:
        scope.reuse_variables()
        
      s_h, s_w = self.output_height, self.output_width
      s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
      s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
      s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
      s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

      h0 = tf.nn.relu(conv2d(image, self.df_dim, name='g_e_h0_conv'))
      h1 = tf.nn.relu(conv2d(h0, self.df_dim*2, name='g_e_h1_conv'))
      h2 = tf.nn.relu(conv2d(h1, self.df_dim*4, name='g_e_h2_conv'))
      h3 = tf.nn.relu(conv2d(h2, self.df_dim*8, name='g_e_h3_conv'))
      h3_s = tf.reshape(h3, [-1, s_h16*s_w16*64])
      h4 = linear(h3_s, self.y_dim, 'g_e_h4_lin')
      h4_p = tf.reduce_max(h4, axis=0, keep_dims=True)
      return h4_p
      
      
  def shape_embeddings(self, image, reuse=False):
    with tf.variable_scope("gen_embedding") as scope:
      if reuse:
        scope.reuse_variables()

      h0 = tf.nn.relu(linear(image, 256, 'g_h0_lin'))
      h1 = tf.nn.relu(linear(h0, 256, 'g_h1_lin'))
      h1_ = tf.reduce_max(h1, axis=0, keep_dims=True)
      h2 = linear(h1_, 256, 'g_h2_lin')

      return h2
      
      

  def generator(self, z, y=None):
    with tf.variable_scope("generator") as scope:
      if self.shapesdataset:
        return self.points_generator(z, y)
      return self.new_generator(z, y)
      
  def sampler(self, z, y=None):
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()
      if self.shapesdataset:
        return self.points_generator(z, y)
      return self.new_generator(z, y)
    
  def new_generator(self, z, y=None):
      s_h, s_w = self.output_height, self.output_width
      s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
      s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
      s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
      s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

      method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
      
      # project `z` and reshape

      z_0 = linear(z, self.y_dim, 'g_h0z_lin')
      z_1 = linear(
          z_0+y, self.gf_dim*8*s_h16*s_w16, 'g_h1z_lin')
      
      h0 = tf.reshape(
          z_1, [-1, s_h16, s_w16, self.gf_dim * 8])
      h0 = tf.nn.relu(self.g_bn0(h0))
      
      y_0 = tf.nn.relu(linear(y, self.y_dim, 'g_h0y_lin'))
      yb = tf.reshape(y_0, [1, 1, 1, self.y_dim])
      h0 = conv_cond_concat(h0, yb)

      h0_up = tf.image.resize_images(h0, (s_h8, s_w8), method=method)
      h1 = tf.nn.relu(self.g_bn1(conv2d(h0_up, self.gf_dim*4, 
          k_h=5, k_w=5, d_h=1, d_w=1, name='g_c1')))
      h1 = conv_cond_concat(h1, yb)

      h1_up = tf.image.resize_images(h1, (s_h4, s_w4), method=method) 
      h2 = tf.nn.relu(self.g_bn2(conv2d(h1_up, self.gf_dim*2, 
          k_h=5, k_w=5, d_h=1, d_w=1, name='g_c2')))
      h2 = conv_cond_concat(h2, yb)

      h2_up = tf.image.resize_images(h2, (s_h2, s_w2), method=method) 
      h3 = tf.nn.relu(self.g_bn3(conv2d(h2_up, self.gf_dim, 
          k_h=5, k_w=5, d_h=1, d_w=1, name='g_c3')))
      h3 = conv_cond_concat(h3, yb)
      
      #additional conv layers
      h3 = tf.nn.relu(self.g_bn4(conv2d(h3, self.gf_dim, 
          k_h=5, k_w=5, d_h=1, d_w=1, name='g_extra_conv')))
      #h3 = conv_cond_concat(h3, yb)
      
      h3_up = tf.image.resize_images(h3, (s_h, s_w), method=method) 
      h4 = conv2d(h3_up, self.c_dim, 
          k_h=5, k_w=5, d_h=1, d_w=1, name='g_c4')
      
      return tf.nn.tanh(h4)
      
      
  def points_generator(self, z, y=None):
      z_ = linear(z, 256, 'g_h0z_lin')
      h0 = tf.nn.relu(z_+y)
      h1 = tf.nn.relu(linear(h0, 1024, 'g_h1_lin'))
      h2 = tf.nn.relu(linear(h1, 1024, 'g_h2_lin'))
      h3 = linear(h2, 2, 'g_h3_lin')
      return h3
      
      
  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.output_height, self.output_width)
      
  def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0
      
      
# -----------------------------------------------------------------------------------
#     JS-Regularizer
# -----------------------------------------------------------------------------------
  def Discriminator_Regularizer(self, D1, D1_logits, D1_arg, D2, D2_logits, D2_arg):
    with tf.name_scope('disc_reg'):
      batch_size = self.batch_size
      self.gamma = 5
      grad_D1_logits = tf.gradients(D1_logits, D1_arg)[0]
      grad_D2_logits = tf.gradients(D2_logits, D2_arg)[0]
      grad_D1_logits_norm = tf.norm(
        tf.reshape(grad_D1_logits, [batch_size,-1]) , axis=1, keep_dims=True)
      grad_D2_logits_norm = tf.norm(
        tf.reshape(grad_D2_logits, [batch_size,-1]) , axis=1, keep_dims=True)
    
      reg_D1 = tf.multiply(tf.square(1.0-D1), tf.square(grad_D1_logits_norm))
      reg_D2 = tf.multiply(tf.square(D2), tf.square(grad_D2_logits_norm))
    
      self.disc_regularizer = tf.reduce_mean(reg_D1 + reg_D2)
      
      return self.disc_regularizer
