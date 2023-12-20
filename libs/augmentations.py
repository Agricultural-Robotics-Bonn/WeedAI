"""
Some transform examples for this data.
"""

import numpy
from PIL import Image
from scipy.ndimage.morphology import binary_dilation
import skimage.transform
from skimage import filters
from skimage import feature
from skimage.draw import disk, rectangle
from skimage.color import rgb2lab, lab2rgb
from skimage.transform import rotate
import sys
import torch
import time

class MeanStdNorm(object):
  """Normalise the image by the mean and standard deviation

  Args:
    RGB_mean_arr: the mean of the images to be normalised to.
    RGB_std_arr:  the std dev of the images to be normalised to.
  """

  def __init__(self, RGB_mean_arr=numpy.array([0.5, 0.5, 0.5]),
                     RGB_std_arr=numpy.array([0.5, 0.5, 0.5]),
              ):
    self.RGB_mean_arr = RGB_mean_arr
    self.RGB_std_arr = RGB_std_arr

  def __call__(self, sample):
    image, GT = sample['rgb'], sample['gt']

    image = (image - self.RGB_mean_arr)/self.RGB_std_arr

    ret_dict = dict(
                rgb=image,
                gt=GT,
                )
    if 'boxes' in sample.keys():
      ret_dict['boxes'] = sample['boxes']
    for k in sample.keys():
      if k not in ['rgb', 'gt', 'boxes']:
        ret_dict[k] = sample[k]
    return ret_dict

class Rescale(object):
  """Rescale the image in a sample to a given size.

  Args:
    output_size (tuple or int): Desired output size. If tuple, output is
        matched to output_size. If int, smaller of image edges is matched
        to output_size keeping aspect ratio the same.
  """

  def __init__( self, output_size, min_pixels=20, ispanoptic=False ):
    assert isinstance(output_size, (int, tuple, list))
    self.output_size = output_size
    self.min_pixels = min_pixels
    self.ispanoptic = ispanoptic

  def __call__( self, sample ):
    image, GT = sample['rgb'], sample['gt']
    sub_labels = sample['sub_labels'] if 'sub_labels' in sample.keys() else None
    h, w = image.shape[:2]
    if isinstance(self.output_size, int):
      if h > w:
        new_h, new_w = self.output_size * h / w, self.output_size
      else:
        new_h, new_w = self.output_size, self.output_size * w / h
    else:
      new_h, new_w = self.output_size

    new_h, new_w = int(new_h), int(new_w)
    img = skimage.transform.resize( image, (new_h, new_w) )
    if GT is not None:
      GT = skimage.transform.resize( GT, (new_h, new_w), anti_aliasing=False,
                               anti_aliasing_sigma=None,
                               order = 0, preserve_range=True ) #.astype('uint8')

    # now remove gt's that are no longer in the image.
    if self.ispanoptic or len( GT.shape )==2:
      omask = GT
    else:
      sls, omask = None, None
      for g in range( GT.shape[2] ):
        (x,y) = numpy.where( GT[:,:,g]>0 )
        ux, uy = numpy.unique( x ), numpy.unique( y )
        if ux.shape[0] > 1 and uy.shape[0] > 1 and x.shape[0] > self.min_pixels:
          try:
            omask = numpy.dstack( (omask, GT[:,:,g]) )
            if sub_labels is not None:
              sls.append( sub_labels[g] )
          except:
            omask = GT[:,:,g]
            if sub_labels is not None:
              sls = [sub_labels[g]]

    if omask is None:
      omask = numpy.zeros( (rgb.shape[0], rgb.shape[1], 1) )
    ret_dict = dict(
                rgb=img, gt=omask
                )
    if sub_labels is not None:
      ret_dict['sub_labels'] = sls

    for k in sample.keys():
      if k not in ['rgb', 'gt', 'sub_labels']:
        ret_dict[k] = sample[k]
    return ret_dict

class RandomCrop( object ):
  def __init__( self, height, width, min_pixels=20, ispanoptic=False, p=0.5 ):
    self.height = height
    self.width = width
    self.min_pixels = min_pixels
    self.ispanoptic = ispanoptic
    self.p = p

  def _normalget( self, sample ):
    # st = time.time()
    rgb, gt = sample['rgb'], sample['gt']
    sub_labels = sample['sub_labels'] if 'sub_labels' in sample.keys() else None
    s = sample.copy()
    h, w = rgb.shape[:2]
    # get the new locations
    top = numpy.random.randint( 0, h - self.height ) if h-self.height > 0 else 0
    left = numpy.random.randint( 0, w - self.width ) if w-self.width > 0 else 0
    # crop
    rgb = rgb[top:top+self.height,
                      left:left+self.width]
    gt = gt[top:top+self.height,
                      left:left+self.width]
    s['rgb'] = rgb
    # now remove gt's that are no longer in the image.
    if len( gt.shape ) == 3: # instances
       sls, omask = None, None
       for g in range( gt.shape[2] ):
         (x,y) = numpy.where( gt[:,:,g]>0 )
         ux, uy = numpy.unique( x ), numpy.unique( y )
         if ux.shape[0] > 1 and uy.shape[0] > 1 and x.shape[0] > self.min_pixels:
           try:
             omask = numpy.dstack( (omask, gt[:,:,g]) )
             if sub_labels is not None:
               sls.append( sub_labels[g] )
           except:
             omask = gt[:,:,g]
             if sub_labels is not None:
               sls = [sub_labels[g]]
       if omask is None:
         omask = numpy.zeros( (rgb.shape[0], rgb.shape[1], 1) )
       s['gt'] = omask
    else: # it's semantic
      s['gt'] = gt

    if sub_labels is not None:
      s['sub_labels'] = sls
    # now get the new boxes if they already exist
    if 'boxes' in sample.keys():
      boxes = []
      for i in range( omask.shape[2] ):
        pos = numpy.where( omask[:,:,i] )
        xmin = numpy.min( pos[1] )
        xmax = numpy.max( pos[1] )
        ymin = numpy.min( pos[0] )
        ymax = numpy.max( pos[0] )
        boxes.append( [xmin, ymin, xmax, ymax] )
      boxes = numpy.array( boxes )
      s['boxes'] = boxes
    return s

  def _panopticget( self, sample ):
    s = sample.copy()
    rgb, gt = sample['rgb'], sample['gt']
    h, w = rgb.shape[:2]
    # get the new locations
    top = numpy.random.randint( 0, h - self.height ) if h-self.height > 0 else 0
    left = numpy.random.randint( 0, w - self.width ) if w-self.width > 0 else 0
    # crop
    rgb = rgb[top:top+self.height,
                      left:left+self.width]
    gt = gt[top:top+self.height,
                      left:left+self.width]
    # store
    s['rgb'] = rgb
    s['gt'] = gt
    # return
    return s

  def __call__( self, sample ):
    v = numpy.random.uniform()
    if v <= self.p:
      if self.ispanoptic:
        return self._panopticget( sample )
      else:
        return self._normalget( sample )
    else:
      return sample

class RandomFlip( object ):
  def __init__( self, mode='lr', p=0.5 ):
    self.mode = mode
    self.p = p

  def lr( self, sample ):
    s = sample.copy()
    rgb, gt = sample['rgb'], sample['gt']
    s['rgb'] = numpy.fliplr( rgb ).copy()
    s['gt'] = numpy.fliplr( gt ).copy()
    # boxes
    if 'boxes' in sample.keys():
      x = rgb.shape[1]
      boxes = sample['boxes']
      boxes[:,2] = x - boxes[:,0]
      boxes[:,0] = x - boxes[:,2]
      s['boxes'] = boxes
    return s

  def ud( self, sample ):
    s = sample.copy()
    rgb, gt = sample['rgb'], sample['gt']
    s['rgb'] = numpy.flipud( rgb ).copy()
    s['gt'] = numpy.flipud( gt ).copy()
    # boxes
    if 'boxes' in sample.keys():
      y = rgb.shape[0]
      boxes = sample['boxes']
      boxes[:,3] = y - boxes[:,1]
      boxes[:,1] = y - boxes[:,3]
      s['boxes'] = boxes
    return s

  def __call__( self, sample ):
    if numpy.random.uniform() <= self.p:
      if self.mode == 'lr':
        return self.lr( sample )
      elif self.mode == 'ud':
        return self.ud( sample )
      else:
        assert False, "Random Flip transform wrong mode"

      return sample
    else:
      return sample

class ColourJitter( object ):
  """
    Colour jitter.

    Expects the image to be on the range 0. to 1.

    Remember that rbg to lab converts from 0 to 1 for the rgb to:
      L = 0 to 100
      a = -127 to 127
      b = -127 to 127
    Then lab to rgb coverts back to 0. to 1.
  """

  def __init__( self, mode='rgb', range=0.05, p=0.5 ):
    self.mode = mode
    self.range = range
    self.p = p

  def rgbjitter( self, rgb ):
    rgb[:,:,0] += numpy.random.uniform( -self.range, self.range )
    rgb[:,:,1] += numpy.random.uniform( -self.range, self.range )
    rgb[:,:,2] += numpy.random.uniform( -self.range, self.range )
    rgb = numpy.clip( rgb, 0., 1. )
    return rgb

  def ljitter( self, rgb ):
    lab = rgb2lab( rgb )
    l = numpy.random.randint( -self.range, self.range )
    lab[:,:,0] += l
    lab[:,:,0] = numpy.clip( lab[:,:,0], 0, 100 )
    return lab2rgb( lab )

  def labcoljitter( self, rgb ):
    return rgb

  def __call__( self, sample ):
    rgb = sample['rgb']
    if numpy.random.uniform() <= self.p:
      if self.mode == 'rgb':
        rgb = self.rgbjitter( rgb )
      elif self.mode == 'l':
        rgb = self.ljitter( rgb )
      else:
        assert False, 'Wrong mode to Colour Jitter transform mode!'

      sample['rgb'] = rgb
    return sample

class RandomRotation( object ):
  def __init__( self, degree, p=0.5, ispanoptic=False ):
    self.degree = degree
    self.p = p
    self.ispanoptic = ispanoptic

  def _normalget( self, sample ):
    s = sample.copy()
    rgb, gt = sample['rgb'], sample['gt']
    d = numpy.random.uniform( -self.degree, self.degree )
    s['rgb'] = rotate( rgb, d )
    gt = rotate( gt, d )
    # now remove gt's that are no longer in the image.
    sls, omask = None, None
    for g in range( gt.shape[2] ):
      (x,y) = numpy.where( gt[:,:,g]>0 )
      ux, uy = numpy.unique( x ), numpy.unique( y )
      if ux.shape[0] > 1 and uy.shape[0] > 1 and x.shape[0] > self.min_pixels:
        try:
          omask = numpy.dstack( (omask, gt[:,:,g]) )
          if sub_labels is not None:
            sls.append( sub_labels[g] )
        except:
          omask = gt[:,:,g]
          if sub_labels is not None:
            sls = [sub_labels[g]]
    if omask is None:
      omask = numpy.zeros( (rgb.shape[0], rgb.shape[1], 1) )
    s['gt'] = omask
    if sub_labels is not None:
      s['sub_labels'] = sls
    # now get the new boxes if they already exist
    if 'boxes' in sample.keys():
      boxes = []
      for i in range( omask.shape[2] ):
        pos = numpy.where( omask[:,:,i] )
        xmin = numpy.min( pos[1] )
        xmax = numpy.max( pos[1] )
        ymin = numpy.min( pos[0] )
        ymax = numpy.max( pos[0] )
        boxes.append( [xmin, ymin, xmax, ymax] )
      boxes = numpy.array( boxes )
      s['boxes'] = boxes
    return s

  def _panopticget( self, sample ):
    s = sample.copy()
    rgb, gt = sample['rgb'], sample['gt']
    d = numpy.random.uniform( -self.degree, self.degree )
    s['rgb'] = rotate( rgb, d )
    s['gt'] = rotate( gt, d, preserve_range=True ).astype( numpy.uint8 )
    return s

  def __call__( self, sample ):
    if numpy.random.uniform() <= self.p:
      if self.ispanoptic:
        return self._panopticget( sample )
      else:
        return self._normalget( sample )
    else:
      return sample

class Edge( object ):
  """
    Creates an edge map from the ground truth image.

    !!!NOTE: Should be used after the reshape!!!!!!
  """
  def __init__( self, dilations=0, semantic=True, boundary_type='full', blur=None, ispanoptic=False ):
    self.dilations = dilations
    self.outname = 'edg'
    self.boundary_type = boundary_type
    self.semantic = semantic
    self.blur = blur
    self.ispanoptic = ispanoptic

  def _domaskextract( self, sample ):
    # get the ground truth mask(s)
    GT = sample['gt']
    # ensure that we have 3 channels
    GT = numpy.expand_dims( GT, axis=2 ) if len( GT.shape ) == 2 else GT
    # convert for iteration
    GT = numpy.transpose( GT, (2,0,1) )
    # iterate over the gths
    for gt in GT:
      # edge on mask - save it as an integer not boolean
      edge = feature.canny( gt ).astype( int )
      # are we dilating and or masking?
      if self.dilations > 0:
        edge = binary_dilation( edge, iterations=self.dilations ).astype( edge.dtype )
        if self.boundary_type == 'mask_fg':
          edge[gt>0] = 0
        elif self.boundary_type == 'mask_bg':
          edge[gt==0] = 0
      if self.blur is not None:
        edge = filters.gaussian( edge, self.blur, preserve_range=True )
        edge /= edge.max()
      try:
        edges = numpy.dstack( (edges, edge) )
      except:
        edges = edge
    # is this a semantic mask?
    edges = numpy.expand_dims( edges, axis=2 ) if len( edges.shape ) == 2 else edges
    if self.semantic:
      edges = numpy.clip( numpy.amax( edges, axis=2 ), 0., 1. )
      edges = numpy.expand_dims( edges, axis=2 )
    # return the output
    sample[self.outname] = edges
    return sample

  def _dopanopticextract( self, sample ):
    # get the imap
    imap = sample['gt'][:,:,1]
    # define the edges map
    un = numpy.unique( imap )[1:]
    # create the edge map
    if un.shape[0] > 0:
      edges = numpy.zeros( (imap.shape[0], imap.shape[1], un.shape[0]) )
    else:
      edges = numpy.zeros( (imap.shape[0], imap.shape[1]) )
    # iterate over the unique values
    for i, u in enumerate( un ):
      smap = numpy.zeros_like( imap )
      smap[imap==u] = 255
      # get the edge
      edge = feature.canny( smap ).astype( int )
      # are we doing dilations?
      if self.dilations > 0:
        edge = binary_dilation( edge, iterations=self.dilations ).astype( edge.dtype )
      # are we bluring
      if self.blur is not None:
        edge = filters.gaussian( edge, self.blur, preserve_range=True )
        if edge.max() > 0:
          edge /= edge.max()
      # are we masking out the background or foreground?
      if self.boundary_type == 'fg':
        edge[smap>0] = 0
      elif self.boundary_type == 'bg':
        edge[smap==0] = 0
      # store the edge in the matrix
      edges[:,:,i] = edge
    # now create the semantic edges.
    if un.shape[0] > 0:
      edges = numpy.amax( edges, axis=2 )
    # store in the sample and return
    sample[self.outname] = edges
    return sample


  def __call__( self, sample ):
    if self.ispanoptic:
      return self._dopanopticextract( sample )
    else:
      return self._domaskextract( sample )

class Panoptic( object ):
  """
    Creates the center and regression from the panoptic imap
    !!! Note, do this after reshape crop etc.
  """

  def __init__( self, radius=1, blur=[9,9] ):
    self.radius = radius
    self.blur = blur

  def __call__( self, sample ):
    # get the ground truth imap
    imap = sample['gt'][:,:,1]
    # storage
    first = True
    tags = numpy.zeros( (imap.shape[0], imap.shape[1]) )
    reg = numpy.zeros( (imap.shape[0], imap.shape[1], 2) )
    # create the centers
    for u in numpy.unique( imap ):
      if u == 0:
        continue
      rr, cc = numpy.where( imap==u )
      r, c = int( rr.mean() ), int( cc.mean() )
      # create the radius
      r, c = disk( (r,c), self.radius, shape=imap.shape )
      # draw
      tag = numpy.zeros( (imap.shape[0], imap.shape[1]) )
      tag[r,c] = 1
      # are we blurring?
      if self.blur:
        tag = filters.gaussian( tag, self.blur, preserve_range=True )
        tag /= tag.max()
      # store the tags channelwise
      if first:
        tags = tag
        first = False
      else:
        tags = numpy.dstack( (tags, tag) )
      # create the regression
      cr = (rr.max() - rr.min())//2 + rr.min()
      cn = (cc.max() - cc.min())//2 + cc.min()
      # install the values into the tregression
      reg[rr,cc,0] = (cr-rr)
      reg[rr,cc,1] = (cn-cc)
    # get the centers as a single channel
    if len( tags.shape ) > 2:
      tags = numpy.clip( numpy.amax( tags, axis=2 ), 0., 1. )
    # store the values
    sample['cnt'] = tags
    sample['reg'] = reg
    return sample

class ToTensor(object):
  """Convert ndarrays in sample to Tensors."""

  def __call__(self, sample):
    image, GT = sample['rgb'], sample['gt']
    image = image.transpose( (2,0,1) )
    if len( GT.shape ) > 2:
      GT = numpy.transpose( GT, (2,0,1) )
    ret_dict = {'rgb':torch.from_numpy( image.copy() ).float(),
                'gt':torch.from_numpy( GT.copy() ).to( torch.int64 )}#long()}

    if 'edg' in sample.keys():
      edge = numpy.transpose( sample['edg'], [2,0,1] ) if len( sample['edg'].shape ) > 2 else sample['edg']
      ret_dict['edg'] = torch.from_numpy( edge ).float() # long for xe
    if 'labels' in sample.keys():
      ret_dict['labels'] = torch.as_tensor( sample['labels'], dtype=torch.int64 )
    if 'sub_labels' in sample.keys():
      ret_dict['sub_labels'] = torch.as_tensor( sample['sub_labels'], dtype=torch.int64 )
    if 'cnt' in sample.keys():
      cnt = numpy.transpose( sample['cnt'], [2,0,1] ) if len( sample['cnt'].shape ) > 2 else sample['cnt']
      ret_dict['cnt'] = torch.from_numpy( cnt ).float()
    if 'reg' in sample.keys():
      reg = numpy.transpose( sample['reg'], [2,0,1] ) if len( sample['reg'].shape ) > 2 else sample['reg']
      ret_dict['reg'] = torch.from_numpy( reg ).float()
    for k in sample.keys():
      if k not in ['rgb', 'gt', 'edg', 'labels', 'sub_labels', 'reg', 'cnt']:
        ret_dict[k] = sample[k]
    return ret_dict
