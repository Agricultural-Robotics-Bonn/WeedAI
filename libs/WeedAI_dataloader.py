"""
  A basic panoptic dataloader from the paper:

  Michael Halstead, Patrick Zimmer, and Chris McCool. A Cross-Domain Challenge with Panoptic Segmentation in Agriculture The International Journal of Robotics Research

  Created MAH 20231220
"""
import os
import sys

import numpy
from skimage.io import imread

import yaml

from pycocotools import mask
from pycocotools.coco import COCO

import PIL

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tvtransforms
import pytorch_lightning as pl

# import dataloaders.modules.numpyTransforms as Mytrans
import libs.augmentations as Mytrans


# Disable
def blockPrint():
    sys.stdout = open( os.devnull, 'w' )
# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


class CoCo( Dataset ):

  def __init__(self,
              root,
              subset,
              dataset_name,
              transforms=None,
              class_type='plant', # plant, super, weed_unknown, subassuper
              extension='png',
              skip_LUT=False,
              ):

    print( '******************** WeedAI21 ******************' )

    self._root_dir = root
    self._extension = extension
    self.dataset_name = dataset_name
    self.subset = subset
    self.transforms = transforms

    assert self.subset == 'train' or self.subset == 'valid' or self.subset == 'eval'

    # Directories and root addresses
    self.annotation_files_dir = os.path.join( self._root_dir, 'datasets', self.dataset_name, self.dataset_name + '.json' )
    self.dataset_config_list_dir = os.path.join( self._root_dir, 'datasets', self.dataset_name, self.dataset_name + '.yaml' )
    with open(self.dataset_config_list_dir) as stream:
        self.dataset_config = yaml.safe_load( stream )
    self.image_sets =  self.dataset_config["image_sets"]

    blockPrint()
    # Initialize COCO api for instance annotations
    self.coco = COCO( self.annotation_files_dir )
    enablePrint()

    # create look up table for the classes or sub-classes
    if not skip_LUT:
      self.LUT = dict()
      for id, c in self.coco.cats.items():
        if class_type == 'plant':
          self.LUT[id] = 1
        elif class_type == 'super': # crop, weed, unknown
          if c['supercategory'] == 'crop':
            self.LUT[id] = 1
          elif c['supercategory'] == 'weed':
            self.LUT[id] = 2
          else:
            self.LUT[id] = 3
        elif class_type == 'weed_unknown':
          if c['supercategory'] == 'crop':
            self.LUT[id] = 1
          else:
            self.LUT[id] = 2
        elif class_type == 'subassuper':
          l = ['Ch', 'unknown', 'An', 'SB', 'VG', 'Ci', 'So', 'Sn', 'Ch_damaged', 'An_damaged', 'REVIEW', 'VG_damaged']
          self.LUT[id] = l.index( c['name'] )+1
        else:
          assert False, 'The class type is incorrect, only plant super or subassuper exists'

  def getTargetID(self, index):
    img_set_ids = self.image_sets[self.subset][index]
    img_list_idx = next( (index for (index, d) in enumerate( self.coco.dataset["images"] ) if d["id"] == img_set_ids), None )
    image_id = self.coco.dataset["images"][img_list_idx]["id"]
    return image_id

  def getImgGT( self, index ):
    # get meta data of called image (ID = index)
    self.img_id = self.getTargetID( index )
    self.img_metadata = self.coco.loadImgs( self.img_id )[0]
    path = self.img_metadata['path'][1:]
    img = imread( os.path.join( self._root_dir, path ) )/255.
    # load the masks and labels
    cat_ids = self.coco.getCatIds()
    anns_ids = self.coco.getAnnIds( imgIds=self.img_id, catIds=cat_ids, iscrowd=None )
    # get the annotations for the current image
    anns = self.coco.loadAnns(anns_ids)
    # get the semantic mask and the imap
    itr = 1
    semmask = numpy.zeros( (self.img_metadata['height'], self.img_metadata['width']) ).astype( numpy.uint8 )
    imap = numpy.zeros( (self.img_metadata['height'], self.img_metadata['width']) ).astype( numpy.uint8 )
    for ann in anns:
      if not ann['segmentation']:
        continue
      ann_mask = self.coco.annToMask( ann )
      semmask[ann_mask>0] = self.LUT[ann['category_id']]
      imap[ann_mask>0] = itr
      itr += 1
    panoptic = numpy.dstack( (semmask, imap) )
    return img, panoptic, path

  def __getitem__( self, index ):
    # get the image, panoptic, and path
    img, panoptic, img_path = self.getImgGT( index )
    # transform where required.
    sample = {'rgb':img.copy(), 'gt':panoptic}
    if self.transforms is not None:
      sample = self.transforms( sample )
    sample['RGB_fname'] = os.path.basename( img_path )
    return sample



  def __len__( self ):
    return len( self.image_sets[self.subset] )


class Parser( pl.LightningDataModule ):
  def __init__( self, config ):
    super( Parser, self ).__init__()
    # config file
    self.config = config
    # transforms (with augmentation)
    transforms = None
    if config['dataset']['transforms']['use']:
      tt, te = [], []
      for k, v in config['dataset']['transforms'].items():
        # mean standard deviation
        if k == 'meanstdnorm' and config['dataset']['transforms']['meanstdnorm']['use']:
          tt.append( Mytrans.MeanStdNorm( RGB_mean_arr=config['dataset']['transforms']['meanstdnorm']['RGB_mean_arr'],
                                            RGB_std_arr=config['dataset']['transforms']['meanstdnorm']['RGB_std_arr'] ) )
          te.append( Mytrans.MeanStdNorm( RGB_mean_arr=config['dataset']['transforms']['meanstdnorm']['RGB_mean_arr'],
                                            RGB_std_arr=config['dataset']['transforms']['meanstdnorm']['RGB_std_arr'] ) )
        # rescale
        if k == 'rescale' and v['use']:
          tt.append( Mytrans.Rescale( output_size=config['dataset']['transforms']['rescale']['output_size'], ispanoptic=config['dataset']['transforms']['rescale']['ispanoptic'] ) )
          te.append( Mytrans.Rescale( output_size=config['dataset']['transforms']['rescale']['output_size'], ispanoptic=config['dataset']['transforms']['rescale']['ispanoptic'] ) )
        if k == 'randomcrop' and config['dataset']['transforms']['randomcrop']['use']:
          tt.append( Mytrans.RandomCrop( height=v['height'], width=v['width'], ispanoptic=True, p=v['p'] ) )
        if k == 'randomflip' and v['use']:
          tt.append( Mytrans.RandomFlip( mode=v['mode'], p=v['p'] ) )
        if k == 'colourjitter' and v['use']:
          tt.append( Mytrans.ColourJitter( mode=v['mode'], range=v['range'], p=v['p'] ) )
        if k == 'randomrotate' and v['use']:
          tt.append( Mytrans.RandomRotation( v['degree'], p=v['p'], ispanoptic=True ) )
        # edge
        if k == 'edge' and v['use']:
          dilations = v.get( 'dilation', 0 )
          boundary_type = v['boundary_type'] if 'boundary_type' in v.keys() else 'full'
          tt.append( Mytrans.Edge( blur=v['blur'], ispanoptic=True, dilations=dilations, boundary_type=boundary_type ) )
          te.append( Mytrans.Edge( blur=v['blur'], ispanoptic=True, dilations=dilations, boundary_type=boundary_type ) )
        if k == 'panoptic' and config['dataset']['transforms']['panoptic']['use']:
          tt.append( Mytrans.Panoptic( radius=config['dataset']['transforms']['panoptic']['radius'],
                                  blur=config['dataset']['transforms']['panoptic']['blur'] ) )
          te.append( Mytrans.Panoptic( radius=config['dataset']['transforms']['panoptic']['radius'],
                                  blur=config['dataset']['transforms']['panoptic']['blur'] ) )
        # to tensor
        if k == 'totensor' and config['dataset']['transforms']['totensor']['use']:
          tt.append( Mytrans.ToTensor() )
          te.append( Mytrans.ToTensor() )
      if len( tt ) > 0:
        print(f"Train transforms: {[type(tts).__name__ for tts in tt]}")
        train_transforms = tvtransforms.Compose( tt )
      if len( te ) > 0:
        print(f"Infer transforms: {[type(its).__name__ for its in te]}")
        infer_transforms = tvtransforms.Compose( te )

    # other parameters for the dataloader
    dataset_name = config['dataset']['dataset_name'] if 'dataset_name' in config['dataset'] else ''
    extension = config['dataset']['extension'] if 'extension' in config['dataset'] else 'png'
    class_type = config['dataset']['class_type'] if 'class_type' in config['dataset'] else 'plant'
    # create the dataloader
    self.loadertrain, self.loadervalid, self.loadereval = None, None, None
    if 'train' in config['dataset']['subsets']:
      self.loadertrain = CoCo( config['dataset']['location'], 'train', config['dataset']['coconame'],
                      transforms=train_transforms,
                      extension=extension,
                      class_type=class_type,
                      )
      print( '******* Training Samples =', len( self.loadertrain ), '*************' )

    if 'valid' in config['dataset']['subsets']:
      self.loadervalid = CoCo( config['dataset']['location'], 'valid', config['dataset']['coconame'],
                      transforms=infer_transforms,
                      extension=extension,
                      class_type=class_type,
                      )
      print( '******* Validation Samples =', len( self.loadervalid ), '*************' )

    if 'eval' in config['dataset']['subsets']:
      self.loadereval = CoCo( config['dataset']['location'], 'eval', config['dataset']['coconame'],
                      transforms=infer_transforms,
                      extension=extension,
                      class_type=class_type,
                      )
      print( '******* Evaluation Samples =', len( self.loadereval ), '*************' )

  # get the training parser
  def train_dataloader( self ):
    if self.loadertrain is None:
      return self.loadertrain
    return DataLoader( self.loadertrain,
                        batch_size=self.config['dataloader']['batch_size'],
                        shuffle=self.config['dataloader']['shuffle'],
                        num_workers=self.config['dataloader']['workers_num'],
                        drop_last=self.config['dataloader']['drop_last'], )

  # get the validation parser
  def val_dataloader( self ):
    if self.loadervalid is None:
      return self.loadervalid
    return DataLoader( self.loadervalid,
                        batch_size=self.config['dataloader']['batch_size'],
                        shuffle=False,
                        num_workers=self.config['dataloader']['workers_num'],
                        drop_last=False, )

  # get the evaluation parser
  def test_dataloader( self ):
    if self.loadereval is None:
      return self.loadereval
    return DataLoader( self.loadereval,
                        batch_size=self.config['dataloader']['batch_size'],
                        shuffle=False,
                        num_workers=self.config['dataloader']['workers_num'],
                        drop_last=False, )
