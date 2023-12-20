"""
  Basic run script for the WeedAI dataset.
"""

import argparse
import yaml

import torch

from libs.WeedAI_dataloader import Parser

# Parse the command line arguments
def parsecmdline():
  parser = argparse.ArgumentParser( description='Extracting command line arguments', add_help=True )
  # config file,
  parser.add_argument( '--config', action='store', default='configs/WeedAI_plant.yml' )
  return parser.parse_args()

if __name__ == "__main__":
  # parse the arguments
  flags = parsecmdline()
  # get the config file
  with open( flags.config, 'r' ) as fid:
    cfg = yaml.load( fid, Loader=yaml.FullLoader )
  # create the dataloaders
  dl = Parser( cfg )
  # get the three dataloaders from the parser, depending on the config file these can be None.
  train = dl.train_dataloader()
  valid = dl.val_dataloader()
  infer = dl.test_dataloader()
  for t in train:
    #              0 is the first in the batch.
    rgb = t['rgb'][0] # normalised image
    print( rgb.shape )
    smap = t['gt'][0,0,:,:] # panoptic semantic map
    print( smap.shape )
    imap = t['gt'][0,1,:,:] # panoptic instance map
    print( imap.shape )
    cnt = t['cnt'][0] # the center ground truth
    print( cnt.shape )
    reg = t['reg'][0] # the regression to center ground truth
    print( reg.shape )
    break
