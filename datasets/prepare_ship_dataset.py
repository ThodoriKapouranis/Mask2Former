import logging
import os
import getopt
import sys
from absl import flags, app
import re
'''
Example Run:
python datasets/prepare_ship_dataset.py --path='/workspaces/final-dl/datasets/ShipRSImageNet_V1'
'''

# The dataset comes with 4 levels of categorization specificity
# 0 : Dock, Ship
# 1 : Dock, 3 ship categories
# 2 : Dock, 24 ship categories
# 3 : Dock, 49 ship categories

# Define flags / Command Line Args
FLAGS = flags.FLAGS
flags.DEFINE_string("path", None, "Number of features in record")
flags.DEFINE_integer("level", 3, "Level of category specification")
flags.DEFINE_bool("debug", False, "Set logging level to debug")
  
def main(argv):
  '''
  This function returns 
    - The folder with the formatted test, train, val annotations.
    - The directory to the image reference folder used
  '''
  logging.basicConfig()

  if FLAGS.debug:
      logging.getLoggers().setLevel(logging.DEBUG)

# The datasets are assumed to exist in a directory specified by the environment variable `DETECTRON2_DATASETS`.
  os.environ['DETECTRON2_DATASETS'] = "/workspaces/final-dl/datasets"
  
  formatted_dir = 'datasets/ShipRSImageNet'
  
  for dir in [
    formatted_dir, 
      f'{formatted_dir}/annotations', 
  ]:
    try: 
      os.mkdir(dir)
    except FileExistsError: 
      pass  
      
  initialize_new_dir(formatted_dir)

def initialize_new_dir(formatted_dir):
  # Copy the requested annotation files over to the new directory
  for file in os.listdir(f'{FLAGS.path}/COCO_Format'):
  # Regex to only grab the files for the given FLAGS.level specification
    match = re.search(f'(train|test|val)(_level_)?({FLAGS.level})', file)
  
    if not match==None:
      # Write the correct level files to our formatted_dir
      copy = open(f'{formatted_dir}/annotations/{match.group(1)}.json', 'w')
      original =  open(f'{FLAGS.path}/COCO_Format/{file}', "r")
      copy.write(original.read())
      
  # Symlink the image folder 
  try:
    os.symlink( src=f'{FLAGS.path}/VOC_Format/JPEGImages', dst=f'{formatted_dir}/images')
  except FileExistsError:
    pass
               
if __name__ == "__main__":
  app.run(main)
  
  
  
  
  