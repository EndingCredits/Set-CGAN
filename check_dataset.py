import os
import time
import math
from glob import glob
import numpy as np

from tqdm import tqdm

from utils import *

dataset_name = "celebA"

paths = glob(os.path.join("./data", dataset_name, "*/"))
#print paths
names = []
data = []
for path in paths:
  _, n = os.path.split(os.path.split(path)[0])
  #print n
  names.append(n)
  d = glob(os.path.join(path, "*.jpg"))
  data.append(d)

input_height = input_width = 192
output_height = output_width = 64

wrong_files = []
  
for i, _ in enumerate(data):
  print names[i]
  for j in tqdm(range(len(data[i]))):
    sample_files = data[i][j:(j+1)]
    try:
        sample = [
            get_image(sample_file,
                      input_height=input_height,
                      input_width=input_width,
                      resize_height=output_height,
                      resize_width=output_width,
                      crop=True,
                      grayscale=False) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)
        assert np.shape(sample_images) == (1, output_height, output_width, 3)
    except:
        print sample_files
        wrong_files = wrong_files + sample_files
        
print wrong_files
        

