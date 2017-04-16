import sys
sys.path.append('./support')
from paths import *
from glob import glob
from os.path import join, basename, isfile
from numpy import *
from multiprocessing import Pool
from functools import partial

def extract_paths(valid_fold, test_fold):
    file_list = set([path for path in glob(join(PATH['NODULES'], 
                                                'subset*', '*.npy'))]
                    + [path for path in glob(join(PATH['VESSELS'], 
                                                  'subset*', '*.npy'))])

    test = [path for path in file_list 
            if ''.join(['subset', test_fold]) in path]

    file_list = list(file_list.difference(test))
    valid = sorted([path for path in file_list 
                    if ''.join(['subset', valid_fold]) in path])
    return asarray(file_list), asarray(valid), asarray(test)


def manipulate_samples(reserved, undersampling=2, oversampling=.7):
    random.shuffle(reserved)
    new_paths = [path for path in reserved 
                 if 'nodule' in path.lower()]
    
    new_paths = new_paths[: int(oversampling * len(new_paths))]
    new_paths += [path for path in reserved 
                  if 'nodule' not in path.lower()][:undersampling * len(new_paths)]
    random.shuffle(new_paths)
    return new_paths


def on_finish(patch_paths, reserved, 
              undersampling, oversampling, shift):
    if undersampling:
        patch_paths = manipulate_samples(reserved, undersampling, oversampling)
    else:
        if shift:
            random.shuffle(patch_paths)
    return patch_paths


def augment_patch(patch, in_shape, lower, upper, shift=2):
    center = array(patch.shape) // 2 
    
    if shift:
        shift = random.randint(-shift, shift, 3)
        center += shift
        if random.randint(0, 2):
            patch = flip(patch, 1)
        if random.randint(0, 2):
            patch = flip(patch, 2)
    
    left = array(in_shape) // 2
    right = array(in_shape) - left
    patch = clip(patch, lower, upper) 
    patch = (patch - lower) / float(upper - lower)  
    return patch[center[0] - left[0]: center[0] + right[0], 
                 center[1] - left[1]: center[1] + right[1], 
                 center[2] - left[2]: center[2] + right[2]].flatten()
    

def batch_generator(patch_paths, in_shape, 
                    batch_size=32, 
                    lower=-1000, upper=400,
                    shift=2, CPU=8, 
                    undersampling=2, planes=3,
                    oversampling=.7
                   ):
    
    counter = 0
    reserved = patch_paths.copy()
    if shift:
        random.shuffle(patch_paths)
        
    if undersampling:
        patch_paths = manipulate_samples(reserved, 
                                         undersampling, 
                                         oversampling)
    
    number_of_batches = ceil(len(patch_paths) 
                             / batch_size)
    
    while True:
        batch_files = patch_paths[batch_size * counter:
                                  batch_size * (counter + 1)]
        
        with Pool(CPU) as pool:
            patch_list = pool.map(load, batch_files) 
        
        augment = partial(augment_patch, 
                          in_shape=in_shape,
                          lower=lower, 
                          upper=upper,
                          shift=shift)
        
        with Pool(CPU) as pool:
            patch_list = pool.map(augment, patch_list)
        
        counter += 1
        labels = [1 if 'NODULE' in patch_path else -1
                  for patch_path in batch_files]
        yield (asarray(patch_list), 
               asarray(labels))
        
        if counter == number_of_batches:
            patch_paths = on_finish(patch_paths, 
                                    reserved, 
                                    undersampling,
                                    oversampling,
                                    shift) 
            counter = 0