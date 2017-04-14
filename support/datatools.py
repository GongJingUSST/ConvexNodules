import sys
sys.path.append('./support')
from paths import *
from glob import glob
from os.path import join, basename, isfile
from numpy import *


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


def augment_patch(patch, in_shape, shift=4):
    
    center = array(patch.shape) // 2 
    
    if shift:
        shift = random.randint(-shift, shift, 3)
        center += shift
        if random.randint(0, 2):
            patch = flip(patch, 1)
        if random.randint(0, 2):
            patch = flip(patch, 2)
    
    
    lefts = array(in_shape)[:, :3] // 2
    rights = array(in_shape)[:, :3] - lefts
    patch = clip(patch, LOWER, UPPER) 
    patch = (patch - LOWER) / float(UPPER - LOWER)  
    
    in_patch = [patch[center[0] - left[0]: center[0] + right[0], 
                      center[1] - left[1]: center[1] + right[1], 
                      center[2] - left[2]: center[2] + right[2]]
                for left, right in zip(lefts, rights)]
    return in_patch


def batch_generator(patch_paths, in_shape, 
                    batch_size=32, 
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
                          shift=shift)
        
        with Pool(CPU) as pool:
            patch_list = pool.map(augment, patch_list)
        
        counter += 1
        labels = [[1, 0] if 'NODULE' in patch_path else [0, 1] 
                  for patch_path in batch_files]
        yield ([expand_dims(asarray([patch[0] for patch in patch_list]), -1), 
                expand_dims(asarray([patch[1] for patch in patch_list]), -1), 
                expand_dims(asarray([patch[2] for patch in patch_list]), -1)], 
               asarray(labels))
        
        if counter == number_of_batches:
            patch_paths = on_finish(patch_paths, 
                                    reserved, 
                                    undersampling,
                                    oversampling,
                                    shift) 
            counter = 0