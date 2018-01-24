"""This script creates a reduced version of the MIT_split dataset. It takes 50 random samples
of each category of the training set."""
import math
import os
from shutil import copyfile


def reduce_MIT_split_train(source_data_dir, destination_data_dir, dataset_size=50, include_test_set=True):
    if not os.path.exists(destination_data_dir):
        try:
            os.makedirs(destination_data_dir)
        except OSError:
            if not os.path.isdir(destination_data_dir):
                raise
    for class_dir in os.listdir(source_data_dir):
        full_dst_path = os.path.join(destination_data_dir, class_dir)
        if not os.path.exists(full_dst_path):
            try:
                os.makedirs(full_dst_path)
            except OSError:
                if not os.path.isdir(full_dst_path):
                    raise
        full_src_path= os.path.join(source_data_dir, class_dir)
        counter = 0
        for imname in os.listdir(full_src_path):
            copyfile(os.path.join(full_src_path, imname), os.path.join(full_dst_path, imname))
            counter = counter + 1
            if counter == dataset_size:
                    break

def divide_MIT_split_test(source_data_dir, destination_test_dir, destination_val_dir, dataset_size=25):
    if not os.path.exists(destination_test_dir):
        try:
            os.makedirs(destination_test_dir)
        except OSError:
            if not os.path.isdir(destination_test_dir):
                raise
    for class_dir in os.listdir(source_data_dir):
        full_dst_test_path = os.path.join(destination_test_dir, class_dir)
        full_dst_val_path = os.path.join(destination_val_dir, class_dir)
        if not os.path.exists(full_dst_test_path) or not os.path.exists(full_dst_val_path):
            try:
                os.makedirs(full_dst_test_path)
                os.makedirs(full_dst_val_path)
            except OSError:
                if not os.path.isdir(full_dst_test_path):
                    raise
                elif not os.path.isdir(full_dst_val_path):
                    raise
        full_src_path = os.path.join(source_data_dir, class_dir)
        if dataset_size > 0: #We want a reduced dataset
            counter = 0
            for imname in os.listdir(full_src_path):
                counter = counter + 1
                if counter <= dataset_size:
                    copyfile(os.path.join(full_src_path, imname), os.path.join(full_dst_test_path, imname))
                elif counter > dataset_size and counter <= dataset_size*2:
                    copyfile(os.path.join(full_src_path, imname), os.path.join(full_dst_val_path, imname))
        else:
            full_src_files = os.listdir(full_src_path)
            len_samples_dir = len(full_src_files)
            num_val_samples = math.floor(len_samples_dir * 0.6) #60% of samples for validation and 40% for test
            counter = 0
            for imname in full_src_files:
                counter = counter + 1
                if counter <= num_val_samples:
                    copyfile(os.path.join(full_src_path, imname), os.path.join(full_dst_val_path, imname))
                else:
                    copyfile(os.path.join(full_src_path, imname), os.path.join(full_dst_test_path, imname))






train_data_dir = '/imatge/froldan/MIT_split/train'
test_data_dir = '/imatge/froldan/MIT_split/test'
#destination_train_data_dir = '/imatge/froldan/MIT_split_400/train'
#destination_test_data_dir = '/imatge/froldan/MIT_split_400/test'
#create_reduced_dataset(train_data_dir, destination_train_data_dir)
#create_reduced_dataset(test_data_dir, destination_test_data_dir, dataset_size=25)
destination_test_dir = '/imatge/froldan/MIT_split/test_set'
destination_val_dir = '/imatge/froldan/MIT_split/val_set'

divide_MIT_split_test(test_data_dir, destination_test_dir, destination_val_dir, dataset_size=0)