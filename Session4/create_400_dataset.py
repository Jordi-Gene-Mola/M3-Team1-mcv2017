"""This script creates a reduced version of the MIT_split dataset. It takes 50 random samples
of each category of the training set."""
import os
from shutil import copyfile


def create_reduced_dataset(source_data_dir, destination_data_dir, dataset_size=50):
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
        full_src_path= os.path.join(train_data_dir, class_dir)
        counter = 0
        for imname in os.listdir(full_src_path):
            copyfile(os.path.join(full_src_path, imname), os.path.join(full_dst_path, imname))
            counter = counter + 1
            if counter == dataset_size:
                break


train_data_dir = '/imatge/froldan/MIT_split/train'
test_data_dir = '/imatge/froldan/MIT_split/test'
destination_train_data_dir = '/imatge/froldan/MIT_split_400/train'
destination_test_data_dir = '/imatge/froldan/MIT_split_400/test'
#create_reduced_dataset(train_data_dir, destination_train_data_dir)
create_reduced_dataset(test_data_dir, destination_test_data_dir, dataset_size=200)