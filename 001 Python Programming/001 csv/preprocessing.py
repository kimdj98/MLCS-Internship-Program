import pdb # for debugging
import os
import csv
import pandas as pd
import numpy as np

# process 1: read the dataset files in 'datasets' directory using relative path.

# make relative paths to read dataset
path_to_train_dataset_1 = os.path.join('datasets','mnist_train.csv')
path_to_train_dataset_2 = os.path.join('datasets','mnist_train_2.csv')
path_to_test_dataset = os.path.join('datasets','mnist_test.csv')

paths_to_datasets = os.path.join('datasets','*.csv') # Returns list of paths
paths_to_datasets = [path_to_train_dataset_1, path_to_train_dataset_2, path_to_test_dataset]

# read train_dataset_1
with open(path_to_train_dataset_1, 'r') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        # print(row)
        pass

# read train_dataset_2
with open(path_to_train_dataset_2, 'r') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        # print(row)
        pass

# read test_dataset
with open(path_to_test_dataset, 'r') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        # print(row)
        pass

# pdb.set_trace() # checkpoint

# process 2: Read the lines (rows) in the csv and split the string into a label and image data.

for path in paths_to_datasets:
    with open(path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            label = row[0]
            image = row[1:]

# pdb.set_trace() # checkpoint

# process 3: Convert the image data which is in 1D vector form into 2D vector using 'for' loops.

for path in paths_to_datasets:
    with open(path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            label = row[0]
            image_1d = row[1:]
            image_2d = []
            for i in range(28):
                image_2d.append(image_1d[1+28*i:29+28*i])     

# pdb.set_trace() # checkpoint

# Process 4: create outputs/train and outputs/test directory if not exist
output_path_train = os.path.join('outputs', 'train')
output_path_test = os.path.join('outputs', 'test')

if not os.path.exists(output_path_train):
    os.makedirs(output_path_train)

if not os.path.exists(output_path_test):
    os.makedirs(output_path_test)

# Process 5: Save each individual 2D vector image into '#{label}-#{row_index}.csv' under train or test outputs directory.

with open(path_to_test_dataset, 'r') as csv_file:
    reader = csv.reader(csv_file)
    index = 0
    
    for row in reader:
        label = row[0]
        image_1d = row[1:]

        output_file_test = open(output_path_test+'/#'+str(label)+'-#' + str(index)+'.csv', 'w')
        csv_writer_test = csv.writer(output_file_test)
        image_1d = row[1:]
        for i in range(28):
            csv_writer_test.writerow(image_1d[1+28*i:29+28*i])

        index += 1
    
# with open(path_to_train_dataset_2, 'r') as csv_file:
#     reader = csv.reader(csv_file)
#     for i, row in enumerate(reader):
#         output_file_train = open(output_path_train+'/#'+str(row[0])+'-#' + str(i+count)+'.csv', 'w')
#         csv_writer_train = csv.writer(output_file_train)
#         for row in reader:
#             label = row[0]
#             image_1d = row[1:]
#             image_2d = []
#             for i in range(28):
#                 image_2d.append(image_1d[1+28*i:29+28*i])     
#         csv_writer_train.writerow(image_2d)



# with open(path_to_train_dataset_2, 'r') as csv_file:
#     reader = csv.reader(csv_file)
#     for row in reader:

pdb.set_trace() # checkpoint