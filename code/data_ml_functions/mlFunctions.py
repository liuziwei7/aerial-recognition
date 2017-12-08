"""
Copyright 2017 The Johns Hopkins University Applied Physics Laboratory LLC
All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


__author__ = 'jhuapl'
__version__ = 0.1

import json
from keras import backend as K
from keras.applications import imagenet_utils
from keras.layers import Dense,Input,merge,Flatten,Dropout,LSTM,Conv2D,GlobalAveragePooling2D
from keras.models import Sequential,Model
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical

import numpy as np

import sys
sys.path.insert(0, './DenseNet')
sys.path.insert(0, './DeformConv')

import densenet
from non_local import non_local_block
from SpatialPyramidPooling import SpatialPyramidPooling
from deform_conv.layers import ConvOffset2D

from data_ml_functions.dataFunctions import get_batch_inds

from concurrent.futures import ProcessPoolExecutor
from functools import partial

import os
import pdb

def get_cnn_model(params):   
    """
    Load base CNN model and add metadata fusion layers if 'use_metadata' is set in params.py
    :param params: global parameters, used to find location of the dataset and json file
    :return model: CNN model with or without depending on params
    """
    
    input_tensor = Input(shape=(params.target_img_size[0],params.target_img_size[1],params.num_channels))
    baseModel = densenet.DenseNetImageNet161(input_shape=(params.target_img_size[0], params.target_img_size[1], params.num_channels), include_top=False, input_tensor=input_tensor)

    modelStruct = baseModel.layers[-1].output

    if params.use_nlm:
        modelStruct = baseModel.layers[-2].output
        modelStruct = non_local_block(modelStruct, computation_compression=1, mode='embedded')
        modelStruct = Conv2D(params.cnn_lstm_layer_length, [3, 3], name='conv_nlm')(modelStruct)
        modelStruct = Flatten()(modelStruct)
        modelStruct = Dense(params.cnn_lstm_layer_length, activation='relu', name='fc_nlm')(modelStruct)
        modelStruct = Dropout(0.5)(modelStruct)

    if params.use_spp:
        modelStruct = baseModel.layers[-2].output
        modelStruct = SpatialPyramidPooling([1, 2, 4], name='spp')(modelStruct)
        modelStruct = Dense(params.cnn_lstm_layer_length, activation='relu', name='fc_spp')(modelStruct)
        modelStruct = Dropout(0.5)(modelStruct)

    if params.use_deform:
        modelStruct = baseModel.layers[-2].output
        modelStruct = ConvOffset2D(params.cnn_lstm_layer_length, name='deform')(modelStruct)
        modelStruct = Conv2D(params.cnn_lstm_layer_length, [3, 3], name='conv_deform')(modelStruct)
        modelStruct = Flatten()(modelStruct)
        modelStruct = Dense(params.cnn_lstm_layer_length, activation='relu', name='fc_deform')(modelStruct)
        modelStruct = Dropout(0.5)(modelStruct)

    if params.use_metadata:
        auxiliary_input = Input(shape=(params.metadata_length,), name='aux_input')
        modelStruct = merge([modelStruct,auxiliary_input],'concat')

    modelStruct = Dense(params.cnn_last_layer_length, activation='relu', name='fc1')(modelStruct)
    modelStruct = Dropout(0.5)(modelStruct)
    modelStruct = Dense(params.cnn_last_layer_length, activation='relu', name='fc2')(modelStruct)
    modelStruct = Dropout(0.5)(modelStruct)
    predictions = Dense(params.num_labels, activation='softmax')(modelStruct)

    if not params.use_metadata:
        model = Model(input=[baseModel.input], output=predictions)
    else:
        model = Model(input=[baseModel.input, auxiliary_input], output=predictions)

    for i,layer in enumerate(model.layers):
        layer.trainable = True

    return model

def get_lstm_model(params, codesStats):
    """
    Load LSTM model and add metadata concatenation to input if 'use_metadata' is set in params.py
    :param params: global parameters, used to find location of the dataset and json file
    :param codesStats: dictionary containing CNN codes statistics, which are used to normalize the inputs
    :return model: LSTM model
    """

    if params.use_metadata:
        layerLength = params.cnn_lstm_layer_length + params.metadata_length
    else:
        layerLength = params.cnn_lstm_layer_length

    model = Sequential()
    if params.use_fusion:
        model.add(Dense(4096, activation='relu', input_shape=(codesStats['max_temporal'], layerLength)))
        model.add(Dropout(0.5))
    else:
        model.add(LSTM(4096, return_sequences=True, input_shape=(codesStats['max_temporal'], layerLength), dropout=0.5))
        model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(params.num_labels, activation='softmax'))
    return model

def img_metadata_generator(params, data, metadataStats):
    """
    Custom generator that yields images or (image,metadata) batches and their
    category labels (categorical format).
    :param params: global parameters, used to find location of the dataset and json file
    :param data: list of objects containing the category labels and paths to images and metadata features
    :param metadataStats: metadata stats used to normalize metadata features
    :yield (imgdata,labels) or (imgdata,metadata,labels): image data, metadata (if params set to use), and labels (categorical form)
    """

    N = len(data)

    idx = np.random.permutation(N)

    batchInds = get_batch_inds(params.batch_size_cnn, idx, N)

    executor = ProcessPoolExecutor(max_workers=params.num_workers)

    while True:
        for inds in batchInds:
            batchData = [data[ind] for ind in inds]
            imgdata, metadata, labels = load_cnn_batch(params, batchData, metadataStats, executor)
            if params.use_metadata:
                yield ([imgdata, metadata], labels)
            else:
                yield (imgdata, labels)

def load_cnn_batch(params, batchData, metadataStats, executor):
    """
    Load batch of images and metadata and preprocess the data before returning.
    :param params: global parameters, used to find location of the dataset and json file
    :param batchData: list of objects in the current batch containing the category labels and paths to CNN codes and images
    :param metadataStats: metadata stats used to normalize metadata features
    :return imgdata,metadata,labels: numpy arrays containing the image data, metadata, and labels (categorical form)
    """

    futures = []
    imgdata = np.zeros((params.batch_size_cnn, params.target_img_size[0],
                        params.target_img_size[1], params.num_channels))
    metadata = np.zeros((params.batch_size_cnn, params.metadata_length))
    labels = np.zeros(params.batch_size_cnn)
    for i in range(0, len(batchData)):
        currInput = {}
        currInput['data'] = batchData[i]
        currInput['metadataStats'] = metadataStats
        task = partial(_load_batch_helper, currInput)
        futures.append(executor.submit(task))

    results = [future.result() for future in futures]

    for i, result in enumerate(results):
        metadata[i, :] = result['metadata']
        imgdata[i, :, :, :] = result['img']
        labels[i] = result['labels']

    for i in len(results):
        img_cur = imgdata[i, :, :, :]
        img_cur = image.array_to_img(img_cur)
        img_cur.save(os.path.join('/home/zwliu/preview/', str(i), '.jpg'))
    
    pdb.set_trace()

    if params.use_aug:
        datagen = image.ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
        for imgdata_aug in datagen.flow(imgdata, batch_size=len(results)):
            imgdata = imgdata_aug
            break;

    for i in len(results):
        img_cur = imgdata[i, :, :, :]
        img_cur = image.array_to_img(img_cur)
        img_cur.save(os.path.join('/home/zwliu/preview/', str(i), '.jpg'))

    pdb.set_trace()

    imgdata = imagenet_utils.preprocess_input(imgdata)
    imgdata = imgdata / 255.0

    labels = to_categorical(labels, params.num_labels)

    return imgdata, metadata, labels

def _load_batch_helper(inputDict):
    """
    Helper for load_cnn_batch that actually loads imagery and supports parallel processing
    :param inputDict: dict containing the data and metadataStats that will be used to load imagery
    :return currOutput: dict with image data, metadata, and the associated label
    """

    data = inputDict['data']
    metadataStats = inputDict['metadataStats']
    metadata = np.divide(json.load(open(data['features_path'])) - np.array(metadataStats['metadata_mean']), metadataStats['metadata_max'])
    img = image.load_img(data['img_path'])
    img = image.img_to_array(img)
    labels = data['category']
    currOutput = {}
    currOutput['img'] = img
    currOutput['metadata'] = metadata
    currOutput['labels'] = labels
    return currOutput

def codes_metadata_generator(params, data, metadataStats, codesStats):
    """
    Custom generator that yields a vector containign the 4096-d CNN codes output by ResNet50 and metadata features (if params set to use).
    :param params: global parameters, used to find location of the dataset and json file
    :param data: list of objects containing the category labels and paths to CNN codes and images 
    :param metadataStats: metadata stats used to normalize metadata features
    :yield (codesMetadata,labels): 4096-d CNN codes + metadata features (if set), and labels (categorical form) 
    """
    
    N = len(data)

    idx = np.random.permutation(N)

    batchInds = get_batch_inds(params.batch_size_lstm, idx, N)
    trainKeys = list(data.keys())

    executor = ProcessPoolExecutor(max_workers=params.num_workers)
    
    while True:
        for inds in batchInds:
            batchKeys = [trainKeys[ind] for ind in inds]
            codesMetadata,labels = load_lstm_batch(params, data, batchKeys, metadataStats, codesStats, executor)
            yield(codesMetadata,labels)
        
def load_lstm_batch(params, data, batchKeys, metadataStats, codesStats, executor):
    """
    Load batch of CNN codes + metadata and preprocess the data before returning.
    :param params: global parameters, used to find location of the dataset and json file
    :param data: dictionary where the values are the paths to the files containing the CNN codes and metadata for a particular sequence
    :param batchKeys: list of keys for the current batch, where each key represents a temporal sequence of CNN codes and metadata
    :param metadataStats: metadata stats used to normalize metadata features
    :param codesStats: CNN codes stats used to normalize CNN codes and define the maximum number of temporal views
    :return codesMetadata,labels: 4096-d CNN codes + metadata (if set) and labels (categorical form)
    """

    if params.use_metadata:
        codesMetadata = np.zeros((params.batch_size_lstm, codesStats['max_temporal'], params.cnn_lstm_layer_length+params.metadata_length))
    else:
        codesMetadata = np.zeros((params.batch_size_lstm, codesStats['max_temporal'], params.cnn_lstm_layer_length))

    labels = np.zeros(params.batch_size_lstm)

    futures = []
    for i,key in enumerate(batchKeys):
        currInput = {}
        currInput['currData'] = data[key]
        currInput['lastLayerLength'] = codesMetadata.shape[2]
        currInput['codesStats'] = codesStats
        currInput['use_metadata'] = params.use_metadata
        currInput['metadataStats'] = metadataStats
        labels[i] = data[key]['category']

        task = partial(_load_lstm_batch_helper, currInput)
        futures.append(executor.submit(task))

    results = [future.result() for future in futures]

    for i,result in enumerate(results):
        codesMetadata[i,:,:] = result['codesMetadata']

    labels = to_categorical(labels, params.num_labels)
    
    return codesMetadata,labels

def _load_lstm_batch_helper(inputDict):

    currData = inputDict['currData']
    codesStats = inputDict['codesStats']
    currOutput = {}

    codesMetadata = np.zeros((codesStats['max_temporal'], inputDict['lastLayerLength']))

    timestamps = []
    for codesIndex in range(len(currData['cnn_codes_paths'])):
        cnnCodes = json.load(open(currData['cnn_codes_paths'][codesIndex]))
        # compute a timestamp for temporally sorting
        timestamp = (cnnCodes[4]-1970)*525600 + cnnCodes[5]*12*43800 + cnnCodes[6]*31*1440 + cnnCodes[7]*60
        timestamps.append(timestamp)

        cnnCodes = np.divide(cnnCodes - np.array(codesStats['codes_mean']), np.array(codesStats['codes_max']))
        codesMetadata[codesIndex,:] = cnnCodes

    sortedInds = sorted(range(len(timestamps)), key=lambda k:timestamps[k])
    codesMetadata[range(len(sortedInds)),:] = codesMetadata[sortedInds,:]

    currOutput['codesMetadata'] = codesMetadata
    return currOutput



