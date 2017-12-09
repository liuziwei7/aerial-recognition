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

import os
from multiprocessing import cpu_count

num_workers = cpu_count() / 2
num_gpus_parallel = 2

use_metadata = True

# batch_size_cnn = 128
batch_size_cnn = 32
# batch_size_lstm = 512
batch_size_lstm = 256
# batch_size_eval = 128
batch_size_eval = 16
metadata_length = 45
num_channels = 3
cnn_last_layer_length = 4096
cnn_lstm_layer_length = 2208

target_img_size = (224,224)

image_format = 'jpg'

train_cnn = False
generate_cnn_codes = False
train_lstm = False
test_cnn = False
test_lstm = False

use_finetune = True
use_fusion = False
use_reweight = True
use_aug = False

use_nlm = False
use_spp = False
use_deform = False

#LEARNING PARAMS
# cnn_adam_learning_rate = 1e-4
cnn_adam_learning_rate = 1e-5
cnn_adam_loss = 'categorical_crossentropy'
# cnn_epochs = 50
cnn_epochs = 2
	
lstm_adam_learning_rate = 1e-4
lstm_loss = 'categorical_crossentropy'
lstm_epochs = 100

#DIRECTORIES AND FILES
directories = {}
# directories['dataset'] = '../../fmow_dataset'
directories['dataset'] = '/home/public/dataset/fMoW/fMoW-rgb'
directories['input'] = os.path.join('..', 'data', 'input')
directories['output'] = os.path.join('..', 'data', 'output')
directories['working'] = os.path.join('..', 'data', 'working')
directories['train_data'] = os.path.join(directories['input'], 'train_data')
directories['test_data'] = os.path.join(directories['input'], 'test_data')
directories['cnn_models'] = os.path.join(directories['working'], 'cnn_models')
directories['lstm_models'] = os.path.join(directories['working'], 'lstm_models')
directories['predictions'] = os.path.join(directories['output'], 'predictions')
directories['cnn_checkpoint_weights'] = os.path.join(directories['working'], 'cnn_checkpoint_weights')
directories['lstm_checkpoint_weights'] = os.path.join(directories['working'], 'lstm_checkpoint_weights')

directories['cnn_codes'] = os.path.join(directories['working'], 'cnn_codes')

directories['cnn_finetune_reweight_models'] = os.path.join(directories['working'], 'cnn_finetune_reweight_models')
directories['cnn_finetune_reweight_checkpoint_weights'] = os.path.join(directories['working'], 'cnn_finetune_reweight_checkpoint_weights')

directories['cnn_finetune_reweight_aug_models'] = os.path.join(directories['working'], 'cnn_finetune_reweight_aug_models')
directories['cnn_finetune_reweight_aug_checkpoint_weights'] = os.path.join(directories['working'], 'cnn_finetune_reweight_aug_checkpoint_weights')

directories['cnn_finetune_reweight_nlm_models'] = os.path.join(directories['working'], 'cnn_finetune_reweight_nlm_models')
directories['cnn_finetune_reweight_nlm_checkpoint_weights'] = os.path.join(directories['working'], 'cnn_finetune_reweight_nlm_checkpoint_weights')

directories['cnn_finetune_reweight_spp_models'] = os.path.join(directories['working'], 'cnn_finetune_reweight_spp_models')
directories['cnn_finetune_reweight_spp_checkpoint_weights'] = os.path.join(directories['working'], 'cnn_finetune_reweight_spp_checkpoint_weights')

directories['cnn_finetune_reweight_deform_models'] = os.path.join(directories['working'], 'cnn_finetune_reweight_deform_models')
directories['cnn_finetune_reweight_deform_checkpoint_weights'] = os.path.join(directories['working'], 'cnn_finetune_reweight_deform_checkpoint_weights')

directories['lstm_finetune_models'] = os.path.join(directories['working'], 'lstm_finetune_models')
directories['lstm_finetune_checkpoint_weights'] = os.path.join(directories['working'], 'lstm_finetune_checkpoint_weights')

directories['fusion_reweight_models'] = os.path.join(directories['working'], 'fusion_reweight_models')
directories['fusion_reweight_checkpoint_weights'] = os.path.join(directories['working'], 'fusion_reweight_checkpoint_weights')

files = {}
files['training_struct'] = os.path.join(directories['working'], 'training_struct.json')
files['test_struct'] = os.path.join(directories['working'], 'test_struct.json')
files['dataset_stats'] = os.path.join(directories['working'], 'dataset_stats.json')
files['class_weight'] = os.path.join(directories['working'], 'class_weights.json')


    
category_names = ['false_detection', 'airport', 'airport_hangar', 'airport_terminal', 'amusement_park', 'aquaculture', 'archaeological_site', 'barn', 'border_checkpoint', 'burial_site', 'car_dealership', 'construction_site', 'crop_field', 'dam', 'debris_or_rubble', 'educational_institution', 'electric_substation', 'factory_or_powerplant', 'fire_station', 'flooded_road', 'fountain', 'gas_station', 'golf_course', 'ground_transportation_station', 'helipad', 'hospital', 'interchange', 'lake_or_pond', 'lighthouse', 'military_facility', 'multi-unit_residential', 'nuclear_powerplant', 'office_building', 'oil_or_gas_facility', 'park', 'parking_lot_or_garage', 'place_of_worship', 'police_station', 'port', 'prison', 'race_track', 'railway_bridge', 'recreational_facility', 'impoverished_settlement', 'road_bridge', 'runway', 'shipyard', 'shopping_mall', 'single-unit_residential', 'smokestack', 'solar_farm', 'space_facility', 'stadium', 'storage_tank','surface_mine', 'swimming_pool', 'toll_booth', 'tower', 'tunnel_opening', 'waste_disposal', 'water_treatment_facility', 'wind_farm', 'zoo']
category_weighting = {0:0.6, 1:0.6, 2:1.0, 3:1.0, 4:1.0, 5:1.0, 6:1.0, 7:1.0, 8:1.4, 9:1.0, 10:1.0, 11:1.4, 12:0.6, 13:1.0, 14:0.6, 15:1.4, 16:1.0, 17:1.4, 18:1.4, 19:0.6, 20:1.0, 21:1.4, 22:1.0, 23:1.0, 24:1.0, 25:1.0, 26:1.0, 27:1.0, 28:1.0, 29:0.6, 30:1.0, 31:0.6, 32:1.0, 33:1.0, 34:1.0, 35:1.0, 36:1.0, 37:1.4, 38:1.0, 39:1.0, 40:1.0, 41:1.0, 42:1.0, 43:1.0, 44:1.4, 45:1.0, 46:1.0, 47:1.0, 48:0.6, 49:1.4, 50:0.6, 51:1.0, 52:1.0, 53:1.0, 54:1.0, 55:1.0, 56:1.0, 57:1.4, 58:0.6, 59:1.0, 60:1.0, 61:0.6, 62:1.0}

num_labels = len(category_names)

for directory in directories.values():
    if not os.path.isdir(directory):
        os.makedirs(directory)

