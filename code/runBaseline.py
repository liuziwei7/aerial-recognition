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

import sys

from fmowBaseline import FMOWBaseline
import params #params.py

def main(argv):
    baseline = FMOWBaseline(params, argv)
    if baseline.params.train_cnn:
        baseline.train_cnn()
    if baseline.params.generate_cnn_codes:
        baseline.generate_cnn_codes()
    if baseline.params.train_lstm:
        baseline.train_lstm()
    if baseline.params.test_cnn or baseline.params.test_lstm:
        baseline.test_models()
    
if __name__ == "__main__":
    main(sys.argv[1:])
