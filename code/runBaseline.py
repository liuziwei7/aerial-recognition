import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import sys

from fmowBaseline import FMOWBaseline
import params_gpu01 #params.py

params = params_gpu01

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
