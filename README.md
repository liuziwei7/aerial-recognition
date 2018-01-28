## Aerial Image Recognition

<img src='./misc/demo.gif' width=360>

Further information please contact [Ziwei Liu](https://liuziwei7.github.io/).

## Requirements
* [Multi-GPU Training](https://github.com/kuza55/keras-extras/blob/master/utils/multi_gpu.py)
* [DenseNet](https://github.com/titu1994/DenseNet)
* [Spatial Pyramid Pooling](https://github.com/yhenon/keras-spp)
* [Non-Local Neural Networks](https://github.com/titu1994/keras-non-local-nets)

## Getting started
* Run the data preparation script:
``` bash
python runBaseline.py -prepare
```
* Run the training script:
``` bash
python runBaseline.py -cnn
```
* Run the testing script:
``` bash
python runBaseline.py -test_cnn
```

## Acknowledgement
This code is heavily borrowed from [fMoW-baseline](https://github.com/fMoW/baseline).
