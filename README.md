导入相关的库:


```python
import os

import numpy as np

import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader

import matplotlib.pyplot as plt


```

先从较小的数据集开始训练，当然越多的数据，模型精度更高:


```python
image_path = tf.keras.utils.get_file(
      'flower_photos.tgz',
      'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
      extract=True)
image_path = os.path.join(os.path.dirname(image_path), 'flower_photos')
```

加载数据集，并将数据集分为训练数据和测试数据:


```python
data = DataLoader.from_folder(image_path)
train_data, test_data = data.split(0.9)
```

    INFO:tensorflow:Load image with size: 3670, num_label: 5, labels: daisy, dandelion, roses, sunflowers, tulips.
    

    INFO:tensorflow:Load image with size: 3670, num_label: 5, labels: daisy, dandelion, roses, sunflowers, tulips.
    

训练Tensorflow模型:


```python
model = image_classifier.create(train_data)
```

    INFO:tensorflow:Retraining the models...
    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     hub_keras_layer_v1v2 (HubKe  (None, 1280)             3413024   
     rasLayerV1V2)                                                   
                                                                     
     dropout (Dropout)           (None, 1280)              0         
                                                                     
     dense (Dense)               (None, 5)                 6405      
                                                                     
    =================================================================
    Total params: 3,419,429
    Trainable params: 6,405
    Non-trainable params: 3,413,024
    _________________________________________________________________
    None
    Epoch 1/5
    

    2023-05-31 02:33:56.249590: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 51380224 exceeds 10% of free system memory.
    2023-05-31 02:33:56.575815: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 51380224 exceeds 10% of free system memory.
    2023-05-31 02:33:56.614415: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 51380224 exceeds 10% of free system memory.
    2023-05-31 02:33:56.645726: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 25690112 exceeds 10% of free system memory.
    2023-05-31 02:33:56.659033: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 154140672 exceeds 10% of free system memory.
    

    103/103 [==============================] - 57s 528ms/step - loss: 0.8779 - accuracy: 0.7561
    Epoch 2/5
    103/103 [==============================] - 53s 512ms/step - loss: 0.6608 - accuracy: 0.8926
    Epoch 3/5
    103/103 [==============================] - 52s 504ms/step - loss: 0.6191 - accuracy: 0.9120
    Epoch 4/5
    103/103 [==============================] - 52s 504ms/step - loss: 0.6005 - accuracy: 0.9229
    Epoch 5/5
    103/103 [==============================] - 52s 505ms/step - loss: 0.5918 - accuracy: 0.9287
    

评估模型：


```python
loss, accuracy = model.evaluate(test_data)
```

    12/12 [==============================] - 8s 478ms/step - loss: 0.5320 - accuracy: 0.9673
    

导出Tensorflow Lite模型：


```python
model.export(export_dir='.')
```

    2023-05-31 02:42:05.786815: W tensorflow/python/util/util.cc:368] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
    

    INFO:tensorflow:Assets written to: /tmp/tmprv6kz6ja/assets
    

    INFO:tensorflow:Assets written to: /tmp/tmprv6kz6ja/assets
    2023-05-31 02:42:10.594876: I tensorflow/core/grappler/devices.cc:66] Number of eligible GPUs (core count >= 8, compute capability >= 0.0): 0
    2023-05-31 02:42:10.595069: I tensorflow/core/grappler/clusters/single_machine.cc:358] Starting new session
    2023-05-31 02:42:10.655973: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:1164] Optimization results for grappler item: graph_to_optimize
      function_optimizer: Graph size after: 913 nodes (656), 923 edges (664), time = 24.988ms.
      function_optimizer: function_optimizer did nothing. time = 0.015ms.
    
    /opt/conda/envs/tf/lib/python3.8/site-packages/tensorflow/lite/python/convert.py:746: UserWarning: Statistics for quantized inputs were expected, but not specified; continuing anyway.
      warnings.warn("Statistics for quantized inputs were expected, but not "
    2023-05-31 02:42:11.462693: W tensorflow/compiler/mlir/lite/python/tf_tfl_flatbuffer_helpers.cc:357] Ignored output_format.
    2023-05-31 02:42:11.462748: W tensorflow/compiler/mlir/lite/python/tf_tfl_flatbuffer_helpers.cc:360] Ignored drop_control_dependency.
    

    INFO:tensorflow:Label file is inside the TFLite model with metadata.
    

    fully_quantize: 0, inference_type: 6, input_inference_type: 3, output_inference_type: 3
    INFO:tensorflow:Label file is inside the TFLite model with metadata.
    

    INFO:tensorflow:Saving labels in /tmp/tmpb39gc28s/labels.txt
    

    INFO:tensorflow:Saving labels in /tmp/tmpb39gc28s/labels.txt
    

    INFO:tensorflow:TensorFlow Lite model exported successfully: ./model.tflite
    

    INFO:tensorflow:TensorFlow Lite model exported successfully: ./model.tflite
    
