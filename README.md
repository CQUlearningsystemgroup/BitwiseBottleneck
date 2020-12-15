# Flexible Neural Network Activation Quantization with Bitwise Information Bottlenecks

### **Introduction**
Recent researches on information bottleneck shed new light on the continuous 
attempts to open the black box of neural signal encoding. Inspired by the 
problem of lossy signal compression for wireless communication, 
this paper presents a Bitwise Information Bottleneck approach for quantizing 
and encoding neural network activations. Based on the rate-distortion theory, 
the Bitwise Information Bottleneck attempts to determine the most significant bits in activation representation by assigning and approximating the sparse coefficient associated with each bit. Given the constraint of a limited average code rate, the information bottleneck minimizes the rate-distortion for optimal activation quantization in a flexible layer-by-layer manner.



### **Dependencies**

+ Python 3.6
+ Tensorflow 1.14.0
+ Sklearn 1.10.0



### **Pre-trained Model**
Please check *[here](https://pan.baidu.com/s/1kaHGmfAsIUgYib9ugRPqXw)

Extraction code: 5hup.



### **Run Bitwise Information Bottleneck**
To begin, you will need to download the ImageNet dataset and convert it to
TFRecord format.

Once your dataset is ready, you can begin training the model as follows:

```bash
python imagenet_main.py --data_dir=/path/to/imagenet
```

The model will begin training and will automatically evaluate itself on the
validation data roughly once per epoch.

Note that there are a number of other options you can specify, including
`--model_dir` to choose where to store the model and `--resnet_size` to choose
the model size (options include ResNet-18 through ResNet-200). See
[`resnet_run_loop.py`](resnet_run_loop.py) for the full list of options.



###**Experiments**
**ImageNet Experiments**

Model|Weights|Activations|Top-1(%)|Top-5(%)
:---:|:---:|:---:|:---:|:---:
ResNet-50|32|32|75.6|92.8
ResNet-50|8|5|75.7|92.7
ResNet-50|8|4|74.8|92.2


###**Customization**
if you want to train you own model, you should do like following steps:
+ Use `dorefa_quantization()` to quantize activations of floating-point model.
+ Print out all activations data by `tensor_print()`.
+ Run `calculate_alpha_coeffient.py`, where you can setting the hyperparameter of threshold of PSNR loss, then obtian the
  alpha coefficient array.
+ Use `bitwise_information_bottleneck()` to replace `dorefa_quantization()`, and run `imagenet_main.py` with only one step.
+ Run `transform_learning.py`, transforming the parameters of the initial quantized model to the parameter of  new model with Bitwise
  Information Bottleneck layers inserted.
+ Retrain 5 epoch with `base_lr = 0.0001`.



