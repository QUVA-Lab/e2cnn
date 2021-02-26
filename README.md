
General E(2)-Equivariant Steerable CNNs
--------------------------------------------------------------------------------
**[Documentation](https://quva-lab.github.io/e2cnn/)** | **[Experiments](https://github.com/QUVA-Lab/e2cnn_experiments)** | **[Paper](https://arxiv.org/abs/1911.08251)** | **[Thesis](https://gabri95.github.io/Thesis/thesis.pdf)**

*e2cnn* is a [PyTorch](https://pytorch.org/) extension for equivariant deep learning.

*Equivariant neural networks* guarantee a specified transformation behavior of their feature spaces under transformations of their input.
For instance, classical convolutional neural networks (*CNN*s) are by design equivariant to translations of their input.
This means that a translation of an image leads to a corresponding translation of the network's feature maps.
This package provides implementations of neural network modules which are equivariant under all *isometries* E(2) of the image plane 
![my equation](https://chart.apis.google.com/chart?cht=tx&chs=19&chl=\mathbb{R}^2)
, that is, under *translations*, *rotations* and *reflections*.
In contrast to conventional CNNs, E(2)-equivariant models are guaranteed to generalize over such transformations, and are therefore more data efficient.

The feature spaces of E(2)-Equivariant Steerable CNNs are defined as spaces of *feature fields*, being characterized by their transformation law under rotations and reflections.
Typical examples are scalar fields (e.g. gray-scale images or temperature fields) or vector fields (e.g. optical flow or electromagnetic fields).

![feature field examples](https://github.com/QUVA-Lab/e2cnn/raw/master/visualizations/feature_fields.png)

Instead of a number of channels, the user has to specify the field *types* and their *multiplicities* in order to define a feature space.
Given a specified input- and output feature space, our ``R2conv`` module instantiates the *most general* convolutional mapping between them.
Our library provides many other equivariant operations to process feature fields, including nonlinearities, mappings to produce invariant features, batch normalization and dropout.
Feature fields are represented by ``GeometricTensor`` objects, which wrap a ``torch.Tensor`` with the corresponding transformation law.
All equivariant operations perform a dynamic type-checking in order to guarantee a geometrically sound processing of the feature fields.

E(2)-Equivariant Steerable CNNs unify and generalize a wide range of isometry equivariant CNNs in one single framework.
Examples include:
- [Group Equivariant Convolutional Networks](https://arxiv.org/abs/1602.07576)
- [Harmonic Networks: Deep Translation and Rotation Equivariance](https://arxiv.org/abs/1612.04642)
- [Steerable CNNs](https://arxiv.org/abs/1612.08498)
- [Rotation equivariant vector field networks](https://arxiv.org/abs/1612.09346)
- [Learning Steerable Filters for Rotation Equivariant CNNs](https://arxiv.org/abs/1711.07289)
- [HexaConv](https://arxiv.org/abs/1803.02108)
- [Roto-Translation Covariant Convolutional Networks for Medical Image Analysis](https://arxiv.org/abs/1804.03393)

For more details we refer to our NeurIPS 2019 paper [General E(2)-Equivariant Steerable CNNs](https://arxiv.org/abs/1911.08251).

--------------------------------------------------------------------------------

The library is structured into four subpackages with different high-level features:

| Component                                                                  | Description                                                      |
| ---------------------------------------------------------------------------| ---------------------------------------------------------------- |
| [**e2cnn.group**](https://github.com/QUVA-Lab/e2cnn/blob/master/group/)                | implements basic concepts of *group* and *representation* theory |
| [**e2cnn.kernels**](https://github.com/QUVA-Lab/e2cnn/blob/master/kernels/)            | solves for spaces of equivariant convolution kernels             |
| [**e2cnn.gspaces**](https://github.com/QUVA-Lab/e2cnn/blob/master/gspaces/)            | defines the image plane and its symmetries                       |
| [**e2cnn.nn**](https://github.com/QUVA-Lab/e2cnn/blob/master/nn/)                      | contains equivariant modules to build deep neural networks       |
-------------------------------------------------------------------------------------------------------------------------------------------------

## Demo

Since E(2)-steerable CNNs are equivariant under rotations and reflections, their inference is independent from the choice of image orientation.
The visualization below demonstrates this claim by feeding rotated images into a randomly initialized E(2)-steerable CNN (left).
The middle plot shows the equivariant transformation of a feature space, consisting of one scalar field (color-coded) and one vector field (arrows), after a few layers.
In the right plot we transform the feature space into a comoving reference frame by rotating the response fields back (stabilized view).

![Equivariant CNN output](https://github.com/QUVA-Lab/e2cnn/raw/master/visualizations/vectorfield.gif)

The invariance of the features in the comoving frame validates the rotational equivariance of E(2)-steerable CNNs empirically.
Note that the fluctuations of responses are discretization artifacts due to the sampling of the image on a pixel grid, which does not allow for exact continuous rotations.
<!-- Note that the fluctuations of responses are due to discretization artifacts coming from the  -->

For comparison, we show a feature map response of a conventional CNN for different image orientations below.

![Conventional CNN output](https://github.com/QUVA-Lab/e2cnn/raw/master/visualizations/conventional_cnn.gif)

Since conventional CNNs are not equivariant under rotations, the response varies randomly with the image orientation.
This prevents CNNs from automatically generalizing learned patterns between different reference frames.


## Experimental results

E(2)-steerable convolutions can be used as a drop in replacement for the conventional convolutions used in CNNs.
Keeping the same training setup and *without performing hyperparameter tuning*, this leads to significant performance boosts compared to CNN baselines (values are test errors in percent):

 model        | CIFAR-10                | CIFAR-100                | STL-10             |
 ------------ | ----------------------- | ------------------------ | ------------------ |
 CNN baseline | 2.6 &nbsp; ± 0.1 &nbsp; | 17.1 &nbsp; ± 0.3 &nbsp; |       12.74 ± 0.23 |
 E(2)-CNN *   | 2.39       ± 0.11       | 15.55       ± 0.13       |       10.57 ± 0.70 |
 E(2)-CNN     | 2.05       ± 0.03       | 14.30       ± 0.09       | &nbsp; 9.80 ± 0.40 |

The models without * are for a fair comparison designed such that the number of parameters of the baseline is approximately preserved while models with * preserve the number of channels, and hence compute.
For more details we refer to our [paper](https://arxiv.org/abs/1911.08251).

## Getting Started

*e2cnn* is easy to use since it provides a high level user interface which abstracts most intricacies of group and representation theory away.
The following code snippet shows how to perform an equivariant convolution from an RGB-image to 10 *regular* feature fields (corresponding to a
[group convolution](https://arxiv.org/abs/1602.07576)).

```python3
from e2cnn import gspaces                                          #  1
from e2cnn import nn                                               #  2
import torch                                                       #  3
                                                                   #  4
r2_act = gspaces.Rot2dOnR2(N=8)                                    #  5
feat_type_in  = nn.FieldType(r2_act,  3*[r2_act.trivial_repr])     #  6
feat_type_out = nn.FieldType(r2_act, 10*[r2_act.regular_repr])     #  7
                                                                   #  8
conv = nn.R2Conv(feat_type_in, feat_type_out, kernel_size=5)       #  9
relu = nn.ReLU(feat_type_out)                                      # 10
                                                                   # 11
x = torch.randn(16, 3, 32, 32)                                     # 12
x = nn.GeometricTensor(x, feat_type_in)                            # 13
                                                                   # 14
y = relu(conv(x))                                                  # 15
```

Line 5 specifies the symmetry group action on the image plane
![my equation](https://chart.apis.google.com/chart?cht=tx&chs=19&chl=\mathbb{R}^2)
under which the network should be equivariant.
We choose the 
[*cyclic group*](https://en.wikipedia.org/wiki/Cyclic_group)
 C<sub>8</sub>, which describes discrete rotations by multiples of 2π/8.
Line 6 specifies the input feature field types.
The three color channels of an RGB image are thereby to be identified as three independent scalar fields, which transform under the
[*trivial representation*](https://en.wikipedia.org/wiki/Trivial_representation)
 of C<sub>8</sub>.
Similarly, the output feature space is in line 7 specified to consist of 10 feature fields which transform under the
[*regular representation*](https://en.wikipedia.org/wiki/Regular_representation)
of C<sub>8</sub>.
The C<sub>8</sub>-equivariant convolution is then instantiated by passing the input and output type as well as the kernel size to the constructor (line 9).
Line 10 instantiates an equivariant ReLU nonlinearity which will operate on the output field and is therefore passed the output field type.

Lines 12 and 13 generate a random minibatch of RGB images and wrap them into a `nn.GeometricTensor` to associate them with their correct field type.
The equivariant modules process the geometric tensor in line 15.
Each module is thereby checking whether the geometric tensor passed to them satisfies the expected transformation law.

Because the parameters do not need to be updated anymore at test time, after training, any equivariant network can be 
converted into a pure PyTorch model with no additional computational overhead in comparison to conventional CNNs.
The code currently supports the automatic conversion of a few commonly used modules through the `.export()` method; 
check the [documentation](https://quva-lab.github.io/e2cnn/api/e2cnn.nn.html) for more details.

A hands-on tutorial, introducing the basic functionality of *e2cnn*, is provided in [introduction.ipynb](https://github.com/QUVA-Lab/e2cnn/blob/master/examples/introduction.ipynb).
Code for training and evaluating a simple model on the [*rotated MNIST*](https://sites.google.com/a/lisa.iro.umontreal.ca/public_static_twiki/variations-on-the-mnist-digits) dataset is given in [model.ipynb](https://github.com/QUVA-Lab/e2cnn/blob/master/examples/model.ipynb).

More complex equivariant *Wide Resnet* models are implemented in [e2wrn.py](https://github.com/QUVA-Lab/e2cnn/blob/master/examples/e2wrn.py).
To try a model which is equivariant under reflections call:
```
cd examples
python e2wrn.py
```
A version of the same model which is simultaneously equivariant under reflections and rotations of angles multiple of 90 degrees can be run via:
```
python e2wrn.py --rot90
```


## Dependencies

The library is based on Python3.7

```
torch>=1.1
numpy
scipy
```
Optional:
```
pymanopt
autograd
```

Check the branch [legacy_py3.6](https://github.com/QUVA-Lab/e2cnn/tree/legacy_py3.6) for a Python 3.6 compatible version of the library.

## Installation

You can install the latest [release](https://github.com/QUVA-Lab/e2cnn/releases) as

```
pip install e2cnn
```

or you can clone this repository and manually install it with
```
pip install git+https://github.com/QUVA-Lab/e2cnn
```


## Cite

The development of this library was part of the work done for our paper
[General E(2)-Equivariant Steerable CNNs](https://arxiv.org/abs/1911.08251). 
Please cite this work if you use our code:

```
@inproceedings{e2cnn,
    title={{General E(2)-Equivariant Steerable CNNs}},
    author={Weiler, Maurice and Cesa, Gabriele},
    booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
    year={2019},
}
```

Feel free to [contact us](mailto:cesa.gabriele@gmail.com,m.weiler.ml@gmail.com).

## License

*e2cnn* is distributed under BSD Clear license. See LICENSE file.
