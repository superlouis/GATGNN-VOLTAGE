# GATGNN-VOLTAGE
This software package implements our work of GATGNN-VOLTAGE for the problem voltage prediction. With GATGNN-VOLTAGE, one can predict the material's voltage from:
1. a materials' formation-energy prediction alone or
1. the reaction of a low and high potential energy materials.

Please read our paper for the detailed implementation of GATGNN-VOLTAGE:

[Accurate Prediction of Voltage of Battery Electrode Materials using Attention based Graph Neural Networks](https://chemrxiv.org/engage/chemrxiv/article-details/6106efa9171fc75328ba29d0)

![GATGNN-VOLTAGE](/assets/imgs/GATGNN-Voltage.png)


[Machine Learning and Evolution Laboratory](http://mleg.cse.sc.edu)<br />
Department of Computer Science and Engineering <br />
University of South Carolina <br />

# Table of Contents
* [How to cite](#how-to-cite)
* [Installation](#installation)
* [Data](#data)
* [Usage](#usage)
* [References](#references)


<a name="how-to-cite"></a>
## How to cite:<br />
```
Louis, S. Y., Siriwardane, E., Joshi, R., Omee, S., Kumar, N., & Hu, J. (2022). Accurate Prediction of Voltage of Battery Electrode Materials Using Attention Based Graph Neural Networks.
```

<a name="installation"></a>
## Installation
1. Inside of your Python environemnt, install the basic dependencies required for GATGNN-VOLTAGE by running code below:
```
pip install -r requirements.txt
```
2. Follow the instructions listed on [Pytorch-Geometric's documentations](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#installation) to install pytorch-geometric for using Graph Neural Network. 

<a name="data"></a>
## Data
To obtain the dataset, run the `get_data.py` file.
```
python get_data.py
```
<a name="usage"></a>
## Usage

### Reaction based voltage
For the reaction based voltage, run the `voltage-reaction.py` file. There are 3 available modes which can be accessed using the `--mode` flag

1. _evaluation_: for evaluating the performance of a trained model. 
```
python voltage-reaction.py --mode evaluation
```
2. _training_: for training a new reaction-based model.
```
python voltage-reaction.py --mode training
```
3. cross-validation or CV:
```
python voltage-reaction.py --mode cross-validation
```
 Using the appropriate `--mode` flag, you can run 



### Formation-energy based voltage

<a name="references"></a>
## References

1. Louis, S. Y., Zhao, Y., Nasiri, A., Wang, X., Song, Y., Liu, F., & Hu, J. (2020). Graph convolutional neural networks with global attention for improved materials property prediction. Physical Chemistry Chemical Physics, 22(32), 18141-18148.

1. Omee, S. S., Louis, S. Y., Fu, N., Wei, L., Dey, S., Dong, R., ... & Hu, J. (2021). Scalable deeper graph neural networks for high-performance materials property prediction. arXiv preprint arXiv:2109.12283.

1. Louis, S. Y., Nasiri, A., Rolland, F. J., Mitro, C., & Hu, J. (2021). NODE-SELECT: A Graph Neural Network Based On A Selective Propagation Technique. arXiv preprint arXiv:2102.08588.
