# GATGNN-VOLTAGE
This software package implements our work of GATGNN-VOLTAGE for the problem voltage prediction. With GATGNN-VOLTAGE, one can predict the material's voltage from:
1. a materials' formation-energy prediction alone or
1. the reaction of a low and high potential energy materials.

Please read our paper for the detailed implementation of GATGNN-VOLTAGE:

[Accurate Prediction of Voltage of Battery Electrode Materials using Attention based Graph Neural Networks](https://chemrxiv.org/engage/chemrxiv/article-details/6106efa9171fc75328ba29d0)

<p align="center">
<img src="/assets/imgs/GATGNN-Voltage.png" alt="GATGNN-VOLTAGE" width="400"/>
</p>

[Machine Learning and Evolution Laboratory](http://mleg.cse.sc.edu)<br />
Department of Computer Science and Engineering <br />
University of South Carolina <br />

# Table of Contents
* [Installation](#installation)
* [Data](#data)
* [Usage](#usage)
* [How to cite](#how-to-cite)
* [References](#references)

<a name="installation"></a>
## Installation
1. Inside of your Python environemnt, install the basic dependencies required for GATGNN-VOLTAGE by running code below:
```bash
pip install -r requirements.txt
```
2. Follow the instructions listed on [Pytorch-Geometric's documentations](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#installation) to install pytorch-geometric for using Graph Neural Network. 

<a name="data"></a>
## Data
To obtain the dataset, run the `get_data.py` file.
```bash
python get_data.py
```
<a name="usage"></a>
## Usage

### Reaction based voltage
For the reaction based voltage, run the `voltage-reaction.py` file. The 3 running options (evaluation, training, cross-validation or CV) can be set by using the `--mode` flag

#### evaluation   

- **Details**

for evaluating the performance of the trained reaction-model. Running this mode predicts voltage of electrodes from the testing-set and saves those results to `RESULTS/voltage--prediction.csv`.

- **Usage example:**
```bash
python voltage-reaction.py --mode evaluation
```
--- 

#### training

- **Details**

for training a new reaction-based model. 

- **Optional arguments**

| Parameter                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
| --train_size	       |	0.8          |ratio size of the training-set
| --batch             | 128          | batch size to use within experinment 
| --graph_size        | small        |graph encoding format by neighborhood size, either 12 (small) or 16 (large)
| --layers            | 3            |number of AGAT layers to use in model (default:3)
| --neurons           | 64           |number of neurons to use per AGAT Layer
| --heads             | 4            |number of Attention-Heads to use  per AGAT Layer

- **Usage example:**

```bash
python voltage-reaction.py --mode training
```
--- 

#### cross-validation or CV:
- **Details**

for running a k-fold cross-validation training/ evaluation method. Running this mode creates `k` different prediction-results which are saved to `RESULTS/{k}-voltage--prediction.csv`; where `k` corresponds to the cross-validation iteration.

- **Optional arguments**

| Parameter                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
| --fold              | 10           | number of folds 
| --train_size	       |	0.8          |ratio size of the training-set
| --batch             | 128          | batch size to use within experinment 
| --graph_size        | small        |graph encoding format by neighborhood size, either 12 (small) or 16 (large)
| --layers            | 3            |number of AGAT layers to use in model (default:3)
| --neurons           | 64           |number of neurons to use per AGAT Layer
| --heads             | 4            |number of Attention-Heads to use  per AGAT Layer

- **Usage example:**

```bash
python voltage-reaction.py --mode cross-validation
```

### Formation-energy based voltage
#### Upcoming soon

<a name="how-to-cite"></a>
## How to cite:<br />
```
Louis, S. Y., Siriwardane, E., Joshi, R., Omee, S., Kumar, N., & Hu, J. (2022). Accurate Prediction of Voltage of Battery Electrode Materials Using Attention Based Graph Neural Networks.
```

<a name="references"></a>
## References

1. Louis, S. Y., Zhao, Y., Nasiri, A., Wang, X., Song, Y., Liu, F., & Hu, J. (2020). Graph convolutional neural networks with global attention for improved materials property prediction. Physical Chemistry Chemical Physics, 22(32), 18141-18148.

1. Omee, S. S., Louis, S. Y., Fu, N., Wei, L., Dey, S., Dong, R., ... & Hu, J. (2021). Scalable deeper graph neural networks for high-performance materials property prediction. arXiv preprint arXiv:2109.12283.

1. Louis, S. Y., Nasiri, A., Rolland, F. J., Mitro, C., & Hu, J. (2021). NODE-SELECT: A Graph Neural Network Based On A Selective Propagation Technique. arXiv preprint arXiv:2102.08588.
