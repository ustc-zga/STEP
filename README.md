### HiCo-STEP: Causality Informed Hierarchical Spatiotemporal Network for Traffic Emission Prediction  ####


1. Install Pytorch and necessary dependencies.
## Requirements:
einops==0.6.0
matplotlib==3.6.3
numpy==1.24.1
omegaconf==2.3.0
scikit_learn==1.2.1
scipy==1.10.0
torch==1.13.1
tqdm==4.64.1

```
pip install -r requirements.txt


2. Datasets and preprocess
--Step 1----------
We provide the datasets of the paper and you can download them from [https://pan.baidu.com/s/1A9TOxjCaTai0qsXhx1sL4Q].(Extraction code: s2x9). The files contains the original traffic emission data and conresponding  adjacent matrix files. Unzip the datasets and place them in the [data folder](./data).


--Step 2----------
Run the data preprocess script [preprocess_data.py] and place the output into the folder [./data/XiAn_City].



#### Model training
 
The hypapameter used in our paper is contained in [./config.yaml].
To train the model, you can run 
---------
python main.py 




