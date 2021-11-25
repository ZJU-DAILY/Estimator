## Introduction
  This folder (i.e., ./CNN, /CNNTCN) holds the source code of our methods.

## Environment Preparation
  - Pytorch 1.9.0
  - Python version: 3.9
  - GPU Server: GeForce RTX 3090, 2.40GHz GPU, and 24-GB RAM
  - Please refer to the source code to install all required packages of libs.

## Dataset Description
  - "./Geolife, ./CNN/Image, ./CNNTCN/Geolife" contains original trajectory data (data1), mapped trajectory data, trajectory data with seasonal data (data2) or partitioning(data1), and boundary information of 3 partitions (bjDistrict).
  - Note, Geolife dataset is consisted of seven kinds of mode datasets.
  - The format of the datasets are respectively:
         Trajectory data: moving object's sampling points consisted of lattitude, longitude, timestamp.
         Mapped trajectory data: grid images (50*50) of trajectory data.
         Boundary information: points on the boundaries consisted of lattitude and longetude.

## Running
  - Remove redundant information (e.g. altitude) of sampling points in GeolifeDataset 1.3;
  - Generate seven mode datasets and compose them as a new Geolife './Geolife';
  - Get mapped trajectory by './CNN/trajectory_mapping.py';
  - Train the CNN model by './CNN/main.py' and save the parameters ('./CNN/res50.py' explains how Resnet50 works)
  - Load the train set and test set from seven mode datasets and Geolife;
  - Run './CNNTCN/geolife_test.py' without cnnTest(), test(), test1() to train TCN model ('./CNNTCN/tcn.py' and './CNNTCN/model.py' explain how TCN works);
  - Run './CNNTCN/geolife_test.py' without train(epoch) to test CNN-TCN model and evaluate our methods;

## Citation
If you use our code for research work, please cite our paper as below:
@article{,
  title={C2TF: A Scalable Framework for Transportation Mode Classification of GPS Trajectories},
  author={Lu Chen, Danlei Hu, Ziquan Fang, Chunhui Sheng, Yunjun Gao},
  year={2022},
}
