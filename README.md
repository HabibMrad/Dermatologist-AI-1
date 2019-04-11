# Dermatologist-AI
- This project was a [task](https://github.com/udacity/dermatologist-ai) on  my [Deep Learning Nanodegree](https://www.udacity.com/course/deep-learning-nanodegree--nd101). The model was trained and tested  on google colab.
## Dataset Downloading
- First, make a directory for the project. So open the terminal
```shell
mkdir dermatologist-ai
cd dermatologist-ai
mkdir data 
cd data
mkdir train; mkdir valid; mkdir test
```
- Download the dataset at `dermatologist-ai/data/` using GNU Wget software package for retrieving files using HTTPS protocol.
```shell
wget https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/train.zip
wget https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/valid.zip
wget https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/test.zip
```
- Extract the zip files to its folders
```shell
unzip train.zip
unzip test.zip
unzip valid.zip
```
- Finaly, delete the zip files because we won't need it again
```shell
rm train.zip
rm valid.zip
rm test.zip
```
Congrats. Now the dataset is ready.
