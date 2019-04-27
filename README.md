[image1]: ./images/ROC_Curve.png 
[image2]: ./images/Confusion_Matrix.png 

# Dermatologist-AI
- This project was a [task](https://github.com/udacity/dermatologist-ai) on my [Deep Learning Nanodegree](https://www.udacity.com/course/deep-learning-nanodegree--nd101). The model was trained and tested on google colab.
## Dataset Downloading
- First, Clone the repo to your machine. So open the terminal
```shell
!git clone https://github.com/gabir-yusuf/Dermatologist-AI
```
- Change directory to the project files:
```shell
%cd Dermatologist-AI/
!mkdir train; mkdir valid; mkdir test
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

- Now to train the model make sure you are in the project directory, then in the terminal write:
```shell
python train.py
```
- To train and test the model in one line write in the terminal:
```shell
python test.py
```
# Results:
- **Accuracy**
```
Category 1 Score: 0.772
Category 2 Score: 0.880
Category 3 Score: 0.826
```



ROC Curve                  |Confusion Matrix        
:-------------------------:|:-------------------------:
![image1]                  |![Confusion Matrix][image2]    



