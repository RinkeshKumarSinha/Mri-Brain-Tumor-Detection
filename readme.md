
## Summary
Brain tumors are abnormal growths of cells in the brain. They can be benign (non-cancerous) or malignant (cancerous). They can arise from various types of brain cells, including glial cells, neurons, and other supportive tissues.

![App Screenshot](https://raw.githubusercontent.com/RinkeshKumarSinha/Mri-Brain-Tumor-Detection/main/brain_tumor_dataset/no/13%20no.jpg?token=GHSAT0AAAAAACOSZ3PCSL24ZK5XPJ7RGZFMZPJGB4Q)

NO Tumor                         
![App Screenshot](https://raw.githubusercontent.com/RinkeshKumarSinha/Mri-Brain-Tumor-Detection/main/brain_tumor_dataset/yes/Y1.jpg?token=GHSAT0AAAAAACOSZ3PC562BUWN2IEIPGDCQZPJF56Q)

Presence of Tumor

CNN for Image Analysis: Convolutional Neural Networks (CNNs) are a type of deep learning algorithm commonly used for image analysis tasks. They are highly effective in detecting patterns and features within images, making them well-suited for tasks such as medical image analysis, including the detection of brain tumors on MRI scans.

Training CNNs with MRI Data: CNNs can be trained using large datasets of MRI images of the brain, both with and without tumors. During training, the CNN learns to identify patterns and features in the images that distinguish between healthy brain tissue and abnormal tumor growth.

Feature Extraction: CNNs automatically learn to extract relevant features from the MRI images that are indicative of the presence of a brain tumor. These features may include variations in tissue density, shape irregularities, and the presence of contrast-enhancing regions within the brain.

Classification: Once trained, the CNN can classify new MRI images as either containing a brain tumor or being tumor-free. This classification process is based on the patterns and features learned by the CNN during training.

Accuracy and Efficiency: CNNs have demonstrated high accuracy and efficiency in the detection of brain tumors on MRI scans. They can analyze images quickly and accurately, providing valuable assistance to radiologists and clinicians in diagnosing and monitoring brain tumors.

Pre-Processing the Data 
Creating a CNN model
Accuracy and Saving model
 
Pre-Processing the Data: 

Pre-processing image data before creating a CNN model is an important step in the machine learning process. It involves cleaning and normalizing the data, extracting relevant features, Pre-processing helps to improve the quality and consistency of the data, which can lead to a more accurate and effective model. In Pre-Processing we convert the image data into NumPy arrays so that the model can understand it and extract the features. Also, as the images vary in size we will resize them to a common size. After preprocessing and resizing we will split the data into test and train datasets and also change the labels into categorical labels so that the model won’t consider the labels as some sought of preference. 

CNN Architecture: 
![App Screenshot](https://miro.medium.com/v2/resize:fit:1400/1*CnNorCR4Zdq7pVchdsRGyw.png)
Our Model will consist of an input Layer. 3 Convolutional Layers, 3 Pooling Layers, 3 Normalization Layers and 3 Dropout Layers. Then the output of the final layer will be sent to a Flatten Layer where we will pass it to 2 dense layers, normalize it again and use dropout and then finally use a dense layer to find the output . This model will be then compiled and saved and used for deployment. All these layers will have the relevant activation function, padding and the kernel size.
