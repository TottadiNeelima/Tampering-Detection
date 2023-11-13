# Tampering-detection-and-localization-on-Copy-Move-Images-using-Deep-Learning-Approaches
Images can be used as legal evidence in forensics, journalism, and other fields. Image tampering is modifying images using modern technologies, which might create false evidence. There are many tampering with images like copy-move image forging, image slicing, and recoloring. Now the detection of those types of images whether tampered with or not and the process of recognizing copied and pasted portions for the copy-move tampered images is a challenging task. In this connection, this work employed CNN, ResNet-50, and VGG-16 for the detection of tampering images. These models were trained on CASIAv2 and COMOFOD datasets. Moreover, SIFT and Gray Scaling were applied to preprocess the Copy-Move images to identify similar features in the image. Furthermore, preprocessed images are fed to the DBSCAN algorithm to locate the portions of the image where it is copied and pasted in copy-move tampered images. From the results, it was observed that the CNN model trained on the CASIAv2 dataset outperformed all other models. 
# Datasets:
CASIAv2(https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset)
COMOFOD(https://www.kaggle.com/code/tusharchauhansoft123/comofod-dataset-forgery)



