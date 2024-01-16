# TensorFlow Developer Certificate Bootcamp

## Contents of notebooks

### 🛠 00. TensorFlow Fundamentals
  * Introduction to tensors (creating tensors)
  * Getting information from tensors (tensor attributes)
  * Manipulating tensors (tensor operations)
  * Tensors and NumPy
  * Using @tf.function (a way to speed up your regular Python functions)
  * Using GPUs with TensorFlow

---

### 🛠 01. Neural Network Regression with TensorFlow
  * Build TensorFlow sequential models with multiple layers
  * Prepare data for use with a machine learning model
  * Learn the different components which make up a deep learning model (loss function, architecture, optimization function)
  * Learn how to diagnose a regression problem (predicting a number) and build a neural network for it

---

### 🛠 02. Neural Network Classification with TensorFlow
  * Learn how to diagnose a classification problem (predicting whether something is one thing or another)
  * Build, compile & train machine learning classification models using TensorFlow
  * Build and train models for binary and multi-class classification
  * Plot modelling performance metrics against each other
  * Match input (training data shape) and output shapes (prediction data target)

---

### 🛠 03. Computer Vision and Convolutional Neural Networks with TensorFlow
  * Build convolutional neural networks with Conv2D and pooling layers
  * Learn how to diagnose different kinds of computer vision problems
  * Learn to how to build computer vision neural networks
  * Learn how to use real-world images with your computer vision models

---

### 🛠 04. Transfer Learning with TensorFlow Part 1: Feature Extraction
  * Learn how to use pre-trained models to extract features from your own data
  * Learn how to use TensorFlow Hub for pre-trained models
  * Learn how to use TensorBoard to compare the performance of several different models

---

### 🛠 05. Transfer Learning with TensorFlow Part 2: Fine-tuning
  * Learn how to setup and run several machine learning experiments
  * Learn how to use data augmentation to increase the diversity of your training data
  * Learn how to fine-tune a pre-trained model to your own custom problem
  * Learn how to use Callbacks to add functionality to your model during training

---

### 🛠 06. Transfer Learning with TensorFlow Part 3: Scaling Up (Food Vision mini)
  * Learn how to scale up an existing model
  * Learn to how evaluate your machine learning models by finding the most wrong predictions
  * Beat the original Food101 paper using only 10% of the data

---

### 🛠 07. Milestone Project 1: Food Vision
  * Combine everything you've learned in the previous 6 notebooks to build Food Vision: a computer vision model able to classify 101 different kinds of foods. Our model well and truly beats the original Food101 paper.

---

### 🛠 08. NLP Fundamentals in TensorFlow
  * Learn to:
    * Preprocess natural language text to be used with a neural network
    * Create word embeddings (numerical representations of text) with TensorFlow
    * Build neural networks capable of binary and multi-class classification using:
      * RNNs (recurrent neural networks)
      * LSTMs (long short-term memory cells)
      * GRUs (gated recurrent units)
      * CNNs
  * Learn how to evaluate your NLP models

---

### 🛠 09. Milestone Project 2: SkimLit
  * Replicate a the model which powers the PubMed 200k paper to classify different sequences in PubMed medical abstracts (which can help researchers read through medical abstracts faster)

---

### 🛠 10. Time Series fundamentals in TensorFlow
  * Learn how to diagnose a time series problem (building a model to make predictions based on data across time, e.g. predicting the stock price of AAPL tomorrow)
  * Prepare data for time series neural networks (features and labels)
  * Understanding and using different time series evaluation methods
    * MAE — mean absolute error
  * Build time series forecasting models with TensorFlow
    * RNNs (recurrent neural networks)
    * CNNs (convolutional neural networks)
   
## Table of materials 📖
This table is the ground truth for course materials. All the links you need for everything will be here.

Key:
* **Number:** The number of the target notebook (this may not match the video section of the course but it ties together all of the materials in the table)
* **Notebook:** The notebook for a particular module with lots of code and text annotations (notebooks from the videos are based on these)
* **Data/model:** Links to datasets/pre-trained models for the associated notebook

| Number | Notebook | Data/Model |
| ----- |  ----- |  ----- |
| 00 | [TensorFlow Fundamentals](https://github.com/ZaQuyn/TensorFlow-Developer-Certificate-Bootcamp/blob/main/00_Tensorflow_Fundamental.ipynb) |  |
| 01 | [TensorFlow Regression](https://github.com/ZaQuyn/TensorFlow-Developer-Certificate-Bootcamp/blob/main/01_Neural_network_regression_with_tensorflow.ipynb) |  |
| 02 | [TensorFlow Classification](https://github.com/ZaQuyn/TensorFlow-Developer-Certificate-Bootcamp/blob/main/02_Classification_neural_network_with_tensorflow.ipynb) |  |
| 03 | [TensorFlow Computer Vision](https://github.com/ZaQuyn/TensorFlow-Developer-Certificate-Bootcamp/blob/main/04_Transfer_learning_in_tensorflow_part_1_feature_extraction.ipynb) | [`pizza_steak`](https://storage.googleapis.com/ztm_tf_course/food_vision/pizza_steak.zip), [`10_food_classes_all_data`](https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_all_data.zip) |
| 04 | [Transfer Learning Part 1: Feature extraction](https://github.com/ZaQuyn/TensorFlow-Developer-Certificate-Bootcamp/blob/main/04_Transfer_learning_in_tensorflow_part_1_feature_extraction.ipynb) | [`10_food_classes_10_percent`](https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_10_percent.zip) |
| 05 | [Transfer Learning Part 2: Fine-tuning](https://github.com/ZaQuyn/TensorFlow-Developer-Certificate-Bootcamp/blob/main/05_Transfer_learning_in_tensorflow_part_2_fine_tuning.ipynb) | [`10_food_classes_10_percent`](https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_10_percent.zip), [`10_food_classes_1_percent`](https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_1_percent.zip), [`10_food_classes_all_data`](https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_all_data.zip) |
| 06 | [Transfer Learning Part 3: Scaling up](https://github.com/ZaQuyn/TensorFlow-Developer-Certificate-Bootcamp/blob/main/06_Transfer_learning_in_tensorflow_part_3_scaling_up.ipynb) | [`101_food_classes_10_percent`](https://storage.googleapis.com/ztm_tf_course/food_vision/101_food_classes_10_percent.zip), [`custom_food_images`](https://storage.googleapis.com/ztm_tf_course/food_vision/custom_food_images.zip), [`fine_tuned_efficientnet_model`](https://storage.googleapis.com/ztm_tf_course/food_vision/06_101_food_class_10_percent_saved_big_dog_model.zip) |
| 07 | [Milestone Project 1: Food Vision 🍔👁](https://github.com/ZaQuyn/TensorFlow-Developer-Certificate-Bootcamp/blob/main/07_Milestone_project_1_food_vision.ipynb), [Template (your challenge)](https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/extras/TEMPLATE_07_food_vision_milestone_project_1.ipynb) | [`feature_extraction_mixed_precision_efficientnet_model`](https://storage.googleapis.com/ztm_tf_course/food_vision/07_efficientnetb0_feature_extract_model_mixed_precision.zip), [`fine_tuned_mixed_precision_efficientnet_model`](https://storage.googleapis.com/ztm_tf_course/food_vision/07_efficientnetb0_fine_tuned_101_classes_mixed_precision.zip) |
| 08 | [TensorFlow NLP Fundamentals](https://github.com/ZaQuyn/TensorFlow-Developer-Certificate-Bootcamp/blob/main/08_Introduction_to_NLP_in_tensorflow.ipynb) | [`diaster_or_no_diaster_tweets`](https://storage.googleapis.com/ztm_tf_course/nlp_getting_started.zip), [`USE_feature_extractor_model`](https://storage.googleapis.com/ztm_tf_course/08_model_6_USE_feature_extractor.zip) |
| 09 | [Milestone Project 2: SkimLit 📄🔥](https://github.com/ZaQuyn/TensorFlow-Developer-Certificate-Bootcamp/blob/main/09_SkimLit_NLP_milestone_project_2.ipynb) | [`pubmed_RCT_200k_dataset`](https://github.com/Franck-Dernoncourt/pubmed-rct.git), [`skimlit_tribrid_model`](https://storage.googleapis.com/ztm_tf_course/skimlit/skimlit_tribrid_model.zip) |
| 10 | [TensorFlow Time Series Fundamentals & Milestone Project 3: BitPredict 💰📈](https://github.com/ZaQuyn/TensorFlow-Developer-Certificate-Bootcamp/blob/main/10_Time_series_forecasting_with_tensorflow.ipynb) | [`bitcoin_price_data_USD_2013-10-01_2021-05-18.csv`](https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/extras/BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv) |
