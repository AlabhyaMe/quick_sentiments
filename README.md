[![PyPI Version](https://img.shields.io/pypi/v/quick-sentiments)](https://pypi.org/project/quick-sentiments/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/quick-sentiments?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/quick-sentiments)
<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" />

#  Quick Sentiments

## Updates
The package is now live! 

```bash
!pip install quick-sentiments
```
Alternatively, you can clone the git and install it locally. (Not recommended, please try the pip install)

```bash
git clone https://github.com/AlabhyaMe/quick_sentiments.git
```
Then run the command in the command prompt or notebook where git is cloned. Make sure you are in the main directory - quick_sentiments

```
pip install .\dist\quick_sentiments-0.4.3-py3-none-any.whl # please note, sometimes I might not have updated the version number to the  latest
```

This Python package is designed to streamline natural language processing (NLP) for sentiment analysis. It achieves this by combining various vectorization techniques with machine learning models. The package automates the often complex and time-consuming vectorization process, allowing users to skip the manual coding typically required for this step. Additionally, users can easily select their preferred machine learning models to conduct sentiment analysis.


##  Features

- **End-to-End Pipeline**: Go from raw text to sentiment predictions with minimal setup.
- **Automated Preprocessing**: Includes robust text cleaning:
  - Lemmatization
  - Stop word removal
  - Punctuation handling
  - URL/emoji/HTML removal, etc.
- **Multiple Text Representation Methods**:
  - Bag-of-Words (BoW)
  - Term Frequency (TF)
  - TF-IDF (Term Frequency-Inverse Document Frequency)
  - Word Embeddings (Word2Vec - pre-trained Google News 300-dim model)
  - Glove Embedding (25,50,100 and 200)
  - Hugging Face (The module is optional but user have to download the required dependencies themself)
- **Multiple Machine Learning Algorithms**:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - Naive Bayesian
  - Neural Network
  - Naive Bayes
  - Tensorflow Models
- **Hyperparameter Tuning Support**:
  - All models are compatible with GridSearchCV. Tensorflow has it's own tuning
  - By default, models run with standard parameters for quick testing.
  - Grid search options are built-in and ready to use if needed.
- **Modular Design**: Each component is cleanly separated into its own module.
- **Prediction on New Data**: Easily apply your trained model to new, unseen data.

---

## 3. INSTRUCTIONS AND DEMO

To help users get started with this package, I have documented comprehensive instructions and a demo workbook. Please begin by reviewing quick_sentiments.pdf for an introduction to the library's capabilities.

Afterward, proceed to the Demo workbook, which contains ready-to-use examples. Please ensure that your file names and column labels are accurately set before proceeding with the instructions within the workbook. As an alternative, you may directly execute the Python script, provided your files and labels are correctly configured.

###  Training Data

Place your training CSV file in the `demo/training_data` folder.

- It must contain:
  - A column for  the raw input text. 
  - A column for sentiments

### New Data for Prediction

Place your new prediction CSV file in the `new_data/` folder.

- It must contain:
  - A column named `RawTextColumn` (or another name you configure in the notebook).

## 📚 Dataset Citation

The demo uses publicly available training data from:

> Madhav Kumar Choudhary. *Sentiment Prediction on Movie Reviews*. Kaggle.  
> [https://www.kaggle.com/datasets/madhavkumarchoudhary/sentiment-prediction-on-movie-reviews](https://www.kaggle.com/datasets/madhavkumarchoudhary/sentiment-prediction-on-movie-reviews)  
> Accessed on: 2025- 07-15

If you use this dataset in your own work, please cite the original creator as per Kaggle's [Terms of Use](https://www.kaggle.com/terms).

