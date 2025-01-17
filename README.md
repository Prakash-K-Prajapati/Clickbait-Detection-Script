# Clickbait Detection System

## Project Overview
The Clickbait Detection System is designed to classify headlines or text into two categories: "Clickbait" and "Non-Clickbait." By leveraging text preprocessing techniques and machine learning models, this system aims to improve content quality evaluation and help identify sensationalized headlines.

---

## Features
- **Text Preprocessing**: Converts text to lowercase, removes punctuation, numbers, and stopwords, and strips extra whitespace.
- **Document-Term Matrix (DTM)**: Transforms textual data into numerical features for model training.
- **Naive Bayes Classification**: Employs a machine learning algorithm suitable for text-based classification tasks.
- **Data Visualization**: Provides graphical representation of data distribution (clickbait vs non-clickbait).
- **Performance Evaluation**: Confusion matrix to assess the model's accuracy, precision, recall, and F1-score.

---

## Project Structure
1. **Input Dataset**: A CSV file containing headlines and their corresponding labels (1 for Clickbait, 0 for Non-Clickbait).
2. **R Script**: Implements preprocessing, training, and evaluation workflows.
3. **Visualization Output**: A bar chart depicting the distribution of labels.

---

## Prerequisites
- R programming environment.
- R packages:
  - `tm` (Text Mining)
  - `SnowballC` (Stemming)
  - `caret` (Machine Learning)
  - `e1071` (Naive Bayes)
  - `ggplot2` (Visualization)
  - `readr` (Data Import)

---

## Installation
1. Install R from [CRAN](https://cran.r-project.org/).
2. Install required R packages by running the following command in the R console:
   ```R
   install.packages(c("tm", "SnowballC", "caret", "e1071", "readr", "ggplot2"))
   ```

---

## Usage
1. **Prepare Dataset**:
   - Ensure the input CSV file contains at least two columns: `headline` (text data) and `clickbait` (labels: 1 for clickbait, 0 for non-clickbait).
2. **Run the Script**:
   - Load the dataset using the file chooser dialog in the script.
   - Execute the R script `Clickbait_Detection_Script.R` to preprocess data, train the model, and evaluate performance.
3. **Output**:
   - View the confusion matrix for model evaluation.
   - Inspect the bar chart of clickbait vs non-clickbait distribution.

---

## Workflow
1. **Data Loading**:
   - The dataset is loaded using `read_csv()`.
2. **Text Preprocessing**:
   - Corpus creation and cleaning (lowercasing, punctuation and stopword removal, etc.).
3. **Feature Engineering**:
   - Create a Document-Term Matrix (DTM) to represent text numerically.
4. **Model Training**:
   - Split the dataset into training (80%) and testing (20%) sets.
   - Train a Naive Bayes classifier using the `caret` and `e1071` packages.
5. **Evaluation**:
   - Use a confusion matrix to measure the model’s accuracy, precision, recall, and F1-score.
6. **Visualization**:
   - Generate a bar plot to understand the data distribution.

---

## Sample Outputs
- **Confusion Matrix**: Evaluates model performance.
- **Bar Plot**: Visualizes the distribution of clickbait and non-clickbait labels in the dataset.

---

## Future Improvements
- Experiment with advanced models (e.g., Random Forests, Gradient Boosting, or deep learning models like LSTMs).
- Improve feature extraction using TF-IDF or contextual embeddings (e.g., Word2Vec, BERT).
- Address potential class imbalance using oversampling or undersampling techniques.
- Add more visualizations, such as word clouds or feature importance plots.

---

## License
This project is licensed under the MIT License. See the `Clickbait_Detection_Script.R` file for more details.

---

## Authors
- **Prakash Kumar**  
  B.Tech in Computer Science and Engineering  
  Noida Institute of Engineering and Technology

