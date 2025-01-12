MIT License

Copyright (c) [2024] [Prakash Kumar]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following condition:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.












# Install the required packages
install.packages(c("tm", "SnowballC", "caret", "e1071","readr", "ggplot2"))

# Load the libraries
library(tm)          # For text mining
library(SnowballC)   # For stemming
library(caret)       # For machine learning
library(e1071)       # For Naive Bayes
library(ggplot2)     # For visualization
library(readr)

data <- read_csv(file.choose()) # Load the dataset
head(data)                      # Check the first few rows of the data

# Convert the 'Headline' column to a corpus (collection of text data)
corpus <- Corpus(VectorSource(data$headline)) #In place of headline use name of the column which has text as data 

# Preprocess the corpus
corpus <- tm_map(corpus, content_transformer(tolower))          # Convert to lowercase
corpus <- tm_map(corpus, removePunctuation)                     # Remove punctuation
corpus <- tm_map(corpus, removeNumbers)                         # Remove numbers
corpus <- tm_map(corpus, removeWords, stopwords("en"))          # Remove stopwords (e.g., "the", "is")
corpus <- tm_map(corpus, stripWhitespace)                       # Remove extra spaces

#Viewing the corpus
inspect(corpus[1:5]) # View the first 5 documents in the corpus
corpus              # View the entire corpus (not recommended for large corpora)
#corpus_df <- data.frame(text = sapply(corpus, content)) # View the corpus into a data frame
#head(corpus_df)   # View the first few rows

# Create the Document-Term Matrix
dtm <- DocumentTermMatrix(corpus)
dtm_matrix <- as.matrix(dtm)     # Convert DTM to a matrix format
dim(dtm_matrix)                 # Check the dimensions of the DTM (number of rows and columns)
head(dtm_matrix)                #view dtm matrx
summary(dtm)

#Split Data into Training and Testing Sets

set.seed(123) # Set a random seed for reproducibility
trainIndex <- createDataPartition(data$clickbait, p = 0.8, list = FALSE) # Split the data into training (80%) and testing (20%) sets
trainData <- dtm_matrix[trainIndex, ]
testData <- dtm_matrix[-trainIndex, ]# Create training and testing datasets
trainLabels <- data$clickbait[trainIndex]
testLabels <- data$clickbait[-trainIndex]# Create labels for training and testing sets

# Train the Naive Bayes Model

model <- naiveBayes(trainData, trainLabels) # Train a Naive Bayes model using the training data
model    # Print the model to check the trained details

# Make predictions on the testing data

predictions <- predict(model, testData)
predictions <- factor(predictions, levels = c("0", "1")) # Ensure both predictions and testLabels are factors with the same levels
testLabels <- factor(testLabels, levels = c("0", "1")) # Ensure both predictions and testLabels are factors with the same levels
head(predictions) # Check the first few predictions
# Evaluate the model using a confusion matrix
confusionMatrix(predictions, testLabels)


# Visualize the Results
ggplot(data, aes(x = clickbait)) +        # Create a bar plot to visualize the distribution of labels (Clickbait vs Non-Clickbait)
  geom_bar() +
  theme_minimal() +
  ggtitle("Distribution of Clickbait vs Non-Clickbait")

