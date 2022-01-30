library(tidyr)
library(NLP)
library(tm)
library(openNLP) 
library(stringr)
library(tidyverse)
library(spacyr)
library(stringr) 
library(tidyverse)
library(gmodels)  # Crosstable
library(tm)
library(wordcloud)
library(e1071)

#Load the data and keep the interesting columns 
films <- read.csv("../input/rotten-tomatoes-movies-and-critic-reviews-dataset/rotten_tomatoes_critic_reviews.csv", sep = ",", na.strings=c("","NA"))
films <- films[,c('review_type','review_score','review_content')]
#Encode the review type from Fresh to 1 and Rotten to 0 to create the factor.
films$review_type<-ifelse(films$review_type=="Fresh",1,0)
#Omit the NA's due to great loads of data
films <- na.omit(films)

#Create a two level factor with Negative and Positive
films$review_type <- factor(films$review_type, levels = c(0, 1),
                            labels = c("Negative","Positive"))

return(films)
#Set the seed for comparing the results and limit the amount of data
set.seed(1234)
films <- films[order(runif(n=15000)),]

#Create the corpus with the review content
corpus <- Corpus(VectorSource(films$review_content))
inspect(corpus[1:5])

#Transform to lower cases and remove the numbers
clean.corpus <- tm_map(corpus, tolower)
clean.corpus <- tm_map(clean.corpus, removeNumbers)
#Remove the stop words and punctuation
clean.corpus <- tm_map(clean.corpus, removeWords, stopwords())
clean.corpus <- tm_map(clean.corpus, removePunctuation)
#Remove extra whitespace
clean.corpus <- tm_map(clean.corpus, stripWhitespace)
inspect(clean.corpus[1:5])

#Create the document term matrix
clean.corpus.dtm <- DocumentTermMatrix(clean.corpus)

#Create the split for train and data
n <- nrow(films)
X_train <- films[1:round(.8 * n),]
X_test  <- films[(round(.8 * n)+1):n,]

nn <- length(clean.corpus)
Y_train <- clean.corpus[1:round(.8 * nn)]
Y_test  <- clean.corpus[(round(.8 * nn)+1):nn]

nnn <- nrow(clean.corpus.dtm)
matrix_train <- clean.corpus.dtm[1:round(.8 * nnn),]
matrix_test  <- clean.corpus.dtm[(round(.8 * nnn)+1):nnn,]

#Separate between positive and negative instances in the train dataset
positive <- subset(X_train, review_type == "Positive")
negative <- subset(X_train, review_type == "Negative")

#Worldclouds of the positive and negative subsets
wordcloud(positive$review_content, max.words = 30, scale=c(10,.3))
wordcloud(negative$review_content, max.words = 30, scale=c(10,.3))

#Find the most frequent terms in the train matrix 
freq.terms <- findFreqTerms(matrix_train, 5)
matrix_freq_train <- DocumentTermMatrix(Y_train, list(dictionary = freq.terms))
matrix_freq_test  <- DocumentTermMatrix(Y_test, list(dictionary = freq.terms))

#Convert the counts and create a factor with  no and yes and apply it to the train and test 
convert_counts <- function(x) {
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels = c(0, 1), labels = c("No", "Yes"))
  return(x)
}


matrix_freq_train <- apply(matrix_freq_train, MARGIN = 2, convert_counts)
matrix_freq_test  <- apply(matrix_freq_test, MARGIN = 2, convert_counts)

#Model implementation
text.classifer.imp <- naiveBayes(matrix_freq_train, 
                                 X_train$review_type,
                                 laplace = 1)

text.pred.imp <- predict(text.classifer.imp, 
                         matrix_freq_test)

CrossTable(text.pred.imp, X_test$review_type,
           prop.chisq = FALSE, 
           prop.t = FALSE,
           dnn = c('predicted', 'actual'))

#Second model
films <- read.csv("../input/rotten-tomatoes-movies-and-critic-reviews-dataset/rotten_tomatoes_critic_reviews.csv", sep = ",", na.strings=c("","NA"))
films <- films[,c('review_type','review_score','review_content')]
films$review_type<-ifelse(films$review_type=="Fresh",1,0)
films <- na.omit(films)

set.seed(1234)
films <- films[order(runif(n=15000)),]

library(stringr)
films$first <- word(films$review_score, 1, sep = fixed('/'))
films$first<- as.numeric(films$first) 
films$third = word(films$review_score, -1, sep = fixed('/'))
films$third<- as.numeric(films$third) 
films$percentage <- (films$first / films$third)*100
head(films$percentage)

films$rating=ifelse(films$percentage>=50,1,0)

films <- films[,c('review_type','review_score','review_content','percentage','rating')]
films <- na.omit(films)

films$rating <- factor(films$rating, levels = c(0, 1),
                       labels = c("Negative","Positive"))

return(films)

#Create the corpus with the review content
corpus <- Corpus(VectorSource(films$review_content))
#Transform to lower cases and remove the numbers
clean.corpus <- tm_map(corpus, tolower)
clean.corpus <- tm_map(clean.corpus, removeNumbers)
#Remove the stop words and punctuation
clean.corpus <- tm_map(clean.corpus, removeWords, stopwords())
clean.corpus <- tm_map(clean.corpus, removePunctuation)
#Remove extra whitespace
clean.corpus <- tm_map(clean.corpus, stripWhitespace)
#Create the document term matrix
clean.corpus.dtm <- DocumentTermMatrix(clean.corpus)

n <- nrow(films)
X_train <- films[1:round(.8 * n),]
X_test  <- films[(round(.8 * n)+1):n,]

nn <- length(clean.corpus)
Y_train <- clean.corpus[1:round(.8 * nn)]
Y_test  <- clean.corpus[(round(.8 * nn)+1):nn]

nnn <- nrow(clean.corpus.dtm)
matrix_train <- clean.corpus.dtm[1:round(.8 * nnn),]
matrix_test  <- clean.corpus.dtm[(round(.8 * nnn)+1):nnn,]

positive <- subset(X_train, rating == "Positive")
negative <- subset(X_train, rating == "Negative")

freq.terms <- findFreqTerms(matrix_train, 3)
matrix_freq_train <- DocumentTermMatrix(Y_train, list(dictionary = freq.terms))
matrix_freq_test  <- DocumentTermMatrix(Y_test, list(dictionary = freq.terms))

#Convert the counts and create a factor with  no and yes and apply it to the train and test 
convert_counts <- function(x) {
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels = c(0, 1), labels = c("No", "Yes"))
  return(x)
}

matrix_freq_train <- apply(matrix_freq_train, MARGIN = 2, convert_counts)
matrix_freq_test  <- apply(matrix_freq_test, MARGIN = 2, convert_counts)

text.classifer.imp <- naiveBayes(matrix_freq_train, 
                                 X_train$rating,
                                 laplace = 1)

text.pred.imp <- predict(text.classifer.imp, 
                         matrix_freq_test)

CrossTable(text.pred.imp, X_test$rating,
           prop.chisq = FALSE, 
           prop.t = FALSE,
           dnn = c('predicted', 'actual'))
