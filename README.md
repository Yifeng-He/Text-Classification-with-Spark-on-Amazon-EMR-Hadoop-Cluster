# Text-Classification-with-Spark
This project aims to classify the customer reviews on automobiles into five classes using Spark MLLib.

# Data Extraction

1. Download the dataset from the link, and then save it into local disk.

http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Automotive_5.json.gz

2. Run this data_extraction.py to extract the fiels of Overall and ReviewTexts, and then save them into  a text file: reviews_Automotive.txt

# Text Classification

Run CustomerReviewClassification.scala to classify the customer review texts into five classes, using the input data: reviews_Automotive.txt.

# Package the Scala Project with SBT

1. Download sbt.rar and unpack it into C:\project\

2. In the folder: C:\project\sbt\, run: $ sbt assembly

3. Copy the executable JAR file from the folder C:\project\sbt\target\scala-2.11\ to the folder C:\project\, copy the data file credit_data.txt to the folder C:\project\

4. Run the Spark program: $ spark-submit CustomerReviewClassification-assembly-1.0.jar
