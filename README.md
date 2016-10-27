# -Text-Classification-with-Spark
This project aims to classify the customer reviews on automobiles into five classes using Spark MLLib.

# Data extraction

1. Download the dataset from the link, and then save it into local disk.

http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Automotive_5.json.gz

2. Run this data_extraction.py to extract the fiels of Overall and ReviewTexts, and then save them into  a text file: reviews_Automotive.txt

# Text Classification

Run CustomerReviewClassification.scala to classify the customer review texts into five classes, using the input data: reviews_Automotive.txt.
