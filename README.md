# Text-Classification-with-Spark
This project aims to classify the customer reviews on automobiles into five classes using Spark MLLib on an Amazon EMR Hadoop cluster.

# Data Extraction

1. Download the dataset from the link, and then save it into local disk.

http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Automotive_5.json.gz

2. Run this data_extraction.py to extract the fiels of Overall and ReviewTexts, and then save them into  a text file: reviews_Automotive.txt

# Package the Scala Project with SBT

1. Download sbt.rar and unpack it into C:\project\

2. In the folder: C:\project\sbt\, run: $ sbt assembly

# Run the Spark Application on Amazon EMR Cluster

1. upload the Executable Jar file (CustomerReviewClassificationEMR-assembly-1.0.jar) located in the folder: C:\project\sbt\target\scala-2.11, to s3 bucket: s3://yifengspark

2. upload the dataset (reviews_Automotive.txt) to s3 bucket: s3://yifengsparkdata

3. sign in to Amazon AWS account, and create an EMR cluster with Spark application

4. click "Add Step" to submit the Spark application: a) in the field of Spark-submit options, enter: --class CustomerReviewClassificationEMR --verbose; b) in the field of Application Location, enter: s3://yifengspark/CustomerReviewClassificationEMR-assembly-1.0.jar

5. run the Spark application, and check the output result from s3 bucket (if you have stored the result to s3 bucket in your Scala source code.)
