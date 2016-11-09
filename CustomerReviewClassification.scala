/*
 * This program classifies the customer reviews on automobiles into five classes using Spark MLLib
 * The dataset is Amazon product data which can be obtained from:
 * http://jmcauley.ucsd.edu/data/amazon/
 */

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature._
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.{Pipeline, PipelineModel}

object CustomerReviewClassification {
  
  // function to set the log level
  def setupLogging() = {
    import org.apache.log4j.{Level, Logger}   
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)   
  }
  
  // the format of each row in the DataFrame. 
  case class Record(label : Double, reviewText : String )
  
  // the function to convert each line of string into a structured row
  def convertLinetoRow(line: String): Record = {
    val fields = line.split("\t")
    if(fields.size < 2) {
      Record(10, "NA")
    }
    else {
      Record(fields(0).toDouble, fields(1).toString)
    }  
  }
 
   // the main function
   def main(args: Array[String]): Unit = {
    
     // step 1: set up spark session 
     val spark =SparkSession.builder.config(key="spark.sql.warehouse.dir", value="file:///C:/Temp").master("local")
     .appName("CustomerReviewClassification").getOrCreate() 
     // set up log level
    setupLogging()
    
    // step 2: load the data into dataFrame
    val dataset_lines = spark.read.textFile("reviews_Automotive.txt")
    // convert the DataSet of lines to DataFrame of Rows
    import spark.implicits._
    val data_raw = dataset_lines.map(convertLinetoRow).filter(x => x.label < 9).toDF
    data_raw.printSchema()
    val min_count = data_raw.groupBy("label").count().show(false)
    println(s"Total number of reviews is ${data_raw.count()}")
    data_raw.show(5)
    
    // step 3: data pre-processing
    // split the review text into words
    val tokenizer = new RegexTokenizer().setPattern("[a-zA-Z']+").setGaps(false)
      .setInputCol("reviewText").setOutputCol("words")
    val data_tokenized = tokenizer.transform(data_raw)  
    // remove the stop words
    val remover = new StopWordsRemover().setInputCol(tokenizer.getOutputCol).setOutputCol("words_filtered")
    val data_filtered = remover.transform(data_tokenized)
    // create 2-gram words
    val ngram2 = new NGram().setN(2).setInputCol("words_filtered").setOutputCol("gram-2")
    val data_prepared = ngram2.transform(data_filtered)
    // check the data
    data_prepared.printSchema()
    data_prepared.show(5)
    // final data to be vectorized
    val data_cleaned = data_prepared.select("label", "words_filtered", "gram-2").cache()
    
    // step 4: split the data into training and test sets
    val Array(trainingData, testData) = data_cleaned.randomSplit(Array(0.8, 0.2)) 
    
    // step 5: construct the pipeline for feature extraction and classification
    // 1-gram words feature
    val hashingTF_1gram = new HashingTF().setInputCol("words_filtered")
    // 2-gram words feature
    val hashingTF_2gram = new HashingTF().setInputCol("gram-2")
    // assembler: concatenating the two feature vectors
    val assembler = new VectorAssembler().setInputCols(Array(hashingTF_1gram.getOutputCol, hashingTF_2gram.getOutputCol))
    
    // classifier: random forest
    val random_forest = new RandomForestClassifier().setLabelCol("label").setFeaturesCol(assembler.getOutputCol)
    
    // pipeline
    val pipeline = new Pipeline().setStages(Array(hashingTF_1gram, hashingTF_2gram, assembler, random_forest))
    // parameters to be tuned
    val paramGrid = new ParamGridBuilder()  
      .addGrid(hashingTF_1gram.numFeatures, Array(1000,2000))
      .addGrid(hashingTF_2gram.numFeatures, Array(1000,2000))
      .addGrid(random_forest.maxDepth, Array(5, 10)).build()
    // grid search over k-fold cross-validation
    val cross_validator = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator())
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(2) // Use 3+ in practice
 
    // step 6: train the pipeline model
    val cvModel = cross_validator.fit(trainingData)
    
    // step 7: prediction and evaluation
    val predictions = cvModel.transform(testData)
    // check the output DataFrame
    predictions.printSchema()
    predictions.show(5)
    // evaluate
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println("Accuracy = " + accuracy) 

    // step 8: check the best parameters obtained by grid search   
    val bestPipelineModel = cvModel.bestModel.asInstanceOf[PipelineModel]  
    val stages = bestPipelineModel.stages  
    val ngram1HashingTFStage = stages(0).asInstanceOf[HashingTF]  
    val ngram2HashingTFStage = stages(1).asInstanceOf[HashingTF]
    val classifier_model = stages(3).asInstanceOf[RandomForestClassificationModel]   
    println("Best Model Parameters:")  
    println("ngram1HashingTFStage numFeatures = " + ngram1HashingTFStage.getNumFeatures)  
    println("ngram2HashingTFStage numFeatures = " + ngram2HashingTFStage.getNumFeatures)
    println("Learned classification random forest model:\n" + classifier_model.toDebugString)
    
    // stop the spark session
    spark.stop()     
  } 
}


