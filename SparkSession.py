# Starting a Spark session
from pyspark.ml.classification import LogisticRegression,RandomForestClassifier,GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.functions import count, mean, when, lit, create_map, regexp_extract, col, split
from itertools import chain
import pyspark
import findspark
findspark.init()
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('Titanic-ML').getOrCreate()


train = spark.read.csv('./data/train.csv', header=True, inferSchema=True)
test  = spark.read.csv('./data/test.csv', header=True, inferSchema=True)
