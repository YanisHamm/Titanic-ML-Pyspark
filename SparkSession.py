# Starting a Spark session
import pyspark
import findspark
findspark.init()
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('Titanic-ML').getOrCreate()