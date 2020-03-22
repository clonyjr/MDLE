#!/usr/bin/env python
# coding: utf-8

# Imports
from pyspark import SparkContext as sparkc
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark.sql import DataFrameReader, SparkSession

spark = SparkSession.builder.master("local[*]").getOrCreate()

schema = StructType([
    StructField("START", StringType(), True),
    StructField("STOP", StringType(), True),
    StructField("PATIENT", StringType(), True),
    StructField("ENCOUNTER", StringType(), True),
    StructField("CODE", IntegerType(), True),
    StructField("DESCRIPTION", StringType(), True)])

df = spark.read.csv("Assignment1/conditionsample.csv",header=True,schema=schema)
df.printSchema()
df.show()

## PATIENT - CODE
patients = df.select('PATIENT', 'CODE')
patients.show()

## CODE - DESCRIPTION (DISTINCT)
diagnosis = df.select('CODE', 'DESCRIPTION').distinct()
diagnosis.show()

## Functions
### Apriori
def apriori(codePatients):
    for basket in codePatients:
        for item in basket:
            item_counts[item] += 1

    frequent_items = frequent_items_table(item_counts)

    # for (item in basket):
    for i in basket:
        if i not in frequent_items:
            continue
        for j in basket:
            if j in frequent_items:
                pair_counts[i, j] += 1

### Spark Functions
codeGroupedByPatients = df.rdd.map(lambda x: (x.PATIENT, x.CODE)).groupByKey().mapValues(set)
codeGroupedByPatients.take(10)