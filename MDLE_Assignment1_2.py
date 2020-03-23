#!/usr/bin/env python
# coding: utf-8

# # Imports
from pyspark import SparkContext as sparkc
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark.sql import DataFrameReader
from pyspark.sql import SparkSession

import collections
from collections import defaultdict
from operator import itemgetter

from itertools import islice


spark = SparkSession.builder.master("local[*]").getOrCreate()

schema = StructType([
    StructField("START", StringType(), True),
    StructField("STOP", StringType(), True),
    StructField("PATIENT", StringType(), True),
    StructField("ENCOUNTER", StringType(), True),
    StructField("CODE", IntegerType(), True),
    StructField("DESCRIPTION", StringType(), True)])

df = spark.read.csv("./Assignment1/conditions.csv",header=True,schema=schema)
# df = spark.read.csv("Assignment1/conditionsample.csv",header=True,schema=schema).select('PATIENT','CODE')

df.printSchema()
df.show()


# ## PATIENT - CODE

patients = df.select('PATIENT', 'CODE')
patients.show()


# ## CODE - DESCRIPTION (DISTINCT)

diagnosis = df.select('CODE', 'DESCRIPTION').distinct()
diagnosis.show()


# ## Functions

def take(n, iterable):
    return list(islice(iterable, n))


# ### Definir Itemset PATIENT-CODEs

codeGroupedByPatients = df.rdd.map(lambda x: (x.PATIENT, x.CODE)).groupByKey().mapValues(set)
# codeGroupedByPatients.take(10)
codeGroupedByPatients.collect()


# # Frequent Items Table

def frequent_items_table(itemCounts):
    frequentItems = list()
    supportThreshold = 1000
    for key, value in itemCounts.items():
        if key is None:
            continue
        if value < supportThreshold:
            frequentItems.append(key)
    return frequentItems


# ## Confidence - Interest

# def confidenceInterest(pairs, occurrences):
#     supportThreshold = 1000
# #     print("PAIR ", pairs)
#     for key, value in pairs.items():
#         print("PAIR ", key, " VALUE", value)
#         confidence = supportThreshold/value
# #         for item in pair:
# #         newItem = str(key.split(', '))
#         item = str(key)
#         newItem = item.split(', ')
# #         if confidence >= 0.9 and str(pair.key() + newItem[2]) in pairs:
#         if confidence >= 0.9 and str(key + newItem[2]) in pairs.items():
#             interest = confidence - occurrences[newItem[2]]/pairs.count()
#             if interest >= 0.9:
#                 print("Confidence: ", confidence, " Interest: ", interest)


def confidenceInterest(pairs, newItem, occurrences):
    supportThreshold = 1000
    for key, value in pairs.items():
        confidence = supportThreshold/value
        interest = confidence - occurrences[newItem]/len(pairs)
        if confidence >= 0.9 and interest >= 0.9:
#             print("Confidence: ", confidence, " Interest: ", interest)
            result = "%s - %s --- Confidence: %f Interest: %f" % (key, newItem, confidence, interest)
        print(result)


# # A-Priori

def apriori(codePatients):
#     codePatient = codePatients.collect()
    codePatient = codePatients.take(500)
    item_counts, pair_counts = defaultdict(int),  defaultdict(int)
    for basket in codePatient:
        for item in basket[1]:
                item_counts[item] += 1
    return item_counts    

def aprioriSecondPass(codePatients, freqItems):
#     codePatient = codePatients.collect()
    codePatient = codePatients.take(500)
    pair_counts = defaultdict(int)
    for basket in codePatient:
        for i in basket[1]:
            if i not in freqItems:
                continue
            for j in basket[1]:
                if j in freqItems:
                    pair_counts[i, j] += 1
#     return take(10, sorted(pair_counts.items(), key=itemgetter(1), reverse=True))
    return pair_counts

def aprioriThirdPass(codePatients, freqItems, pairs, occurrences):
#     codePatient = codePatients.collect()
    codePatient = codePatients.take(500)
    triple_counts = defaultdict(int)
    for basket in codePatient:
        for i in basket[1]:
            if i not in freqItems:
                continue
            for j in basket[1]:
                if j not in freqItems:
                    continue
                for z in basket[1]:
                    if z in freqItems:
                        triple_counts[i, j, z] += 1
                    confidenceInterest(pairs, z, occurrences)
    return take(10, sorted(triple_counts.items(), key=itemgetter(1), reverse=True))


result = apriori(codeGroupedByPatients)
print(result.items())


freq_items = frequent_items_table(result)
print(freq_items)


second_pass = aprioriSecondPass(codeGroupedByPatients, freq_items)
# print(second_pass.items())
print(second_pass)

third_pass = aprioriThirdPass(codeGroupedByPatients, freq_items, second_pass, result)
# print(third_pass)