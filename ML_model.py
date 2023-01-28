from Preprocess import *

indexer = StringIndexer(inputCols=['Sex','Embarked'],outputCols=['SexIndex', 'EmbarkedIndex'])
indexer_model = indexer.fit(train)
train_1 = indexer_model.transform(train).drop('Sex','Embarked') 

assembler = VectorAssembler(inputCols=train_1.columns[1:], outputCol = 'features')
train_1 = assembler.transform(train_1).select('features','Survived')


train_ds, valid_ds = train_1.randomSplit([0.7,0.3])

accuracy = MulticlassClassificationEvaluator(labelCol='Survived', metricName = 'accuracy')




