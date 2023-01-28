from ML_model import *
from Preprocess import *

gbt = GBTClassifier(labelCol='Survived')
model = gbt.fit(train_ds)
predictions = model.transform(valid_ds)
accuracy.evaluate(predictions)