from ML_model import *
from Preprocess import *

lr = LogisticRegression(labelCol='Survived')
model = lr.fit(train_ds)
prediction = model.transform(valid_ds)
accuracy.evaluate(prediction)