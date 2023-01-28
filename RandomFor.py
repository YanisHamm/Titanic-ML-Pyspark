from ML_model import *
from Preprocess import *

rf = RandomForestClassifier(labelCol='Survived')
model = rf.fit(train_ds)
prediction = model.transform(valid_ds)
accuracy.evaluate(prediction)