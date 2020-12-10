import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.externals.joblib import dump

seed = 42
np.random.seed(seed)
test_size = 0.2
batch_size = 25
epochs = 200
verbose = 0

df = pd.read_csv("model_training_data.csv")
df['Match Date'] = pd.to_datetime(df['Match Date'])
df = df[df['Match Date'] > '01/01/2005']  # Ignore first year of matches while skill level was being assessed

X = df[['Win prob','Team skill','Opp skill','Team ranking pts','Opposition ranking pts']]
Y_score = df[['For', 'Aga']]
Y_diff = df[['Diff']]

X = X.values
Y_score = Y_score.values
Y_diff = Y_diff.values

'''
def baseline_model():

    # create model
    model = Sequential()
    model.add(Dense(15, input_dim=5, kernel_initializer='normal', activation='relu'))
    model.add(Dense(8, kernel_initializer='normal', activation='relu'))
    model.add(Dense(2, kernel_initializer='normal'))

    # Compile model
    model.compile(loss='mse', optimizer='adam')
    return model

# Evaluate model
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=epochs, batch_size=batch_size, verbose=verbose)))
pipeline = Pipeline(estimators)

kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y_score, cv=kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))
'''

# Train final model

sc = StandardScaler()
X = sc.fit_transform(X)

# Save scaler
dump(sc, 'std_scaler.bin', compress=True)

model = Sequential()
model.add(Dense(15, input_dim=5, kernel_initializer='normal', activation='relu'))
model.add(Dense(8, kernel_initializer='normal', activation='relu'))
model.add(Dense(2, kernel_initializer='normal'))

model.compile(loss='mse', optimizer='adam')
out = model.fit(X, Y_score, validation_split=0.25, epochs=epochs, batch_size=batch_size, verbose=verbose)

plt.plot(out.history['loss'])
plt.plot(out.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

# Save model and architecture to single file
model.save("model.h5")
print("Saved model to disk")