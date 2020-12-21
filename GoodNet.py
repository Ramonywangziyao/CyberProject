import keras
import sys
import h5py
import numpy as np
import keras.backend as K
from keras import initializers

data_filename = str(sys.argv[1])
clean_data_filename = str(sys.argv[2])
model_filename = str(sys.argv[3])
# entropy threshold
ethd = 0.6

def data_loader(path):
    data = h5py.File(filepath, 'r')
    X = np.array(data['data'])
    y = np.array(data['label'])
    X = X.transpose((0,2,3,1))
    X /= 255
    return X, y

def superimpose(img1, img2):
  result = cv2.addWeighted(img1, 1, img2, 1, 0)
  return result

def getEntropy(img, cleanX, model, entropyRange):
    entropy_sum = [0] * entropyRange
    overlaid_x = [0] * entropyRange

    for i in range(entropyRange):
        result = (superimpose(img, cleanX[i]))
        overlaid_x[i] = result

    overlaid_x = np.array(overlaid_x)
    overlaid_y = model.predict(overlaid_x)
    entropySum = -np.nansum(overlaid_y * np.log2(overlaid_y))

    return entropySum

# data preparation
X, y = data_loader(data_filename)
n = len(X)
cleanX, cleanY = data_loader(clean_data_filename)
cleanN = len(cleanX)
entropyRange = int(cleanN * 0.1)
model = keras.models.load_model(model_filename)

# processing
predicts = []
for i in range(n):
    x = X[i]
    entropyX = getEntropy(x, cleanX, model, entropyRange)
    if entropyX < ethd:
        predicts.append(n + 1)
    else:
        label = model.predict(np.array(x))
        predict.append(label)


print(predicts)
