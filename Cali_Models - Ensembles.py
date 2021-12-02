import matplotlib.pyplot as plt
import numpy as np

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
hists = np.load('lagged_hists_Ifeanyi_35day.npy')
# restore np.load for future normal usage
np.load = np_load_old
print(hists.shape)

# Separate histograms from yields
lagged_hists = []
for i in range(len(hists)):
    lagged_hists.append(hists[i,0])
lagged_hists = np.array(lagged_hists)
print(lagged_hists.shape)

''
lagged_hists = np.delete(lagged_hists, [0,1,2,3,4,5,6],3)
print(lagged_hists.shape)
''

lagged_yields = []
for i in range(len(hists)):
    lagged_yields.append(hists[i,1])
lagged_yields = np.array(lagged_yields)
print(lagged_yields.shape)

'''
# Normalizing Attempt
from sklearn import preprocessing
lagged_yields = lagged_yields.reshape(-1, 1)
minmax_scale = preprocessing.MinMaxScaler().fit(lagged_yields)
lagged_yields = minmax_scale.transform(lagged_yields)
lagged_yields = np.squeeze()
'''

# Reshape
lagged_hists = np.transpose(lagged_hists, [0,2,1,3])
lagged_hists = np.reshape(lagged_hists,[lagged_hists.shape[0],-1,lagged_hists.shape[2]*lagged_hists.shape[3]])
print('Reshaped:', lagged_hists.shape)

split = int(0.75 * len(lagged_hists))
hists_train = lagged_hists[:split]
yields_train = lagged_yields[:split]
hists_val = lagged_hists[split:]
yields_val = lagged_yields[split:]
print('Train:', hists_train.shape, yields_train.shape)
print('Validate:', hists_val.shape, yields_val.shape)
#%%
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint
tf.keras.backend.clear_session()

n = 2

for i in range(n):
    print('Model',i + 1)

    model = models.Sequential()
    model.add(layers.BatchNormalization(input_shape=(hists_train.shape[1], hists_train.shape[2])))
    model.add(layers.Conv1D(filters=32, kernel_size=3, strides=1, padding='causal', activation='linear'))#, input_shape=(hists_train.shape[1], hists_train.shape[2])))
    model.add(layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='causal', activation='linear'))
    model.add(layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='causal', activation='linear'))
    model.add(layers.MaxPool1D(pool_size=3, strides=3, padding='same'))
    #model.add(layers.Dropout(0.2))
    #model.add(layers.BatchNormalization()) # <--- This is good
    model.add(layers.LSTM(128))
    #model.add(layers.Dropout(0.3))
    
    
    #model.add(layers.Flatten())
    
    model.add(layers.Dense(units=128, activation='linear'))
    model.add(layers.Dense(units=256, activation='linear'))
    #model.add(layers.Dense(units=512, activation='linear'))
    #model.add(layers.Dense(units=1024, activation='relu'))
    #model.add(layers.Dropout(0.3))
    model.add(layers.Dense(units=1, activation='linear'))
    optimizer = tf.keras.optimizers.Adadelta(learning_rate=6e-2)
    loss = 'mean_absolute_error'
    model.compile(optimizer=optimizer, loss = loss)


    # Model training

    def decay_schedule(epoch, lr):
        # decay by 0.1 in these specific epochs
        if (epoch %10 == 0) and (epoch != 0):
            lr = lr * 0.5
        return lr

    lr_scheduler = LearningRateScheduler(decay_schedule)
    mcp_save = ModelCheckpoint('best_ensemble35' + str(i) + '.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    history = model.fit(hists_train, yields_train, validation_data=(hists_val, yields_val), epochs=30, batch_size=32, callbacks=[lr_scheduler, mcp_save], verbose=1)
#%%
from sklearn.metrics import r2_score

n = 2
ens_preds = [[]] * n
for i in range(n):
    model = models.load_model('Yield NSR/best_ensemblensr35' + str(i) + '.hdf5')
    ens_preds[i] = model.predict(hists_val).flatten()
    RMSE_CNN = np.sqrt(np.mean((ens_preds[i] - yields_val)**2))
    MAE_CNN = np.mean(np.abs(ens_preds[i] - yields_val))
    r2_CNN = r2_score(yields_val, ens_preds[i])
    agm_CNN = ((RMSE_CNN + MAE_CNN)/2)*(1-r2_CNN)
    print ("MAE of CNN:",MAE_CNN)
    print ("RMSE of CNN:", RMSE_CNN)
    print ("R2 score of CNN:",r2_CNN)
    print ("AGM score of CNN:",agm_CNN)

avg_preds = np.mean(ens_preds,axis=0)
RMSE_CNN = np.sqrt(np.mean((avg_preds - yields_val)**2))
MAE_CNN = np.mean(np.abs(avg_preds - yields_val))
r2_CNN = r2_score(yields_val, avg_preds)
agm_CNN = ((RMSE_CNN + MAE_CNN)/2)*(1-r2_CNN)
print ("MAE of CNN:",MAE_CNN)
print ("RMSE of CNN:", RMSE_CNN)
print ("R2 score of CNN:",r2_CNN)
print ("AGM score of CNN:",agm_CNN)

from scipy.stats import spearmanr
print(spearmanr(avg_preds, yields_val))
m = 365
num = np.mean(np.abs(yields_val - avg_preds))
den = np.sum(np.abs(yields_train[m + 1:] - yields_train[:-(m + 1)])) / (len(yields_train) - m)
print('MASE:', num/den)

plt.plot(yields_val, label='True Values');
plt.plot(avg_preds, label='Predicted Values');
plt.legend();

#%% SINGLE MODEL HISTORY

train_loss = history.history['loss']
val_loss = history.history['val_loss']

print('End Train:', np.round(np.min(train_loss),3), ' End Val:', np.round(val_loss[-1],3), ' Min Val:', np.round(np.min(val_loss),3))
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs'); plt.ylabel('Loss')
plt.legend(); plt.grid(); plt.show(); 

from sklearn.metrics import r2_score
preds = model.predict(hists_val).flatten()

RMSE_CNN = np.sqrt(np.mean((preds - yields_val)**2))
MAE_CNN = np.mean(np.abs(preds - yields_val))
r2_CNN = r2_score(yields_val, preds)
agm_CNN = ((RMSE_CNN + MAE_CNN)/2)*(1-r2_CNN)
print ("MAE of CNN:",MAE_CNN)
print ("RMSE of CNN:", RMSE_CNN)
print ("R2 score of CNN:",r2_CNN)
print ("AGM score of CNN:",agm_CNN)
plt.plot(yields_val, label='True Values');
plt.plot(preds, label='Predicted Values');
plt.legend();
#%%
import pandas as pd

out = pd.DataFrame(avg_preds[:456])
out.to_csv('predictions1.csv')



#%%
#%% REALLY OLD EXPERIMENT
plt.plot(yields_val, label='True Values');
n = 2
ens_preds = [[]] * n
for i in range(n):
    model = models.load_model('Yield NSR/best_ensemblensr35' + str(i) + '.hdf5')
    ens_preds[i] = model.predict(hists_val).flatten()
preds_val = np.mean(ens_preds,axis=0)
plt.plot(preds_val, label='SIM_CNN-LSTM_Ens');

plt.legend(); plt.xlabel('Sample'); plt.ylabel('Yield'); plt.grid();
plt.title('Predicted and True Yield Values ');
