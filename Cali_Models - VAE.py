#%%
import matplotlib.pyplot as plt
import numpy as np

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
hists = np.load('lagged_hists_ox.npy')
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
#lagged_hists = np.delete(lagged_hists, [0,1,2,3,4,5,6],3)
#print(lagged_hists.shape)
''

lagged_yields = []
for i in range(len(hists)):
    lagged_yields.append(hists[i,1])
lagged_yields = np.array(lagged_yields)
print(lagged_yields.shape)

# Reshape
lagged_hists = np.transpose(lagged_hists, [0,2,1,3])
lagged_hists = np.reshape(lagged_hists,[lagged_hists.shape[0],-1,lagged_hists.shape[2]*lagged_hists.shape[3]])
print('Reshaped:', lagged_hists.shape)

split = int(0.8 * len(lagged_hists))
hists_train = lagged_hists[:split]
yields_train = lagged_yields[:split]
hists_val = lagged_hists[split:]
yields_val = lagged_yields[split:]
print('Train:', hists_train.shape, yields_train.shape)
print('Validate:', hists_val.shape, yields_val.shape)
#%%
import tensorflow as tf
#from keras_self_attention import SeqSelfAttention
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
tf.keras.backend.clear_session()
from numpy.random import seed
seed(1)
tf.random.set_seed(5)

n_past = hists_train.shape[1]
n_features = hists_train.shape[2]
n_future = 16
encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))
encoder_l1 = tf.keras.layers.LSTM(100,return_sequences = True, return_state=True)
encoder_outputs1 = encoder_l1(encoder_inputs)
encoder_states1 = encoder_outputs1[1:]
encoder_l2 = tf.keras.layers.LSTM(100, return_state=True)
encoder_outputs2 = encoder_l2(encoder_outputs1[0])
encoder_states2 = encoder_outputs2[1:]
#
decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs2[0])
#
decoder_l1 = tf.keras.layers.LSTM(100, return_sequences=True)(decoder_inputs,initial_state = encoder_states1)
decoder_l2 = tf.keras.layers.LSTM(100, return_sequences=True)(decoder_l1,initial_state = encoder_states2)

decoder_outputs2 = tf.keras.layers.Dense(1)(tf.keras.layers.Flatten()(decoder_l2))
#
model_e2d2 = tf.keras.models.Model(encoder_inputs,decoder_outputs2)
#
model_e2d2.summary()
l1 = tf.keras.losses.Huber()
l2 = 'mean_absolute_error'
model_e2d2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4), loss=l2)

from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint
def decay_schedule(epoch, lr):
    # decay by 0.1 in these specific epochs
    if (epoch %10 == 0) and (epoch != 0):
        lr = lr * 0.5
    return lr

lr_scheduler = LearningRateScheduler(decay_schedule)
mcp_save = ModelCheckpoint('best_s2y_vae_ox.hdf5', save_best_only=True, monitor='val_loss', mode='min')
history = model_e2d2.fit(hists_train, yields_train, validation_data=(hists_val, yields_val), epochs=20\
                    , batch_size=32, callbacks=[mcp_save], verbose=1)

'''
    TESTING
'''

train_loss = history.history['loss']
val_loss = history.history['val_loss']

print('End Train:', np.round(np.min(train_loss),3), ' End Val:', np.round(val_loss[-1],3), ' Min Val:', np.round(np.min(val_loss),3))
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs'); plt.ylabel('Loss')
plt.legend(); plt.show();

from sklearn.metrics import r2_score
preds = model_e2d2.predict(hists_val).flatten()

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
#model_e2d2.load_weights('Best Models VAE/best_s2p_vae_sa.hdf5')
model_e2d2.load_weights('best_s2y_vae_sa.hdf5')
from sklearn.metrics import r2_score
preds_val = model_e2d2.predict(hists_val).flatten()

RMSE_CNN = np.sqrt(np.mean((preds_val - yields_val)**2))
MAE_CNN = np.mean(np.abs(preds_val - yields_val))
r2_CNN = r2_score(yields_val, preds_val)
agm_CNN = ((RMSE_CNN + MAE_CNN)/2)*(1-r2_CNN)
print ("MAE of CNN:",MAE_CNN)
print ("RMSE of CNN:", RMSE_CNN)
print ("R2 score of CNN:",r2_CNN)
print ("AGM score of CNN:",agm_CNN)

from scipy.stats import spearmanr
print(spearmanr(preds_val, yields_val))
m = 30
num = np.mean(np.abs(yields_val - preds_val))
den = np.sum(np.abs(yields_train[m + 1:] - yields_train[:-(m + 1)])) / (len(yields_train) - m)
print('MASE:', num/den)


plt.plot(yields_val, label='True Values');
plt.plot(preds_val, label='Predicted Values');
plt.legend(); plt.show()
# %%
# Best Models Santa Maria/best_s2p_sm2.hdf5 all of them have linear layers
# Best Models Santa Maria/best_s2p_sm2.hdf5 and sm3 have no max pool
# Best Models Santa Maria/best_s2p_sm3.hdf5 has 128 LSTM the rest have 64
tf.keras.backend.clear_session()
model = models.Sequential()
model.add(layers.BatchNormalization(input_shape=(hists_train.shape[1], hists_train.shape[2])))
model.add(layers.Conv1D(filters=32, kernel_size=3, strides=1, padding='causal', activation='linear'))#, input_shape=(hists_train.shape[1], hists_train.shape[2])))
model.add(layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='causal', activation='linear'))
model.add(layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='causal', activation='linear'))
#model.add(layers.MaxPool1D(pool_size=3, strides=3, padding='same'))
model.add(layers.LSTM(64))
#model.add(layers.Dropout(0.3))
model.add(layers.Dense(units=128, activation='linear'))
model.add(layers.Dense(units=256, activation='linear'))
model.add(layers.Dense(units=1, activation='linear'))
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
loss = 'mean_absolute_error'
model.compile(optimizer=optimizer, loss = loss)
model.summary()

model.load_weights('Best Models Salinas/best_s2p_sa_no_maxpool.hdf5')
from sklearn.metrics import r2_score
preds_val = model.predict(hists_val).flatten()

RMSE_CNN = np.sqrt(np.mean((preds_val - yields_val)**2))
MAE_CNN = np.mean(np.abs(preds_val - yields_val))
r2_CNN = r2_score(yields_val, preds_val)
agm_CNN = ((RMSE_CNN + MAE_CNN)/2)*(1-r2_CNN)
print ("MAE of CNN:",MAE_CNN)
print ("RMSE of CNN:", RMSE_CNN)
print ("R2 score of CNN:",r2_CNN)
print ("AGM score of CNN:",agm_CNN)

from scipy.stats import spearmanr
print(spearmanr(preds_val, yields_val))
m = 30
num = np.mean(np.abs(yields_val - preds_val))
den = np.sum(np.abs(yields_train[m + 1:] - yields_train[:-(m + 1)])) / (len(yields_train) - m)
print('MASE:', num/den)


plt.plot(yields_val, label='True Values');
plt.plot(preds_val, label='Predicted Values');
plt.legend(); plt.show()
# %%
import pandas as pd
df = pd.DataFrame(preds_val)
df.to_csv('CNN-LSTM_sm_results.csv')
# %%
df_vae = pd.read_csv('VAE_results.csv',index_col=0).values
df_cl = pd.read_csv('CNN-LSTM_sm_results.csv',index_col=0).values
avg = np.mean(np.append(df_vae,df_cl,axis=1),axis=1)

y_val = yields_val.copy()

df_vae = df_vae.flatten()
df_cl = df_cl.flatten()

RMSE_CNN = np.sqrt(np.mean((df_vae - y_val)**2))
MAE_CNN = np.mean(np.abs(df_vae - y_val))
r2_CNN = r2_score(y_val, df_vae)
agm_CNN = ((RMSE_CNN + MAE_CNN)/2)*(1-r2_CNN)
print ("MAE of VAE:",MAE_CNN)
print ("RMSE of VAE:", RMSE_CNN)
print ("R2 score of VAE:",r2_CNN)
print ("AGM score of VAE:",agm_CNN)

RMSE_CNN = np.sqrt(np.mean((df_cl - y_val)**2))
MAE_CNN = np.mean(np.abs(df_cl - y_val))
r2_CNN = r2_score(y_val, df_cl)
agm_CNN = ((RMSE_CNN + MAE_CNN)/2)*(1-r2_CNN)
print ("MAE of CNN:",MAE_CNN)
print ("RMSE of CNN:", RMSE_CNN)
print ("R2 score of CNN:",r2_CNN)
print ("AGM score of CNN:",agm_CNN)

RMSE_CNN = np.sqrt(np.mean((avg - y_val)**2))
MAE_CNN = np.mean(np.abs(avg - y_val))
r2_CNN = r2_score(y_val, avg)
agm_CNN = ((RMSE_CNN + MAE_CNN)/2)*(1-r2_CNN)
print ("MAE of avg:",MAE_CNN)
print ("RMSE of avg:", RMSE_CNN)
print ("R2 score of avg:",r2_CNN)
print ("AGM score of avg:",agm_CNN)

plt.plot(y_val, label='True Values');
plt.plot(df_vae, label='VAE Values',c='y');
plt.plot(df_cl, label='CNN-LSTM Values');
plt.plot(avg, label='AVG Values',c='r');
plt.legend(); plt.show()
# %%
