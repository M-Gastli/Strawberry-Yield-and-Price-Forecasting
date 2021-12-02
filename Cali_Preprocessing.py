#%%
import matplotlib.pyplot as plt
import numpy as np
import gdal
gdal.UseExceptions()
#%%
# Surface Reflectance
filename = 'Midwest/sr01_047_2019-01-04'
sr_img = np.array(gdal.Open(filename).ReadAsArray(), dtype='uint16')
print(sr_img.shape)
plt.imshow(sr_img[0])
#%%
# Temperature
filename = 'Cali Temperature/06_083_2019.tif'
temp_img = np.array(gdal.Open(filename).ReadAsArray(), dtype='uint16')
print(temp_img.shape)
plt.imshow(temp_img[0]); plt.colorbar();
plt.title('MODIS Santa Barbara County: Temperature');
#%%
# Moisture
filename = 'Cali Moisture/06_083_2019.tif'
moist_img = np.array(gdal.Open(filename).ReadAsArray(), dtype='uint16')
print(moist_img.shape)
plt.imshow(moist_img[501]); plt.colorbar();
plt.title('Santa Barbara County: Moisture');
#%%
plt.imshow(sr_img[15]); plt.title('MODIS Santa Barbara: Surface Reflectance');
plt.colorbar(); plt.show()
plt.imshow(temp_img[5]); plt.title('MODIS Santa Barbara: Temperature');
plt.colorbar(); plt.show()
plt.imshow(moist_img[5]); plt.title('MODIS Santa Barbara: Moisture');
plt.colorbar(); plt.show()
#%%
# Landcover
filename = 'Cali Landcover/06_083_2019.tif'
mask_img = np.array(gdal.Open(filename).ReadAsArray(), dtype='uint16')
print(mask_img.shape)
plt.imshow(mask_img[0])
#%%
def preprocess(img, tag, bands, mask):
    masks = mask.copy()
    #Adding empty images
    temp = []
    for i in range(0,len(img), bands):
        temp.append(img[i:i + bands])
    temp = np.array(temp)
    print(temp.shape)
    plt.imshow(temp[0,0,:,:]);
    plt.show()
    
    # UNBLOCK ONLY FOR MISSING IMAGES IN SR 2016
    """
    a = np.append(np.append(np.append(temp[:410], np.zeros((1,7,460,396)), axis=0),\
                            temp[410:413], axis=0), np.zeros((9,7,460,396)), axis=0)
    temp = np.append(a, temp[413:], axis=0)
    print(temp.shape)
    """
    
    # UNBLOCK ONLY FOR MISSING IMAGES IN SR 2019
    """
    a = np.append(np.append(np.append(np.append(np.zeros((1,7,460,396)), temp[:38], axis=0),\
                        np.zeros((1,7,460,396)), axis=0), temp[38:56], axis=0), np.zeros((1,7,460,396)), axis=0)
    temp = np.append(a, temp[56:], axis=0)
    print(temp.shape)
    """
    
    # UNBLOCK ONLY FOR MISSING IMAGES IN TEMP 2016
    """
    a = np.append(temp[:414], np.zeros((9,2,460,396)), axis=0)
    temp = np.append(a, temp[414:], axis=0)
    print(temp.shape)
    """
    # CODE BLOCK ONLY FOR MOISTURE
    if tag == 'moist':
        ttemp_imgs = []
        for i in range(temp.shape[0]):
            if i == 0:
                ttemp_imgs.append(temp[i]); ttemp_imgs.append(temp[i]); ttemp_imgs.append(temp[i]); ttemp_imgs.append(temp[i])
            if i == (temp.shape[0] - 1):
                ttemp_imgs.append(temp[i]);ttemp_imgs.append(temp[i]); ttemp_imgs.append(temp[i]); ttemp_imgs.append(temp[i]);
                break
            ttemp_imgs.append(temp[i]); ttemp_imgs.append(temp[i]); ttemp_imgs.append(temp[i])
        temp = np.array(ttemp_imgs)
        print(temp.shape)
    
    # Masking
    masks[masks != 12] = 0
    masks[masks == 12] = 1

    masked_img = temp.copy()
    for i in range(temp.shape[0]):
        for b in range(temp.shape[1]):
            if int(i/365) >= masks.shape[0]:
                masked_img[i,b] = temp[i,b]*masks[-1]
                continue
            else:
                masked_img[i,b] = temp[i,b]*masks[int(i/365)]
    print('masked shape:',masked_img.shape)
    plt.imshow(masked_img[0,0,:,:]);
    plt.show()
    print('max 0', np.max(masked_img[:,0]))
    print('max 1', np.max(masked_img[:,1]))
    # Histograms
    print('\nHistograms:')
    if tag == 'sr':
        # Surface Reflectance
        bin_seq = np.linspace(1, 16999, 33)
    if tag == 'temp':
        # Temperature
        bin_seq = np.linspace(12999, 16999, 33)
    if tag == 'moist':
        bin_seq = np.linspace(1, 149, 33)
        
    n_imgs = masked_img.shape[0]
    bands = masked_img.shape[1]
    bins = 32
    hist_temp = np.zeros((n_imgs,bands,bins))
    
    for i in range(masked_img.shape[0]):
        for b in range(masked_img.shape[1]):
            density, _ = np.histogram(masked_img[i,b], bin_seq, density=False)
            if float(density.sum()) == 0:
                continue
            hist_temp[i,b] = density / float(density.sum())

    total = 0
    e = 0
    print(hist_temp.shape)
    for i in range(hist_temp.shape[0]):
        for b in range(hist_temp.shape[1]):
            total+=1
            if np.sum(hist_temp[i,b]) == 0:
                e+=1
    print('Total Histograms:',total)
    print('Missing Histograms:',e)
    total = 0
    sum = 0
    print(masked_img.shape)
    for i in range(masked_img.shape[0]):
        for b in range(masked_img.shape[1]):
            total+=1
            if np.sum(masked_img[i,b]) == 0:
                sum+=1
    print('Total Images:',total)
    print('Missing Images',sum)
    
    if tag == 'sr':
        np.save('Cali Surface Reflectance/hist_sr_2019', hist_temp)
        print('saved')
    if tag == 'temp':
        np.save('Cali Temperature/hist_temp_2019', hist_temp)
        print('saved')
    if tag == 'moist':
        np.save('Cali Moisture/hist_moist_2019', hist_temp)
        print('saved')
    
preprocess(moist_img, 'moist', 2, mask_img[0:])
mask_img.shape

#%%

'''
    Combining Histograms
'''

# Surface Reflectance
sr_hist1 = np.load('Cali Surface Reflectance/hist_sr_2011-2012.npy')
sr_hist2 = np.load('Cali Surface Reflectance/hist_sr_2013-2014.npy')
sr_hist3 = np.load('Cali Surface Reflectance/hist_sr_2015-2016.npy')
sr_hist4 = np.load('Cali Surface Reflectance/hist_sr_2017-2018.npy')
sr_hist5 = np.load('Cali Surface Reflectance/hist_sr_2019.npy')
sr_hists = np.append(np.append(np.append(np.append(sr_hist1, sr_hist2, axis=0), sr_hist3, axis=0), sr_hist4, axis=0), sr_hist5, axis=0)

# Temperature
temp_hist1 = np.load('Cali Temperature/hist_temp_2011-2014.npy')
temp_hist2 = np.load('Cali Temperature/hist_temp_2015-2018.npy')
temp_hist3 = np.load('Cali Temperature/hist_temp_2019.npy')
temp_hists = np.append(np.append(temp_hist1, temp_hist2, axis=0), temp_hist3, axis=0)

# Moisture
moist_hists = np.load('Cali Moisture/hist_moist_2019.npy')

final_hist = np.zeros([moist_hists.shape[0],11,32])
for i in range(final_hist.shape[0]):
    final_hist[i] = np.concatenate((sr_hists[i], temp_hists[i], moist_hists[i]))
print(final_hist.shape)
np.save('final_hists', final_hist)

for i in range(len(final_hist)):
    for j in range(len(final_hist[0])):
        if np.sum(final_hist[i, j]) == 0:
            if i - 365 >= 0:
                final_hist[i, j] = final_hist[i - 365, j]
            else:
                final_hist[i, j] = final_hist[i - 1, j]    


total = 0
e = 0
for i in range(final_hist.shape[0]):
    for b in range(final_hist.shape[1]):
        total+=1
        if np.sum(final_hist[i,b]) == 0:
            e+=1
print('Total Histograms:',total)
print('Missing Histograms:',e)

#%%

import matplotlib.pyplot as plt

l = 8 * 365
plt.subplots(7,5,figsize=(15, 15))

for j in range(32):
    maxi = []
    for i in range(len(moist_hists)):
        m = moist_hists[i,0,j]
        maxi.append(m)
    
    plt.subplot(7,5,j+1);
    plt.scatter(range(len(maxi[:l])), maxi[:l])
#%%

'''
    Lagging Histograms with Old Yield
'''
import pandas as pd
import numpy as np

final_hists = np.load('final_hists.npy')
final_hists = np.transpose(final_hists, (0,2,1))

yld = pd.read_excel('Imputed Yield Ifeanyi.xlsx')
print(final_hists.shape, yld.shape)

# Aligning histogram dates with available yield
final_hists = np.delete(final_hists, np.arange(31+28+31+3), axis=0)
print(final_hists.shape)


yields = yld['Yield'][:len(final_hists)].values

lagged_hists = []
a = 28
for i in range(len(final_hists)):
    if i + 140 + a >= len(final_hists):
        break
    lagged_hists.append([final_hists[i:i+140], yields[i+140+a]])
lagged_hists = np.array(lagged_hists)
print(lagged_hists.shape)
np.save('lagged_hists_28day', lagged_hists, allow_pickle=True)
#%%

final_hists = np.load('final_hists.npy')
final_hists = np.transpose(final_hists, (0,2,1))

prices = pd.read_excel('Imputed Price Ifeanyi.xlsx')['Yield'].values
print(final_hists.shape, prices.shape)

# Aligning histogram dates with available yield
final_hists = np.delete(final_hists, np.arange(31+28+31+30+31+30+31+31+24 - 140 - 35), axis=0)
print(final_hists.shape)

lagged_hists = []
for i in range(len(prices)):
    lagged_hists.append([final_hists[i:i+140], prices[i]])
lagged_hists = np.array(lagged_hists)
print(lagged_hists.shape)
np.save('lagged_hists_Ifeanyi_35day', lagged_hists, allow_pickle=True)

#%%
test1 = np.load('lagged_hists_35day.npy', allow_pickle=True)
#%%
test2 = np.load('cpami-cube files/lagged_hists_35day_newres.npy', allow_pickle=True)

#%%
plt.plot(test1[1000:,1], label='1 Day');
plt.plot(test2[1000:,1], label='7 Day');
plt.legend();
np.sum(test1[:,1] == test2[:,1])
