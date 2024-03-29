# GEOL0069week4

## Background
Prior to this assignment, colocation of Sentinel-3 OLCI (Ocean and Land Colour Instrument) data with Sentinel-2 optical data have already been performed, which allows a richer and more detailed Earth's surface investigation by combining the high spatial resolution of Sentinel-2 and the comprehensive coverage and colocated altimeter data from Sentinel-3. In this assignment, we apply two unsupervised methods - K-means clustering and Gaussian Mixture Models to distinguish sea ice from leads with Sentinel-2 imagery and Sentinel-3 altimetry dataset, classify the echos in leads and sea ice, and produce an average echo shape and standard deviation for these 2 classes. 


## Sentinel-2
### K-means clustering 
We first apply K-means clustering to distinguish between sea ice and leads in Sentinel-2 imagery.
```
!pip install rasterio
```
```
import rasterio
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

base_path = "/content/drive/MyDrive/GEOL0069/week4/Week_4/S2A_MSIL1C_20190301T235611_N0207_R116_T01WCU_20190302T014622.SAFE/S2A_MSIL1C_20190301T235611_N0207_R116_T01WCU_20190302T014622.SAFE/GRANULE/L1C_T01WCU_A019275_20190301T235610/IMG_DATA/" # You need to specify the path
bands_paths = {
    'B4': base_path + 'T01WCU_20190301T235611_B04.jp2',
    'B3': base_path + 'T01WCU_20190301T235611_B03.jp2',
    'B2': base_path + 'T01WCU_20190301T235611_B02.jp2'
}

# Read and stack the band images
band_data = []
for band in ['B4', 'B3', 'B2']:
    with rasterio.open(bands_paths[band]) as src:
        band_data.append(src.read(1))

# Stack bands and create a mask for valid data (non-zero values in all bands)
band_stack = np.dstack(band_data)
valid_data_mask = np.all(band_stack > 0, axis=2)

# Reshape for K-means, only including valid data
X = band_stack[valid_data_mask].reshape((-1, 3))

# K-means clustering
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
labels = kmeans.labels_

# Create an empty array for the result, filled with a no-data value (e.g., -1)
labels_image = np.full(band_stack.shape[:2], -1, dtype=int)

# Place cluster labels in the locations corresponding to valid data
labels_image[valid_data_mask] = labels

# Plotting the result
plt.imshow(labels_image, cmap='viridis')
plt.title('K-means clustering on Sentinel-2 Bands')
plt.colorbar(label='Cluster Label')
plt.show()
```
![K-means Sentinel 2](https://github.com/eunicewly/GEOL0069week4/assets/159627060/64cc2590-abdb-4f9b-8eca-fc0557a38a61)

### Gaussian Mixture Models (GMM)
Next, we apply Gaussian Mixture Models (GMM) to distinguish between sea ice and leads in Sentinel-2 imagery.
```
import rasterio
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Paths to the band images
# Paths to the band images
base_path = "/content/drive/MyDrive/GEOL0069/week4/Week_4/S2A_MSIL1C_20190301T235611_N0207_R116_T01WCU_20190302T014622.SAFE/S2A_MSIL1C_20190301T235611_N0207_R116_T01WCU_20190302T014622.SAFE/GRANULE/L1C_T01WCU_A019275_20190301T235610/IMG_DATA/" # You need to specify the path
bands_paths = {
    'B4': base_path + 'T01WCU_20190301T235611_B04.jp2',
    'B3': base_path + 'T01WCU_20190301T235611_B03.jp2',
    'B2': base_path + 'T01WCU_20190301T235611_B02.jp2'
}

# Read and stack the band images
band_data = []
for band in ['B4', 'B3', 'B2']:
    with rasterio.open(bands_paths[band]) as src:
        band_data.append(src.read(1))

# Stack bands and create a mask for valid data (non-zero values in all bands)
band_stack = np.dstack(band_data)
valid_data_mask = np.all(band_stack > 0, axis=2)

# Reshape for GMM, only including valid data
X = band_stack[valid_data_mask].reshape((-1, 3))

# GMM clustering
gmm = GaussianMixture(n_components=2, random_state=0).fit(X)
labels = gmm.predict(X)

# Create an empty array for the result, filled with a no-data value (e.g., -1)
labels_image = np.full(band_stack.shape[:2], -1, dtype=int)

# Place GMM labels in the locations corresponding to valid data
labels_image[valid_data_mask] = labels

# Plotting the result
plt.imshow(labels_image, cmap='viridis')
plt.title('GMM clustering on Sentinel-2 Bands')
plt.colorbar(label='Cluster Label')
plt.show()
```
![GMM Sentinel 2](https://github.com/eunicewly/GEOL0069week4/assets/159627060/9986cc2a-3c3b-4942-bf76-5b952b67a39a)

## Sentinel-3 Altimetry
Now, we apply these GMM to altimetry classification tasks, focusing specifically on distinguishing between sea ice and leads in Sentinel-3 altimetry dataset.
We first transform the raw data into meaningful variables, such as peakniness and stack standard deviation (SSD), etc. to ensure compatibility with our analytical models. 

```
! pip install netCDF4
```
The codes below create functions that calculate the peakiness of waveforms, unpacks data, and calculates the Sum of Squared Differences (SSD) for each RIP (Radar Imagery Product) waveform. 
```
#
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
# from mpl_toolkits.basemap import Basemap
import numpy.ma as ma
import glob
from matplotlib.patches import Polygon
import scipy.spatial as spatial
from scipy.spatial import KDTree

import pyproj
# import cartopy.crs as ccrs
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, fcluster

#=========================================================================================================
#===================================  SUBFUNCTIONS  ======================================================
#=========================================================================================================

#*args and **kwargs allow you to pass an unspecified number of arguments to a function,
#so when writing the function definition, you do not need to know how many arguments will be passed to your function
#**kwargs allows you to pass keyworded variable length of arguments to a function.
#You should use **kwargs if you want to handle named arguments in a function.
#double star allows us to pass through keyword arguments (and any number of them).
def peakiness(waves, **kwargs):

    "finds peakiness of waveforms."

    #print("Beginning peakiness")
    # Kwargs are:
    #          wf_plots. specify a number n: wf_plots=n, to show the first n waveform plots. \

    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import time

    print("Running peakiness function...")

    size=np.shape(waves)[0] #.shape property is a tuple of length .ndim containing the length of each dimensions
                            #Tuple of array dimensions.

    waves1=np.copy(waves)

    if waves1.ndim == 1: #number of array dimensions
        print('only one waveform in file')
        waves2=waves1.reshape(1,np.size(waves1)) #numpy.reshape(a, newshape, order='C'), a=array to be reshaped
        waves1=waves2

    # *args is used to send a non-keyworded variable length argument list to the function
    def by_row(waves, *args):
        "calculate peakiness for each waveform"
        maximum=np.nanmax(waves)
        if maximum > 0:

            maximum_bin=np.where(waves==maximum)
            #print(maximum_bin)
            maximum_bin=maximum_bin[0][0]
            waves_128=waves[maximum_bin-50:maximum_bin+78]

            waves=waves_128

            noise_floor=np.nanmean(waves[10:20])
            where_above_nf=np.where(waves > noise_floor)

            if np.shape(where_above_nf)[1] > 0:
                maximum=np.nanmax(waves[where_above_nf])
                total=np.sum(waves[where_above_nf])
                mean=np.nanmean(waves[where_above_nf])
                peaky=maximum/mean

            else:
                peaky = np.nan
                maximum = np.nan
                total = np.nan

        else:
            peaky = np.nan
            maximum = np.nan
            total = np.nan

        if 'maxs' in args:
            return maximum
        if 'totals' in args:
            return total
        if 'peaky' in args:
            return peaky

    peaky=np.apply_along_axis(by_row, 1, waves1, 'peaky') #numpy.apply_along_axis(func1d, axis, arr, *args, **kwargs)

    if 'wf_plots' in kwargs:
        maximums=np.apply_along_axis(by_row, 1, waves1, 'maxs')
        totals=np.apply_along_axis(by_row, 1, waves1, 'totals')

        for i in range(0,kwargs['wf_plots']):
            if i == 0:
                print("Plotting first "+str(kwargs['wf_plots'])+" waveforms")

            plt.plot(waves1[i,:])#, a, col[i],label=label[i])
            plt.axhline(maximums[i], color='green')
            plt.axvline(10, color='r')
            plt.axvline(19, color='r')
            plt.xlabel('Bin (of 256)')
            plt.ylabel('Power')
            plt.text(5,maximums[i],"maximum="+str(maximums[i]))
            plt.text(5,maximums[i]-2500,"total="+str(totals[i]))
            plt.text(5,maximums[i]-5000,"peakiness="+str(peaky[i]))
            plt.title('waveform '+str(i)+' of '+str(size)+'\n. Noise floor average taken between red lines.')
            plt.show()


    return peaky

#=========================================================================================================
#=========================================================================================================
#=========================================================================================================


def unpack_gpod(variable):

    from scipy.interpolate import interp1d

    time_1hz=SAR_data.variables['time_01'][:]
    time_20hz=SAR_data.variables['time_20_ku'][:]
    time_20hzC = SAR_data.variables['time_20_c'][:]

    out=(SAR_data.variables[variable][:]).astype(float)  # convert from integer array to float.

    #if ma.is_masked(dataset.variables[variable][:]) == True:
    #print(variable,'is masked. Removing mask and replacing masked values with nan')
    out=np.ma.filled(out, np.nan)

    if len(out)==len(time_1hz):

        print(variable,'is 1hz. Expanding to 20hz...')
        out = interp1d(time_1hz,out,fill_value="extrapolate")(time_20hz)

    if len(out)==len(time_20hzC):
        print(variable, 'is c band, expanding to 20hz ku band dimension')
        out = interp1d(time_20hzC,out,fill_value="extrapolate")(time_20hz)
    return out


#=========================================================================================================
#=========================================================================================================
#=========================================================================================================

def calculate_SSD(RIP):

    from scipy.optimize import curve_fit
    from scipy import asarray as ar,exp
    do_plot='Off'

    def gaussian(x,a,x0,sigma):
            return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

    SSD=np.zeros(np.shape(RIP)[0])*np.nan
    x=np.arange(np.shape(RIP)[1])

    for i in range(np.shape(RIP)[0]):

        y=np.copy(RIP[i])
        y[(np.isnan(y)==True)]=0

        if 'popt' in locals():
            del(popt,pcov)

        SSD_calc=0.5*(np.sum(y**2)*np.sum(y**2)/np.sum(y**4))
        #print('SSD calculated from equation',SSD)

        #n = len(x)
        mean_est = sum(x * y) / sum(y)
        sigma_est = np.sqrt(sum(y * (x - mean_est)**2) / sum(y))
        #print('est. mean',mean,'est. sigma',sigma_est)

        try:
            popt,pcov = curve_fit(gaussian, x, y, p0=[max(y), mean_est, sigma_est],maxfev=10000)
        except RuntimeError as e:
            print("Gaussian SSD curve-fit error: "+str(e))
            #plt.plot(y)
            #plt.show()

        except TypeError as t:
            print("Gaussian SSD curve-fit error: "+str(t))

        if do_plot=='ON':

            plt.plot(x,y)
            plt.plot(x,gaussian(x,*popt),'ro:',label='fit')
            plt.axvline(popt[1])
            plt.axvspan(popt[1]-popt[2], popt[1]+popt[2], alpha=0.15, color='Navy')
            plt.show()

            print('popt',popt)
            print('curve fit SSD',popt[2])

        if 'popt' in locals():
            SSD[i]=abs(popt[2])


    return SSD

```
The codes below unload data (latitude (SAR_lat), longitude (SAR_lon), waveforms (waves), signal-to-noise ratio (sig_0) and RIP (Radar Imagery Product)), filter data points, calculate Peakiness (PP) and Sum of Squared Differences (SSD) and standardise data. 

```
path = '/content/drive/MyDrive/GEOL0069/week4/S3B_SR_2_LAN_SI_20190301T231304_20190301T233006_20230405T162425_1021_022_301______LN3_R_NT_005.SEN3/' # You need to specify the path
SAR_file='S3B_SR_2_LAN_SI_20190301T231304_20190301T233006_20230405T162425_1021_022_301______LN3_R_NT_005.SEN3'
print('overlapping SAR file is',SAR_file)
SAR_data=Dataset(path + SAR_file+'/enhanced_measurement.nc')

SAR_lat, SAR_lon, waves, sig_0, RIP, flag = unpack_gpod('lat_20_ku'), unpack_gpod('lon_20_ku'), unpack_gpod('waveform_20_ku'),unpack_gpod('sig0_water_20_ku'),unpack_gpod('rip_20_ku'),unpack_gpod('surf_type_class_20_ku') #unpack_gpod('Sigma0_20Hz')
SAR_index=np.arange(np.size(SAR_lat))

find=np.where(SAR_lat >= -99999)#60
SAR_lat=SAR_lat[find]
SAR_lon=SAR_lon[find]
SAR_index=SAR_index[find]
waves=waves[find]
sig_0=sig_0[find]
RIP=RIP[find]

PP=peakiness(waves)
SSD=calculate_SSD(RIP)
sig_0_np = np.array(sig_0)  # Replace [...] with your data
RIP_np = np.array(RIP)
PP_np = np.array(PP)
SSD_np = np.array(SSD)

data = np.column_stack((sig_0_np,PP_np, SSD_np))
# Standardize the data
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)
```

Next, we need to clean and delete the NaN values in the dataset:
```
nan_count = np.isnan(data_normalized).sum()
print(f"Number of NaN values in the array: {nan_count}")
data_cleaned = data_normalized[~np.isnan(data_normalized).any(axis=1)]
flag_cleaned = flag[~np.isnan(data_normalized).any(axis=1)]
waves_cleaned = waves[~np.isnan(data_normalized).any(axis=1)][(flag_cleaned==1)|(flag_cleaned==2)]
```

## Echos of sea ice and lead
### 2 classes
We now run the GMM to classify data into 2 classes to distinguish sea ice from lead, and then plot the the average echo shape of sea ice and lead:
```
gmm = GaussianMixture(n_components=2, random_state=0)
gmm.fit(data_cleaned[(flag_cleaned==1)|(flag_cleaned==2)])
clusters_gmm = gmm.predict(data_cleaned[(flag_cleaned==1)|(flag_cleaned==2)])

plt.plot(np.mean(waves_cleaned[clusters_gmm==0],axis=0),label='ice')
plt.plot(np.mean(waves_cleaned[clusters_gmm==1],axis=0),label='lead')
plt.title('Plot of the average echos shape of sea ice and lead')
plt.xlabel('Number of datapoints')
plt.ylabel('Wave amplitude')
plt.legend()
```
![average echos](https://github.com/eunicewly/GEOL0069week4/assets/159627060/72a9f3ac-984d-4a1b-ba06-6ec7e9de8ab4)

Then we plot the standard deviation of the echos in sea ice and leads:
```
plt.plot(np.std(waves_cleaned[clusters_gmm==0],axis=0),label='ice')
plt.plot(np.std(waves_cleaned[clusters_gmm==1],axis=0),label='lead')
plt.title('Plot of the standard deviation of the echos of sea ice and lead')
plt.xlabel('Number of datapoints')
plt.ylabel('Wave amplitude')
plt.legend()
```
![std echos](https://github.com/eunicewly/GEOL0069week4/assets/159627060/b6fb5a07-9753-4b3d-9713-16b7118332b3)

We can inspect the number of data points in each class of your GMM clustering prediction:
```
unique, counts = np.unique(clusters_gmm, return_counts=True)
class_counts = dict(zip(unique, counts))

print(class_counts)
```
```
{0: 5013, 1: 1836}
```
The results show that there is 5013 data points classified as sea ice, and 1836 as lead. 


### 5 classes
We can also change the number of classes from 2 to 5 and 10 to compare and validate whether a 2-class classification is appropriate. 
PLotting the average echo shapes:
```
# 5 classes
gmm = GaussianMixture(n_components=5, random_state=0)
gmm.fit(data_cleaned[(flag_cleaned==1)|(flag_cleaned==2)])
clusters_gmm = gmm.predict(data_cleaned[(flag_cleaned==1)|(flag_cleaned==2)])

plt.plot(np.mean(waves_cleaned[clusters_gmm==0],axis=0),label='class0')
plt.plot(np.mean(waves_cleaned[clusters_gmm==1],axis=0),label='class1')
plt.plot(np.mean(waves_cleaned[clusters_gmm==2],axis=0),label='class2')
plt.plot(np.mean(waves_cleaned[clusters_gmm==3],axis=0),label='class3')
plt.plot(np.mean(waves_cleaned[clusters_gmm==4],axis=0),label='class4')
plt.title('Plot of the average echo shape of the 5 classes in GMM')
plt.xlabel('Number of datapoints')
plt.ylabel('Wave amplitude')
plt.legend()
```
![gmm5class](https://github.com/eunicewly/GEOL0069week4/assets/159627060/6fe0e06c-63b2-464f-94a4-067942aad5c3)


The standard deviations:
```
plt.plot(np.std(waves_cleaned[clusters_gmm==0],axis=0),label='class0')
plt.plot(np.std(waves_cleaned[clusters_gmm==1],axis=0),label='class1')
plt.plot(np.std(waves_cleaned[clusters_gmm==2],axis=0),label='class2')
plt.plot(np.std(waves_cleaned[clusters_gmm==3],axis=0),label='class3')
plt.plot(np.std(waves_cleaned[clusters_gmm==4],axis=0),label='class4')
plt.title('Plot of the echo standard deviation of the 5 classes in GMM')
plt.xlabel('Number of datapoints')
plt.ylabel('Wave amplitude')
plt.legend()
```
![gmm5class_std](https://github.com/eunicewly/GEOL0069week4/assets/159627060/13717530-4dc3-4d2d-81e7-b80023da6785)


### 10 classes
The average echo shapes:
```
# 10 classes
gmm = GaussianMixture(n_components=10, random_state=0)
gmm.fit(data_cleaned)
clusters_gmm = gmm.predict(data_cleaned[(flag_cleaned==1)|(flag_cleaned==2)])

plt.plot(np.mean(waves_cleaned[clusters_gmm==0],axis=0),label='class0')
plt.plot(np.mean(waves_cleaned[clusters_gmm==1],axis=0),label='class1')
plt.plot(np.mean(waves_cleaned[clusters_gmm==2],axis=0),label='class2')
plt.plot(np.mean(waves_cleaned[clusters_gmm==3],axis=0),label='class3')
plt.plot(np.mean(waves_cleaned[clusters_gmm==4],axis=0),label='class4')
plt.plot(np.mean(waves_cleaned[clusters_gmm==5],axis=0),label='class5')
plt.plot(np.mean(waves_cleaned[clusters_gmm==6],axis=0),label='class6')
plt.plot(np.mean(waves_cleaned[clusters_gmm==7],axis=0),label='class7')
plt.plot(np.mean(waves_cleaned[clusters_gmm==8],axis=0),label='class8')
plt.plot(np.mean(waves_cleaned[clusters_gmm==9],axis=0),label='class9')
plt.title('Plot of the average echo shape of the 10 classes in GMM')
plt.xlabel('Number of datapoints')
plt.ylabel('Wave amplitude')
plt.legend()
```
![gmm10class](https://github.com/eunicewly/GEOL0069week4/assets/159627060/fa26c0f6-05e7-4d1b-8e3b-336554ee9f58)


The standard deviations:
```
plt.plot(np.std(waves_cleaned[clusters_gmm==0],axis=0),label='class0')
plt.plot(np.std(waves_cleaned[clusters_gmm==1],axis=0),label='class1')
plt.plot(np.std(waves_cleaned[clusters_gmm==2],axis=0),label='class2')
plt.plot(np.std(waves_cleaned[clusters_gmm==3],axis=0),label='class3')
plt.plot(np.std(waves_cleaned[clusters_gmm==4],axis=0),label='class4')
plt.plot(np.std(waves_cleaned[clusters_gmm==5],axis=0),label='class5')
plt.plot(np.std(waves_cleaned[clusters_gmm==6],axis=0),label='class6')
plt.plot(np.std(waves_cleaned[clusters_gmm==7],axis=0),label='class7')
plt.plot(np.std(waves_cleaned[clusters_gmm==8],axis=0),label='class8')
plt.plot(np.std(waves_cleaned[clusters_gmm==9],axis=0),label='class9')
plt.title('Plot of the echo standard deviation of the 10 classes in GMM')
plt.xlabel('Number of datapoints')
plt.ylabel('Wave amplitude')
plt.legend()
```
![gmm10class_std](https://github.com/eunicewly/GEOL0069week4/assets/159627060/57647d79-e10c-42c6-8663-9e07283cf054)


## Confusion Matrix

Finally, we will quantify our echo classification by GMM (2 class) against the ESA official classification using a confusion matrix.

We first need to modify the ESA classification to make it compatible with our classifcation for the confusion matrix, by extracting the corresponding data from flag (ESA classification) that matches those in data_cleaned, and changing the numbers that indicate the sea ice and lead classification to match those in GMM:
```
# extract the values in flag that are not corresponding to the NaN values in the data_normalized, just like for data_cleaned 
esa_flag_cleaned = flag[~np.isnan(data_normalized).any(axis=1)][(flag_cleaned==1)|(flag_cleaned==2)]
# changing 1 to 0 (sea ice), and 2 to 1 (lead), so that it is the same as in clusters_gmm classification
esa_flag_cleaned = np.where(esa_flag_cleaned == 1, 0, esa_flag_cleaned)
esa_flag_cleaned = np.where(esa_flag_cleaned == 2, 1, esa_flag_cleaned)
```
Then we do the confusion matrix:
```
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Compute confusion matrix
cm = confusion_matrix(esa_flag_cleaned, clusters_gmm)

# Calculate accuracy
accuracy = accuracy_score(esa_flag_cleaned, clusters_gmm)

# Display classification report
report = classification_report(esa_flag_cleaned, clusters_gmm)
print("Classification Report:\n", report)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Display accuracy
print(f"Accuracy: {accuracy:.2f}")
```
The results from the codes above:
```
Classification Report:
               precision    recall  f1-score   support

         0.0       1.00      0.99      0.99      5086
         1.0       0.96      1.00      0.98      1763

    accuracy                           0.99      6849
   macro avg       0.98      0.99      0.99      6849
weighted avg       0.99      0.99      0.99      6849
```
![confusion matrix with esa](https://github.com/eunicewly/GEOL0069week4/assets/159627060/ee07d0a4-15bf-4eaf-9198-559bf4f68d15)
