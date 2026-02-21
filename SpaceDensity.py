# 1. Process density distribution data from the file
# in this case it is file e744a8ed50e64f5b8f56d3bab9f2cd1d-0.parquet
# but ideally one would parse along all files in all directories to build and train the model
# I cannot do it dure to the limitation in RAM, CPU resources. The purpose is to show how 
# I would approach this problem given access to all the necessary resources.

# Another approach would be to reduce the dimensional resolution by autoencoder technique. First, encode
# the grid to the smaller resolution latent space, then apply Convolutional LSTM in that space,
# and finally decode the forecasted image to the original resolution. That would introduce some extra
# error (plus to the actual extrapolational error) due to inevitable image distortion after encoding/decoding,
# however it would be more feasible at limited resources. I started this approach, but did not
# have time to finish it. I am ready to demonstrate it if more time is given.

import pyarrow.parquet as pa
import pandas as pd
import numpy as np
import dateutil.parser

# Gets 3D resolution for latitude/longitude/altitude box from dataframe df
# df is supposed to be preliminary sorted in latitude/longitude/altitude order.
def get_resolution(df):
    res = []
    alt_res = -1
    alt_np = df['altitude'].to_numpy()
    alt_start = alt_np[0]
    for i in range (0, len(alt_np)):
        alt_res += 1
        if i != 0 and alt_np[i] == alt_start:
            break
        
    lon_res = -1
    lon_np = df['longitude'].to_numpy()
    lon_start = lon_np[0]
    for i in range(0, len(lon_np), alt_res):
        lon_res += 1
        if i != 0 and lon_np[i] == lon_start:
            break
        
    lat_res = -1
    lat_np = df['latitude'].to_numpy()
    lat_start = lat_np[0]
    for i in range(0, len(lat_np), alt_res * lon_res):
        lat_res += 1
        if i != 0 and lat_np[i] == lat_start:
            break
    res.append(lat_res + 1)
    res.append(lon_res)
    res.append(alt_res)
        
    return res

# Get pandajs dataframe from parquet files
def get_dataframe_from_parquet(par_name):
    table = pa.read_table(par_name)
    df = table.to_pandas() 
    return df

df = get_dataframe_from_parquet('e744a8ed50e64f5b8f56d3bab9f2cd1d-0.parquet')


# Builds grid for each solar time stamp
# The data is contained in individual (for each time) dataframe df
def build_grid_for_timestamp(df, res):
    df_arr = df.to_numpy()
    lat_size = res[0]
    lon_size = res[1]
    alt_size = res[2]
    grid = np.zeros([lat_size, lon_size, alt_size], dtype = np.float32)
    row = 0
    for i in range(lat_size):
        for j in range(lon_size):
            for k in range(alt_size):
                grid[i][j][k] = df_arr[row][3]
                row += 1
    return grid

# Sorts data in the original dataframe in local_solar_time - latitude - longitude - altitude order.
# Split the original dataframe into groups of dataframes for each solar timestamp
# and generate grid for each of them. Keep only the data of the same (smallest) resolution.
# Upscaling to higher resolution is possible via extrapolation/interpolation techniques,
# but a) have no time, b) it will introduce extra error.
def split_df(df):
    df_sort = df.sort_values(by=['local_solar_time','latitude', 'longitude', 'altitude'], ignore_index=True)
    frames = []
    times = []
    resolutions = []
    while df_sort.empty == False:
        # get top solar time data into separate dataframe
        cur_time = df_sort['local_solar_time'][0]
        df0 = df_sort[df_sort['local_solar_time'] == cur_time]
        times.append(cur_time)
        df0 = df0.drop(['datetime', 'local_solar_time'], axis=1)
        frames.append(df0)
        cur_res = get_resolution(df0)
        resolutions.append(cur_res)
        print('cur_time = ', cur_time)
        print('cur_size = ', len(df0))
        print('cur_res = ', cur_res)
        # remove top solar time data
        df_sort = df_sort[df_sort['local_solar_time'] > cur_time]
        df_sort = df_sort.reset_index(drop=True)

    # find smallest resolution and drop other
    # it is possible to restore the smallest available grid to larger sizes by extrapolation
    # but I don't have time to do that. The purpose is to show the methodology and way of thinking.

    # find smallest resolution
    minSize = 9223372036854775807
    for res in resolutions:
        size = res[0] * res[1] * res[2]
        if size < minSize:
            minSize = size

    # choose only data frames with minSize resolution
    fin_res = []
    fin_times = []
    fin_grids = []
    for i in range(len(frames)):
        size = resolutions[i][0] * resolutions[i][1] * resolutions[i][2]
        if size == minSize:
            cur_df = frames[i]
            cur_time = times[i]
            cur_res = resolutions[i]
            cur_grid = build_grid_for_timestamp(cur_df, cur_res)
            fin_res.append(cur_res)
            fin_times.append(cur_time)
            fin_grids.append(cur_grid)

    return fin_res, fin_times, fin_grids

res, times, grids = split_df(df)


# Build Convolutional LSTM Neural Network to process sequential 3D grids and to generate the next one.
# Using TensorFlow Keras, not PyTorch as the former has ConvLSTM3D method which Torch still lacks.

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, ConvLSTM3D, Conv3D, Dense, Flatten, Concatenate, Reshape, BatchNormalization, Lambda
from tensorflow.keras.callbacks import Callback
import tensorflow.image as tfi
import random

print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))

# Enable GPU memory growth (avoids TF grabbing all GPU RAM at once)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU(s): {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found, running on CPU.")

# Here I assume that `grids` and `times` are produced by preprocessing above and given as input:
# grids is a list of 3D numpy arrays, each with shape (19, 24, 137)
# times is a list of corresponding timestamps (which are float values)
# Times are important as, due to the dropping of some grids of larger sizes
# I introduced unevenness in the samplings.

# set the number of samples to 10 (can be any number), so we try to predict the 11th element
# Convert `grids` and `times` into training sequences
def prepare_data(grids, times, seq_len=10):
    x_images = []
    y_images = []
    x_times = []
    
    for i in range(len(grids) - seq_len):
        x_images.append(grids[i:i+seq_len])
        y_images.append(grids[i+seq_len])  # The 11th image as the target
        x_times.append([times[i+j+1] - times[i+j] for j in range(seq_len)])  # Time deltas between consecutive steps
    
    return np.array(x_images), np.array(y_images), np.array(x_times)

# Prepare training data
x_train, y_train, time_sequences = prepare_data(grids, times)

# Reorganize data for keras processing foramts.
# Expand dimensions to add channel (e.g., (num_sequences, seq_len, H, W, D, 1))
x_train = np.expand_dims(x_train, axis=-1)  # Add channel dimension to (num_sequences, seq_len, H, W, D)
y_train = np.expand_dims(y_train, axis=-1)  # Add channel dimension to (num_sequences, H, W, D)

# Input shapes
seq_len = x_train.shape[1]  # Sequence length (10)
H, W, D = x_train.shape[2], x_train.shape[3], x_train.shape[4]  # Extract H, W, D from the data

# Model architecture
image_input = Input(shape=(seq_len, H, W, D, 1))  # (seq_len, H, W, D, channels)
time_input = Input(shape=(seq_len,))  # Time delta sequence

# ConvLSTM3D layers to process the image sequence
x = ConvLSTM3D(filters=64, kernel_size=(3, 3, 3), padding='same', return_sequences=True)(image_input)
x = BatchNormalization()(x)
x = ConvLSTM3D(filters=64, kernel_size=(3, 3, 3), padding='same', return_sequences=False)(x)
x = BatchNormalization()(x)

# Flatten and process time deltas
time_dense = Dense(64, activation='relu')(time_input)

# Reshape time feature and broadcast it to match image feature dimensions
# Broadcast `time_dense` to match the spatial dimensions (H, W, D)
# Use Lambda layer to apply tf.tile and broadcast time features
time_broadcast = Reshape((1, 1, 1, 64))(time_dense)  # Start with (1, 1, 1, 64)
time_broadcast = Lambda(lambda t: tf.tile(t, [1, H, W, D, 1]))(time_broadcast)  # Tile to (H, W, D, 64)

# Concatenate time delta and image features, otherwise it will assume equal time deltas
combined = Concatenate()([x, time_broadcast])

# Final Conv3D layer to predict the next image
output = Conv3D(filters=1, kernel_size=(3, 3, 3), activation='sigmoid', padding='same')(combined)

# Define the model
model = Model(inputs=[image_input, time_input], outputs=output)
# Use Adam optimizer and mean square error to compare original and generated image
model.compile(optimizer='adam', loss='mean_squared_error')

# Print the model summary
model.summary()

# Custom callback to calculate SSIM and PSNR during training based on MSE metrics
class MetricsCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.predict([x_train, time_sequences])
        ssim_values = []
        psnr_values = []
        
        # Loop over each image and calculate SSIM and PSNR for comparison
        for i in range(len(predictions)):
            predicted_image = predictions[i]
            actual_image = y_train[i]
            
            # Cast predicted and actual images to float32
            predicted_image_float = tf.cast(predicted_image, tf.float32)
            actual_image_float = tf.cast(actual_image, tf.float32)
            
            # Compute SSIM and PSNR using TensorFlow image functions
            ssim = tfi.ssim(predicted_image_float, actual_image_float, max_val=1.0)
            psnr = tfi.psnr(predicted_image_float, actual_image_float, max_val=1.0)
            
            # Append the mean SSIM and PSNR values for each image
            ssim_values.append(np.mean(ssim))
            psnr_values.append(np.mean(psnr))
        
        # Calculate and print average SSIM and PSNR across all images
        avg_ssim = np.mean(ssim_values)
        avg_psnr = np.mean(psnr_values)
        print(f"Epoch {epoch+1} - SSIM: {avg_ssim:.4f}, PSNR: {avg_psnr:.4f}")

if __name__ == "__main__":
    # Fit the model
    metrics_callback = MetricsCallback()

    # Train the model
    model.fit([x_train, time_sequences], y_train, epochs=10, batch_size=4, callbacks=[metrics_callback])

    # --- TEST EXAMPLE AFTER TRAINING ---

    # Randomly select 10 samples from the training set
    test_indices = random.sample(range(len(x_train)), 10)
    test_images = x_train[test_indices]
    test_times = time_sequences[test_indices]
    actual_next_images = y_train[test_indices]

    # Predict the next images
    predicted_next_images = model.predict([test_images, test_times])

    # Calculate the loss (Mean Squared Error) between the predicted and actual next images
    # Option 1: instantiate the loss object (use new method)
    mse = tf.keras.losses.MeanSquaredError()
    mse_loss = mse(actual_next_images, predicted_next_images)


    # Print the MSE loss for each test sample
    for i, loss in enumerate(mse_loss):
        print(f"Test Sample {i+1} - MSE Loss: {loss.numpy():.6f}")

    # Print the average MSE loss across the 10 test samples
    avg_mse_loss = tf.reduce_mean(mse_loss).numpy()
    print(f"Average MSE Loss for 10 test samples: {avg_mse_loss:.6f}")