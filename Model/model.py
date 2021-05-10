NN_model = Sequential()

# The Input Layer :
NN_model.add(Dense(25, kernel_initializer='normal',input_dim = 12, activation='relu'))
from keras.utils.vis_utils import plot_model
# The Hidden Layers :
NN_model.add(Dense(51, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(128, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(207, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
#NN_model.add(BatchNormalization())
# The Output Layer 
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
NN_model.summary()
plot_model(NN_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]
