import keras
from keras.preprocessing.image import ImageDataGenerator

img_width, img_height = 227, 227
train_data_dir = '../transfer_classes/train'
validation_data_dir = '../transfer_classes/validation'
nb_train_samples = 1438
nb_validation_samples = 546
epochs = 10
batch_size = 16

model=keras.models.load_model('towards_transfer_class_2_inc.hdf5')


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        train_data_dir,  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels


validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical', shuffle=False)

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    verbose=2, workers=12)

score = model.evaluate_generator(validation_generator, nb_validation_samples/batch_size, workers=4)

#scores = model.predict_generator(validation_generator, nb_validation_samples/batch_size, workers=4)

#correct = 0
#for i, n in enumerate(validation_generator.filenames):
#    if n.startswith("cats") and scores[i][0] <= 0.5:
#        correct += 1
#    if n.startswith("dogs") and scores[i][0] > 0.5:
#        correct += 1

#print("Correct:", correct, " Total: ", len(validation_generator.filenames))
print("Loss: ", score[0], "Accuracy: ", score[1])
