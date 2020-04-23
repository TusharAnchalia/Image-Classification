from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

for j in range(1, 33):
  datagen = ImageDataGenerator(
          rotation_range=30,
          width_shift_range=0.1,
          height_shift_range=0.1,
          shear_range=0.1,
          zoom_range=0.1,
          fill_mode='nearest')

  img = load_img('C:/machine learning/Machine Learning A-Z Template Folder/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/project/pre process/dataset/train_data/telugu/'+str(j)+'.jpg')  # this is a PIL image
  x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
  x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
  
  # the .flow() command below generates batches of randomly transformed images
  # and saves the results to the `preview/` directory
  i = 0
  for batch in datagen.flow(x, batch_size=1,
                            save_to_dir='C:/machine learning/Machine Learning A-Z Template Folder/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/project/pre process/dataset/train_data/telugu/dir/', save_prefix='telugu', save_format='jpg'):
      i += 1
      if i > 20:
          break   

