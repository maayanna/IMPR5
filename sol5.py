# ##########################################################
# # By Maayann Affriat
# # username : maayanna
# # filename : sol5.py
# ##########################################################


import imageio
import skimage.color
import numpy as np
import random
import sol5_utils
from keras.layers import Conv2D, Activation, Input, Add
from keras.models import Model
from keras.optimizers import Adam
from scipy.ndimage.filters import convolve
import matplotlib.pylab as plt


NORMALIZE = 255


def read_image(filename, representation=1):
    """
    Function that reads an image file and convert it into a given representation
    :param filename: the filename of an image on disk
    :param representation: representation code, either 1 or 2 defining whether the output should
                           be a grayscale image (1) or an RGB image (2)
    :return: an image represented by a matrix of type np.float64
    """

    color_flag = True  # if RGB image
    image = imageio.imread(filename)

    float_image = image.astype(np.float64)

    if not np.all(image <= 1):
        float_image /= NORMALIZE  # Normalized to range [0,1]

    if len(float_image.shape) != 3:  # Checks if RGB or Grayscale
        color_flag = False

    if color_flag and representation == 1:  # Checks if need RGB to Gray
        return skimage.color.rgb2gray(float_image)

    # Same coloring already
    return float_image


def load_dataset(filenames, batch_size, corruption_func, crop_size):
    """
    This function generates pairs of image patches on the fly, each time picking a random image, applying a random
    corruption, and extracting a random patch
    :param filenames: A list of filenames of clean images
    :param batch_size: The size of the batch of images for each iteration of Stochastic Gradient Descent
    :param corruption_func: A function receiving a np array representation of an image as a single argument.
                            and returns a randomly corrupted version of the input image
    :param crop_size: A tuple (h,w) specifying the crop size of the patches to extract
    :return: data_generator : a Python's generator object which outputs random tuples of the form
    (source_batch, target_batch) where each output variable is an array of shape (batch_size, h, w, 1)

    """

    cache_images = dict() # Pq en dehors du while

    while True:

        # Code for randomly creating x

        src = list()
        target = list()

        for index in range(batch_size):

            index_random = random.randint(0, len(filenames)-1)

            file = filenames[index_random]

            if file not in cache_images:
                cache_images[file] = read_image(file)  # 1 for gray

            image_file = cache_images[file]

            # Taking a bigger patch 3x3

            height = random.randint(0,image_file.shape[0] - crop_size[0] * 3)  # -1 ?
            width = random.randint(0,image_file.shape[1] - crop_size[1] * 3)  # -1 ?

            im_big3 = image_file[height: height + crop_size[0] * 3, width: width + crop_size[1] * 3]

            # Taking a smaller patch in the big one

            h = random.randint(0, im_big3.shape[0] - crop_size[0])
            w = random.randint(0, im_big3.shape[1] - crop_size[1])

            func_after = corruption_func(im_big3)

            final_patch = image_file[ height + h:height+h + crop_size[0], w + width : width + w + crop_size[1]]

            corrupted = func_after[h:h + crop_size[0], w:w + crop_size[1]]

            final_patch = final_patch.reshape(crop_size[0], crop_size[1], 1)
            target.append(final_patch - 0.5)

            corrupted = corrupted.reshape(crop_size[0], crop_size[1], 1)
            src.append(corrupted - 0.5)

        x = (np.array(src), np.array(target))
        # x = (src, target)

        yield x



def resblock(input_tensor, num_channels):
    """
    This function takes as input a symbolic input tensor and the number of channels for each
    of its convolutional layers and returns teh symbolic output tensor of the layer configuration
    described in figure 1 in the exercuce description
    :param input_tensor:
    :param num_channels:
    :return:
    """

    first_conv = Conv2D(num_channels, (3, 3), padding='same')(input_tensor)
    activated = Activation('relu')(first_conv)
    output_tensor1 = Conv2D(num_channels, (3, 3), padding='same')(activated)
    adding_all = Add()([output_tensor1, input_tensor])
    return Activation('relu')(adding_all)
    # return adding_all


def build_nn_model(height, width, num_channels, num_res_block):
    """
    This function should return an untrained Keras model
    :param height:
    :param width:
    :param num_channels:
    :param num_res_block:
    :return:
    """

    input = Input(shape=(height, width, 1))
    first_conv = Conv2D(num_channels, (3, 3), padding = 'same')(input)
    activated = Activation('relu')(first_conv)
    # res = resblock(activated, num_channels)
    res = activated

    for index in range(num_res_block):
        res = resblock(res, num_channels)

    out = Conv2D(1, (3,3), padding = 'same')(res)
    output = Add()([out, input])
    return Model(inputs = input, outputs = output)



def train_model(model, images, corruption_func, batch_size, steps_per_epoch, num_epochs,num_valid_samples):
    """
    This function divides the images into a training set and validation set, using an 80-20 split,
    and generate from each set a dataset with the given batch size and corruption function
    :param model: a general neural network model for image restoration
    :param images: a list of file paths pointing to image files
    :param corruption_func: same as above
    :param batch_size: the size of the batch of examples for each iteration of SGD
    :param steps_per_epoch: number of update steps in each epoch
    :param num_epochs: number of epochs for which the optimization will run
    :param num_valid_samples: number of samples in the validation set to test on after every epoch
    :return:
    """

    # Getting a 80-20 split

    # random.shuffle(images) # Not taking the first images but randomally taking them #TODO do we have to shuffle or not ???
    index = int(len(images)*0.8)
    train_images = images[:index] # 80 first percent
    valid_images = images[index:]

    # the crop_size to use

    crop = list()
    crop.append(model.input_shape[1])
    crop.append(model.input_shape[2])

    # create the datasets

    train_set = load_dataset(train_images, batch_size, corruption_func, crop)
    valid_set = load_dataset(valid_images, batch_size, corruption_func, crop)

    model.compile(loss = 'mean_squared_error', optimizer=Adam(beta_2=0.9))

    # num_valid = int(num_valid_samples/batch_size)

    model.fit_generator(train_set, steps_per_epoch=steps_per_epoch, epochs=num_epochs,
                        validation_data=valid_set, validation_steps=num_valid_samples)


def restore_image(corrupted_image, base_model):
    """

    :param corrupted_image:
    :param base_model:
    :return:
    """
    # print(corrupted_image.shape)

    input = Input(shape=(corrupted_image.shape[0], corrupted_image.shape[1], 1))

    base = base_model(input)

    my_model = Model( inputs=input, outputs=base)

    # corrupted_image = corrupted_image.reshape(corrupted_image.shape[0], corrupted_image.shape[1] , 1) # pq ???

    my_image = (corrupted_image.reshape(1, corrupted_image.shape[0], corrupted_image.shape[1], 1)) - 0.5

    predicted = my_model.predict(my_image)

    new_image = predicted[0]
    new_image += 0.5
    # new_image = new_image.clip(0, 1).reshape(corrupted_image.shape[0], corrupted_image.shape[1])
    # print(new_image.shape)
    new_image = new_image.clip(0, 1)[:, :,0]
    return new_image.astype(np.float64)


def add_gaussian_noise(image, min_sigma, max_sigma):
    """

    :param image:
    :param min_sigma:
    :param max_sigma:
    :return:
    """

    sig = random.uniform(min_sigma, max_sigma)
    to_add = np.random.normal(0, sig, image.shape)

    my_image =  image + to_add
    my_image = np.round(my_image * NORMALIZE)
    my_image /= NORMALIZE
    return my_image.clip(0,1)


def learn_denoising_model(num_res_blocks=5, quick_mode=False):
    """

    :param num_res_blocks:
    :param quick_mode:
    :return:
    """

    my_model = build_nn_model(24, 24, 48, num_res_blocks)
    corrupted_func = lambda image : add_gaussian_noise(image, 0, 0.2)
    images = sol5_utils.images_for_denoising()

    if quick_mode:
        train_model(my_model, images, corrupted_func, 10, 3, 2, 30)

    else:
        train_model(my_model, images, corrupted_func, 100, 100, 5, 1000)

    return my_model


def add_motion_blur(image, kernel_size, angle):
    """

    :param image:
    :param kernel_size:
    :param angle:
    :return:
    """

    return convolve(image, sol5_utils.motion_blur_kernel(kernel_size, angle))


def random_motion_blur(image, list_of_kernel_sizes):
    """

    :param image:
    :param list_if_kernel_sizes:
    :return:
    """
    my_angle = np.random.uniform(0, np.pi)
    size = random.randint(0, len(list_of_kernel_sizes) - 1)
    kernel_size = list_of_kernel_sizes[size]

    my_image = add_motion_blur(image, kernel_size, my_angle)

    my_image = np.round(my_image*NORMALIZE)
    my_image /= NORMALIZE

    return my_image.clip(0,1)


def learn_deblurring_model(num_res_blocks=5, quick_mode=False):
    """

    :param num_res_blocks:
    :param quick_mode:
    :return:
    """

    my_model = build_nn_model(16,16,32,num_res_blocks)
    corrupted_func = lambda image: random_motion_blur(image, [7])
    images = sol5_utils.images_for_deblurring()

    if quick_mode:
        train_model(my_model, images, corrupted_func, 10, 3, 2, 30)

    else:
        train_model(my_model, images, corrupted_func, 100, 100, 10, 1000)

    return my_model


# # model1 = learn_denoising_model()
# model2 = learn_deblurring_model()
# # model_json = model.to_json()
# # with open("model_debluring.json","w") as json_file:
# #     json_file.write(model_json)
# # model.save_weights("model.h5")
# # model.save("model_denoising.h5")
#
# # model = load_model("model22.h5")
# cor1 = read_image("/cs/usr/maayanna/PycharmProjects/ImageProcessing/ex5-maayanna/examples/0000018_2_corrupted.png",1)
#
# # cor1 =  read_image("/cs/usr/ilan.benhamou/PycharmProjects/IMPR/ex5/ex5-ilan.benhamou/examples/0000018_2_corrupted.png",1)
# cor2 = read_image("/cs/usr/maayanna/PycharmProjects/ImageProcessing/ex5-maayanna/examples/163004_2_corrupted_0.10.png",1)
# # img1 = restore_image(cor2,model1)
# # plt.imshow(img1, cmap="gray")
# # plt.show()
#
# img2 = restore_image(cor1,model2)
# plt.imshow(img2, cmap="gray")
# plt.show()
#
#
# # if __name__ == '__main__':
# #     validation_error_denoise = []
# #     # validation_error_deblur = []
# #     for i in range(1, 6):
# #         denoise_model = learn_denoising_model(i)
# #         # deblur_model = learn_deblurring_model(i)
# #         validation_error_denoise.append(denoise_model.history.history['val_loss'][-1])
# #         # validation_error_deblur.append(deblur_model.history.history['val_loss'][-1])
# #
# #     arr = np.arange(1, 6)
# #
# #     plt.plot(arr, validation_error_denoise)
# #     plt.title('validation error - denoise')
# #     plt.xlabel('number res blocks')
# #     plt.ylabel('validation loss denoise')
# #     plt.savefig('denoise2.png')
# #     plt.show()
# #
# #     # plt.plot(arr, validation_error_deblur)
# #     # plt.title('validation error - deblur')
# #     # plt.xlabel('number res blocks')
# #     # plt.ylabel('validation loss deblur')
# #     # plt.savefig('deblur.png')
# #     # plt.show()