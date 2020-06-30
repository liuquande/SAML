import tensorflow as tf
import numpy as np
import os
# from matplotlib import pyplot as plt
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
import skimage as sk
from skimage import transform
import SimpleITK as sitk

IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)


class ImageDataGenerator(object):

    def __init__(self, txt_file, mode, batch_size, num_classes, shuffle=True, buffer_size=5):

        """Create a new ImageDataGenerator.
        Receives a path string to a text file, where each line has a path string to an image and
        separated by a space, then with an integer referring to the class number.

        Args:
            txt_file: path to the text file.
            mode: either 'training' or 'validation'. Depending on this value, different parsing functions will be used.
            batch_size: number of images per batch.
            num_classes: number of classes in the dataset.
            shuffle: wether or not to shuffle the data in the dataset and the initial file list.
            buffer_size: number of images used as buffer for TensorFlows shuffling of the dataset.

        Raises:
            ValueError: If an invalid mode is passed.
        """

        self.txt_file = txt_file
        self.num_classes = num_classes

        # retrieve the data from the text file
        self._read_txt_file()

        # number of samples in the dataset
        self.data_size = len(self.img_paths)

        # initial shuffling of the file and label lists together
        if shuffle:
            self._shuffle_lists()

        # convert lists to TF tensor
        self.img_paths = convert_to_tensor(self.img_paths, dtype=dtypes.string)

        # create dataset
        data = tf.data.Dataset.from_tensor_slices((self.img_paths))

        # repeat indefinitely (train.py will count the epochs)
        data = data.repeat()

        # distinguish between train/infer. when calling the parsing functions
        self.get_patches_fn = lambda filename: tf.py_func(self.extract_patch, [filename, [384,384,3], 2], [tf.float32, tf.float32])

        if mode == 'training':
            data = data.map(self.get_patches_fn, num_parallel_calls=8)

        elif mode == 'inference':
            data = data.map(self._parse_function_inference, num_parallel_calls=8)

        else:
            raise ValueError("Invalid mode '%s'." % (mode))

        # shuffle the first `buffer_size` elements of the dataset
        if shuffle:
            data = data.shuffle(buffer_size=buffer_size)

        # create a new dataset with batches of images
        data = data.batch(batch_size)

        self.data = data

    def _read_txt_file(self):
        """Read the content of the text file and store it into lists."""
        with open(self.txt_file, 'r') as f:
            rows = f.readlines()
            self.img_paths = [row[:-1] for row in rows]

    def _shuffle_lists(self):
        """Conjoined shuffling of the list of paths and labels."""
        path = self.img_paths
        permutation = np.random.permutation(self.data_size)
        self.img_paths = []
        for i in permutation:
            self.img_paths.append(path[i])

    def extract_patch(self, filename, patch_size, num_class, num_patches=1):
        """Input parser for samples of the training set."""
        # convert label number into one-hot-encoding

        image, mask = self.parse_fn(filename) # get the image and its mask
        image_patches = []
        mask_patches = []
        num_patches_now = 0

        while num_patches_now < num_patches:
            # z = np.random.randint(1, mask.shape[2]-1)
            z = self.random_patch_center_z(mask, patch_size=patch_size) # define the centre of current patch
            image_patch = image[:, :, z-1:z+2]
            mask_patch  =  mask[:, :, z]
            
            image_patches.append(image_patch)
            mask_patches.append(mask_patch)
            num_patches_now += 1
        image_patches = np.stack(image_patches) # make into 4D (batch_size, patch_size[0], patch_size[1], patch_size[2])
        mask_patches = np.stack(mask_patches) # make into 4D (batch_size, patch_size[0], patch_size[1], patch_size[2])

        mask_patches = self._label_decomp(mask_patches, num_cls=num_class) # make into 5D (batch_size, patch_size[0], patch_size[1], patch_size[2], num_classes)
        #print image_patches.shape
        return image_patches[0,...].astype(np.float32), mask_patches[0,...].astype(np.float32)

    def random_patch_center_z(self, mask, patch_size):
        # bounded within the brain mask region
        limX, limY, limZ = np.where(mask>0)
        if (np.min(limZ) + patch_size[2] // 2 + 1) < (np.max(limZ) - patch_size[2] // 2):
            z = np.random.randint(low = np.min(limZ) + patch_size[2] // 2 + 1, high = np.max(limZ) - patch_size[2] // 2)
        else:
            z = np.random.randint(low = patchsize[2]//2, high = mask.shape[2] - patchsize[2]//2)

        limX, limY, limZ = np.where(mask>0)

        z = np.random.randint(low = max(1, np.min(limZ)), high = min(np.max(limZ), mask.shape[2] - 2))
        # z = np.random.randint(low = max(1, np.min(limZ)), high = min(np.max(limZ), mask.shape[2] - 2))

        return z

    def parse_fn(self, data_path):
        '''
        :param image_path: path to a folder of a patient
        :return: normalized entire image with its corresponding label
        In an image, the air region is 0, so we only calculate the mean and std within the brain area
        For any image-level normalization, do it here
        '''
        path = data_path.split(",")
        image_path = path[0]
        label_path = path[1]
        #itk_image = zoom2shape(image_path, [512,512])#os.path.join(image_path, 'T1_unbiased_brain_rigid_to_mni.nii.gz'))
        #itk_mask = zoom2shape(label_path, [512,512], label=True)#os.path.join(image_path, 'T1_brain_seg_rigid_to_mni.nii.gz'))
        itk_image = sitk.ReadImage(image_path)#os.path.join(image_path, 'T1_unbiased_brain_rigid_to_mni.nii.gz'))
        itk_mask = sitk.ReadImage(label_path)#os.path.join(image_path, 'T1_brain_seg_rigid_to_mni.nii.gz'))
        # itk_image = sitk.ReadImage(os.path.join(image_path, 'T2_FLAIR_unbiased_brain_rigid_to_mni.nii.gz'))

        image = sitk.GetArrayFromImage(itk_image)
        mask = sitk.GetArrayFromImage(itk_mask)
        #image[image >= 1000] = 1000
        binary_mask = np.ones(mask.shape)
        mean = np.sum(image * binary_mask) / np.sum(binary_mask)
        std = np.sqrt(np.sum(np.square(image - mean) * binary_mask) / np.sum(binary_mask))
        image = (image - mean) / std  # normalize per image, using statistics within the brain, but apply to whole image

        mask[mask==2] = 1

        return image.transpose([1,2,0]), mask.transpose([1,2,0]) # transpose the orientation of the


    def _label_decomp(self, label_vol, num_cls):
        """
        decompose label for softmax classifier
        original labels are batchsize * W * H * 1, with label values 0,1,2,3...
        this function decompse it to one hot, e.g.: 0,0,0,1,0,0 in channel dimension
        numpy version of tf.one_hot
        """
        one_hot = []
        for i in xrange(num_cls):
            _vol = np.zeros(label_vol.shape)
            _vol[label_vol == i] = 1
            one_hot.append(_vol)

        return np.stack(one_hot, axis=-1)
    # def augment(self, x):
    #     # add more types of augmentations here
    #     augmentations = [self.flip]
    #     for f in augmentations:
    #         x = tf.cond(tf.random_uniform([], 0, 1) < 0.25, lambda: f(x), lambda: x)
            
    #     return x

    # def flip(self, x):
    #     """Flip augmentation
    #     Args:
    #         x: Image to flip
    #     Returns:
    #         Augmented image
    #     """
    #     x = tf.image.random_flip_left_right(x)
    #     # x = tf.image.random_flip_up_down(x)

    #     return x

