""" Utility functions. """
import numpy as np
import os
import random
import tensorflow as tf

from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.python.platform import flags
import SimpleITK as sitk
from scipy import ndimage
import itertools
from tensorflow.contrib import slim
from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion,\
    generate_binary_structure
FLAGS = flags.FLAGS

## Image reader
def get_images(paths, labels, nb_samples=None, shuffle=True):
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images = [(i, os.path.join(path, image)) \
        for i, path in zip(labels, paths) \
        for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images)
    return images
 
## Loss functions
def mse(pred, label):
    pred = tf.reshape(pred, [-1])
    label = tf.reshape(label, [-1])
    return tf.reduce_mean(tf.square(pred-label))

def xent(pred, label):
    return tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=label)

def kd(data1, label1, data2, label2, bool_indicator, n_class=7, temperature=2.0):

    kd_loss = 0.0
    eps = 1e-16

    prob1s = []
    prob2s = []

    for cls in range(n_class):
        mask1 = tf.tile(tf.expand_dims(label1[:, cls], -1), [1, n_class])
        logits_sum1 = tf.reduce_sum(tf.multiply(data1, mask1), axis=0)
        num1 = tf.reduce_sum(label1[:, cls])
        activations1 = logits_sum1 * 1.0 / (num1 + eps) # add eps for prevent un-sampled class resulting in NAN
        prob1 = tf.nn.softmax(activations1 / temperature)
        prob1 = tf.clip_by_value(prob1, clip_value_min=1e-8, clip_value_max=1.0)  # for preventing prob=0 resulting in NAN

        mask2 = tf.tile(tf.expand_dims(label2[:, cls], -1), [1, n_class])
        logits_sum2 = tf.reduce_sum(tf.multiply(data2, mask2), axis=0)
        num2 = tf.reduce_sum(label2[:, cls])
        activations2 = logits_sum2 * 1.0 / (num2 + eps)
        prob2 = tf.nn.softmax(activations2 / temperature)
        prob2 = tf.clip_by_value(prob2, clip_value_min=1e-8, clip_value_max=1.0)

        KL_div = (tf.reduce_sum(prob1 * tf.log(prob1 / prob2)) + tf.reduce_sum(prob2 * tf.log(prob2 / prob1))) / 2.0
        kd_loss += KL_div * bool_indicator[cls]

        prob1s.append(prob1)
        prob2s.append(prob2)

    kd_loss = kd_loss / n_class

    return kd_loss, prob1s, prob2s

def JS(data1, label1, data2, label2, bool_indicator, n_class=7, temperature=2.0):

    kd_loss = 0.0
    eps = 1e-16

    prob1s = []
    prob2s = []

    for cls in range(n_class):
        mask1 = tf.tile(tf.expand_dims(label1[:, cls], -1), [1, n_class])
        logits_sum1 = tf.reduce_sum(tf.multiply(data1, mask1), axis=0)
        num1 = tf.reduce_sum(label1[:, cls])
        activations1 = logits_sum1 * 1.0 / (num1 + eps) # add eps for prevent un-sampled class resulting in NAN
        prob1 = tf.nn.softmax(activations1 / temperature)
        prob1 = tf.clip_by_value(prob1, clip_value_min=1e-8, clip_value_max=1.0)  # for preventing prob=0 resulting in NAN

        mask2 = tf.tile(tf.expand_dims(label2[:, cls], -1), [1, n_class])
        logits_sum2 = tf.reduce_sum(tf.multiply(data2, mask2), axis=0)
        num2 = tf.reduce_sum(label2[:, cls])
        activations2 = logits_sum2 * 1.0 / (num2 + eps)
        prob2 = tf.nn.softmax(activations2 / temperature)
        prob2 = tf.clip_by_value(prob2, clip_value_min=1e-8, clip_value_max=1.0)

        mean_prob = (prob1 + prob2) / 2

        JS_div = (tf.reduce_sum(prob1 * tf.log(prob1 / mean_prob)) + tf.reduce_sum(prob2 * tf.log(prob2 / mean_prob))) / 2.0
        kd_loss += JS_div * bool_indicator[cls]

        prob1s.append(prob1)
        prob2s.append(prob2)

    kd_loss = kd_loss / n_class

    return kd_loss, prob1s, prob2s

def contrastive(feature1, label1, feature2, label2, bool_indicator=None, margin=50):

    l1 = tf.argmax(label1, axis=1)
    l2 = tf.argmax(label2, axis=1)
    pair = tf.to_float(tf.equal(l1,l2))

    delta = tf.reduce_sum(tf.square(feature1-feature2), 1) + 1e-10
    match_loss = delta

    delta_sqrt = tf.sqrt(delta + 1e-10)
    mismatch_loss = tf.square(tf.nn.relu(margin - delta_sqrt))

    if bool_indicator is None:
        loss = tf.reduce_mean(0.5 * (pair * match_loss + (1-pair) * mismatch_loss))
    else:
        loss = 0.5 * tf.reduce_sum(match_loss*pair)/tf.reduce_sum(pair)

    debug_dist_positive = tf.reduce_sum(delta_sqrt * pair)/tf.reduce_sum(pair)
    debug_dist_negative = tf.reduce_sum(delta_sqrt * (1-pair))/tf.reduce_sum(1-pair)

    return loss, pair, delta, debug_dist_positive, debug_dist_negative

def compute_distance(feature1, label1, feature2, label2):
    l1 = tf.argmax(label1, axis=1)
    l2 = tf.argmax(label2, axis=1)
    pair = tf.to_float(tf.equal(l1,l2))

    delta = tf.reduce_sum(tf.square(feature1-feature2), 1)
    delta_sqrt = tf.sqrt(delta + 1e-16)

    dist_positive_pair = tf.reduce_sum(delta_sqrt * pair)/tf.reduce_sum(pair)
    dist_negative_pair = tf.reduce_sum(delta_sqrt * (1-pair))/tf.reduce_sum(1-pair)

    return dist_positive_pair, dist_negative_pair

def _get_segmentation_cost(softmaxpred, seg_gt, n_class=2):
    """
    calculate the loss for segmentation prediction
    :param seg_logits: probability segmentation from the segmentation network
    :param seg_gt: ground truth segmentaiton mask
    :return: segmentation loss, according to the cost_kwards setting, cross-entropy weighted loss and dice loss
    """
    dice = 0

    for i in xrange(n_class):
        #inse = tf.reduce_sum(softmaxpred[:, :, :, i]*seg_gt[:, :, :, i])
        inse = tf.reduce_sum(softmaxpred[:, :, :, i]*seg_gt[:, :, :, i])
        l = tf.reduce_sum(softmaxpred[:, :, :, i])
        r = tf.reduce_sum(seg_gt[:, :, :, i])
        dice += 2.0 * inse/(l+r+1e-7) # here 1e-7 is relaxation eps
    dice_loss = 1 - 1.0 * dice / n_class

    # ce_weighted = 0
    # for i in xrange(n_class):
    #     gti = seg_gt[:,:,:,i]
    #     predi = softmaxpred[:,:,:,i]
    #     ce_weighted += -1.0 * gti * tf.log(tf.clip_by_value(predi, 0.005, 1))
    # ce_weighted_loss = tf.reduce_mean(ce_weighted)

    # total_loss =  dice_loss 


    return dice_loss#, dice_loss, ce_weighted_loss

def _get_compactness_cost(y_pred, y_true): 

    """
    y_pred: BxHxWxC
    """
    """
    lenth term
    """

    # y_pred = tf.one_hot(y_pred, depth=2)
    # print (y_true.shape)
    # print (y_pred.shape)
    y_pred = y_pred[..., 1]
    y_true = y_pred[..., 1]

    x = y_pred[:,1:,:] - y_pred[:,:-1,:] # horizontal and vertical directions 
    y = y_pred[:,:,1:] - y_pred[:,:,:-1]

    delta_x = x[:,:,1:]**2
    delta_y = y[:,1:,:]**2

    delta_u = tf.abs(delta_x + delta_y) 

    epsilon = 0.00000001 # where is a parameter to avoid square root is zero in practice.
    w = 0.01
    length = w * tf.reduce_sum(tf.sqrt(delta_u + epsilon), [1, 2])

    area = tf.reduce_sum(y_pred, [1,2])

    compactness_loss = tf.reduce_sum(length ** 2 / (area * 4 * 3.1415926))

    return compactness_loss, tf.reduce_sum(length), tf.reduce_sum(area), delta_u

# def _get_sample_masf(y_true):
#     """
#     y_pred: BxHxWx2
#     """
#     positive_mask = np.expand_dims(y_true[..., 1], axis=3)
#     metrix_label_group = np.expand_dims(np.array([1, 0, 1, 1, 0]), axis = 1)
#     # print (positive_mask.shape)
#     coutour_group = np.zeros(positive_mask.shape)

#     for i in range(positive_mask.shape[0]):
#         slice_i = positive_mask[i]
        
#         if metrix_label_group[i] == 1:
#             sample = (slice_i == 1)
#         elif metrix_label_group[i] == 0:
#             sample = (slice_i == 0)

#         coutour_group[i] = sample

#     return coutour_group, metrix_label_group

def _get_coutour_sample(y_true):
    """
    y_true: BxHxWx2
    """
    positive_mask = np.expand_dims(y_true[..., 1], axis=3)
    metrix_label_group = np.expand_dims(np.array([1, 0, 1, 1, 0]), axis = 1)
    coutour_group = np.zeros(positive_mask.shape)

    for i in range(positive_mask.shape[0]):
        slice_i = positive_mask[i]

        if metrix_label_group[i] == 1:
            # generate coutour mask
            erosion = ndimage.binary_erosion(slice_i[..., 0], iterations=1).astype(slice_i.dtype)
            sample = np.expand_dims(slice_i[..., 0] - erosion, axis = 2)

        elif metrix_label_group[i] == 0:
            # generate background mask
            dilation = ndimage.binary_dilation(slice_i, iterations=5).astype(slice_i.dtype)
            sample = dilation - slice_i 

        coutour_group[i] = sample
    return coutour_group, metrix_label_group

# def _get_negative(y_true):
def _get_boundary_cost(y_pred, y_true): 

    """
    y_pred: BxHxWxC
    """
    """
    lenth term
    """

    # y_pred = tf.one_hot(y_pred, depth=2)
    # print (y_true.shape)
    # print (y_pred.shape)
    y_pred = y_pred[..., 1]
    y_true = y_pred[..., 1]

    x = y_pred[:,1:,:] - y_pred[:,:-1,:] # horizontal and vertical directions 
    y = y_pred[:,:,1:] - y_pred[:,:,:-1]

    delta_x = x[:,:,1:]**2
    delta_y = y[:,1:,:]**2

    delta_u = tf.abs(delta_x + delta_y) 

    epsilon = 0.00000001 # where is a parameter to avoid square root is zero in practice.
    w = 0.01
    length = w * tf.reduce_sum(tf.sqrt(delta_u + epsilon), [1, 2]) # equ.(11) in the paper

    area = tf.reduce_sum(y_pred, [1,2])

    compactness_loss = tf.reduce_sum(length ** 2 / (area * 4 * 3.1415926))

    return compactness_loss, tf.reduce_sum(length), tf.reduce_sum(area)

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        print ("Allocating '{:}'".format(log_dir))
        os.makedirs(log_dir)
    return log_dir

def _eval_dice(gt_y, pred_y, detail=False):

    class_map = {  # a map used for mapping label value to its name, used for output
        "0": "bg",
        "1": "CZ",
        "2": "prostate"
    }

    dice = []

    for cls in xrange(1,2):

        gt = np.zeros(gt_y.shape)
        pred = np.zeros(pred_y.shape)

        gt[gt_y == cls] = 1
        pred[pred_y == cls] = 1

        dice_this = 2*np.sum(gt*pred)/(np.sum(gt)+np.sum(pred))
        dice.append(dice_this)

        if detail is True:
            #print ("class {}, dice is {:2f}".format(class_map[str(cls)], dice_this))
            logging.info("class {}, dice is {:2f}".format(class_map[str(cls)], dice_this))
    return dice

def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()
            
    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)
    
    # test for emptiness
    if 0 == np.count_nonzero(result): 
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(reference): 
        raise RuntimeError('The second supplied array does not contain any binary object.')    
            
    # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)
    
    # compute average surface distance        
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]
    
    return sds

def asd(result, reference, voxelspacing=None, connectivity=1):
  
    sds = __surface_distances(result, reference, voxelspacing, connectivity)
    asd = sds.mean()
    return asd

def calculate_hausdorff(lP,lT,spacing):

    return asd(lP, lT, spacing)

def _eval_haus(pred, gt, spacing, detail=False):
    '''
    :param pred: whole brain prediction
    :param gt: whole
    :param detail:
    :return: a list, indicating Dice of each class for one case
    '''
    haus = []

    for cls in range(1,2):
        pred_i = np.zeros(pred.shape)
        pred_i[pred == cls] = 1
        gt_i = np.zeros(gt.shape)
        gt_i[gt == cls] = 1

        # hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
        # hausdorff_distance_filter.Execute(gt_i, pred_i)

        haus_cls = calculate_hausdorff(gt_i, (pred_i), spacing)

        haus.append(haus_cls)

        if detail is True:
            logging.info("class {}, haus is {:4f}".format(class_map[str(cls)], haus_cls))
    # logging.info("4 class average haus is {:4f}".format(np.mean(haus)))

    return haus

def _connectivity_region_analysis(mask):
    s = [[0,1,0],
         [1,1,1],
         [0,1,0]]
    label_im, nb_labels = ndimage.label(mask)#, structure=s)

    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))

    # plt.imshow(label_im)        
    label_im[label_im != np.argmax(sizes)] = 0
    label_im[label_im == np.argmax(sizes)] = 1

    return label_im

def _crop_object_region(mask, prediction):

    limX, limY, limZ = np.where(mask>0)
    min_z = np.min(limZ)
    max_z = np.max(limZ)

    prediction[..., :np.min(limZ)] = 0
    prediction[..., np.max(limZ)+1:] = 0

    return prediction

def parse_fn(data_path):
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


def parse_fn_haus(data_path):
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
    spacing = itk_mask.GetSpacing()

    image = sitk.GetArrayFromImage(itk_image)
    mask = sitk.GetArrayFromImage(itk_mask)
    #image[image >= 1000] = 1000
    binary_mask = np.ones(mask.shape)
    mean = np.sum(image * binary_mask) / np.sum(binary_mask)
    std = np.sqrt(np.sum(np.square(image - mean) * binary_mask) / np.sum(binary_mask))
    image = (image - mean) / std  # normalize per image, using statistics within the brain, but apply to whole image

    mask[mask==2] = 1

    return image.transpose([1,2,0]), mask.transpose([1,2,0]), spacing

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

