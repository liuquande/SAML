import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
from data_generator import ImageDataGenerator
import logging
from utils import _eval_dice, _connectivity_region_analysis, parse_fn, _crop_object_region, _get_coutour_sample, parse_fn_haus,_eval_haus
import time 
import os
import SimpleITK as sitk

def train(model, saver, sess, train_file_list, test_file, args, resume_itr=0):

    if args.log:
        train_writer = tf.summary.FileWriter(args.log_dir + '/' + args.phase + '/', sess.graph)

    # Data loaders
    with tf.device('/cpu:0'):
        tr_data_list, train_iterator_list, train_next_list = [],[],[]
        for i in range(len(train_file_list)):
            tr_data = ImageDataGenerator(train_file_list[i], mode='training', \
                                         batch_size=args.meta_batch_size, num_classes=args.n_class, shuffle=True)
            tr_data_list.append(tr_data)
            train_iterator_list.append(tf.data.Iterator.from_structure(tr_data.data.output_types,tr_data.data.output_shapes))
            train_next_list.append(train_iterator_list[i].get_next())

    # Ops for initializing different iterators
    training_init_op = []
    train_batches_per_epoch = []
    for i in range(len(train_file_list)):
        training_init_op.append(train_iterator_list[i].make_initializer(tr_data_list[i].data))
        sess.run(training_init_op[i])  # initialize training sample generator at itr=0

    # Training begins
    best_test_dice = 0
    best_test_haus = 0
    for epoch in xrange(0, args.epoch):
    	for itr in range(resume_itr, args.train_iterations):
			start = time.time()
			# Sampling training and test tasks
			num_training_tasks = len(train_file_list)
			num_meta_train = 2#num_training_tasks-1
			num_meta_test = 1#num_training_tasks-num_meta_train  # as setting num_meta_test = 1

			# Randomly choosing meta train and meta test domains
			task_list = np.random.permutation(num_training_tasks)
			meta_train_index_list = task_list[:2]
			meta_test_index_list = task_list[-1:]

			# Sampling meta-train, meta-test data
			for i in range(num_meta_train):
			    task_ind = meta_train_index_list[i]
			    if i == 0:
			        inputa, labela = sess.run(train_next_list[task_ind])
			    elif i == 1:
			        inputa1, labela1 = sess.run(train_next_list[task_ind])
			    else:
			        raise RuntimeError('check number of meta-train domains.')

			for i in range(num_meta_test):
			    task_ind = meta_test_index_list[i]
			    if i == 0:
			        inputb, labelb = sess.run(train_next_list[task_ind])
			    else:
			        raise RuntimeError('check number of meta-test domains.')
			
			input_group = np.concatenate((inputa[:2],inputa1[:1],inputb[:2]), axis=0)
			label_group = np.concatenate((labela[:2],labela1[:1],labelb[:2]), axis=0)

			contour_group, metric_label_group = _get_coutour_sample(label_group)

			feed_dict = {model.inputa: inputa, model.labela: labela, \
			             model.inputa1: inputa1, model.labela1: labela1, \
			             model.inputb: inputb, model.labelb: labelb, \
			             model.input_group:input_group, \
			             model.label_group:label_group, \
			             model.contour_group:contour_group, \
			             model.metric_label_group:metric_label_group, \
			             model.KEEP_PROB: 1.0}

			output_tensors = [model.task_train_op, model.meta_train_op, model.metric_train_op]
			output_tensors.extend([model.summ_op, model.seg_loss_b, model.compactness_loss_b, model.smoothness_loss_b, model.target_loss, model.source_loss])
			_, _, _, summ_writer, seg_loss_b, compactness_loss_b, smoothness_loss_b, target_loss, source_loss = sess.run(output_tensors, feed_dict)
			# output_tensors = [model.task_train_op]
			# output_tensors.extend([model.source_loss])
			# _, source_loss = sess.run(output_tensors, feed_dict)

			if itr % args.print_interval == 0:
			    logging.info("Epoch: [%2d] [%6d/%6d] time: %4.4f inner lr:%.8f outer lr:%.8f" % (epoch, itr, args.train_iterations, (time.time()-start), model.inner_lr.eval(), model.outer_lr.eval()))
			    logging.info('sou_loss: %.7f, tar_loss: %.7f, tar_seg_loss: %.7f, tar_compactness_loss: %.7f, tar_smoothness_loss: %.7f' % (source_loss, target_loss, seg_loss_b, compactness_loss_b, smoothness_loss_b))

			if itr % args.summary_interval == 0:
			    train_writer.add_summary(summ_writer, itr)
			    train_writer.flush()

			if (itr!=0) and itr % args.save_freq == 0:
			    saver.save(sess, args.checkpoint_dir + '/epoch_' + str(epoch) + '_itr_'+str(itr) + ".model.cpkt")

			# Testing periodically
			if (itr!=0) and itr % args.test_freq == 0:
			    test_dice, test_dice_arr, test_haus, test_haus_arr = test(sess, test_file, model, args)

			    if test_dice > best_test_dice:
			        best_test_dice = test_dice

			    with open((os.path.join(args.log_dir,'eva.txt')), 'a') as f:
			        print >> f, 'Iteration %d :' % (itr)
			        print >> f, '	Unseen domain testing results: Dice: %f' %(test_dice), test_dice_arr
			        print >> f, '	Current best accuracy %f' %(best_test_dice)
			        print >> f, '	Unseen domain testing results: Haus: %f' %(test_haus), test_haus_arr
			        print >> f, '	Current best accuracy %f' %(best_test_haus)
			    # Save model

def test(sess, test_list, model, args):
    
    dice = []
    haus = []
    start = time.time()

    with open(test_list, 'r') as fp:
        rows = fp.readlines()
    test_list  = [row[:-1] if row[-1] == '\n' else row for row in rows]

    for fid, filename in enumerate(test_list):
        image, mask, spacing = parse_fn_haus(filename)
        pred_y = np.zeros(mask.shape)

        frame_list = [kk for kk in range(1, image.shape[2] - 1)]

        for ii in xrange(int(np.floor(image.shape[2] // model.test_batch_size))):
            vol = np.zeros([model.test_batch_size, model.volume_size[0], model.volume_size[1], model.volume_size[2]])

            for idx, jj in enumerate(frame_list[ii * model.test_batch_size: (ii + 1) * model.test_batch_size]):
                vol[idx, ...] = image[..., jj - 1: jj + 2].copy()

            pred_student = sess.run((model.outputs), feed_dict={model.test_input: vol, \
                                                                    model.KEEP_PROB: 1.0,\
                                                                    model.training_mode: True})

            for idx, jj in enumerate(frame_list[ii * model.test_batch_size: (ii + 1) * model.test_batch_size]):
                pred_y[..., jj] = pred_student[idx, ...].copy()

        processed_pred_y = _connectivity_region_analysis(pred_y)

        dice_subject = _eval_dice(mask, processed_pred_y)

        # print spacing
        dice.append(dice_subject)
        # haus.append(haus_subject)
        # _save_nii_prediction(mask, processed_pred_y, pred_y, args.result_dir, '_' + filename[-26:-20])
    dice_avg = np.mean(dice, axis=0).tolist()[0]
    # haus_avg = np.mean(haus, axis=0).tolist()[0]
    
    logging.info("dice_avg %.4f" % (dice_avg))
    # logging.info("haus_avg %.4f" % (haus_avg))

    return dice_avg, dice, 0, 0
    # return dice_avg, dice, haus_avg, haus

def _save_nii_prediction(gth, comp_pred, pre_pred, out_folder, out_bname):
    sitk.WriteImage(sitk.GetImageFromArray(gth), out_folder + out_bname + 'gth.nii.gz') 
    sitk.WriteImage(sitk.GetImageFromArray(pre_pred), out_folder + out_bname + 'premask.nii.gz') 
    sitk.WriteImage(sitk.GetImageFromArray(comp_pred), out_folder + out_bname + 'mask.nii.gz') 
