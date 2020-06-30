import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
from data_generator import ImageDataGenerator
from saml_func import SAML
from train import train
from train import test
import datetime
import argparse
from utils import check_folder, show_all_variables
import logging

currtime = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
tf.set_random_seed(2)

def parse_args(train_date):
    desc = "Tensorflow implementation of DenseUNet for prostate segmentation"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--gpu', type=str, default='0', help='train or test or guide')
    parser.add_argument('--phase', type=str, default='train', help='train or test or guide')
    parser.add_argument('--n_class', type=int, default=2, help='The size of class')

    ## Training operations
    parser.add_argument('--target_domain', type=str, default='ISBI', help='dataset_name')
    parser.add_argument('--volume_size', type=list, default=[384, 384, 3], help='The size of input data')
    parser.add_argument('--label_size', type=list, default=[384, 384, 1], help='The size of label')
    parser.add_argument('--epoch', type=int, default=1, help='The number of epochs to run')
    parser.add_argument('--train_iterations', type=int, default=10000, help='The number of training iterations')
    parser.add_argument('--meta_batch_size', type=int, default=5, help='number of images sampled per source domain')
    parser.add_argument('--test_batch_size', type=int, default=1, help='number of images sampled per source domain')
    parser.add_argument('--inner_lr', type=float, default=1e-4, help='The learning rate')
    parser.add_argument('--outer_lr', type=float, default=1e-3, help='The learning rate')
    parser.add_argument('--metric_lr', type=float, default=1e-3, help='The learning rate')
    parser.add_argument('--margin', type=float, default=10.0, help='The learning rate')
    parser.add_argument('--compactness_loss_weight', type=float, default=1.0, help='The learning rate')
    parser.add_argument('--smoothness_loss_weight', type=float, default=0.005, help='The learning rate')
    parser.add_argument('--clipNorm', type=int, default=True, help='number of images sampled per source domain')
    parser.add_argument('--gradients_clip_value', type=float, default=10.0, help='The learning rate')

    # Logging, saving, and testing options
    parser.add_argument('--resume', type=int, default=False, help='number of images sampled per source domain')
    parser.add_argument('--log', type=int, default=True, help='write tensorboard')
    parser.add_argument('--decay_step', type=float, default=500, help='The learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.95, help='The learning rate')
    parser.add_argument('--test_freq', type=int, default=200, help='The number of ckpt_save_freq')
    parser.add_argument('--save_freq', type=int, default=200, help='The number of ckpt_save_freq')
    parser.add_argument('--print_interval', type=int, default=5, help='The frequency to write tensorboard')
    parser.add_argument('--summary_interval', type=int, default=20, help='The frequency to write tensorboard')
    parser.add_argument('--restored_model', type=str, default=None, help='Model to restore')
    parser.add_argument('--test_model', type=str, default=None, help='Model to restore')
    # parser.add_argument('--dropout', type=str, default=1, help='dropout rate')
    # parser.add_argument('--cost_kwargs', type=str, default=1, help='cost_kwargs')
    # parser.add_argument('--opt_kwargs', type=str, default=1, help='opt_kwargs')

    parser.add_argument('--checkpoint_dir', type=str, default='../output/' + train_date + '/checkpoints/' ,
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='../output/' + train_date + '/results/',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='../output/' + train_date + '/logs/',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='../output/' + train_date + '/samples/',
                        help='Directory name to save the samples on training')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)
    # --result_dir
    check_folder(args.result_dir)
    # --result_dir
    check_folder(args.log_dir)
    # --sample_dir
    check_folder(args.sample_dir)

    return args

def main():
    train_date = 'xxx'
    args = parse_args(train_date)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # define logger
    logging.basicConfig(filename=args.log_dir+"/"+args.phase+'_log.txt', level=logging.DEBUG, format='%(asctime)s %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())

    # print all parameters
    logging.info("Usage:")
    logging.info("    {0}".format(" ".join([x for x in sys.argv]))) 
    logging.debug("All settings used:")

    os.system('cp main.py %s' % (args.log_dir)) # bkp of train procedure
    os.system('cp saml_func.py %s' % (args.log_dir)) # bkp of train procedure
    os.system('cp train.py %s' % (args.log_dir)) # bkp of train procedure
    os.system('cp utils.py %s' % (args.log_dir)) # bkp of train procedure
    os.system('cp data_generator.py %s' % (args.log_dir))


    filelist_root = '../dataset'
    source_list = ['HK', 'ISBI', 'ISBI_1.5', 'I2CVB','UCL', 'BIDMC']#'ISBI_1.5', 'I2CVB', 'UCL','BIDMC']#, 'I2CVB', 'ISBI_1.5', 'UCL', 'BIDMC']#'I2CVB', 'UCL', 'BIDMC', 'HK']
    source_list.remove(args.target_domain)

    # Constructing model
    model = SAML(args)
    model.construct_model_train()
    model.construct_model_test()
    
    model.summ_op = tf.summary.merge_all()
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
    sess = tf.InteractiveSession()

    tf.global_variables_initializer().run()
    show_all_variables()

    # restore model ----
    resume_itr = 0
    model_file = None
    if args.resume:
        model_file = tf.train.latest_checkpoint(args.checkpoint_dir)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1+5:])
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)

    train_file_list = [os.path.join(filelist_root, source_domain+'_train_list') for source_domain in source_list]
    test_file_list = [os.path.join(filelist_root, args.target_domain+'_train_list')]

    # start training ----
    if args.phase == 'train':
        train(model, saver, sess, train_file_list, test_file_list[0], args, resume_itr)
    else:
        args.test_model = 'xxx'
        saver.restore(sess, args.test_model)
        logging.info("testing model restored %s" % args.test_model)

        test_dice, test_dice_arr, test_haus, test_haus_arr = test(sess, test_file_list[0], model, args)
        with open((os.path.join(args.log_dir,'test.txt')), 'a') as f:
            print >> f, 'testing model %s :' % (args.test_model)
            print >> f, '   Unseen domain testing results: Dice: %f' %(test_dice), test_dice_arr
            print >> f, '   Unseen domain testing results: Haus: %f' %(test_haus), test_haus_arr

if __name__ == "__main__":
    main()
