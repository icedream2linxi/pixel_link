import tensorflow as tf
from tensorflow.python.framework import graph_util
from nets import pixel_link_symbol
from preprocessing import ssd_vgg_preprocessing
import config

tf.app.flags.DEFINE_string('train_dir', None, 
                           'the path to store checkpoints and eventfiles for summaries')

tf.app.flags.DEFINE_string('checkpoint_path', None, 
   'the path of pretrained model to be used. If there are checkpoints\
    in train_dir, this config will be ignored.')

tf.app.flags.DEFINE_string(
    'output_file', '', 'Where to save the resulting file to.')

tf.app.flags.DEFINE_integer('export_image_width', 512, 'Train image size')
tf.app.flags.DEFINE_integer('export_image_height', 512, 'Train image size')


FLAGS = tf.app.flags.FLAGS

def main(_):
    image_shape = (FLAGS.export_image_height, FLAGS.export_image_width)
    config.load_config(FLAGS.train_dir)
    config.init_config(image_shape, 
                       batch_size = 1, 
                       pixel_conf_threshold = 0.8,
                       link_conf_threshold = 0.8,
                       num_gpus = 1, 
                   )

    image = tf.placeholder(dtype=tf.int32, shape = [None, None, 3])
    processed_image, _, _, _, _ = ssd_vgg_preprocessing.preprocess_image(image, None, None, None, None, 
                                                   out_shape = config.image_shape,
                                                   data_format = config.data_format, 
                                                   is_training = False)
    b_image = tf.expand_dims(processed_image, axis = 0)
    net = pixel_link_symbol.PixelLinkNet(b_image, is_training = True)
    pixel_pos_scores = tf.identity(net.pixel_pos_scores, name='pixel_pos_scores')
    link_pos_scores = tf.identity(net.link_pos_scores, name='link_pos_scores')

    saver = tf.train.Saver()

    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    checkpoint_exists = ckpt and ckpt.model_checkpoint_path
    if not checkpoint_exists:
        tf.logging.info('Checkpoint not exists in FLAGS.train_dir')
        return


    with tf.Session() as sess:
        saver.restore(sess, ckpt.model_checkpoint_path)
        output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=['pixel_pos_scores', 'link_pos_scores'])

        with tf.gfile.FastGFile(FLAGS.output_file, mode='wb+') as f:
            print('write file : ' + FLAGS.output_file)
            ss = output_graph_def.SerializeToString()
            f.write(output_graph_def.SerializeToString())
            print('Write finish!')


if __name__ == '__main__':
    tf.app.run()