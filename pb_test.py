import tensorflow as tf
import cv2
import numpy as np
import pixel_link
import config
import util

def main(_):
    pb_file = 'e:/ocr/pixel_link.pb'
    image_path = 'd:/tt/TextVOC/JPEGImages/000000.jpg'

    with tf.Graph().as_default():
        graph_def = tf.GraphDef()

        with tf.gfile.FastGFile(pb_file, mode='rb') as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def)


        config.init_config((512, 512))

        with tf.Session() as sess:
            # with tf.summary.FileWriter(logdir='e:/ocr/logdir', graph=sess.graph) as w:
            #     w.flush()
            input_image = sess.graph.get_tensor_by_name('import/input_image:0')
            pixel_pos_scores = sess.graph.get_tensor_by_name('import/pixel_pos_scores:0')
            link_pos_scores = sess.graph.get_tensor_by_name('import/link_pos_scores:0')

            img = cv2.imread(image_path)
            pps, lps = sess.run([pixel_pos_scores, link_pos_scores], feed_dict={input_image:img})

            mask = pixel_link.decode_batch(pps, lps)[0, ...]
            bboxes = pixel_link.mask_to_bboxes(mask, img.shape)
            print(bboxes)

            for points in bboxes:
                points = np.reshape(points, (4, 2))
                cnts = util.img.points_to_contours(points)
                util.img.draw_contours(img, cnts, -1, color = util.img.COLOR_GREEN, border_width = 3)

            cv2.imwrite('d:/tt/000000.jpg', img)


if __name__ == "__main__":
    main(None)