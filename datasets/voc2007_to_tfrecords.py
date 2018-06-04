#encoding=utf-8
import numpy as np
import tensorflow as tf
import util
from dataset_utils import convert_to_example
import config
import xml.etree.ElementTree as ET
        

def cvt_to_tfrecords(output_path , images_dir, annotations_dir, files_txt):
    with tf.python_io.TFRecordWriter(output_path) as tfrecord_writer:
        lines = util.io.read_lines(files_txt)
        count = len(lines)
        for idx, filename in enumerate(lines):
            oriented_bboxes = []
            bboxes = []
            labels = []
            labels_text = []

            filename = filename.replace('\n', '')
            print('{} / {} {}'.format(idx, count, filename))

            annotation_path = util.io.join_path(annotations_dir, filename + '.xml')
            annotation_xml = ET.ElementTree(file=annotation_path)
            image_filename = annotation_xml.find('filename').text
            image_path = util.io.join_path(images_dir, image_filename)
            image_data = tf.gfile.FastGFile(image_path, 'rb').read()
            image = util.img.imread(image_path, rgb=True)
            shape = image.shape
            h, w = shape[0:2]
            h *= 1.0
            w *= 1.0

            for obj_ele in annotation_xml.iterfind('object'):
                content = obj_ele.find('content').text
                content = str.encode(content)
                labels_text.append(content)
                labels.append(config.text_label)

                bndbox = obj_ele.find('bndbox')
                xmin = int(bndbox.find('xmin').text) / w
                ymin = int(bndbox.find('ymin').text) / h
                xmax = int(bndbox.find('xmax').text) / w
                ymax = int(bndbox.find('ymax').text) / h
                bboxes.append([xmin, ymin, xmax, ymax])

                if xmin < 0 or ymin < 0 or xmax > 1 or ymax > 1:
                    print(filename)
                    return

                oriented_box = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymin]
                oriented_bboxes.append(oriented_box)
            
            filename = str.encode(filename)
            example = convert_to_example(image_data, filename, labels, labels_text, bboxes, oriented_bboxes, shape)
            tfrecord_writer.write(example.SerializeToString())
        
if __name__ == "__main__":
    root_dir = util.io.get_absolute_path('d:/tt/TextVOC/')
    output_dir = util.io.get_absolute_path('d:/tt/TextVOC/')
    util.io.mkdir(output_dir)

    images_dir = util.io.join_path(root_dir, 'JPEGImages')
    annotations_dir = util.io.join_path(root_dir,'Annotations')
    files_txt = util.io.join_path(root_dir, 'ImageSets/Main/train.txt')
    cvt_to_tfrecords(util.io.join_path(output_dir, 'voc2007_train.tfrecord'), images_dir, annotations_dir, files_txt)

    files_txt = util.io.join_path(root_dir, 'ImageSets/Main/test.txt')
    cvt_to_tfrecords(util.io.join_path(output_dir, 'voc2007_test.tfrecord'), images_dir, annotations_dir, files_txt)
