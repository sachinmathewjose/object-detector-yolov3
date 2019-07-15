import numpy as np
import tensorflow as tf
from networks import *
from PIL import Image, ImageDraw

#defining the command line arguments
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input_img', 'horses.jpg', 'Input image')
tf.app.flags.DEFINE_string('output_img', 'out.jpg', 'Output image')
tf.app.flags.DEFINE_string('class_names', 'coco.names', 'File with class names')
tf.app.flags.DEFINE_string('weights_file', 'yolov3.weights', 'Binary file with detector weights')

tf.app.flags.DEFINE_float('conf_threshold', 0.5, 'Confidence threshold')
tf.app.flags.DEFINE_float('iou_threshold', 0.4, 'IoU threshold')


def load_class_names(file_name):
	img = Image.open(FLAGS.input_img)
	img_resized = img.resize(size=(416, 416))
	
	#load class names into the memory from file
	names = {}
	with open(file_name) as f:
		for id, name in enumerate(f):
			names[id] = name
	return names

def draw_boxes(boxes, img, cls_names, detection_size):
	draw = ImageDraw.Draw(img)

	for cls, bboxs in boxes.items():
		color = tuple(np.random.randint(0, 256, 3))
		for box, score in bboxs:
			box = convert_to_original_size(box, np.array(detection_size), np.array(img.size))
			draw.rectangle(box, outline=color)
			draw.text(box[:2], '{} {:.2f}%'.format(cls_names[cls], score * 100), fill=color)


def convert_to_original_size(box, size, original_size):
	ratio = original_size / size
	box = box.reshape(2, 2) * ratio
	return list(box.reshape(-1))

def main(argv=None):
	#code for running the module
	img = Image.open(FLAGS.input_img)
	img_resized = img.resize(size=(416,416))
	classes = load_class_names(FLAGS.class_names) #class names
	inputs = tf.placeholder(tf.float32, [None, 416, 416, 3])
									#None indicates the placeholder is of any batch size
	with tf.variable_scope('detector'):
		yolo_out = yolo_v3(inputs, len(classes), data_format='NCHW')
		load_ops = load_weights(tf.global_variables(scope='detector'), FLAGS.weights_file)
	detection = convert_to_box(yolo_out)
	with tf.Session() as sess:
		#~ writer = tf.summary.FileWriter("/home/nvidia/work/computervision/tensorflow_yolo/out",sess.graph)
		sess.run(load_ops)
		print "loaded weights"
		detected_boxes = sess.run(detection, feed_dict={inputs: [np.array(img_resized, dtype=np.float32)]})
		#~ writer.close()
	filtered_boxes = non_max_suppression(detected_boxes, confidence_threshold=FLAGS.conf_threshold,
																iou_threshold=FLAGS.iou_threshold)
	draw_boxes(filtered_boxes, img, classes, (FLAGS.size, FLAGS.size))
	img.save(FLAGS.output_img)


if __name__ == '__main__':
    tf.app.run()
