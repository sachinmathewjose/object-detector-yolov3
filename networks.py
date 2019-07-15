import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

_BatchNorm_momentum = 0.9
_Batchnorm_E = 1e-05
_LeakeyRELU_alpha = 0.1

_Coco_anchor = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)]

def upsample_layer(inputs, out_shape, data_format='NCHW'):
	if data_format == 'NCHW':
		inputs = tf.transpose(inputs, [0, 2, 3, 1])
	if data_format == 'NCHW':
		height = out_shape[3]
		width = out_shape[2]
	else:
		height = out_shape[2]
		width = out_shape[1]
	inputs = tf.image.resize_bilinear(inputs, (height, width))
	if data_format == 'NCHW':
		inputs = tf.transpose(inputs, [0, 3, 1, 2])
	inputs = tf.identity(inputs, name='upsampled')
	return inputs

def conv2d_layer(inputs, filters, kernel_size, strides=1):
	inputs = slim.conv2d(inputs, filters, kernel_size, stride=strides, padding='SAME')
	return inputs
	
def shortcut_layer(inputs, filters):
	start_layer = inputs
	inputs = conv2d_layer(inputs, filters, 1)
	inputs = conv2d_layer(inputs, filters * 2, 3)
	inputs = inputs + start_layer
	return inputs
	
def darknet53_layer(inputs):
	inputs = conv2d_layer(inputs, 32, 3)
	inputs = conv2d_layer(inputs, 64, 3, strides=2)
	inputs = shortcut_layer(inputs, 32)
	inputs = conv2d_layer(inputs, 128, 3, strides=2)
	for i in range(2):
		inputs = shortcut_layer(inputs, 64)
	inputs = conv2d_layer(inputs, 256, 3, strides=2)
	for i in range(8):
		inputs = shortcut_layer(inputs, 128)
	route_layer36 = inputs
	inputs = conv2d_layer(inputs, 512, 3, strides=2)
	for i in range(8):
		inputs = shortcut_layer(inputs, 256)
	route_layer61 = inputs
	inputs = conv2d_layer(inputs, 1024, 3, strides=2)
	for i in range(4):
		inputs = shortcut_layer(inputs, 512)
	return route_layer36, route_layer61, inputs
	
def yolo_layer(inputs, num_classes, img_size, anchors, data_format = 'NCHW'):
	box_size = num_classes + 5
	num_anchors = len(anchors)
	inputs = slim.conv2d(inputs, num_anchors * box_size, 1, stride=1, normalizer_fn=None,
										activation_fn=None, biases_initializer=tf.zeros_initializer())
	input_shape = inputs.get_shape().as_list()
	if (len(input_shape) == 4): grid_dim = input_shape[2:4] if data_format == 'NCHW' else input_shape[1:3]
	else: grid_dim = input_shape[1:3] if data_format == 'NCHW' else input_shape[0:2]
	input_dim = grid_dim[0] * grid_dim[1]
	if data_format == 'NCHW':
		inputs = tf.reshape(inputs, [-1, num_anchors * box_size, input_dim])
		inputs = tf.transpose(inputs,[0,2,1])
	inputs = tf.reshape(inputs, [-1, input_dim * num_anchors, box_size])
	box_center, box_size, box_confidence, box_class = tf.split(inputs, [2, 2, 1, num_classes], axis=-1)
	box_confidence = tf.nn.sigmoid(box_confidence)
	box_class = tf.nn.sigmoid(box_class)
	stride = (img_size[0] // grid_dim[0], img_size[1] // grid_dim[1])
	
	anchors = [(a[0] / stride[0], a[1] / stride[1]) for a in anchors]
	anchors = tf.tile(anchors, [input_dim, 1])
	tf.cast(box_size, tf.int32)
	anchors = tf.cast(anchors, tf.float32)
	box_size = tf.exp(box_size)
	print anchors
	box_size = box_size * anchors
	box_size = box_size * stride
	
	box_center = tf.nn.sigmoid(box_center)
	grid_x = tf.range(grid_dim[0], dtype=tf.float32)
	grid_y = tf.range(grid_dim[1], dtype=tf.float32)
	a, b = tf.meshgrid(grid_x, grid_y)
	x_offset = tf.reshape(a, (-1, 1))
	y_offset = tf.reshape(b, (-1, 1))
	x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
	x_y_offset = tf.reshape(tf.tile(x_y_offset, [1, num_anchors]), [1, -1, 2]) #could be [-1,2]
	box_center = box_center + x_y_offset
	box_center = box_center * stride
	detections = tf.concat([box_center, box_size, box_confidence, box_class], axis=-1)
	return detections
	
def convert_to_box(detections):
	center_x, center_y, width, height, attrs = tf.split(detections, [1, 1, 1, 1, -1], axis=-1)
	w2 = width / 2
	h2 = height / 2
	x0 = center_x - w2
	y0 = center_y - h2
	x1 = center_x + w2
	y1 = center_y + h2
	boxes = tf.concat([x0, y0, x1, y1, attrs], axis=-1)
	return boxes


def yolo_v3(inputs, numclasses, data_format='NCHW', is_training=False, reuse=False):
	# input data model is NHWC, for better performance in GPU we use NCHW 
	# data_format = data format to be used
	img_size = inputs.get_shape().as_list()[1:3]
	if data_format == 'NCHW':
		inputs = tf.transpose(inputs, [0, 3, 1, 2])#convert to NCHW
	#normalize the input
	batch_norm_params = {
	'decay': _BatchNorm_momentum,
	'epsilon': _Batchnorm_E,
	'scale': True,
	'is_training': False,
	'fused': None,  # Use fused batch norm if possible.
	}
	
	with slim.arg_scope([slim.conv2d, slim.batch_norm], data_format=data_format, reuse=reuse):
		#This will create a scope for the operators specified in the list and pass all the other arguments into the ops
		with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params,
                            biases_initializer=None, activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=_LeakeyRELU_alpha)):
				layer36_out, layer61_out, inputs = darknet53_layer(inputs)				
				#first detection layer
				filters = 512
				inputs = conv2d_layer(inputs, filters, 1)
				inputs = conv2d_layer(inputs, filters * 2, 3)
				inputs = conv2d_layer(inputs, filters, 1)
				inputs = conv2d_layer(inputs, filters * 2, 3)
				inputs = conv2d_layer(inputs, filters, 1)
				layer79_out = inputs
				inputs = conv2d_layer(inputs, filters * 2, 3)
				detection1 = yolo_layer(inputs, numclasses, img_size, _Coco_anchor[6:9], data_format)
				
				inputs = conv2d_layer(layer79_out, 256, 1)
				upsample_size = layer61_out.get_shape().as_list()
				inputs = upsample_layer(inputs, upsample_size, data_format)
				inputs = tf.concat([inputs, layer61_out], axis=1 if data_format == 'NCHW' else 3)
				#second detection layer
				filters = 256
				inputs = conv2d_layer(inputs, filters, 1)
				inputs = conv2d_layer(inputs, filters * 2, 3)
				inputs = conv2d_layer(inputs, filters, 1)
				inputs = conv2d_layer(inputs, filters * 2, 3)
				inputs = conv2d_layer(inputs, filters, 1)
				layer91_out = inputs
				inputs = conv2d_layer(inputs, filters * 2, 3)
				detection2 = yolo_layer(inputs, numclasses, img_size, _Coco_anchor[3:6], data_format)
				
				inputs = conv2d_layer(layer91_out, 128, 1)
				upsample_size = layer36_out.get_shape().as_list()
				inputs = upsample_layer(inputs, upsample_size, data_format)
				inputs = tf.concat([inputs, layer36_out], axis=1 if data_format == 'NCHW' else 3)
				
				#third detection layer
				filters = 128
				inputs = conv2d_layer(inputs, filters, 1)
				inputs = conv2d_layer(inputs, filters * 2, 3)
				inputs = conv2d_layer(inputs, filters, 1)
				inputs = conv2d_layer(inputs, filters * 2, 3)
				inputs = conv2d_layer(inputs, filters, 1)
				inputs = conv2d_layer(inputs, filters * 2, 3)
				detection3 = yolo_layer(inputs, numclasses, img_size, _Coco_anchor[0:3], data_format)
				return tf.concat([detection1, detection2, detection3], axis = 1)


def load_weights(var_list, weights_file):
	with open(weights_file, "rb") as fp:
		_ = np.fromfile(fp, dtype=np.int32, count=5)
		weights = np.fromfile(fp, dtype=np.float32)
	ptr = 0
	i = 0
	assign_ops = []
	while i < len(var_list) - 1:
		var1 = var_list[i]
		var2 = var_list[i + 1]
		if 'Conv' in var1.name.split('/')[-2]:
			if 'BatchNorm' in var2.name.split('/')[-2]:
				gamma, beta, mean, var = var_list[i + 1:i + 5]
				batch_norm_vars = [beta, gamma, mean, var]
				for var in batch_norm_vars:
					shape = var.shape.as_list()
					num_params = np.prod(shape)
					var_weights = weights[ptr:ptr + num_params].reshape(shape)
					ptr += num_params
					assign_ops.append(tf.assign(var, var_weights, validate_shape=True))
				i += 4
			elif 'Conv' in var2.name.split('/')[-2]:
				bias = var2
				bias_shape = bias.shape.as_list()
				bias_params = np.prod(bias_shape)
				bias_weights = weights[ptr:ptr + bias_params].reshape(bias_shape)
				ptr += bias_params
				assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))
				i += 1
			shape = var1.shape.as_list()
			num_params = np.prod(shape)
			var_weights = weights[ptr:ptr + num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
			# remember to transpose to column-major
			var_weights = np.transpose(var_weights, (2, 3, 1, 0))
			ptr += num_params
			assign_ops.append(tf.assign(var1, var_weights, validate_shape=True))
			i += 1
	return assign_ops

def _iou(box1, box2):
    b1_x0, b1_y0, b1_x1, b1_y1 = box1
    b2_x0, b2_y0, b2_x1, b2_y1 = box2
    int_x0 = max(b1_x0, b2_x0)
    int_y0 = max(b1_y0, b2_y0)
    int_x1 = min(b1_x1, b2_x1)
    int_y1 = min(b1_y1, b2_y1)
    int_area = (int_x1 - int_x0) * (int_y1 - int_y0)
    b1_area = (b1_x1 - b1_x0) * (b1_y1 - b1_y0)
    b2_area = (b2_x1 - b2_x0) * (b2_y1 - b2_y0)
    iou = int_area / (b1_area + b2_area - int_area + 1e-05)
    return iou

def non_max_suppression(predictions_with_boxes, confidence_threshold, iou_threshold=0.4):
	conf_mask = np.expand_dims((predictions_with_boxes[:, :, 4] > confidence_threshold), -1)
	predictions = predictions_with_boxes * conf_mask

	result = {}
	for i, image_pred in enumerate(predictions):
		shape = image_pred.shape
		non_zero_idxs = np.nonzero(image_pred)
		image_pred = image_pred[non_zero_idxs]
		image_pred = image_pred.reshape(-1, shape[-1])
		
		bbox_attrs = image_pred[:, :5]
		classes = image_pred[:, 5:]
		classes = np.argmax(classes, axis=-1)

		unique_classes = list(set(classes.reshape(-1)))

		for cls in unique_classes:
			cls_mask = classes == cls
			cls_boxes = bbox_attrs[np.nonzero(cls_mask)]
			cls_boxes = cls_boxes[cls_boxes[:, -1].argsort()[::-1]]
			cls_scores = cls_boxes[:, -1]
			cls_boxes = cls_boxes[:, :-1]

			while len(cls_boxes) > 0:
				box = cls_boxes[0]
				score = cls_scores[0]
				if not cls in result:
					result[cls] = []
				result[cls].append((box, score))
				cls_boxes = cls_boxes[1:]
				ious = np.array([_iou(box, x) for x in cls_boxes])
				iou_mask = ious < iou_threshold
				cls_boxes = cls_boxes[np.nonzero(iou_mask)]
				cls_scores = cls_scores[np.nonzero(iou_mask)]
	return result
