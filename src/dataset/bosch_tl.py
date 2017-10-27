import cv2
import os
import numpy as np
import subprocess
import yaml
from dataset.imdb import imdb
from utils.util import bbox_transform_inv, batch_iou

class bosch_tl(imdb):
    def __init__(self, image_set, data_path, mc):

        imdb.__init__(self, 'bosch'+image_set, mc)

        self._image_set = image_set     # train, test or trainval

        self._data_root_path = data_path

        #self._image_path = os.path.join(self._data_root_path, 'training', 'image_2')
        #self._label_path = os.path.join(self._data_root_path, 'training', 'label_2')

        self._classes = self.mc.CLASS_NAMES
        self._class_to_idx = dict(zip(self.classes, range(self.num_classes)))

        self._input_yaml = image_set+".yaml"
        self._bosch_data = yaml.load(open(os.path.join(data_path,self._input_yaml), 'rb').read())

        #image_dict= bosch_data[0]  # bosch_data content image dictionary 'boxes' and 'path'
        # a list of string indices of images in the directory
        self._image_idx = []
        self._image_path = []

        # a dict of image_idx -> [[cx, cy, w, h, cls_idx]]. x,y,w,h are not divided by
        # the image width and height
        # '006303': [[75.735, 243.735, 151.47, 127.01000000000002, 0], [111.485, 238.64499999999998, 222.97, 114.95000000000002, 0], [237.58, 216.33999999999997, 116.94, 64.84, 0], [303.685, 210.755, 151.11, 58.44999999999999, 0], [434.94, 197.68, 52.72000000000003, 37.78, 0], ]
        self._rois = {}

        idx_counter=0
        for imgdict in self._bosch_data:
            image_path = os.path.join(data_path, imgdict['path'])

            roi ={}
            bboxes=[]
            for box in imgdict['boxes']:
                w = box["x_max"] - box["x_min"]
                h = box["y_max"] - box["y_min"]
                if w<0:
                    print("Warning negative value of width %f, %s" % (w, imgdict['path']))
                    w=abs(w)
                if h < 0:
                    print("Warning negative value of height %f, %s"% (h, imgdict['path']))
                    h=abs(h)
                cx = box["x_min"] + w/2
                cy = box["y_min"] + h/2
                class_idx = 3
                if "Red" in box['label']:
                    class_idx = 0
                elif 'Yellow' in box['label']:
                    class_idx = 1
                elif 'Green' in box['label']:
                    class_idx = 2
                bboxes.append([cx, cy, w, h, class_idx])

            self._image_idx.append(idx_counter)

            self._rois[idx_counter] =bboxes
            self._image_path.append(image_path)

            idx_counter = idx_counter + 1

        ## batch reader ##
        self._perm_idx = None
        self._cur_idx = 0
        self._shuffle_image_idx()

    def _image_path_at(self, idx):
        image_path = self._image_path[int(idx)]
        assert os.path.exists(image_path), \
            'Image does not exist: {}'.format(image_path)
        return image_path

def gen_batch_function_Bosch(data_path):
    """
    Generate function to create batches of training data
    :param data_path: Path to folder that train.yaml

    ################
    # Bosch Dataset #
    ################

    """
    def get_batches_fn():
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """

        # Shuffle training data for each epoch
        np.random.shuffle(self._bosch_data)

        for batch_i in range(0, len(self._bosch_data), batch_size):

            images = []
            labels = []

            for image_dict in self._bosch_data[batch_i:batch_i+batch_size]:

                image_path = os.path.abspath(os.path.join(os.path.dirname(self._input_yaml), image_dict['path']))
                image = get_image(image_path)

                if image is None:
                    raise IOError('Could not open image path', image_dict['path'])

                roi ={}
                bbox=[]
                for box in image_dict['boxes']:
                    w = box["xmax"] - box["xmin"]
                    h = box["ymax"] - box["ymin"]
                    cx = w/2
                    cy = h/2
                    class_idx = 3
                    if "Red" in box['label']:
                        class_idx = 0
                    elif 'Yellow' in box['label']:
                        class_idx = 1
                    elif 'Green' in box['label']:
                        class_idx = 2
                    bbox.append([cx, cy, w,h, class_idx])

                roi[str(idx_counter)] = bbox
                image_idx.append(idx_counter)
                idx_counter = idx_counter + 1
                rois.append(roi)
                image_paths.append(image_path)
                images =[image]

                images.append(image)
                labels.append(label)

#                print('>>>>>', label)
#                cv2.imshow('labeled_image', image)
#                cv2.waitKey(3000)

            # Augment images
            images, labels = augment_images(images, labels)
            # Yield
            yield np.array(images), np.array(labels)


    # Get X_test and y_test
    #print('Generating test set... {}% train, {}% testing'.format(RATIO*100, (1-RATIO)*100))
    X_test = []
    y_test = []

    return get_batches_fn, np.array(X_test), np.array(y_test)