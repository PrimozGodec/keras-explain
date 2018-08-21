import unittest
import os

import numpy as np

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image

from keras_explain.bach_lrp import BachLRP
from keras_explain.ribeiro_lime import RibeiroLime
from keras_explain.selvaraju import SelvarjuGradCam, SelvarjuGuidedGradCam
from keras_explain.simonyan import Simonyan
from keras_explain.spingenberg import SpingenbergGuidedBP
from keras_explain.sundararajan import SundararajanIntegrated
from keras_explain.zeiler import Zeiler
from keras_explain.zintgraf import Zintgraf

class TestBasicFunction(unittest.TestCase):

    def setUp(self):
        path = "data/input_data/impro_imagenet"
        images_names = os.listdir(path)

        self.model = InceptionV3(include_top=True, weights='imagenet',
                                 input_shape=(299, 299, 3))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')
        images = [image.load_img(os.path.join(path, name), target_size=(299, 299)) for name in images_names]
        images = [image.img_to_array(img) for img in images]
        images = np.stack(images)
        self.images = preprocess_input(images)

    def _test_approach(self, approach, kwargs, images):
        explainer = approach(**kwargs)
        for image in images:
            exp_pos, exp_neg = explainer.explain(image, 15)
            self.assertTupleEqual(exp_pos.shape, (299, 299))
            self.assertTrue(exp_neg is None or exp_neg.shape == (299, 299))

    def test_zeiler(self):
       self._test_approach(Zeiler, {"model": self.model}, self.images)

    def test_zintgraf(self):
        self._test_approach(
            Zintgraf, {"model": self.model, "all_images": self.images},
            self.images)

    def test_sundarajan(self):
        self._test_approach(
            SundararajanIntegrated, {"model": self.model}, self.images)

    def test_springenberg(self):
        self._test_approach(
            SpingenbergGuidedBP, {"model": self.model}, self.images)

    def test_simonyan(self):
        self._test_approach(
            Simonyan, {"model": self.model, "layer": 312}, self.images)

    def test_lime(self):
        self._test_approach(
            RibeiroLime, {"model": self.model}, self.images)

    def test_grad_cam(self):
        self._test_approach(
            SelvarjuGradCam, {"model": self.model, "layer": 312}, self.images)

    def test_guided_grad_cam(self):
        self._test_approach(
            SelvarjuGuidedGradCam, {"model": self.model, "layer": 312}, self.images)


if __name__ == '__main__':
    unittest.main()
