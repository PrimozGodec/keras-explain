import unittest
import pytest
import os

import numpy as np

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image

from keras_explain.lrp import LRP
from keras_explain.lime_ribeiro import Lime
from keras_explain.grad_cam import GradCam, GuidedGradCam
from keras_explain.saliency import Saliency
from keras_explain.guided_bp import GuidedBP
from keras_explain.integrated_gradients import IntegratedGrad
from keras_explain.graying_out import GrayingOut
from keras_explain.prediction_diff import PredictionDiff

class TestBasicFunction(unittest.TestCase):

    def setUp(self):
        path = "data/input_data/impro_imagenet"
        images_names = os.listdir(path)

        self.model = InceptionV3(include_top=True, weights='imagenet',
                                 input_shape=(299, 299, 3))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')

        images = [image.load_img(
            os.path.join(path, name), target_size=(299, 299))
            for name in images_names]
        images = [image.img_to_array(img) for img in images]
        images = np.stack(images)
        self.images = preprocess_input(images)

    def _test_approach(self, approach, kwargs, images):
        print("Testing", approach)
        explainer = approach(**kwargs)
        for image in images:
            exp_pos, exp_neg = explainer.explain(image, 15)
            self.assertTupleEqual(exp_pos.shape, (299, 299))
            self.assertTrue(exp_neg is None or exp_neg.shape == (299, 299))

    def test_graying_out(self):
       self._test_approach(GrayingOut, {"model": self.model}, self.images)

    def test_prediction_diff(self):
        self._test_approach(
            PredictionDiff, {"model": self.model, "all_images": self.images},
            self.images)

    def test_integrated_grad(self):
        self._test_approach(
            IntegratedGrad, {"model": self.model}, self.images)

    def test_guided_bp(self):
        self._test_approach(
            GuidedBP, {"model": self.model}, self.images)

    def test_saliency(self):
        self._test_approach(
            Saliency, {"model": self.model, "layer": 312}, self.images)

    def test_lime(self):
        self._test_approach(
            Lime, {"model": self.model}, self.images)

    def test_grad_cam(self):
        self._test_approach(
            GradCam, {"model": self.model, "layer": 312}, self.images)

    def test_guided_grad_cam(self):
        self._test_approach(
            GuidedGradCam, {"model": self.model, "layer": 312}, self.images)

    @pytest.mark.skip(reason="No way to test it with Inception, not all"
                             "layers supported.")
    def test_lrp(self):
        self._test_approach(
            LRP, {"model": self.model}, self.images)


if __name__ == '__main__':
    unittest.main()
