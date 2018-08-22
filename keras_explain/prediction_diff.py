# -*- coding: utf-8 -*-
import time

import numpy as np
import keras_explain.zintgraf_utils.utils_sampling as utlS

from keras_explain.zintgraf_utils.prediction_difference_analysis import \
    PredDiffAnalyser


class PredictionDiff:

    name = "Prediction Difference Analysis"
    authors = "Zintgraf et al."

    # window size (i.e., the size of the pixel patch that is marginalised out
    # in each step); k in alg 1 (see paper)
    win_size = 10

    # indicate whether windows should be overlapping or not
    overlapping = True

    # sampling
    num_samples = 10
    padding_size = 2  # for conditional: l = win_size+2*padding_size in alg 1

    # set the batch size - the larger, the faster computation will be
    batch_size = 128

    def __init__(self, model, all_images):
        self.model = model
        self.all_images = all_images

    def explain(self, image, target_class):

        # change images to channel first
        image = np.moveaxis(image, 2, 0)
        all_images = np.moveaxis(self.all_images, 3, 1)

        imsize = image.shape

        # target function (mapping input features to output probabilities)
        def target_func(images_batch):
            if np.ndim(images_batch) == 3:
                images_batch = images_batch[np.newaxis]
            if np.ndim(images_batch) < 4:
                images_batch = images_batch.reshape(
                    [images_batch.shape[0]] + list(imsize))

            images_batch = np.moveaxis(images_batch, 1, 3)  # to channel last
            prediction = self.model.predict(images_batch)

            return [prediction]

        # get the specific image
        x_test = image

        start_time = time.time()

        # we are using only conditional sampling
        sampler = utlS.cond_sampler_imagenet(
            win_size=self.win_size, padding_size=self.padding_size,
            image_dims=image.shape[1:], X=all_images)

        pda = PredDiffAnalyser(
            x_test, target_func, sampler, num_samples=self.num_samples,
            batch_size=self.batch_size)
        pred_diff = pda.get_rel_vect(
            win_size=self.win_size, overlap=self.overlapping)

        print("--- Total computation took {:.4f} minutes ---".format(
            (time.time() - start_time)/60))

        results = pred_diff[0].reshape(
            (imsize[1], imsize[2], -1))[:, :, target_class]
        abs_max = np.max(np.abs(results))
        results /= abs_max

        positive_impacts = results[results > 0]
        positive_impacts_mask = np.zeros(results.shape)
        positive_impacts_mask[results > 0] = positive_impacts

        negative_impacts = - results[results < 0]
        negative_impacts_mask = np.zeros(results.shape)
        negative_impacts_mask[results < 0] = negative_impacts

        return positive_impacts_mask, negative_impacts_mask
