import numpy as np


class GrayingOut:

    name = "Graying out"
    authors = "Zeiler et al."

    def __init__(self, model, kernel_size=10, jump=5):
        self.model = model
        self.k = kernel_size
        self.jump = jump

    def produce_images(self, image, k=10, jump=5):
        ims = \
            np.zeros((((image.shape[0] - k) // jump + 1) *
                      ((image.shape[1] - k) // jump + 1),
                      image.shape[0], image.shape[1], image.shape[2]))
        mask = \
            np.zeros(((((image.shape[0] - k) // jump + 1) *
                      ((image.shape[1] - k) // jump + 1)),
                      image.shape[0], image.shape[1]))
        idx = 0
        for i in range(0, image.shape[0] - k + 1, jump):
            for j in range(0, image.shape[1] - k + 1, jump):
                temp = image.copy()
                temp[i:i+k, j:j+k, :] = 0
                mask[idx, i:i+k, j:j+k] = 1
                ims[idx] = temp
                idx += 1
        return ims, mask


    def explain(self, image, target_class):

        # generate images with removed parts
        input_im, mask = self.produce_images(image, k=self.k, jump=self.jump)

        # make predictions
        full_im_pred = self.model.predict(image[None, ...])
        errased_im_pred = self.model.predict(input_im)

        # inspect the differeces
        diff = errased_im_pred - full_im_pred
        diff_observed = - diff[:, target_class]

        results = mask * diff_observed[:, np.newaxis, np.newaxis]

        results = results.sum(axis=0) / np.count_nonzero(results, axis=0)

        results[np.isnan(results)] = 0
        abs_max = np.max(np.abs(results))
        results /= abs_max

        positive_impacts = results[results > 0]
        positive_impacts_mask = np.zeros(results.shape)
        positive_impacts_mask[results > 0] = positive_impacts

        negative_impacts = - results[results < 0]
        negative_impacts_mask = np.zeros(results.shape)
        negative_impacts_mask[results < 0] = negative_impacts

        return positive_impacts_mask, negative_impacts_mask
