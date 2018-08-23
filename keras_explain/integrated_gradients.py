import numpy as np

from keras_explain.deep_viz_keras.integrated_gradients import IntegratedGradients


class IntegratedGrad:

    name = "Integrated gradients"
    authors = "Sundararajan et al."

    def __init__(self, model):
        self.model = model

    def explain(self, image, target_class):
        print(self.model)
        try:
            ig = IntegratedGradients(self.model)
            mask = ig.get_mask(image)
        except AttributeError:
            print("Your model has no optimizer defined. Please use the complie"
                  "function on your model to define the optimizer.")
            raise

        # TODO: check if positive really means positive impact and
        # TODO: negative value negative impact
        # that idea found at
        # https://github.com/hiranumn/IntegratedGradients/blob/master/examples/example.ipynb
        # i cant get so nice representations

        mask = np.sum(mask, axis=2)
        mask /= np.abs(mask).max()
        # print(mask.min(), mask.max())
        #
        # positive_impacts = np.copy(mask)
        # positive_impacts[positive_impacts < 0] = 0
        #
        # negative_impacts = np.copy(mask) * -1
        # negative_impacts[negative_impacts < 0] = 0
        mask = np.abs(mask)

        return mask, None  # positive_impacts, negative_impacts
