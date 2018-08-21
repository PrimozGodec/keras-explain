import numpy as np
from keras_explain.deep_viz_keras.guided_backprop import GuidedBackprop


class SpingenbergGuidedBP:

    name = "SpingenbergGuidedBP"

    def __init__(self, model):
        self.model = model

    def explain(self, image, target_class):
        guided_bprop = GuidedBackprop(self.model,
                                      output_index=target_class)
        mask = guided_bprop.get_mask(image)

        # ignoring negative activations since they come only from the
        # last layer, since guided backprop does not propagate negative values
        # through the ReLU

        mask[mask < 0] = 0
        mask = np.sum(mask, axis=2)
        # mask = np.abs(mask)
        mask /= mask.max()  # values should be between 0 and 1
        print(mask.max())

        return mask, None
