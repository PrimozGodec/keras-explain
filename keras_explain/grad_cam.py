import numpy as np
from vis.visualization import visualize_cam

from keras_explain.deep_viz_keras.guided_backprop import GuidedBackprop


class GradCam:

    name = "GradCam"

    def __init__(self, model, layer=None):
        self.model = model

        # if layer not set it is automatically set to last layer
        if layer is None:
            self.layer = len(model.layers) - 1
        else:
            self.layer = layer

    def explain(self, image, target_class):

        # generate images with removed parts
        res = visualize_cam(
            model=self.model,
            layer_idx=self.layer,
            filter_indices=target_class, seed_input=image)

        return res, None

class GuidedGradCam:

    name = "GuidedGradCam"

    def __init__(self, model, layer=None):
        self.model = model

        # if layer not set it is automatically set to last layer
        if layer is None:
            self.layer = len(model.layers) - 1
        else:
            self.layer = layer

    def explain(self, image, target_class):
        # generate images with removed parts
        grad_cam_res = visualize_cam(
            model=self.model,
            layer_idx=self.layer,
            filter_indices=target_class, seed_input=image)

        guided_bprop = GuidedBackprop(self.model, output_index=target_class)

        guided_backprop_res = guided_bprop.get_mask(image)

        # ignoring negative activations since they come only from the
        # last layer, since guided backprop does not propagate negative values
        # through the ReLU

        guided_backprop_res[guided_backprop_res < 0] = 0
        guided_backprop_res = np.sum(guided_backprop_res, axis=2)
        # mask = np.abs(mask)
        # values should be between 0 and 1
        guided_backprop_res /= guided_backprop_res.max()

        # pintvise multiplication
        res = guided_backprop_res * grad_cam_res

        # values should be between 0 and 1
        res /= res.max()

        return res, None
