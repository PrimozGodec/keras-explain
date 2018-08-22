from vis.visualization import visualize_saliency


class Saliency:

    name = "Saliency"
    authors = "Simonyan et al."

    def __init__(self, model, layer=None):
        self.model = model

        # if layer not set it is automatically set to last layer
        if layer is None:
            self.layer = len(model.layers) - 1
        else:
            self.layer = layer

    def explain(self, image, target_class):
        # generate images with removed parts
        res = visualize_saliency(
            model=self.model, layer_idx=self.layer,
            filter_indices=target_class, seed_input=image)

        return res, None
