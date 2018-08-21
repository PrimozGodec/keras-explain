from vis.visualization import visualize_saliency


class Simonyan:

    name = "Simonyan"

    def __init__(self, model, layer):
        self.model = model
        self.layer = layer

    def explain(self, image, target_class):
        # generate images with removed parts
        res = visualize_saliency(
            model=self.model, layer_idx=self.layer,
            filter_indices=target_class, seed_input=image)

        return res, None
