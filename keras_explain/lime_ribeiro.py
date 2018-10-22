from lime import lime_image


class Lime:

    name = "LIME"
    authors = "Ribeiro et al."

    def __init__(self, model):
        self.model = model
        pass

    def explain(self, image, target_class):
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            image, self.model.predict, top_labels=None,
            labels=(target_class,), num_samples=1000)
        temp, mask = explanation.get_image_and_mask(
            target_class, positive_only=False, num_features=10, hide_rest=True)

        return (mask == 2).astype(int), (mask == 1).astype(int)
