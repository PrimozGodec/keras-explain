import numpy as np
from keras_explain.deep_viz_keras.guided_backprop import GuidedBackprop


class Enhanced_GuidedBP:

    name = "Enhanced Guided Back Propagation"
    paper = "Jindong Gu, Volker Tresp; Saliency Methods for Explaining Adversarial Attacks"

    def __init__(self, model):
        self.model = model


    def channel_energy_map(self, mask, mask_contra):
        
        mask = mask/mask.sum(axis=(0, 1)).reshape((1,1,3))
        mask_contra = mask_contra/mask_contra.sum(axis=(0, 1)).reshape((1,1,3))
        
        mask = mask - mask_contra
        return mask
        
    def explain(self, image, target_class):
        guided_bprop1 = GuidedBackprop(self.model, output_index=target_class)
        mask = guided_bprop1.get_mask(image)
        mask = np.maximum(0, mask)

        output = self.model.predict(image)
        index = np.argsort(output)[0]

        if index[-1] == target_class: contra_class = index[-2]
        else: contra_class = index[-1]

        guided_bprop2 = GuidedBackprop(self.model, output_index=contra_class)
        mask_contra = guided_bprop2.get_mask(image)
        mask_contra = np.maximum(0, mask_contra)

        mask = self.channel_energy_map(mask, mask_contra)
        
        mask = np.sum(mask, axis=2)
        mask /= mask.max()
        
        return mask, None



    
