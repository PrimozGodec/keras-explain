import numpy as np
import keras.backend as K
from keras.engine import InputLayer
from keras.layers import MaxPooling2D, Conv2D
from keras.layers.core import Dense, Dropout, Flatten


class LayerNotImplementedException(Exception):
    pass


class BachLRP:

    name = "Bach_LRP"

    def __init__(self, model, alpha=2, beta=1):
        self.model = model
        self.alpha = alpha
        self.beta = beta

    def explain(self, image, target_class):
        model = self.model

        # retrieve outputs of all layers
        outputs = self.get_layers_outputs(model, image)

        relevances = {}  # key: layer name, value: relevance of input tensor

        # retrieve the output layer
        output_layer = model.layers[-1]  # TODO: Check if true

        target_activations = outputs[output_layer.name]
        print(target_activations.shape)
        target_activations_modified = np.zeros(target_activations.shape)
        target_activations_modified[:, target_class] = \
            target_activations[:, target_class]

        previous = [output_layer]
        new_r = None

        while len(previous) > 0:
            curr_layer = previous[0]
            print("curr", curr_layer)
            print("curr", curr_layer.name)

            # get layer before

            previous_layer = self._get_layer_before(curr_layer)

            previous = previous[1:] + \
                       [previous_layer] if previous_layer is not None else []

            # layer after
            out_node = curr_layer._outbound_nodes

            next_layer = None
            if len(out_node) > 0:
                next_layer = out_node[0].outbound_layer
            if next_layer is not None:
                print("next", next_layer.name)

            previous = previous[1:] + \
                       [previous_layer] if previous_layer is not None else []

            # print(previous_layer)


            # get previous layer relevance
            r = target_activations_modified if next_layer is None \
                else relevances[next_layer.name]
            if not isinstance(curr_layer, InputLayer):
                x = outputs[previous_layer.name]
                w = curr_layer.get_weights()

            if isinstance(curr_layer, Dense):
                new_r = self.lrp_dense(r, x, w)
            elif isinstance(curr_layer, Dropout):
                new_r = self.lrp_layer_skip(r)
            elif isinstance(curr_layer, Flatten):
                new_r = self.lrp_flatten(r, x)
            elif isinstance(curr_layer, MaxPooling2D):
                new_r = self.lrp_max_pooling(
                    r, x, outputs[curr_layer.name],
                    curr_layer.pool_size, curr_layer.strides)
            elif isinstance(curr_layer, InputLayer):
                new_r = r.copy()
                print("here")
            elif isinstance(curr_layer, Conv2D):
                new_r = self.lrp_conv2D(r, x, w, curr_layer.strides)
            else:
                print("Layer %s is not supported by the LRP implemented in"
                      "this framework. You can implement it and create the "
                      "pull request on the GitHub or write the issue." %
                      curr_layer)
                raise LayerNotImplementedException

            relevances[curr_layer.name] = new_r
            print(new_r.shape)

        print(new_r.shape)
        return np.sum(new_r[0, ...], axis=-1), None

    def _get_layer_before(self, layer):
        previous_layer = None
        in_node = layer._inbound_nodes
        if len(in_node) > 0 and len(in_node[0].inbound_layers) > 0:
            previous_layer = in_node[0].inbound_layers[0]
        return previous_layer


    def get_layers_outputs(self, model, input):
        # retrieve layers
        layers = model.layers

        # get activations
        activations = {}  # each layer will be accessible by name
        for layer in layers:
            out_fun = K.function([model.layers[0].input], [layer.output])
            layer_output = out_fun([input[None, ...]])[0]
            activations[layer.name] = layer_output

            # check if layer has input layer attached
            if hasattr(layer, "batch_input_shape"):
                print(layer.batch_input_shape)
                input_layer = self._get_layer_before(layer)
                out_fun = K.function([model.layers[0].input], [input_layer.output])
                layer_output = out_fun([input[None, ...]])[0]
                activations[input_layer.name] = layer_output

        return activations

    def lrp_dense(self, r, x, w):
        b = w[1]  # bias
        w = w[0]  # weights

        z = w[np.newaxis, :] * x[:, :, np.newaxis]

        zp = z * (z > 0)
        zn = z * (z < 0)

        zsp = np.sum(zp, axis=1) + (b * (b > 0))[np.newaxis, :]
        zsn = np.sum(zn, axis=1) + (b * (b > 0))[np.newaxis, :]

        zp = zp / zsp
        zn = zn / zsn

        r_new = self.alpha * np.sum(zp * r[:, np.newaxis, :], axis=2) + \
            self.beta * np.sum(zn * r[:, np.newaxis, :], axis=2)

        return r_new

    def lrp_layer_skip(self, r):
        return r

    def lrp_flatten(self, r, x):
        return np.reshape(r, x.shape)

    def lrp_max_pooling(self, r, x, y, pool_size, strides):
        n, h, w, d = x.shape

        h_pool, w_pool = pool_size
        h_stride, w_stride = strides

        hout, wout = r.shape[1:3]

        r_x = np.zeros_like(x, dtype=np.float)

        for i in range(hout):
            for j in range(wout):
                z = y[:,i:i+1,j:j+1,:] == x[:, i * h_stride: i * h_stride + h_pool, j * w_stride: j * w_stride + w_pool, :]
                zs = z.sum(axis=(1,2),keepdims=True,dtype=np.float)
                r_x[:, i * h_stride: i * h_stride + h_pool, j * w_stride: j * w_stride + w_pool, :] \
                    += (z / zs) * r[:, i:i+1,j:j+1, :]
        return r_x

    def lrp_conv2D(self, r, x, w, stride):
        b = w[1]  # bias
        w = w[0]  # weights

        na = np.newaxis

        N, Hout, Wout, NF = r.shape
        hf, wf, df, NF = w.shape
        hstride, wstride = stride

        Rx = np.zeros_like(x, dtype=np.float)

        for i in range(Hout):
            for j in range(Wout):
                Z = w[na, ...] * x[:, i * hstride:i * hstride + hf,
                                      j * wstride:j * wstride + wf, :, na]
                # alpha part
                Zp = Z * (Z > 0)
                Bp = (b * (b > 0))[na, na, na, na, ...]
                Zsp = Zp.sum(axis=(1, 2, 3), keepdims=True) + Bp
                Ralpha = self.alpha * (
                    (Zp / Zsp) * r[:, i:i + 1, j:j + 1, na, :]).sum(axis=4)
                Ralpha[np.isnan(Ralpha)] = 0

                Zn = Z * (Z < 0)
                Bn = (b * (b < 0))[na, na, na, na, ...]
                Zsn = Zn.sum(axis=(1, 2, 3), keepdims=True) + Bn
                Rbeta = self.beta * (
                    (Zn / Zsn) * r[:, i:i + 1, j:j + 1, na, :]).sum(axis=4)
                Rbeta[np.isnan(Rbeta)] = 0

                Rx[:, i * hstride:i * hstride + hf:,
                j * wstride:j * wstride + wf:, :] += Ralpha + Rbeta

        return Rx



