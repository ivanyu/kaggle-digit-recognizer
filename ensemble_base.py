#!/usr/bin/env python
# -*- coding: utf-8 -*-
from keras.models import model_from_json


def load_nn_model(fname_model, fname_model_weights):
    with open(fname_model, 'r') as f:
        model_json = f.read()

    model = model_from_json(model_json)

    model.load_weights(fname_model_weights)
    return model

