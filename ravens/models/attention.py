# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
"""Attention module."""

from ravens.models.resnet import ResNet43_8s
import numpy as np
import torch
from torch import nn, optim
from torchvision import transforms
import PIL

class Attention(nn.Module):
    """Attention module."""

    def __init__(self, in_shape, n_rotations, preprocess, device, lite=False):
        super().__init__()
        self.n_rotations = n_rotations
        self.preprocess = preprocess
        self.device = device

        max_dim = np.max(in_shape[:2])

        self.padding = np.zeros((3, 2), dtype=int)
        pad = (max_dim - np.array(in_shape[:2])) / 2
        self.padding[:2] = pad.reshape(2, 1)

        # Initialize fully convolutional Residual Network with 43 layers and
        # 8-stride (3 2x2 max pools and 3 2x bilinear upsampling)
        self.model = ResNet43_8s(in_shape[2], 1).to(device)
        self.optim = optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        # self.metric = tf.keras.metrics.Mean(name='loss_attention')

    def forward(self, in_img, softmax=True, train=False):
        """Forward pass."""
        self.model.train(train)
        in_data = np.pad(in_img, self.padding, mode='constant')
        in_data = self.preprocess(in_data)
        in_shape = (1,) + in_data.shape
        in_data = in_data.reshape(in_shape)
        in_tens = torch.tensor(in_data, dtype=torch.float32).to(self.device)
        in_tens = in_tens.permute(0,3,1,2)

        # Rotate angles.
        angles = [i*360/self.n_rotations if i*360/self.n_rotations <= 180 
            else i*360/self.n_rotations-360 for i in range(self.n_rotations)]

        # Forward pass.
        logits = ()
        for angle in angles:
            rotated_tensor = transforms.functional.affine(in_tens, angle=angle, translate=[0,0], 
                                                          scale=1, shear=[0,0], resample=PIL.Image.NEAREST)
            rotated_out = self.model(rotated_tensor)
            out = transforms.functional.affine(rotated_out, angle= -angle, translate=[0,0], 
                                               scale=1, shear=[0,0], resample=PIL.Image.NEAREST)
            logits += (out,)
        logits = torch.cat(logits, dim=0)
        
        logits = logits.permute(1, 2, 3, 0)
        c0 = self.padding[:2, 0]
        c1 = c0 + in_img.shape[:2]
        logits = logits[:, c0[0]:c1[0], c0[1]:c1[1], :]
        output = torch.reshape(logits, (1, np.prod(logits.shape)))
        if softmax:
            output = nn.functional.softmax(output, dim=1)
            output = output.detach().numpy().reshape(logits.shape[1:])
        return output

    def train(self, in_img, p, theta, backprop=True):
        """Train."""
        # self.metric.reset_states()
        output = self.forward(in_img, softmax=False, train=True)

        # Get label.
        theta_i = theta / (2 * np.pi / self.n_rotations)
        theta_i = np.int32(np.round(theta_i)) % self.n_rotations
        label_size = in_img.shape[:2] + (self.n_rotations,)
        label = np.zeros(label_size)
        label[p[0], p[1], theta_i] = 1
        label = label.reshape(np.prod(label.shape),)
        label = np.where(label==1)[0][0]
        label = torch.tensor(label, dtype=torch.int64).reshape(1,).to(self.device)
        # Get loss.
        loss = self.criterion(output, label)

        # Backpropagate
        if backprop:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        return loss.item()

    def load(self, path, device):
        checkpoint = torch.load(path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def save(self, filename):
        state = {'model_state_dict': self.model.state_dict(),
                }
        torch.save(state, filename)
        print("checkpoint saved at epoch", filename)