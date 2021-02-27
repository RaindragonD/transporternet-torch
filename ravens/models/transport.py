# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transport module."""

from ravens.models.resnet import ResNet43_8s
import numpy as np
import torch
from torch import nn, optim
from torchvision import transforms
import PIL

class Transport(nn.Module):
    """Transport module."""

    def __init__(self, in_shape, n_rotations, crop_size, preprocess):
        """Transport module for placing.

        Args:
            in_shape: shape of input image.
            n_rotations: number of rotations of convolving kernel.
            crop_size: crop size around pick argmax used as convolving kernel.
            preprocess: function to preprocess input images.
        """
        super().__init__()
        self.iters = 0
        self.n_rotations = n_rotations
        self.crop_size = crop_size    # crop size must be N*16 (e.g. 96)
        self.preprocess = preprocess

        self.pad_size = int(self.crop_size / 2)
        self.padding = np.zeros((3, 2), dtype=int)
        self.padding[:2, :] = self.pad_size

        in_shape = np.array(in_shape)
        in_shape[0:2] += self.pad_size * 2
        in_shape = tuple(in_shape)

        # Crop before network (default for Transporters in CoRL submission).
        # kernel_shape = (self.crop_size, self.crop_size, in_shape[2])

        if not hasattr(self, 'output_dim'):
            self.output_dim = 3
        if not hasattr(self, 'kernel_dim'):
            self.kernel_dim = 3

        # 2 fully convolutional ResNets with 57 layers and 16-stride
        self.querynet = ResNet43_8s(in_shape[2], self.kernel_dim)
        self.keynet = ResNet43_8s(in_shape[2], self.output_dim)
        self.optim = optim.Adam(list(self.querynet.parameters())+list(self.keynet.parameters()), lr=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        # self.metric = tf.keras.metrics.Mean(name='loss_transport')

    def correlate(self, in0, in1, softmax):
        """Correlate two input tensors."""
        output = nn.functional.conv2d(in0, in1)
        output = output.permute(0,2,3,1)
        if softmax:
            output_shape = output.shape
            output = torch.reshape(output, (1, np.prod(output.shape)))
            output = nn.Softmax(dim=1)(output)
            output = output.detach().numpy().reshape(output_shape[1:])
        return output

    def forward(self, in_img, p, softmax=True, train=False):
        """Forward pass."""
        self.keynet.train(train)
        self.querynet.train(train)

        img_unprocessed = np.pad(in_img, self.padding, mode='constant')
        input_data = self.preprocess(img_unprocessed.copy())
        in_shape = (1,) + input_data.shape
        input_data = input_data.reshape(in_shape)
        in_tensor = torch.tensor(input_data, dtype=torch.float32)
        in_tensor = in_tensor.permute(0,3,1,2)

        logits = self.keynet(in_tensor) # (1, self.output_dim, H, W)
        kernel_raw = self.querynet(in_tensor) # (1, self.kernel_dim, H, W)
        crop = kernel_raw[:, :, p[0]:(p[0] + self.crop_size), p[1]:(p[1] + self.crop_size)] # (1, self.kernel_dim, crop_size, crop_size)
        # Rotate angles.
        angles = [i*360/self.n_rotations if i*360/self.n_rotations <= 180 
                else i*360/self.n_rotations-360 for i in range(self.n_rotations)]
        crops = ()
        for angle in angles:
                rotated_crop = transforms.functional.affine(crop, angle=angle, translate=[0,0], 
                                                              scale=1, shear=[0,0], resample=PIL.Image.NEAREST)
                                                       
                crops += (rotated_crop,)
        kernel = torch.cat(crops, dim=0) # (n_rotations, self.kernel_dim, self.crop_size, self.crop_size)
        # Obtain kernels for cross-convolution.
        kernel_paddings = (0,1,0,1)
        kernel = nn.functional.pad(kernel, kernel_paddings, mode='constant')
        output = self.correlate(logits, kernel, softmax)
        return output

    def train(self, in_img, p, q, theta, backprop=True):
        """Transport pixel p to pixel q.

        Args:
            in_img: input image.
            p: pixel (y, x)
            q: pixel (y, x)
            theta: rotation label in radians.
            backprop: True if backpropagating gradients.

        Returns:
            loss: training loss.
        """

        # self.metric.reset_states()

        output = self.forward(in_img, p, softmax=False, train=True)
        output = output.reshape(1, np.prod(output.shape))

        itheta = theta / (2 * np.pi / self.n_rotations)
        itheta = np.int32(np.round(itheta)) % self.n_rotations

        # Get one-hot pixel label map.
        label_size = in_img.shape[:2] + (self.n_rotations,)
        label = np.zeros(label_size)
        label[q[0], q[1], itheta] = 1
        # Get loss.
        label = label.reshape(np.prod(label.shape),)
        label = np.where(label==1)[0][0]
        label = torch.tensor(label, dtype=torch.int64).reshape(1,)
        loss = self.criterion(output, label)

        if backprop:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        self.iters += 1
        return loss.item()

    def load(self, transport_fname, device):
        checkpoint = torch.load(transport_fname, map_location=device)
        self.keynet.load_state_dict(checkpoint['keynet_state_dict'])
        self.querynet.load_state_dict(checkpoint['querynet_state_dict'])

    def save(self, transport_fname):
        state = {'keynet_state_dict': self.keynet.state_dict(),
                'querynet_state_dict': self.querynet.state_dict()}
        torch.save(state, transport_fname)