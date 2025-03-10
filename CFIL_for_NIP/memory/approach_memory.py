from collections import deque
import numpy as np
import torch
import pickle
import joblib
import os, random


class ApproachMemory(dict):
    ''' LazyMemory is memory-efficient but time-inefficient. '''
    keys = ['image', 'position']

    def __init__(self, capacity, device):
        super(ApproachMemory, self).__init__()
        self.capacity = int(capacity)
        self.device = device
        self.reset()
        self.new_append = False
        self.initialize = False

    def __str__(self):
        return 'Image shape: {}, position shape: {}'.format(self.image_shape, self.position_shape)

    def __repr__(self):
        return 'LazyMemory(capacity={}, device={})'.format(self.capacity, self.device)

    def reset(self):
        self.is_set_init = False
        self._p = 0
        self._n = 0
        for key in self.keys:
            self[key] = [None] * self.capacity

    def initial_settings(self, image, position):
        self.image_shape = image.shape
        self.position_shape = position.shape
        self.is_set_init = True
        self.reset()
        self.initialize = True

    def append(self, image, position):
        # assert self.is_set_init is True
        self._append(image, position)

    def append_episode(self, images, positions):
        # assert self.is_set_init is True
        for idx in range(len(images)):
            image = images[idx]
            position = positions[idx]
            self._append(image, position)

    def _append(self, image, position):
        self['image'][self._p] = image
        self['position'][self._p] = position
        self.new_append = True

        self._n = min(self._n + 1, self.capacity)
        self._p = (self._p + 1) % self.capacity

    def sample(self, batch_size):
        '''
        Returns:
            images_seq  : (N, S, *observation_shape) shaped tensor.
            positions_seq  : (N, S, *position_shape) shaped tensor.
        '''

        images_seq = np.empty((batch_size, *self.image_shape), dtype=np.uint8)
        positions_seq = np.empty((batch_size, *self.position_shape), dtype=np.float32)
        indices = np.random.randint(low=0, high=self._n, size=batch_size)

        for i, index in enumerate(indices):
            # Convert LazeFrames into np.ndarray here.
            # print("i" , i)
            # print("index" , index)
            images_seq[i, ...] = self['image'][index]
            positions_seq[i, ...] = self['position'][index]

        images_seq = np.transpose(images_seq, [0, 3, 1, 2])

        # positions_seq = positions_seq[:, [0, 1, 2, 3, 4, 5]]
        

        images_seq = torch.ByteTensor(images_seq).to(self.device).float() / 255.
        positions_seq = torch.FloatTensor(positions_seq).to(self.device)
        return {'images_seq': images_seq,
                'positions_seq': positions_seq,
                }

    def save_pickle(self, filepath):
        data = {
            'image': self['image'][:self._n],
            'position': self['position'][:self._n],
        }
        save_dir = os.path.dirname(filepath)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load_pickle(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        dn = len(data['image'])
        self['image'][self._p:self._p + dn] = data['image']
        self['position'][self._p:self._p + dn] = data['position']

        self._n = min(self._n + dn, self.capacity)
        self._p = (self._p + dn) % self.capacity

    def save_joblib(self, filepath):
        data = {
            'image': self['image'][:self._n],
            'position': self['position'][:self._n],
        }
        save_dir = os.path.dirname(filepath)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        joblib.dump(data, filepath, compress=3)

    def load_joblib(self, filepath):
        data = joblib.load(filepath)
        dn = len(data['image'])
        self['image'][self._p:self._p + dn] = data['image']
        self['position'][self._p:self._p + dn] = data['position']
        
        self.image_shape = data['image'][0].shape
        self.position_shape = data['position'][0].shape
        
        self._n = min(self._n + dn, self.capacity)
        self._p = (self._p + dn) % self.capacity

    def __len__(self):
        return self._n
