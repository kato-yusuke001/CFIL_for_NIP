from collections import deque
import numpy as np
import torch
import pickle
import joblib
import os , random

class LazyFrames:
    ''' LazyFrames memory-efficiently stores stacked data. '''

    def __init__(self, frames):
        self._frames = frames
        self.dtype = frames[0].dtype
        self.out = None

    def __str__(self):
        return 'Data type: {}, Data: {}'.format(self.dtype, self._frames)

    def _force(self):
        return np.array(self._frames, dtype=self.dtype)

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


class LazySequenceBuff:
    '''
    The mask is defined as 0 for absorbing states, 1 for other states.
    State is set to zero by multiplying the mask at the absorbing state.
    '''
    keys = ['image', 'state', 'action', 'reward', 'done', 'mask']

    def __init__(self, num_sequences=8):
        self.num_sequences = int(num_sequences)

    def reset(self):
        self.memory = {
            'image': deque(maxlen=self.num_sequences + 1),
            'state': deque(maxlen=self.num_sequences + 1),
            'mask': deque(maxlen=self.num_sequences + 1),
            'action': deque(maxlen=self.num_sequences),
            'reward': deque(maxlen=self.num_sequences),
            'done': deque(maxlen=self.num_sequences)}

    def set_init_state(self, image, state):
        self.reset()
        self.memory['image'].append(image)
        self.memory['state'].append(state)
        self.memory['mask'].append(np.array([1.0]))

    def append(self, action, reward, next_image, next_state, done, mask):
        self.memory['image'].append(next_image)
        self.memory['state'].append(next_state)
        self.memory['mask'].append(np.array([mask], dtype=np.float32))
        self.memory['action'].append(action)
        self.memory['reward'].append(np.array([reward], dtype=np.float32))
        self.memory['done'].append(np.array([done], dtype=np.bool))

    def get(self):
        # It's memory-efficient, but slow.
        images = LazyFrames(list(self.memory['image']))
        states = LazyFrames(list(self.memory['state']))
        masks = LazyFrames(list(self.memory['mask']))
        actions = LazyFrames(list(self.memory['action']))
        rewards = LazyFrames(list(self.memory['reward']))
        dones = LazyFrames(list(self.memory['done']))

        return images, states, actions, rewards, dones, masks

    def __len__(self):
        return len(self.memory['state'])


class LazyMemory(dict):
    ''' LazyMemory is memory-efficient but time-inefficient. '''
    keys = ['image', 'state', 'action', 'reward', 'done', 'mask']

    def __init__(self, capacity, num_sequences, observation_shape,
                 state_shape,
                 action_shape, device):
        super(LazyMemory, self).__init__()
        self.capacity = int(capacity)
        self.num_sequences = int(num_sequences)
        self.observation_shape = observation_shape
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.reset()
        self.expert_done_index=[]
        self.novice_done_index=[]
        self.expert_not_done_index=[]
        self.novice_not_done_index=[]
        self.get_initial_expert_index = False
        self.get_initial_novice_index = False
        self.new_append = False
        self.agent_done_index =[]
        self.agent_not_done_index =[]

    def __str__(self):
        return 'Image shape: {}, state shape: {}'.format(self.initial_image_shape, self.initial_state_shape)

    def __repr__(self):
        return 'LazyMemory(capacity={}, num_sequences={}, ' \
               'observation_shape={}, state_shape={}, action_shape={}, ' \
               'device={})'.format(self.capacity, self.num_sequences,
                                   self.observation_shape, self.state_shape,
                                   self.action_shape, self.device)

    def reset(self):
        self.is_set_init = False
        self._p = 0
        self._n = 0
        for key in self.keys:
            self[key] = [None] * self.capacity
        self.buff = LazySequenceBuff(num_sequences=self.num_sequences)

    def set_initial_state(self, image, state):
        self.initial_image_shape = image.shape
        self.initial_state_shape = state.shape
        self.buff.set_init_state(image, state)
        self.is_set_init = True

    def append(self, action, reward, next_image, next_state, done, episode_done=False, absorb_state=False):
        assert self.is_set_init is True

        mask = 0.0 if absorb_state else 1.0
        self.buff.append(action, reward, next_image, next_state, done, mask)

        if len(self.buff) == self.num_sequences + 1:
            images, states, actions, rewards, dones, masks = self.buff.get()
            self._append(images, states, actions, rewards, dones, masks)

        if done or episode_done:
            self.buff.reset()

    def _append(self, image, state, action, reward, done, mask):
        self['image'][self._p] = image
        self['state'][self._p] = state
        self['mask'][self._p] = mask
        self['action'][self._p] = action
        self['reward'][self._p] = reward
        self['done'][self._p] = done
        self.new_append = True

        self._n = min(self._n + 1, self.capacity)
        self._p = (self._p + 1) % self.capacity


    def sample(self, batch_size, weighting_sample_num, memory_type):
        '''
        Returns:
            images_seq  : (N, S+1, *observation_shape) shaped tensor.
            states_seq  : (N, S+1, *state_shape) shaped tensor.
            actions_seq : (N, S, *action_shape) shaped tensor.
            rewards_seq : (N, S, 1) shaped tensor.
            dones_seq   : (N, S, 1) shaped tensor.
            rewards     : (N, 1) shaped tensor.
            masks       : (N, 1) shaped tensor.
        '''

        images_seq = np.empty((batch_size, self.num_sequences+1, *self.observation_shape), dtype=np.uint8)
        states_seq = np.empty((batch_size, self.num_sequences+1, *self.state_shape), dtype=np.float32)
        actions_seq = np.empty((batch_size, self.num_sequences, *self.action_shape), dtype=np.float32)
        rewards_seq = np.empty((batch_size, self.num_sequences, 1), dtype=np.float32)
        dones_seq = np.empty((batch_size, self.num_sequences, 1), dtype=np.bool)
        rewards = np.empty((batch_size, 1), dtype=np.float32)
        masks = np.empty((batch_size, 1), dtype=np.float32)

        if weighting_sample_num > 0.0:
            important_seq_num = int ( batch_size * weighting_sample_num )
            # print("important_seq_num" , important_seq_num)

            num = 0
            if memory_type == "agent":
                if self.new_append :
                    for i in range(self._n):
                        if self['done'][i][7] == [True]:
                            self.agent_done_index.append(i)
                        else:
                            self.agent_not_done_index.append(i)
                    print("agent_done_index :", len(self.agent_done_index) , "  agent_not_done_index:",   len(self.agent_not_done_index) )
                    self.new_append = False
                if len(self.agent_done_index) > important_seq_num:
                    indices = np.random.randint(low=0, high=len(self.agent_done_index)-1, size=important_seq_num)
                    for index in indices:
                        # Convert LazeFrames into np.ndarray here.
                        # print("i" , i)
                        # print("index" , index)
                        images_seq[num, ...] = self['image'][index]
                        states_seq[num, ...] = self['state'][index]
                        actions_seq[num, ...] = self['action'][index]
                        actions_seq[num][7] = random.uniform(-1,1)
                        rewards_seq[num, ...] = self['reward'][index]
                        dones_seq[num, ...] = self['done'][index]
                        actions_seq[num][7] = [False]
                        rewards[num, ...] = self['reward'][index][-1]
                        masks[num, ...] = self['mask'][index][-1]
                        num += 1
                else :
                    indices = self.agent_done_index
                    for index in indices:
                        # Convert LazeFrames into np.ndarray here.
                        # print("i" , i)
                        # print("index" , index)
                        images_seq[num, ...] = self['image'][index]
                        states_seq[num, ...] = self['state'][index]
                        actions_seq[num, ...] = self['action'][index]
                        actions_seq[num][7] = random.uniform(-1,1)
                        rewards_seq[num, ...] = self['reward'][index]
                        dones_seq[num, ...] = self['done'][index]
                        actions_seq[num][7] = [False]
                        rewards[num, ...] = self['reward'][index][-1]
                        masks[num, ...] = self['mask'][index][-1]
                        num += 1
                    # print("agent is low data !!!! num=" , num)

                indices = np.random.randint(low=0, high=len(self.agent_not_done_index)-1, size=batch_size-num)
                for index in indices:
                    # Convert LazeFrames into np.ndarray here.
                    # print("i" , i)
                    # print("index" , index)
                    images_seq[num, ...] = self['image'][index]
                    states_seq[num, ...] = self['state'][index]
                    actions_seq[num, ...] = self['action'][index]
                    rewards_seq[num, ...] = self['reward'][index]
                    dones_seq[num, ...] = self['done'][index]
                    rewards[num, ...] = self['reward'][index][-1]
                    masks[num, ...] = self['mask'][index][-1]
                    num += 1
                
            elif memory_type == "expert":
                if not self.get_initial_expert_index :
                    for i in range(self._n):
                        if self['done'][i][7] == [True]:
                            self.expert_done_index.append(i)
                        else:
                            self.expert_not_done_index.append(i)
                    print("get done_seq expert : " , len(self.expert_done_index))
                    self.get_initial_expert_index = True

                if len(self.expert_done_index) > important_seq_num:
                    indices = np.random.randint(low=0, high=len(self.expert_done_index)-1, size=important_seq_num)
                    for index in indices:
                        # Convert LazeFrames into np.ndarray here.
                        # print("i" , i)
                        # print("index" , index)
                        images_seq[num, ...] = self['image'][index]
                        states_seq[num, ...] = self['state'][index]
                        actions_seq[num, ...] = self['action'][index]
                        actions_seq[num][7] = random.uniform(-1,1)
                        rewards_seq[num, ...] = self['reward'][index]
                        dones_seq[num, ...] = self['done'][index]
                        actions_seq[num][7] = [False]
                        rewards[num, ...] = self['reward'][index][-1]
                        masks[num, ...] = self['mask'][index][-1]
                        num += 1
                else :
                    indices = self.expert_done_index
                    for index in indices:
                        # Convert LazeFrames into np.ndarray here.
                        # print("i" , i)
                        # print("index" , indexpert_not_done_indexex)
                        images_seq[num, ...] = self['image'][index]
                        states_seq[num, ...] = self['state'][index]
                        actions_seq[num, ...] = self['action'][index]
                        actions_seq[num][7] = random.uniform(-1,1)
                        rewards_seq[num, ...] = self['reward'][index]
                        dones_seq[num, ...] = self['done'][index]
                        actions_seq[num][7] = [False]
                        rewards[num, ...] = self['reward'][index][-1]
                        masks[num, ...] = self['mask'][index][-1]
                        num += 1
                    # print("agent is low data !!!! num=" , num)

                indices = np.random.randint(low=0, high=len(self.expert_not_done_index)-1, size=batch_size-num)
                for index in indices:
                    # Convert LazeFrames into np.ndarray here.
                    # print("i" , i)
                    # print("index" , index)
                    images_seq[num, ...] = self['image'][index]
                    states_seq[num, ...] = self['state'][index]
                    actions_seq[num, ...] = self['action'][index]
                    rewards_seq[num, ...] = self['reward'][index]
                    dones_seq[num, ...] = self['done'][index]
                    rewards[num, ...] = self['reward'][index][-1]
                    masks[num, ...] = self['mask'][index][-1]
                    num += 1                

            elif memory_type == "novice":
                if not self.get_initial_novice_index:
                    for i in range(self._n):
                        if self['done'][i][7] == [True]:
                            self.novice_done_index.append(i)
                        else:
                            self.novice_not_done_index.append(i)
                    print("get done_seq novice : " , len(self.novice_done_index))
                    self.get_initial_novice_index = True

                if len(self.novice_done_index) > important_seq_num:
                    indices = np.random.randint(low=0, high=len(self.novice_done_index)-1, size=important_seq_num)
                    for index in indices:
                        # Convert LazeFrames into np.ndarray here.
                        # print("i" , i)
                        # print("index" , index)
                        images_seq[num, ...] = self['image'][index]
                        states_seq[num, ...] = self['state'][index]
                        actions_seq[num, ...] = self['action'][index]
                        actions_seq[num][7] = random.uniform(-1,1)
                        rewards_seq[num, ...] = self['reward'][index]
                        dones_seq[num, ...] = self['done'][index]
                        actions_seq[num][7] = [False]
                        rewards[num, ...] = self['reward'][index][-1]
                        masks[num, ...] = self['mask'][index][-1]
                        num += 1
                else :
                    indices = self.novice_done_index
                    for index in indices:
                        # Convert LazeFrames into np.ndarray here.
                        # print("i" , i)
                        # print("index" , index)
                        images_seq[num, ...] = self['image'][index]
                        states_seq[num, ...] = self['state'][index]
                        actions_seq[num, ...] = self['action'][index]
                        actions_seq[num][7] = random.uniform(-1,1)
                        rewards_seq[num, ...] = self['reward'][index]
                        dones_seq[num, ...] = self['done'][index]
                        actions_seq[num][7] = [False]
                        rewards[num, ...] = self['reward'][index][-1]
                        masks[num, ...] = self['mask'][index][-1]
                        num += 1
                    # print("agent is low data !!!! num=" , num)

                indices = np.random.randint(low=0, high=len(self.novice_not_done_index)-1, size=batch_size-num)
                for index in indices:
                    # Convert LazeFrames into np.ndarray here.
                    # print("i" , i)
                    # print("index" , index)
                    images_seq[num, ...] = self['image'][index]
                    states_seq[num, ...] = self['state'][index]
                    actions_seq[num, ...] = self['action'][index]
                    rewards_seq[num, ...] = self['reward'][index]
                    dones_seq[num, ...] = self['done'][index]
                    rewards[num, ...] = self['reward'][index][-1]
                    masks[num, ...] = self['mask'][index][-1]
                    num += 1                

        else:
            indices = np.random.randint(low=0, high=self._n, size=batch_size)

            for i, index in enumerate(indices):
                # Convert LazeFrames into np.ndarray here.
                # print("i" , i)
                # print("index" , index)
                images_seq[i, ...] = self['image'][index]
                states_seq[i, ...] = self['state'][index]
                actions_seq[i, ...] = self['action'][index]
                rewards_seq[i, ...] = self['reward'][index]
                dones_seq[i, ...] = self['done'][index]
                rewards[i, ...] = self['reward'][index][-1]
                masks[i, ...] = self['mask'][index][-1]

        images_seq = torch.ByteTensor(images_seq).to(self.device).float()/255.
        states_seq = torch.FloatTensor(states_seq).to(self.device)
        actions_seq = torch.FloatTensor(actions_seq).to(self.device)
        rewards_seq = torch.FloatTensor(rewards_seq).to(self.device)
        dones_seq = torch.BoolTensor(dones_seq).to(self.device).float()
        rewards = torch.FloatTensor(rewards).to(self.device)
        masks = torch.FloatTensor(masks).to(self.device)

        return {'images_seq': images_seq,
                'states_seq': states_seq,
                'actions_seq': actions_seq,
                'rewards_seq': rewards_seq,
                'dones_seq': dones_seq,
                'rewards': rewards,
                'masks': masks
                }

    # def sample(self, batch_size):
    #     '''
    #     Returns:
    #         images_seq  : (N, S+1, *observation_shape) shaped tensor.
    #         states_seq  : (N, S+1, *state_shape) shaped tensor.
    #         actions_seq : (N, S, *action_shape) shaped tensor.
    #         rewards_seq : (N, S, 1) shaped tensor.
    #         dones_seq   : (N, S, 1) shaped tensor.
    #         rewards     : (N, 1) shaped tensor.
    #         masks       : (N, 1) shaped tensor.
    #     '''
    #     indices = np.random.randint(low=0, high=self._n, size=batch_size)
    #     images_seq = np.empty((batch_size, self.num_sequences+1, *self.observation_shape), dtype=np.uint8)
    #     states_seq = np.empty((batch_size, self.num_sequences+1, *self.state_shape), dtype=np.float32)
    #     actions_seq = np.empty((batch_size, self.num_sequences, *self.action_shape), dtype=np.float32)
    #     rewards_seq = np.empty((batch_size, self.num_sequences, 1), dtype=np.float32)
    #     dones_seq = np.empty((batch_size, self.num_sequences, 1), dtype=np.bool)
    #     rewards = np.empty((batch_size, 1), dtype=np.float32)
    #     masks = np.empty((batch_size, 1), dtype=np.float32)

    #     for i, index in enumerate(indices):
    #         # Convert LazeFrames into np.ndarray here.
    #         images_seq[i, ...] = self['image'][index]
    #         states_seq[i, ...] = self['state'][index]
    #         actions_seq[i, ...] = self['action'][index]
    #         rewards_seq[i, ...] = self['reward'][index]
    #         dones_seq[i, ...] = self['done'][index]
    #         rewards[i, ...] = self['reward'][index][-1]
    #         masks[i, ...] = self['mask'][index][-1]

    #     images_seq = torch.ByteTensor(images_seq).to(self.device).float()/255.
    #     states_seq = torch.FloatTensor(states_seq).to(self.device)
    #     actions_seq = torch.FloatTensor(actions_seq).to(self.device)
    #     rewards_seq = torch.FloatTensor(rewards_seq).to(self.device)
    #     dones_seq = torch.BoolTensor(dones_seq).to(self.device).float()
    #     rewards = torch.FloatTensor(rewards).to(self.device)
    #     masks = torch.FloatTensor(masks).to(self.device)

    #     return {'images_seq': images_seq,
    #             'states_seq': states_seq,
    #             'actions_seq': actions_seq,
    #             'rewards_seq': rewards_seq,
    #             'dones_seq': dones_seq,
    #             'rewards': rewards,
    #             'masks': masks
    #             }

    def sample_latent(self, batch_size):
        '''
        Returns:
            images_seq  : (N, S+1, *observation_shape) shaped tensor.
            states_seq  : (N, S+1, *state_shape) shaped tensor.
            actions_seq : (N, S, *action_shape) shaped tensor.
            rewards_seq : (N, S, 1) shaped tensor.
            dones_seq   : (N, S, 1) shaped tensor.
        '''        
        indices = np.random.randint(low=0, high=self._n, size=batch_size)

        images_seq = np.empty((
            batch_size, self.num_sequences+1, *self.observation_shape),
            dtype=np.uint8)
        states_seq = np.empty((
            batch_size, self.num_sequences+1, *self.state_shape),
            dtype=np.float32)
        actions_seq = np.empty((
            batch_size, self.num_sequences, *self.action_shape),
            dtype=np.float32)
        rewards_seq = np.empty((
            batch_size, self.num_sequences, 1), dtype=np.float32)
        dones_seq = np.empty((
            batch_size, self.num_sequences, 1), dtype=np.bool)

        for i, index in enumerate(indices):
                # Convert LazeFrames into np.ndarray here.
                images_seq[i, ...] = self['image'][index]
                states_seq[i, ...] = self['state'][index]
                actions_seq[i, ...] = self['action'][index]
                rewards_seq[i, ...] = self['reward'][index]
                dones_seq[i, ...] = self['done'][index]

        images_seq = torch.ByteTensor(images_seq).to(self.device).float()/255.
        states_seq = torch.FloatTensor(states_seq).to(self.device)
        actions_seq = torch.FloatTensor(actions_seq).to(self.device)
        rewards_seq = torch.FloatTensor(rewards_seq).to(self.device)
        dones_seq = torch.BoolTensor(dones_seq).to(self.device).float()

        return images_seq, states_seq, actions_seq, rewards_seq, dones_seq


    def sample_sac(self, batch_size):
        '''
        Returns:
            images_seq  : (N, S+1, *observation_shape) shaped tensor.
            states_seq  : (N, S+1, *state_shape) shaped tensor.
            actions_seq : (N, S, *action_shape) shaped tensor.
            rewards     : (N, 1) shaped tensor.
            masks       : (N, 1) shaped tensor.
        '''
        indices = np.random.randint(low=0, high=self._n, size=batch_size)

        images_seq = np.empty((
            batch_size, self.num_sequences+1, *self.observation_shape),
            dtype=np.uint8)
        states_seq = np.empty((
            batch_size, self.num_sequences+1, *self.state_shape),
            dtype=np.float32)
        actions_seq = np.empty((
            batch_size, self.num_sequences, *self.action_shape),
            dtype=np.float32)
        rewards = np.empty((batch_size, 1), dtype=np.float32)
        masks = np.empty((batch_size, 1), dtype=np.float32)

        for i, index in enumerate(indices):
            # Convert LazeFrames into np.ndarray here.
            images_seq[i, ...] = self['image'][index]
            states_seq[i, ...] = self['state'][index]
            actions_seq[i, ...] = self['action'][index]
            rewards[i, ...] = self['reward'][index][-1]
            masks[i, ...] = self['mask'][index][-1]

        images_seq = torch.ByteTensor(images_seq).to(self.device).float()/255.
        states_seq = torch.FloatTensor(states_seq).to(self.device)
        actions_seq = torch.FloatTensor(actions_seq).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        masks = torch.FloatTensor(masks).to(self.device)

        return images_seq, states_seq, actions_seq, rewards, masks

    def save_pickle(self, filepath):
        data = {
            'image': self['image'][:self._n],
            'state': self['state'][:self._n],
            'mask': self['mask'][:self._n],
            'action': self['action'][:self._n],
            'reward': self['reward'][:self._n],
            'done': self['done'][:self._n],
        }
        # View data
        # tmp = np.empty((self.num_sequences+1, *self.observation_shape), dtype=np.uint8)
        # tmp[...] = data['image'][0]
        # print('Saved image shape:', len(data['image']), 'x', tmp.shape)
        save_dir = os.path.dirname(filepath)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load_pickle(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        # View data
        # tmp = np.empty((self.num_sequences+1, *self.observation_shape), dtype=np.uint8)
        # tmp[...] = data['image'][0]
        # print('Loaded image shape:', len(data['image']), 'x', tmp.shape)
        dn = len(data['image'])
        self['image'][self._p:self._p + dn] = data['image']
        self['state'][self._p:self._p + dn] = data['state']
        self['action'][self._p:self._p + dn] = data['action']
        self['reward'][self._p:self._p + dn] = data['reward']
        self['done'][self._p:self._p + dn] = data['done']
        if 'mask' in data.keys():
            self['mask'][self._p:self._p + dn] = data['mask']

        self._n = min(self._n + dn, self.capacity)
        self._p = (self._p + dn) % self.capacity

    def save_joblib(self, filepath):
        data = {
            'image': self['image'][:self._n],
            'state': self['state'][:self._n],
            'mask': self['mask'][:self._n],
            'action': self['action'][:self._n],
            'reward': self['reward'][:self._n],
            'done': self['done'][:self._n],
        }
        save_dir = os.path.dirname(filepath)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        joblib.dump(data, filepath, compress=3)

    def load_joblib(self, filepath):
        data = joblib.load(filepath)
        # View data
        # tmp = np.empty((self.num_sequences+1, *self.observation_shape), dtype=np.uint8)
        # tmp[...] = data['image'][0]
        # print('Loaded image shape:', len(data['image']), 'x', tmp.shape)
        dn = len(data['image'])
        self['image'][self._p:self._p + dn] = data['image']
        self['state'][self._p:self._p + dn] = data['state']
        self['action'][self._p:self._p + dn] = data['action']
        self['reward'][self._p:self._p + dn] = data['reward']
        self['done'][self._p:self._p + dn] = data['done']
        if 'mask' in data.keys():
            self['mask'][self._p:self._p + dn] = data['mask']

        self._n = min(self._n + dn, self.capacity)
        self._p = (self._p + dn) % self.capacity

    def __len__(self):
        return self._n
