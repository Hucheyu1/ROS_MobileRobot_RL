import numpy as np
import random

import util
import torch
    
class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        #  add
        self.obs_dim = 43
        self.act_dim = 2
        self.obses_t = np.zeros((self._maxsize, self.obs_dim), dtype=np.float32)
        self.obses_tp1 = np.zeros((self._maxsize, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self._maxsize, self.act_dim), dtype=np.float32)
        self.rewards = np.zeros(self._maxsize, dtype=np.float32)
        self.dones = np.zeros(self._maxsize, dtype=np.float32)
        

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
            # add
            self.obses_t[self._next_idx] = obs_t
            self.obses_tp1[self._next_idx] = obs_tp1
            self.actions[self._next_idx] = action
            self.rewards[self._next_idx] = reward
            self.dones[self._next_idx] = done
        else:
            self._storage[self._next_idx] = data
            # add
            self.obses_t[self._next_idx] = (np.array(obs_t, copy=False))
            self.obses_tp1[self._next_idx] = (np.array(obs_tp1, copy=False))
            self.actions[self._next_idx] = (np.array(action.flatten().tolist(), copy=False))
            self.rewards[self._next_idx] = (np.array(reward))
            self.dones[self._next_idx] = (np.array(done))
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)
    
    def sample_batch_with_history(self, batch_size=64, max_hist_len=10):
        idxs = np.random.randint(max_hist_len, len(self._storage), size=batch_size)
        # print(idxs)
        # data = list(zip(*self._storage))
        # obses_t, actions, rewards, obses_tp1, dones = [list(item) for item in data]
        if max_hist_len == 0:
            hist_obs = np.zeros([batch_size, 1, self.obs_dim])
            hist_act = np.zeros([batch_size, 1, self.act_dim])
            hist_obs2 = np.zeros([batch_size, 1, self.obs_dim])
            hist_act2 = np.zeros([batch_size, 1, self.act_dim])
            hist_obs_len = np.zeros(batch_size)
            hist_obs2_len = np.zeros(batch_size)
        else:
            hist_obs = np.zeros([batch_size, max_hist_len, self.obs_dim])
            hist_act = np.zeros([batch_size, max_hist_len, self.act_dim])
            hist_obs_len = max_hist_len * np.ones(batch_size)
            hist_obs2 = np.zeros([batch_size, max_hist_len, self.obs_dim])
            hist_act2 = np.zeros([batch_size, max_hist_len, self.act_dim])
            hist_obs2_len = max_hist_len * np.ones(batch_size)
        # Extract history experiences before sampled index
        for i, id in enumerate(idxs):
            hist_start_id = id - max_hist_len + 1
            id = id + 1
            if hist_start_id < 0:
                hist_start_id = 0
            # If exist done before the last experience (not including the done in id), start from the index next to the done.
            if len(np.where(self.dones[hist_start_id:id-1] == 1)[0]) != 0:
                hist_start_id = hist_start_id + (np.where(self.dones[hist_start_id:id-1] == 1)[0][-1]) + 1
            hist_seg_len = id - hist_start_id
            hist_obs_len[i] = hist_seg_len
            # hist_obs = np.zeros([batch_size, max_hist_len, self.obs_dim])
            hist_obs[i, :hist_seg_len, :] = self.obses_t[hist_start_id:id]
            hist_act[i, :hist_seg_len, :] = self.actions[hist_start_id:id]
            # If the first experience of an episode is sampled, the hist lengths are different for obs and obs2.
            if hist_seg_len == 0:
                hist_obs2_len[i] = 1
            else:
                hist_obs2_len[i] = hist_seg_len
            hist_obs2[i, :hist_seg_len, :] = self.obses_tp1[hist_start_id:id]
            hist_act2[i, :hist_seg_len, :] = self.actions[hist_start_id+1:id+1]

        batch = dict(obs = self.obses_t[idxs],
                     obs2 = self.obses_tp1[idxs],
                     act = self.actions[idxs],
                     rew = self.rewards[idxs],
                     done= self.dones[idxs],
                     hist_obs=hist_obs,
                     hist_act=hist_act,
                     hist_obs2=hist_obs2,
                     hist_act2=hist_act2,
                     hist_obs_len=hist_obs_len,
                     hist_obs2_len=hist_obs2_len)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}
    
    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = util.SumSegmentTree(it_capacity)
        self._it_min = util.MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)