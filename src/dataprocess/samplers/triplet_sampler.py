# -*- coding: utf-8 -*-
'''
* @Author       : jiangtao
* @Date         : 2021-04-20 09:08:55
* @Email        : jiangtaoo2333@163.com
* @LastEditTime : 2021-07-03 10:32:45
'''
import copy
import itertools
from collections import defaultdict
from typing import Optional

import numpy as np
from torch.utils.data.sampler import Sampler


class NaiveIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, xml_path, pid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source: str, batch_size: int, num_instances: int, seed: Optional[int] = None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // self.num_instances

        self.index_pid = defaultdict(list)
        self.pid_index = defaultdict(list)

        for index, info in enumerate(data_source):
            pid = info[2]
            self.index_pid[index] = pid
            self.pid_index[pid].append(index)

        self.pids = sorted(list(self.pid_index.keys()))
        self.num_identities = len(self.pids)

        if seed is None:
            seed = 666
        self._seed = int(seed)

        # self._rank = 1
        # self._world_size = 1

    def __iter__(self):
        # start = self._rank
        yield from self._infinite_indices()

    def __len__(self):
        return len(self.data_source)

    def _infinite_indices(self):
        np.random.seed(self._seed)
        while True:
            avai_pids = copy.deepcopy(self.pids)

            batch_idxs_dict = {}
            batch_indices = []

            while len(avai_pids) >= self.num_pids_per_batch:
                selected_pids = np.random.choice(avai_pids, self.num_pids_per_batch, replace=False).tolist()
                for pid in selected_pids:
                    # Register pid in batch_idxs_dict if not
                    if pid not in batch_idxs_dict:
                        idxs = copy.deepcopy(self.pid_index[pid])
                        if len(idxs) < self.num_instances:
                            idxs = np.random.choice(idxs, size=self.num_instances, replace=True).tolist()
                        np.random.shuffle(idxs)
                        batch_idxs_dict[pid] = idxs

                    avai_idxs = batch_idxs_dict[pid]
                    for _ in range(self.num_instances):
                        batch_indices.append(avai_idxs.pop(0))
                    # if left is small than num_instances,then remove
                    if len(avai_idxs) < self.num_instances: avai_pids.remove(pid)

                assert len(batch_indices) == self.batch_size
                yield from batch_indices
                del batch_indices
                batch_indices = []