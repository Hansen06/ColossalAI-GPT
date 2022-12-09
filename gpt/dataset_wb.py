# -*- coding: utf-8 -*-
# Some functions come from the Internet, if you violate your rights, please contact us.
import os
from itertools import chain

import torch
from torch.utils.data import Dataset

SPECIAL_TOKENS = ["[CLS]", "[SEP]",'[user]', '[assistant]']
MODEL_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

class DatasetBase(Dataset):

    def __init__(self, data_path=None):
        self.data_path = data_path
        self.data_files = list()
        self.data_files_offset = list()
        self.data_len = 0
        self._check_files()

    def _check_files(self):
        if self.data_path is None:
            raise RuntimeError("Data path cannot be \
                empty at same time.")

        if self.data_path:
            if not os.path.exists(self.data_path):
                raise RuntimeError("Training files does not exist at " + self.data_path)
            prepare_files_offset(self.data_path, self.data_files,
                                 self.data_files_offset)
            # print(self.data_files_offset)
            self.data_len = len(self.data_files_offset)

    def __len__(self):
        return self.data_len

    def _get_line(self, index):
        tup = self.data_files_offset[index]
        target_file = self.data_files[tup[0]]
        with open(target_file, "r", encoding="utf-8") as f:
            f.seek(tup[1])
            line = f.readline()
        return line.strip()


class WBdistDataset(DatasetBase):

    def __init__(self, tokenizer, max_history=15, n_ctx=1024, batch_first=True, lm_labels=True, *inputs, **kwargs):
        super(WBdistDataset, self).__init__(*inputs, **kwargs)
        self.tokenizer = tokenizer
        self.max_history = max_history
        self.n_ctx = n_ctx
        self.n_ctx = 760
        self.pad = tokenizer.pad_token_id
        self.batch_first = batch_first
        self.lm_labels = lm_labels

    def __getitem__(self, index):
        tokenizer = self.tokenizer
        dialog = self._get_line(index)
        dialog = dialog.strip().split("\t")
        _dialog = []
        department = '[gynecology]' if dialog[2] == '妇产科' else '[pediatrics]'
        for line in dialog[5:]:
            if '[next]' in line:
                sp = line.split('[next]')
                cur = ''
                for i in sp:
                    cur = cur + ' [next] ' + ' '.join(list(i))
                _dialog.append(cur[8:])
            else:
                _dialog.append(' '.join(list(line)))

        # print('dialog :{}'.format(dialog))
        # print('dialog[5:] :{}'.format(dialog[5:]))
        # print('_dialog :{}'.format(_dialog))
        dialog = _dialog #空格切分

        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)

        dialog = tokenize(dialog)
        history = dialog[:-1]
        candidates = dialog[-1]
        return self.process(history, candidates, tokenizer.convert_tokens_to_ids(department))

    def process(self, history, resposne, department, with_eos=True):
        bos, eos, user, assistant = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
        sequence = [[bos]] + history + [resposne + ([eos] if with_eos else [])]
        sequence = [sequence[0]] + [[assistant if i % 2 else user] + s
                                    for i, s in enumerate(sequence[1:])]
        instance = {}
        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [bos] + [assistant if i % 2 else user for i, s in
                                              enumerate(sequence[1:])
                                              for _ in s]
        instance["department_ids"] = [department] * len(instance["input_ids"])
        instance["lm_labels"] = [-1] * len(instance["input_ids"])
        if self.lm_labels:
            instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]

        return instance

    def collate(self, batch):
        """
         计算该batch中的所有sample的最长的input，并且将其他input的长度向其对齐
        """
        input_ids = []
        token_type_ids = []
        labels = []
        department_ids = []
        max_input_len = 0  # 该batch中最长的input，用于该batch的数据对齐
        btc_size = len(batch)
        flag = False
        # 计算该batch中input的最大长度
        for btc_idx in range(btc_size):
            if max_input_len < len(batch[btc_idx]['input_ids']):
                max_input_len = len(batch[btc_idx]['input_ids'])
        if max_input_len > self.n_ctx:
            max_input_len = self.n_ctx
            flag = True  # 本batch有长度超过n_ctx的样本

        if flag:
            # 使用pad id对小于max_input_len的input_id进行补全,超过n_ctx的进行截断
            for btc_idx in range(btc_size):
                input_len = len(batch[btc_idx]['input_ids'])
                if input_len > self.n_ctx:
                    input_ids.append(batch[btc_idx]['input_ids'][-self.n_ctx:])
                    token_type_ids.append(batch[btc_idx]['token_type_ids'][-self.n_ctx:])
                    department_ids.append(batch[btc_idx]['department_ids'][-self.n_ctx:])
                    labels.append(batch[btc_idx]['lm_labels'][-self.n_ctx:])
                else:
                    input_ids.append(batch[btc_idx]['input_ids'])
                    input_ids[btc_idx].extend([self.pad] * (max_input_len - input_len))

                    token_type_ids.append(batch[btc_idx]['token_type_ids'])
                    token_type_ids[btc_idx].extend([self.pad] * (max_input_len - input_len))

                    department_ids.append(batch[btc_idx]['department_ids'])
                    department_ids[btc_idx].extend([self.pad] * (max_input_len - input_len))

                    labels.append(batch[btc_idx]['lm_labels'])
                    labels[btc_idx].extend([-1] * (max_input_len - input_len))
        else:
            # 使用pad id对小于max_input_len的input_id进行补全
            for btc_idx in range(btc_size):
                input_len = len(batch[btc_idx]['input_ids'])
                input_ids.append(batch[btc_idx]['input_ids'])
                input_ids[btc_idx].extend([self.pad] * (max_input_len - input_len))

                token_type_ids.append(batch[btc_idx]['token_type_ids'])
                token_type_ids[btc_idx].extend([self.pad] * (max_input_len - input_len))

                department_ids.append(batch[btc_idx]['department_ids'])
                department_ids[btc_idx].extend([self.pad] * (max_input_len - input_len))

                labels.append(batch[btc_idx]['lm_labels'])
                labels[btc_idx].extend([-1] * (max_input_len - input_len))

        return {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long)
                }, torch.tensor(labels, dtype=torch.long)


def prepare_files_offset(path, files_list, offset_list):
    """Fill the file index and offsets of each line in files_list in offset_list
    Args:
        path: string of file path, support single file or file dir
        files_list: the list contains file names
        offset_list: the list contains the tuple of file name index and offset
    """
    if os.path.isdir(path):  # for multi-file, its input is a dir
        files_list.extend([os.path.join(path, f) for f in os.listdir(path)])
    elif os.path.isfile(path):  # for single file, its input is a file
        files_list.append(path)
    else:
        raise RuntimeError(path + " is not a normal file.")
    print(files_list)
    for i, f in enumerate(files_list):
        offset = 0
        with open(f, "r", encoding="utf-8") as single_file:
            for line in single_file:
                tup = (i, offset)
                offset_list.append(tup)
                offset += len(bytes(line, encoding='utf-8'))
