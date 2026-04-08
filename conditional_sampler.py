import numpy as np
import torch
import torch.nn.functional as F

def cond_loss(fake, cond_vec, transformer):

    loss = 0
    cond_offset = 0

    for info in transformer.column_info:

        if info.col_type == "categorical":

            start = info.start
            end = info.end

            fake_logits = fake[:, start:end]

            cond_slice = cond_vec[:, cond_offset:cond_offset + info.output_dim]

            target = torch.argmax(cond_slice, dim=1)

            loss += F.cross_entropy(fake_logits, target)

            cond_offset += info.output_dim

    return loss


class ConditionalSampler:

    def __init__(self, data, transformer):

        self.data = data
        self.transformer = transformer

        self.categorical_info = [
            info for info in transformer.column_info
            if info.col_type == "categorical"
        ]

        self._prepare_category_probabilities()

    def _prepare_category_probabilities(self):

        self.category_probs = []

        for info in self.categorical_info:
            col_data = self.data[:, info.start:info.end]
            freq = col_data.sum(axis=0)
            prob = freq / freq.sum()
            self.category_probs.append(prob)

    def sample(self, batch_size):

        col_idx = np.random.randint(len(self.categorical_info))
        info = self.categorical_info[col_idx]

        prob = self.category_probs[col_idx]

        categories = np.random.choice(
            np.arange(info.output_dim),
            size=batch_size,
            p=prob
        )

        total_cat_dim = sum(i.output_dim for i in self.categorical_info)

        cond_vec = np.zeros((batch_size, total_cat_dim), dtype="float32")

        offset = 0
        for i in range(col_idx):
            offset += self.categorical_info[i].output_dim

        cond_vec[np.arange(batch_size), offset + categories] = 1

        mask = np.zeros((batch_size, len(self.categorical_info)), dtype="float32")
        mask[:, col_idx] = 1

        return cond_vec, mask, col_idx, categories

    def sample_data(self, batch_size, col_idx, categories):

        info = self.categorical_info[col_idx]

        start = info.start
        end = info.end

        column_data = self.data[:, start:end]

        idx_list = []

        for cat in categories:

            idx = np.where(column_data[:, cat] == 1)[0]

            if len(idx) == 0:
                idx = np.arange(len(self.data))

            chosen = np.random.choice(idx)

            idx_list.append(chosen)

        return self.data[idx_list]