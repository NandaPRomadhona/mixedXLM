from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score

from logging import getLogger
import os
import copy
import time
import json
from collections import OrderedDict
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

import random

from ..optim import get_optimizer
from ..utils import concat_batches, truncate, to_cuda
from ..data.dataset import Dataset
from ..data.loader import load_mix_binarized, set_dico_parameters

from sklearn.neighbors import KNeighborsClassifier


logger = getLogger()


class FEWSHOT:

    def __init__(self, embedder, scores, params):
        """
        Initialize XNLI trainer / evaluator.
        Initial `embedder` should be on CPU to save memory.
        """
        self._embedder = embedder
        self.params = params
        self.scores = scores
        self.clf = KNeighborsClassifier(n_neighbors=params.n_support, p=2)

    def get_iterator(self, splt, lang):
        """
        Get a monolingual data iterator.
        """
        return self.data[lang][splt]['x'].get_iterator(
            shuffle=(splt == 'train'),
            return_indices=True
        )

    def run(self):
        """
        Run XNLI training / evaluation.
        """
        params = self.params
        params.out_features = 2

        # load data
        self.data = self.load_data()
        logger.info("====== NANDA KEPO")
        logger.info(self.data)
        if not self.data['dico'] == self._embedder.dico:
            raise Exception(("Dictionary in evaluation data (%i words) seems different than the one " +
                             "in the pretrained model (%i words). Please verify you used the same dictionary, " +
                             "and the same values for max_vocab and min_count.") % (len(self.data['dico']), len(self._embedder.dico)))

        # embedder
        self.embedder = copy.deepcopy(self._embedder)
        self.embedder.cuda()

        #human_looping:
        self.train()

        #testing
        # self.eval('test')


    def get_random_index(self, y, id_class, n_support):
        idx = []
        while len(idx) != n_support:
            idx_y = random.choice([i for i, a in enumerate(y) if a == id_class])
            if idx_y not in idx:
                idx.append(idx_y)

        return idx

    def train(self):
        """
        Finetune for one epoch on the XNLI English training set.
        """
        params = self.params
        self.embedder.train()
        # self.proj.train()

        # training variables
        losses = []
        ns = 0  # number of sentences
        nw = 0  # number of words
        t = time.time()

        iterator = self.get_iterator('train', 'mix')
        lang_id = params.lang2id['mix']

        end_iteration = False
        self.epoch = 0
        while end_iteration == False and self.epoch < (params.n_epochs)-1:
            logger.info("============== Start epoch %i ....." %self.epoch )

            # batch
            try:
                batch = next(iterator)
            except StopIteration:
                break
            (x, lengths), idx = batch
            x, lengths = truncate(x, lengths, params.max_len, params.eos_index)
            y = self.data['mix']['train']['y'][idx]
            bs = len(lengths)

            x, y, lengths = to_cuda(x, y, lengths)

            output_embed = self.embedder.get_embeddings(x, lengths, positions=None, langs=None)
            #chose N_sample data for each classes
            idx_c0 = self.get_random_index(y, 0, params.n_support)
            idx_c1 = self.get_random_index(y, 1, params.n_support)
            y0 = [0] * params.n_support
            y1 = [1] * params.n_support
            selected_y = y0 + y1

            all_idx = idx_c0 + idx_c1

            selected_embed = [output_embed[idx] for idx in all_idx]
            mean_selected_embed = [[torch.mean(dt).tolist()] for dt in selected_embed]

            
            self.clf.fit(mean_selected_embed, selected_y)

            #need eval
            scores = self.eval('valid')
            self.scores.update(scores)


            logger.info("============== End epoch %i ....." %self.epoch )
            self.epoch += 1



    def eval(self, splt):
        """
        Evaluate on XNLI validation and test sets, for all languages.
        """
        params = self.params
        self.embedder.eval()

        assert splt in ['valid', 'test']
        has_labels = 'y' in self.data['mix'][splt]

        scores = OrderedDict({'epoch': self.epoch})

        idxs = []  # sentence indices
        prob = []  # probabilities
        pred = []  # predicted values
        gold = []  # real values

        for batch in self.get_iterator(splt, 'mix'):
            (x, lengths), idx = batch
            y = self.data['mix'][splt]['y'][idx] if has_labels else None

            # cuda
            x, y, lengths = to_cuda(x, y, lengths)

            # prediction
            output_embed = self.embedder.get_embeddings(x, lengths, positions=None, langs=None)

            mean_embed = [[torch.mean(dt).tolist()] for dt in output_embed]
            p =  self.clf.predict(mean_embed)
            output = self.clf.predict_proba(mean_embed)
            idxs.append(idx)
            prob.append(output)
            pred.append(p)
            if has_labels:
                gold.append(y.cpu().numpy())

        logger.info("============= NANDA KEPO Pred")
        logger.info(pred)
        pred = np.array(pred).flatten()
        logger.info(pred)
        logger.info("============= NANDA KEPO Gold")
        gold = np.array(gold).flatten()
        logger.info(gold)


        # score the predictions if we have labels
        if has_labels:
            # gold = np.concatenate(gold)
            prefix = f'{splt}'
            # if self.is_classif:
            # scores['%s_acc' % prefix] = (pred == gold).sum() / len(pred)
            scores['%s_acc' % prefix] = accuracy_score(gold,pred)
            scores['%s_f1' % prefix] = f1_score(gold, pred, average='binary' if params.out_features == 2 else 'micro')
            scores['%s_mc' % prefix] = matthews_corrcoef(gold, pred)
            # else:
            #     scores['%s_prs' % prefix] = 100. * pearsonr(pred, gold)[0]
            #     scores['%s_spr' % prefix] = 100. * spearmanr(pred, gold)[0]
            logger.info("__log__:%s" % json.dumps(scores))

        # output predictions
        # pred_path = os.path.join(params.dump_path, f'{splt}.pred.{self.epoch}')
        # with open(pred_path, 'w') as f:
        #     for i, p in zip(idxs, prob):
        #         f.write('%i\t%s\n' % (i, ','.join([str(x) for x in p])))
        # logger.info(f"Wrote {len(idxs)} {splt} predictions to {pred_path}")

        return scores
    def load_data(self):
        """
        Load XNLI cross-lingual classification data.
        """
        params = self.params
        data = {'mix':{splt: {} for splt in ['train', 'valid', 'test']}}
        label2id = {'negative': 0, 'positive': 1}
        lang = 'mix'
        dpath = params.data_path

        for splt in ['train', 'valid', 'test']:

            # load data and dictionary
            data1 = load_mix_binarized(os.path.join(dpath, '%s.%s.pth' % (splt, lang)), params)
            logger.info("========== NANDA KEPO")
            logger.info(data1)
            data['dico'] = data.get('dico', data1['dico'])

            # set dictionary parameters
            set_dico_parameters(params, data, data1['dico'])
            # create dataset
            # test = Dataset(data1['sentences'], data1['positions'],params)
            data[lang][splt]['x'] = Dataset(data1['sentences'], data1['positions'],params)

            # load labels
            with open(os.path.join(dpath, '%s.label.%s' % (splt, lang)), 'r') as f:
                labels = [label2id[l.rstrip()] for l in f]
            data[lang][splt]['y'] = torch.LongTensor(labels)
            assert len(data[lang][splt]['x']) == len(data[lang][splt]['y'])

        return data