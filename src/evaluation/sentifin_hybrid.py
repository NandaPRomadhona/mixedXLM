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

from ..optim import get_optimizer
from ..utils import concat_batches, truncate, to_cuda
from ..data.dataset import Dataset
from ..data.loader import load_mix_binarized, set_dico_parameters


from graphviz import Digraph
from torch.autograd import Variable
from torch.autograd import Variable
import torchvision.models as models

# from torch.utils.tensorboard import SummaryWriter

logger = getLogger()


class SENTIFIN:

    def __init__(self, embedder, scores, params):
        """
        Initialize XNLI trainer / evaluator.
        Initial `embedder` should be on CPU to save memory.
        """
        self._embedder = embedder
        self.params = params
        self.scores = scores

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

        # projection layer
        self.proj = nn.Sequential(*[
            nn.Dropout(params.dropout),
            nn.Linear(self.embedder.out_dim+11, 2)
        ]).cuda()

        # optimizers
        self.optimizer_e = get_optimizer(list(self.embedder.get_parameters(params.finetune_layers)), params.optimizer_e)
        self.optimizer_p = get_optimizer(self.proj.parameters(), params.optimizer_p)



        # train and evaluate the model
        for epoch in range(params.n_epochs):

            # update epoch
            self.epoch = epoch

            # training
            logger.info("SENTIFIN - Training epoch %i ..." % epoch)
            self.train()

            # evaluation
            logger.info("SENTIFIN - Evaluating epoch %i ..." % epoch)
            with torch.no_grad():
                scores = self.eval('valid')
                self.scores.update(scores)
                self.eval('test')

    def train(self):
        """
        Finetune for one epoch on the XNLI English training set.
        """
        params = self.params
        self.embedder.train()
        self.proj.train()

        # training variables
        losses = []
        ns = 0  # number of sentences
        nw = 0  # number of words
        t = time.time()

        iterator = self.get_iterator('train', 'mix')
        try:
            lang_id = params.lang2id['mix']
        except KeyError:
            lang_id = 2

        while True:

            # batch
            try:
                batch = next(iterator)
            except StopIteration:
                break
            (x, lengths), idx = batch
            x, lengths = truncate(x, lengths, params.max_len, params.eos_index)
            y = self.data['mix']['train']['y'][idx]
            add_feature = self.data['mix']['train']['add_feature'][idx]
            bs = len(lengths)

            # cuda
            x, y, lengths = to_cuda(x, y, lengths)


            # loss
            concatenate = torch.cat((self.embedder.get_embeddings(x, lengths, positions=None, langs=None), 
                torch.from_numpy(add_feature).float().cuda()), 1)
            output = self.proj(concatenate)
            loss = F.cross_entropy(output, y)

            # backward / optimization
            self.optimizer_e.zero_grad()
            self.optimizer_p.zero_grad()
            loss.backward()
            self.optimizer_e.step()
            self.optimizer_p.step()

            # update statistics
            ns += bs
            nw += lengths.sum().item()
            losses.append(loss.item())

            # log
            if ns % (10*bs) < bs:
                logger.info("SENTIFIN - Epoch %i - Train iter %7i - %.1f words/s - Loss: %.4f" % (self.epoch, ns, nw / (time.time() - t), sum(losses) / len(losses)))
                nw, t = 0, time.time()
                losses = []

            # epoch size
            if params.epoch_size != -1 and ns >= params.epoch_size:
                break

    def eval(self, splt):
        params = self.params
        self.embedder.eval()
        self.proj.eval()

        assert splt in ['valid', 'test']
        has_labels = 'y' in self.data['mix'][splt]

        scores = OrderedDict({'epoch': self.epoch})

        idxs = []  # sentence indices
        prob = []  # probabilities
        pred = []  # predicted values
        gold = []  # real values

        for batch in self.get_iterator(splt, 'mix'):
            (x, lengths), idx = batch
            add_feature = self.data['mix'][splt]['add_feature'][idx]
            y = self.data['mix'][splt]['y'][idx] if has_labels else None

            # cuda
            x, y, lengths = to_cuda(x, y, lengths)

            # prediction
            concatenate = torch.cat((self.embedder.get_embeddings(x, lengths, positions=None, langs=None), 
                torch.from_numpy(add_feature).float().cuda()), 1)
            output = self.proj(concatenate)
            p = output.data.max(1)[1]
            idxs.append(idx)
            # prob.append(output.cpu().numpy())
            pred.append(p.cpu().numpy())
            if has_labels:
                gold.append(y.cpu().numpy())

        # indices / predictions
        idxs = np.concatenate(idxs)
        # prob = np.concatenate(prob)
        pred = np.concatenate(pred)
        assert len(idxs) == len(pred), (len(idxs), len(pred))
        assert idxs[-1] == len(idxs) - 1, (idxs[-1], len(idxs) - 1)

        # score the predictions if we have labels
        if has_labels:
            gold = np.concatenate(gold)
            prefix = f'{splt}'
            # if self.is_classif:
            scores['%s_acc' % prefix] = (pred == gold).sum() / len(pred)
            scores['%s_acc' % prefix] = accuracy_score(gold,pred)
            scores['%s_f1' % prefix] = f1_score(gold, pred, average='binary' if params.out_features == 2 else 'micro')
            scores['%s_mc' % prefix] = matthews_corrcoef(gold, pred)
            # else:
            #     scores['%s_prs' % prefix] = 100. * pearsonr(pred, gold)[0]
            #     scores['%s_spr' % prefix] = 100. * spearmanr(pred, gold)[0]
            logger.info("__log__:%s" % json.dumps(scores))

        # output predictions
        pred_path = os.path.join(params.dump_path, f'{splt}.pred.{self.epoch}')
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
            data['dico'] = data.get('dico', data1['dico'])

            # set dictionary parameters
            set_dico_parameters(params, data, data1['dico'])

            max_idx = 1*len(data1['positions'])
            position = data1['positions'][0:int(max_idx)]
            sentence = data1['sentences'][0:(position[len(position)-1][1]+1)]
            # create dataset
            # test = Dataset(data1['sentences'], data1['positions'],params)
            
            if splt== 'train':
                data[lang][splt]['x'] = Dataset(sentence, position,params)
            else :
                data[lang][splt]['x'] = Dataset(data1['sentences'], data1['positions'],params)

            data[lang]['train']['add_feature'] = np.genfromtxt("/home/nanda/nanda/Fix/Dataset/M4_feature_training_new.csv", delimiter=',')
            data[lang]['test']['add_feature'] = np.genfromtxt("/home/nanda/nanda/Fix/Dataset/M4_feature_testing_new.csv", delimiter=',')
            data[lang]['valid']['add_feature'] = np.genfromtxt("/home/nanda/nanda/Fix/Dataset/M4_feature_validation_new.csv", delimiter=',')

            # load labels
            with open(os.path.join(dpath, '%s.label.%s' % (splt, lang)), 'r') as f:
                labels = [label2id[l.rstrip()] for l in f]
            if splt == 'train':
                data[lang][splt]['y'] = torch.LongTensor(labels[0:int(max_idx)])
            else:
                data[lang][splt]['y'] = torch.LongTensor(labels)
            logger.info(splt)
            logger.info((data[lang][splt]['x']))
            logger.info(len(data[lang][splt]['y']))
            assert len(data[lang][splt]['x']) == len(data[lang][splt]['y'])

        return data



