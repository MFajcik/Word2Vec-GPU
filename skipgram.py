# Created as experimental part of my research at
# FIT@BUT 2018
#
from evaluation.intrinstric_evaluation.wordsim.wordsim import intrinstric_eval

__modelname__ = "Skipgram"
__author__ = "Martin Fajčík"

import argparse
import numpy as np
import torch
import logging
import torch.nn as nn

from collections import deque
from nlpfit.preprocessing.nlp_io import read_word_lists
from word2vec import init_argparser_general, DataProcessor, Word2Vec, init_logging


class Skipgram(Word2Vec):
    def intristric_eval(self):
        return intrinstric_eval(self.u_embeddings, self.dp.w2id, use_cuda=self.use_cuda)

    def create_embedding_matrices(self):
        # create U embedding (target word) matrix
        self.u_embeddings = nn.Embedding(self.dp.vocab_size, self.dp.embedding_size, sparse=True)
        # create V embedding (context word) matrix
        if self.dp.share_weights:
            # share weights in case of shared weigts experiment
            self.v_embeddings = self.u_embeddings
        else:
            self.v_embeddings = nn.Embedding(self.dp.vocab_size, self.dp.embedding_size, sparse=True)
        self.init_embeddings(self.u_embeddings, self.v_embeddings)

    def forward(self, batch):
        """Forward process.
        As pytorch designed, all variables must be batch format, so all input of this method is a list of word id.
        Args:
            pos_u: list of center word ids for positive word pairs.
            pos_v: list of neighbor word ids for positive word pairs.
            neg_v: list of neighbor word ids for negative word pairs.
        Returns:
            Loss of this process, a pytorch variable.

        The sizes of input variables are as following:
            pos_u: [batch_size]
            pos_v: [batch_size]
            neg_v: [batch_size, neg_sampling_count]
        """

        pos_u = torch.LongTensor([pair[0] for pair in batch])
        pos_v = torch.LongTensor([pair[1] for pair in batch])
        neg_v = torch.LongTensor(self.dp.get_neg_v_neg_sampling())
        if self.use_cuda:
            pos_u = pos_u.cuda()
            pos_v = pos_v.cuda()
            neg_v = neg_v.cuda()

        # pick embeddings for words pos_u, pos_v
        u_emb_batch = self.u_embeddings(pos_u)
        v_emb_batch = self.v_embeddings(pos_v)

        # o is sigmoid function
        # NS loss for 1 sample and max objective is
        ##########################################################
        # log o(v^T*u) + sum of k samples log o(-negative_v^T *u)#
        ##########################################################

        # Multiply element wise
        score = torch.mul(u_emb_batch, v_emb_batch)
        # Sum so we get dot product for each row
        score = torch.sum(score, dim=1)
        score = self.logsigmoid(score)
        v_neg_emb_batch = self.v_embeddings(neg_v)

        # v_neg_emb_batch has shape [BATCH_SIZE,NUM_OF_NEG_SAMPLES,EMBEDDING_DIMENSIONALITY]
        # u_emb_batch has shape [BATCH_SIZE,EMBEDDING_DIMENSIONALITY]
        neg_score = torch.bmm(v_neg_emb_batch, u_emb_batch.unsqueeze(2))
        neg_score = self.logsigmoid(-1. * neg_score)

        return -1. * (torch.sum(score) + torch.sum(neg_score)) / self.dp.batch_size


class WordTargetDataProcessor(DataProcessor):
    """
    This dataprocessor creates batches of words and their contexts
    So each sample has pattern `(t,c)`, where `t` is the target word and `c` is the context word
    """

    def create_batch_gen(self):
        # Create word list generator
        wordgen = read_word_lists(self.corpus, bytes_to_read=self.bytes_to_read, report_bytesread=True)
        # Create queue of random choices
        rchoices = deque(np.random.choice(np.arange(1, self.window_size + 1), self.randints_to_precalculate))
        # create doubles
        word_from_last_list = []
        word_pairs = []
        si = 0
        for wlist_ in wordgen:
            wlist = wlist_[0]
            self.bytes_read = wlist_[1]
            # Discard words with min_freq or less occurences
            # Subsample of Frequent Words
            # hese words are removed from the text before generating the contexts
            wlist_clean = []
            for w in wlist:
                try:
                    if not (self.frequency_vocab_with_OOV[w] < self.min_freq or self.should_be_subsampled(w)):
                        wlist_clean.append(w)
                except KeyError as e:
                    logging.error("Encountered unknown word!")
                    logging.error(e)
                    logging.error(f"Wlist: {wlist}")
            wlist = wlist_clean

            # TODO: Phrase clustering here
            # FIXME: Using small number of bytes like 50 for file reading results into failure, WHY?

            if not wlist:
                # This happens if we reached the end of dataset
                # Maybe this is redundant code...
                while word_from_last_list:
                    if not rchoices:
                        rchoices = deque(
                            np.random.choice(np.arange(1, self.window_size + 1), self.randints_to_precalculate))
                    r = rchoices.pop()
                    for i in range(len(word_from_last_list)):
                        for c in range(-r, r + 1):
                            if c == 0 or i + c < 0:
                                continue
                            elif i + c > len(word_from_last_list) - 1:
                                break
                            elif len(word_pairs) == self.batch_size:
                                word_from_last_list = word_from_last_list[i:]
                                break
                            word_pairs.append((word_from_last_list[i], word_from_last_list[i + c]))
                        if i == len(word_from_last_list) - 1:
                            word_from_last_list = []
                    for _ in range(self.batch_size - len(word_pairs)):
                        word_pairs.append((0, 0))
                    assert len(word_pairs) == self.batch_size
                    yield word_pairs
                return

            wlist = list(map(lambda x: self.w2id[x], wlist))
            wlist = word_from_last_list + wlist
            word_from_last_list = []
            for i in range(si, len(wlist)):
                if (i + self.window_size > len(wlist) - 1):
                    # find index m, that points on leftmost word still in a window
                    # of central word
                    m = max(i - self.window_size, 0)

                    # save the index of central word, with respect to start at leftmost word at position m
                    si = i - m

                    # throw away words before leftmost word, they have already been processed
                    word_from_last_list = wlist[m:]
                    break
                if not rchoices:
                    rchoices = deque(
                        np.random.choice(np.arange(1, self.window_size + 1), self.randints_to_precalculate))
                r = rchoices.pop()
                for c in range(-r, r + 1):
                    if c == 0 or i + c < 0:
                        continue
                    word_pairs.append((wlist[i], wlist[i + c]))
            if len(word_pairs) > self.batch_size:
                self.log_epoch_progress()
                yield word_pairs[:self.batch_size]
                word_pairs = word_pairs[self.batch_size:]


def init_argparser_skipgram(parser):
    # Obligatory arguments
    # Optional switch arguments
    # Optional arguments with value
    parser.add_argument("-lr", "--learning_rate", help="initial learning rate",
                        default=0.0025
                        # 10x smaller than used by Tomas Mikolov, because we use SparseAdam, not the SGD
                        )

    ## Step count parameters
    parser.add_argument("--lossreport_step", help="number of steps after which loss value is reported",
                        default=20000)
    parser.add_argument("--epoch_state_step", help="number of steps after epoch progress is reported", default=1000)
    parser.add_argument("--eval_aq_step",
                        help="number of steps after which analogy questions evaluation if performed",
                        default=20000)
    parser.add_argument("--eval_intrx_step", help="number of steps after which intrinstric evaluation if performed",
                        default=20000)
    parser.add_argument("--sanity_check_step", help="number of steps after which sanity check if performed",
                        default=20000)
    parser.add_argument("--eval_extrx_step", help="number of steps after which extrinstric evaluation if performed",
                        default=50000)
    parser.add_argument("--visdom_step", help="number of steps after which data are recorded in visdom",
                        default=1000)
    parser.add_argument("--tensorboard_step",
                        help="number of steps after which embeddings are dumped for tensorboard",
                        default=30000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    init_argparser_general(parser)
    init_argparser_skipgram(parser)
    args = parser.parse_args()
    init_logging(args)

    with WordTargetDataProcessor(args, __modelname__) as data_proc:
        skipgram_model = Skipgram(data_proc)
        bytes_read = 0
        epochs = 100
        for e in range(epochs):
            logging.info(f"Starting epoch: {e}")
            bytes_read = skipgram_model._train(previously_read=bytes_read, epoch=e)
        skipgram_model.save(f"trained/embeddings_e{epochs}.vec")
