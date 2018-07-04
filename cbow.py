# Created as experimental part of my research at
# FIT@BUT 2018
#
__modelname__ = "CBOW"
__author__ = "Martin Fajčík"

import sys
import argparse
import os
import pickle
import numpy as np
import torch
import torch.nn as nn

from nlpfit.other.logging_config import init_logging
from nlpfit.preprocessing.nlp_io import read_word_lists
from collections import deque
from word2vec import init_argparser_general, DataProcessor, Word2Vec


class CBOW(Word2Vec):

    def create_embedding_matrices(self):
        self.u_embeddings = nn.EmbeddingBag(num_embeddings=data_proc.vocab_size,
                                            embedding_dim=data_proc.embedding_size,
                                            sparse=True)
        self.v_embeddings = nn.Embedding(data_proc.vocab_size, data_proc.embedding_size, sparse=True)
        self.init_embeddings(self.u_embeddings, self.v_embeddings)

    def forward(self, batch):
        ##########################
        # Parse input from batch
        ##########################

        # batch contains list that looks like
        # i.e. [([xx],y),([aaa],y2),([bbbb],y3)
        # since we are using embedding bag, we need to parse this example format where:

        # pos: flatten list of all word sequences i.e. [xxaaabbbb]
        # indices: indices of split points in pos i.e. [0,2,5]
        # targets: [y1,y2,y3]

        indices = [0]
        for p in batch:
            indices.append(len(p[0]) + indices[-1])
        indices = indices[:-1]

        targets = torch.LongTensor([item[1] for item in batch])
        pos = torch.LongTensor([item for sublist in batch for item in sublist[0]])
        indices = torch.LongTensor(indices)
        neg_v = self.dp.get_neg_v_neg_sampling()
        neg_v = torch.LongTensor(neg_v)

        if self.use_cuda:
            pos = pos.cuda()
            targets = targets.cuda()
            indices = indices.cuda()
            neg_v = neg_v.cuda()

        # Forward pass
        # Input format:
        #     pos: list of neighbor word ids for positive word samples.
        #     indices: split points in pos list for each sample
        #     targets: list of center word ids for positive word samples.
        #     neg_v: list of neighbor word ids for negative word samples.
        # Returns:
        #     Loss of this process, a pytorch variable.
        #
        # The sizes of input variables are as following:
        #     pos: [batch_size, random_window_size]
        #     indices: [batch_size]
        #     targets:
        #     neg_v: [batch_size, neg_sampling_count]

        # pick embeddings for words pos_u, pos_v
        u_emb_batch = self.u_embeddings(pos, indices)
        v_emb_batch = self.v_embeddings(targets)

        # o is sigmoid function
        # NS loss for 1 sample and max objective is
        ##########################################################
        # log o(v^T*u) + sum of k samples log o(-negative_v^T *u)#
        ##########################################################
        # log o(v^T*u)  = score
        # sum of k samples log o(-negative_v^T *u) = neg_score

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


class CBDataProcessor(DataProcessor):
    def create_batch_gen(self):
        # Create word list generator
        wordgen = read_word_lists(self.corpus, bytes_to_read=self.bytes_to_read, report_bytesread=True)
        # Create queue of random choices
        rchoices = deque(np.random.choice(np.arange(1, self.window_size + 1), self.randints_to_precalculate))
        # create doubles
        word_from_last_list = []
        window_datasamples = []
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
                    self.logging.critical("Encountered unknown word!")
                    self.logging.critical(e)
                    self.logging.critical(f"Wlist: {wlist}")
            wlist = wlist_clean

            # TODO: Phrase clustering here

            if not wlist:
                return

            wlist = list(map(lambda x: self.w2id[x], wlist))
            wlist = word_from_last_list + wlist
            word_from_last_list = []
            for i in range(si, len(wlist)):
                # if the window exceeds the buffered part
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
                if i - r < 0:
                    continue
                window_datasamples.append((wlist[i - r:i] + wlist[i + 1:i + r + 1], wlist[i]))

            if len(window_datasamples) > self.batch_size:
                self.log_epoch_progress()
                yield window_datasamples[:self.batch_size]
                window_datasamples = window_datasamples[self.batch_size:]


def init_argparser_cbow(parser):
    # Obligatory arguments
    # Optional switch arguments
    # Optional arguments with value
    parser.add_argument("-lr", "--learning_rate", help="initial learning rate", default=0.005)

    ## Step count parameters
    parser.add_argument("--lossreport_step", help="number of steps after which loss value is reported",
                        default=5000)
    parser.add_argument("--epoch_state_step", help="number of steps after epoch progress is reported", default=1000)
    parser.add_argument("--eval_aq_step",
                        help="number of steps after which analogy questions evaluation if performed",
                        default=5000)
    parser.add_argument("--eval_intrx_step", help="number of steps after which intrinstric evaluation if performed",
                        default=5000)
    parser.add_argument("--sanity_check_step", help="number of steps after which sanity check if performed",
                        default=5000)
    parser.add_argument("--eval_extrx_step", help="number of steps after which extrinstric evaluation if performed",
                        default=13000)
    parser.add_argument("--visdom_step", help="number of steps after which data are recorded in visdom",
                        default=1000)
    parser.add_argument("--tensorboard_step",
                        help="number of steps after which embeddings are dumped for tensorboard",
                        default=7500)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    init_argparser_general(parser)
    init_argparser_cbow(parser)
    args = parser.parse_args()
    logging = init_logging(os.path.basename(sys.argv[0]).split(".")[0], logpath=args.logging)

    data_proc = CBDataProcessor(args, __modelname__, logging=logging)
    cbow_model = CBOW(data_proc)

    # We need to carefully choose optimizer and its parameters to guarantee no global update will be excuted when training.
    # For example, parameters like weight_decay and momentum in torch.optim. SGD require the global calculation
    # on embedding matrix, which is extremely time-consuming.
    bytes_read = 0
    epochs = 100
    for e in range(epochs):
        logging.info(f"Starting epoch: {e}")
        bytes_read = cbow_model._train(previously_read=bytes_read, epoch=e)

    with open(f"trained/u_embeddings_e{epochs}.pkl", "wb") as f:
        pickle.dump(cbow_model.u_embeddings.weight, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"trained/v_embeddings_e{epochs}.pkl", "wb") as f:
        pickle.dump(cbow_model.v_embeddings.weight, f, protocol=pickle.HIGHEST_PROTOCOL)
    cbow_model.save(f"trained/embeddings_test_e{epochs}.vec")
