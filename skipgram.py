import os
import time

import numpy as np
import torch
import argparse
import visdom
from nlpfit.preprocessing.nlp_io import read_word_lists
import torch.nn as nn
from torch.autograd import Variable
from collections import deque
import torch.optim as optimizer
import torch.nn.functional as F
import math

from evaluation.analogy_questions import read_analogies, eval_analogy_questions

__author__="Martin Fajčík"
# Small parts of this code were inspired by snippets from
#  https://adoni.github.io/2017/11/08/word2vec-pytorch/

# The wisdom server can be started with command
#  python -m visdom.server

# TODO
# Phrase clustering
# Implement save option
# Vocabulary parsing
# Visualise embedding training via dim-reduced 2D space

# TODO for optimization
# Make batch generator to run in parallel (producer-consumer architecture)
# Precalculate random ints!
# according to cProfile,
# <method 'choice' of 'mtrand.RandomState' objects> took 7% of program time

# FIXME
# Using small number of bytes like 50 for file reading results into failure

class Skipgram(nn.Module):

    def __init__(self, data_proc):
        super(Skipgram, self).__init__()
        self.u_embeddings = nn.Embedding(data_proc.vocab_size, data_proc.embedding_size, sparse=True)
        self.v_embeddings = nn.Embedding(data_proc.vocab_size, data_proc.embedding_size, sparse=True)
        self.data_processor = data_proc
        self.logsigmoid = nn.LogSigmoid()
        self.init_embeddings()
        self.use_cuda = torch.cuda.is_available()
        # according to my benchmarks ~10 times faster with batch 1024
        # self.use_cuda = False
        if self.use_cuda:
            self.cuda()
        self.initial_lr = args.learning_rate
        self.optimizer = optimizer.SparseAdam(self.parameters(),
                                              lr=args.learning_rate)

        if data_proc.visdom:
            self.loss_window = data_proc.visdom.line(X=torch.zeros((1,)).cpu(),
                                                     Y=torch.zeros((1)).cpu(),
                                                     opts=dict(xlabel='Bytes processed',
                                                               ylabel='Loss',
                                                               ytype="log",
                                                               title=f"Training {type(self.optimizer).__name__}, lr={args.learning_rate}",
                                                               legend=['Loss']))

    def init_embeddings(self):
        # Initialize with 0.5/embedding dimension  uniform distribution
        initrange = 0.5 / data_proc.embedding_size
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(0, 0)

    def forward(self, pos_u, pos_v, neg_v):
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
        neg_score = None
        try:
            neg_score = torch.bmm(v_neg_emb_batch, u_emb_batch.unsqueeze(2))
            neg_score = self.logsigmoid(-1. * neg_score)
        except Exception as e:
            print(e)
            print(f"v_emb: {v_neg_emb_batch.shape}")
            print(f"u_emb: {u_emb_batch.unsqueeze(2).shape}")

        return -1. * (torch.sum(score) + torch.sum(neg_score))

    def _train(self, previously_read=0, epoch=0):
        # Calculate expected size of pairs for 1 word
        # pairs_per_word = (self.data_processor.window_size + 1.) / 2.
        # Total expected number of pairs
        # DBG: corpus size of ebooks
        # corpus_size = 812546
        # total_pairs = corpus_size * pairs_per_word
        # batch_count = total_pairs / self.data_processor.batch_size

        # progress_bar = tqdm(range(int(batch_count)))
        batch_gen = data_proc.create_batch_gen(previously_read)
        iteration = 0
        for pos_pairs in batch_gen:
            pos_u = [pair[0] for pair in pos_pairs]
            pos_v = [pair[1] for pair in pos_pairs]
            # prog,ress_bar.update(1)
            neg_v = self.data_processor.get_neg_v_neg_sampling()

            pos_u = Variable(torch.LongTensor(pos_u))
            pos_v = Variable(torch.LongTensor(pos_v))
            neg_v = Variable(torch.LongTensor(neg_v))
            if self.use_cuda:
                pos_u = pos_u.cuda()
                pos_v = pos_v.cuda()
                neg_v = neg_v.cuda()

            self.optimizer.zero_grad()
            loss = self.forward(pos_u, pos_v, neg_v)
            loss.backward()
            self.optimizer.step()
            if iteration % 200 == 0:
                if self.data_processor.visdom:
                    self.data_processor.visdom.line(
                        X=(torch.ones((1, 1)).cpu() * self.data_processor.bytes_read).squeeze(1),
                        Y=torch.Tensor([loss.data]).cpu(),
                        win=self.loss_window,
                        update='append')
                if iteration % 10000 == 0:
                    print(f"\nEpoch {epoch}, Loss: {loss.data}")
                    if self.data_processor.analogy_questions is not None:
                        eval_analogy_questions(data_processor= self.data_processor,
                                               u_embeddings=self.u_embeddings,
                                               use_cuda = self.use_cuda)

                    if iteration % 50000 == 0:
                        print("\nSANITY CHECK")
                        print(
                            "----------------------------------------------------------------------------------------------------------------------------------")
                        for testword in self.data_processor.sanitycheck:
                            print(f"Nearest words to '{testword}' are: {', '.join(self.find_nearest(testword))}")
                        print(
                            "----------------------------------------------------------------------------------------------------------------------------------")

            iteration += 1
        return self.data_processor.bytes_read

    def find_nearest(self, word, k=10):
        nembs = torch.transpose(F.normalize(self.u_embeddings.weight), 0, 1)
        word_id = torch.LongTensor([self.data_processor.w2id[word]])
        if self.use_cuda:
            word_id = word_id.cuda()
        embedding = self.u_embeddings(word_id)
        dist = torch.matmul(embedding, nembs)

        top_predicted = torch.topk(dist, dim=1, k=k + 1)[1].cpu().numpy().tolist()[0][1:]
        return list(map(lambda x: self.data_processor.id2w[x], top_predicted))

    def find_nearest_emb(self, embedding, k=10):
        nembs = torch.transpose(F.normalize(self.u_embeddings.weight), 0, 1)
        dist = torch.matmul(embedding, nembs)

        top_predicted = torch.topk(dist, dim=1, k=k + 1)[1].cpu().numpy().tolist()[0][1:]
        return list(map(lambda x: self.data_processor.id2w[x], top_predicted))

    def translate_emb(self, embedding):
        nembs = torch.transpose(F.normalize(self.u_embeddings.weight), 0, 1)
        dist = torch.matmul(embedding, nembs)
        id = torch.topk(dist, dim=1, k=1)[1].cpu().numpy().tolist()[0][0]
        return self.data_processor.id2w[id]

    def save(self):
        pass


class DataProcessor():
    def __init__(self, args):
        self.min_freq = args.min_freq
        self.bytes_to_read = args.bytes_to_read
        self.corpus = args.corpus
        self.vocab_path = args.vocab
        self.batch_size = args.batch_size
        self.window_size = args.window
        self.thresold = args.subsfqwords_tr
        self.randints_to_precalculate = args.random_ints
        self.nsamples = args.nsamples
        self.embedding_size = args.dimension
        self.sanitycheck = args.sanitycheck.split()
        self.visdom_enabled = args.visdom

        if self.visdom_enabled:
            self.visdom = visdom.Visdom()

        # Load corpus vocab, and calculate prerequisities
        self.frequency_vocab_with_OOV = self.load_vocab() if args.vocab else self.parse_vocab()
        self.corpus_size = self.calc_corpus_size()
        # Precalculate term used in subsampling of frequent words
        self.t_cs = self.thresold * self.corpus_size

        self.frequency_vocab = self.calc_frequency_vocab()
        self.vocab_size = len(self.frequency_vocab) + 1  # +1 For unknown

        self.sample_table = self.init_sample_table()

        # Create id mapping used for fast U embedding matrix indexing
        self.w2id = self.create_w2id()
        self.id2w = {v: k for k, v in self.w2id.items()}

        # Preload eval analogy questions
        if args.eval_aq:
            self.eval_data_aq = args.eval_aq
            self.analogy_questions = read_analogies(file=self.eval_data_aq,w2id=self.w2id)

        self.cnt = 0
        self.benchmarktime = time.time()
        self.bytes_read = 0

    # For fast negative sampling
    def init_sample_table(self):
        self.sample_table = []
        sample_table_size = 1e8

        # Create proper uniform distribution raised on 3/4
        pow_frequency = np.array(list(self.frequency_vocab.values())) ** 0.75
        normalizer = sum(pow_frequency)
        normalized_freqs = pow_frequency / normalizer

        # Calculate how much table cells should each distribution element have
        table_distribution = np.round(normalized_freqs * sample_table_size)

        # Create vector table, holding number of items with element ID proprotional
        # to element id's probability in distribution\

        for wid, c in enumerate(table_distribution):
            self.sample_table += [wid] * int(c)
        return np.array(self.sample_table)

    def get_neg_v_neg_sampling(self):
        neg_v = np.random.choice(
            self.sample_table, size=(self.batch_size, self.nsamples)).tolist()
        return neg_v

    # This formula is not exactly the one from the original paper,
    # but it is inspired from tensorflow/models skipgram implementation.
    # The shape of this subsampling function is in fact similar, but
    # it's new behavior now adds relation to the corpus size to the formula
    # and also "it works with the large numbers" from frequency vocab
    # Also see my SO question&answer: https://stackoverflow.com/questions/49012064/skip-gram-implementation-in-tensorflow-models-subsampling-of-frequent-words
    def should_be_subsampled(self, w):
        f = self.frequency_vocab_with_OOV[w]
        keep_prob = (np.sqrt(f / self.t_cs) + 1.) * (self.t_cs / f)
        roll = np.random.uniform()
        return not keep_prob > roll

    def create_batch_gen(self, previously_read=0):
        fsize = os.path.getsize(self.corpus)
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
            self.bytes_read = wlist_[1]  # + previously_read

            # print(word_pairs)
            self.cnt += 1
            if self.cnt % 5000 == 0:
                t = time.time()
                p = t
                # Derive epoch from bytes read
                total_size = fsize * (math.floor(self.bytes_read / fsize) + 1)
                print(
                    f"Time: {p/60:.2f} min - epoch state {self.bytes_read/total_size *100:.2f}% ({int(self.bytes_read/p/1e3)} KB/s)")

            # Discard words with min_freq or less occurences
            # Subsample of Frequent Words
            # hese words are removed from the text before generating the contexts
            wlist_clean = []
            for w in wlist:
                try:
                    if not (self.frequency_vocab_with_OOV[w] < self.min_freq or self.should_be_subsampled(w)):
                        wlist_clean.append(w)
                except KeyError as e:
                    print("Encountered unknown word!")
                    print(e)
                    print(f"Wlist: {wlist}")
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
                    m = max(i - self.window_size, 0)
                    si = i - m
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
                yield word_pairs[:self.batch_size]
                word_pairs = word_pairs[self.batch_size:]

    def load_vocab(self):
        from nlpfit.preprocessing.tools import read_frequency_vocab
        print("Loading vocabulary...")
        return read_frequency_vocab(self.vocab_path)

    def parse_vocab(self):
        # TODO: implement
        pass

    def calc_corpus_size(self):
        return sum(self.frequency_vocab_with_OOV.values())

    def calc_frequency_vocab(self):
        fvocab = dict()
        fvocab['UNK'] = 0
        for k, v in self.frequency_vocab_with_OOV.items():
            if v >= self.min_freq:
                fvocab[k] = v
        return fvocab

    def create_w2id(self):
        w2id = dict()
        w2id['UNK'] = 0
        for i, k in enumerate(self.frequency_vocab, start=1):
            w2id[k] = i
        return w2id

def init_parser(parser):
    parser.add_argument("-c", "--corpus", help="input data corpus", required=True)
    parser.add_argument("--vocab", help="precalculated vocabulary")
    parser.add_argument("--eval_aq", "--eval_analogy_questions",
                        help="file with analogy questions to do the evaluation on", default=None)
    parser.add_argument("-v", "--verbose", help="increase the model verbosity", action="store_true")
    parser.add_argument("-w", "--window", help="size of a context window",
                        default=5)
    parser.add_argument("-ns", "--nsamples", help="number of negative samples",
                        default=25)
    parser.add_argument("-mf", "--min_freq", help="minimum frequence of occurence for a word",
                        default=5)
    parser.add_argument("-lr", "--learning_rate", help="initial learning rate",
                        default=0.0025  # 10x smaller than used by Tomas Mikolov, because we use SparseAdam, not the SGD
                        )
    parser.add_argument("-d", "--dimension", help="size of the embedding dimension",
                        default=300)
    parser.add_argument("-br", "--bytes_to_read", help="how much bytes to read from corpus file per chunk",
                        default=512)
    parser.add_argument("-bs", "--batch_size", help="size of 1 batch in training iteration", default=512)
    parser.add_argument("-pc", "--phrase_clustering", help="enable phrase clustering as described by Mikolov (i.e. New York becomes New_York)",
                        default=True)
    parser.add_argument("-ri", "--random_ints",
                        help="how many random ints for window subsampling to precalculate at once",
                        default=1310720  # 5 megabytes of int32s
                        )
    parser.add_argument("-tr", "--subsfqwords_tr", help="subsample frequent words threshold", default=1e-4)
    parser.add_argument("--visdom", help="visualize training via visdom_enabled library", default=True)
    parser.add_argument("--sanitycheck",
                        help='list of words for which the nearest word embeddings are found during training, '
                             'serves as sanity check, i.e. "dog family king eye"',
                        default="dog family king eye")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    init_parser(parser)
    args = parser.parse_args()

    data_proc = DataProcessor(args)
    skipgram_model = Skipgram(data_proc)

    # We need to carefully choose optimizer and its parameters to guarantee no global update will be excuted when training.
    # For example, parameters like weight_decay and momentum in torch.optim. SGD require the global calculation
    # on embedding matrix, which is extremely time-consuming.
    bytes_read = 0
    for e in range(10):
        print(f"Starting epoch: {e}")
        bytes_read = skipgram_model._train(previously_read=bytes_read, epoch=e)
    # TODO: Implement save functionality
    # skipgram_model.save()
