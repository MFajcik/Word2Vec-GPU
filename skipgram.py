# Created as experimental part of my research at
# FIT@BUT 2018
#

import os
import sys
import time
import numpy as np
import torch
import argparse
import visdom
import torch.nn as nn
import torch.optim as optimizer
import torch.nn.functional as F
import math

from collections import deque
from tensorboardX import SummaryWriter
from nlpfit.preprocessing.nlp_io import read_word_lists
from nlpfit.other.logging_config import init_logging
from nlpfit.preprocessing.tools import read_frequency_vocab

from evaluation.analogy_questions.analogy_questions import read_analogies, eval_analogy_questions
from evaluation.intrinstric_evaluation.wordsim.wordsim import intrinstric_eval

__author__ = "Martin Fajčík"


# The wisdom server can be started with command
# python -m visdom.server

# Tensorboard can be started in project's home directory with
# tensorboard --logdir runs

# TODO
# Phrase clustering
# Vocabulary parsing
# Evaluate solution on extrinstric properties

# TODO for optimization
# Make batch generator to run in parallel (producer-consumer architecture)
# Precalculate random ints!
# according to cProfile,
# <method 'choice' of 'mtrand.RandomState' objects> took 7% of program time


# FIXME
# Using small number of bytes like 50 for file reading results into failure

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Skipgram(nn.Module):
    def __init__(self, data_proc):
        super(Skipgram, self).__init__()
        self.dp = data_proc

        # create U embedding (target word) matrix
        self.u_embeddings = nn.Embedding(data_proc.vocab_size, data_proc.embedding_size, sparse=True)

        # create V embedding (context word) matrix
        if data_proc.share_weights:
            # share weights in case of shared weigts experiment
            self.v_embeddings = self.u_embeddings
        else:
            self.v_embeddings = nn.Embedding(data_proc.vocab_size, data_proc.embedding_size, sparse=True)

        # NS loss uses sigmoid
        self.logsigmoid = nn.LogSigmoid()

        self.init_embeddings()

        self.initial_lr = args.learning_rate
        logging.info(f"Optimizing {count_parameters(self)} parameters!")
        self.optimizer = optimizer.SparseAdam(self.parameters(),
                                              lr=args.learning_rate)

        # Move everything on GPU, if possible
        self.use_cuda = torch.cuda.is_available()
        # according to my benchmarks ~10 times faster on GPU 1080Ti with batch 1024
        # BUT with proper CPU optimization, CPU should be faster (see blog about GenSim)

        if self.use_cuda:
            self.cuda()

        if self.dp.tensorboard_enabled:
            self.global_step = 0
        if data_proc.visdom_enabled:
            self.loss_window = data_proc.visdom.line(X=torch.zeros((1,)).cpu(),
                                                     Y=torch.zeros((1)).cpu(),
                                                     opts=dict(xlabel='Bytes processed',
                                                               ylabel='Loss',
                                                               ytype="log",
                                                               title=f"Model: Skipgram, TM: {type(self.optimizer).__name__}, lr={args.learning_rate}",
                                                               legend=['Loss']))

    def init_embeddings(self):
        # Initialize with 0.5/embedding dimension  uniform distribution
        initrange = 0.5 / data_proc.embedding_size
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        if not self.dp.share_weights:
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
        neg_score = torch.bmm(v_neg_emb_batch, u_emb_batch.unsqueeze(2))
        neg_score = self.logsigmoid(-1. * neg_score)

        return -1. * (torch.sum(score) + torch.sum(neg_score)) / self.dp.batch_size

    def _train(self, previously_read=0, epoch=0):
        batch_gen = data_proc.create_batch_gen()
        iteration = 0
        for pos_pairs in batch_gen:
            pos_u = [pair[0] for pair in pos_pairs]
            pos_v = [pair[1] for pair in pos_pairs]
            neg_v = self.dp.get_neg_v_neg_sampling()

            pos_u = torch.LongTensor(pos_u)
            pos_v = torch.LongTensor(pos_v)
            neg_v = torch.LongTensor(neg_v)
            if self.use_cuda:
                pos_u = pos_u.cuda()
                pos_v = pos_v.cuda()
                neg_v = neg_v.cuda()
            # Zero gradient
            self.optimizer.zero_grad()
            # Do forward pass
            loss = self.forward(pos_u, pos_v, neg_v)
            # Calculate gradients
            loss.backward()
            # Perform optimization step
            self.optimizer.step()
            # Validate results on various metrics
            self.validate_step(epoch, loss, iteration)
            # Log/Visualise current learning state
            self.log_step(epoch, loss, iteration, previously_read=previously_read)

            iteration += 1
        return self.dp.bytes_read + previously_read

    def validate_step(self, epoch, loss, iteration):

        if iteration % self.dp.lossreport_step == 0:
            logging.info(f"Epoch {epoch}, Loss: {loss.data}")

        # Simple sanity check shows nearest words for
        # words in self.data_processor.sanitycheck
        if iteration % self.dp.sanity_step == 0:
            if self.dp.sanity_check_enabled:
                self.run_sanity_check()

        # Evaluate solution on analogy questions task
        if iteration % self.dp.eval_aq_step == 0:
            if self.dp.analogy_questions is not None:
                eval_analogy_questions(data_processor=self.dp,
                                       embeddings=self.u_embeddings,
                                       use_cuda=self.use_cuda)

        ################################################################################################
        # Evaluate solution for intrinstric word similarity properties on following tasks
        # - [MC-30](http://www.tandfonline.com/doi/pdf/10.1080/01690969108406936)
        # - [MEN-TR](http://clic.cimec.unitn.it/~elia.bruni/MEN.html)
        # - [MTurk-287](http://tx.technion.ac.il/~kirar/files/Radinsky-TemporalSemantics.pdf)
        # - [MTurk-771](http://dl.acm.org/citation.cfm?id=1963455)
        # - [RG-65](http://dl.acm.org/citation.cfm?id=365657)
        # - [RW-STANFORD](http://nlp.stanford.edu/~lmthang/data/papers/conll13_morpho.pdf)
        # - [WS-353-ALL](http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/)
        # - [WS-353-REL](http://alfonseca.org/eng/research/wordsim353.html)
        # - [WS-353-SIM](http://alfonseca.org/eng/research/wordsim353.html)
        # - [YP-130](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.214.7538&rep=rep1&type=pdf)
        if iteration % self.dp.eval_intrx_step == 0:
            if self.dp.eval_intrinstric:
                intrinstric_eval(self.u_embeddings, self.dp.w2id, use_cuda=self.use_cuda)

        # Evaluate solution on extrinstric properties
        # TODO: Implement
        if iteration % self.dp.eval_extrx_step == 0:
            pass

    def run_sanity_check(self):
        logging.info("\nSANITY CHECK")
        logging.info(
            "----------------------------------------------------------------------------------------------------------------------------------")
        for testword in self.dp.sanitycheck:
            logging.info(f"Nearest words to '{testword}' are: {', '.join(self.find_nearest(testword))}")
        logging.info(
            "----------------------------------------------------------------------------------------------------------------------------------")

    def log_step(self, epoch, loss, iteration, previously_read=0):
        if iteration % self.dp.visdom_step == 0:
            if self.dp.visdom_enabled:
                self.dp.visdom.line(
                    X=(torch.ones((1, 1)).cpu() * (self.dp.bytes_read + previously_read)).squeeze(1),
                    Y=torch.Tensor([loss.data]).cpu(),
                    win=self.loss_window,
                    update='append')

        if iteration % self.dp.tensorboard_step == 0:
            if self.dp.tensorboard_enabled:
                tag = f"Skipgram_UEMB_Epoch_{epoch}_iter_{iteration}"
                logging.info(f"Saving U embeddings {tag} for tensorboard...")
                self.dp.writer.add_embedding(self.u_embeddings.weight,
                                             metadata=list(
                                                 self.dp.frequency_vocab.keys()),
                                             tag=tag,
                                             global_step=self.global_step)
                self.global_step += 1

    def find_nearest(self, word, k=10):
        nembs = torch.transpose(F.normalize(self.u_embeddings.weight), 0, 1)
        word_id = torch.LongTensor([self.dp.w2id[word]])
        if self.use_cuda:
            word_id = word_id.cuda()
        embedding = self.u_embeddings(word_id)
        dist = torch.matmul(embedding, nembs)

        top_predicted = torch.topk(dist, dim=1, k=k + 1)[1].cpu().numpy().tolist()[0][1:]
        return list(map(lambda x: self.dp.id2w[x], top_predicted))

    def find_nearest_emb(self, embedding, k=10):
        nembs = torch.transpose(F.normalize(self.u_embeddings.weight), 0, 1)
        dist = torch.matmul(embedding, nembs)

        top_predicted = torch.topk(dist, dim=1, k=k + 1)[1].cpu().numpy().tolist()[0][1:]
        return list(map(lambda x: self.dp.id2w[x], top_predicted))

    def translate_emb(self, embedding):
        nembs = torch.transpose(F.normalize(self.u_embeddings.weight), 0, 1)
        dist = torch.matmul(embedding, nembs)
        id = torch.topk(dist, dim=1, k=1)[1].cpu().numpy().tolist()[0][0]
        return self.dp.id2w[id]

    # The vec file is a text file that contains the word vectors, one per line for each word in the vocabulary.
    # The first line is a header containing the number of words and the dimensionality of the vectors.
    # Subsequent lines are the word vectors for all words in the vocabulary, sorted by decreasing frequency.
    # Example:
    # 218316 100
    # the -0.10363 -0.063669 0.032436 -0.040798...
    # of -0.0083724 0.0059414 -0.046618 -0.072735...
    # one 0.32731 0.044409 -0.46484 0.14716...
    def save(self, vec_path):
        vocab_size = self.dp.vocab_size
        embedding_dimension = self.dp.embedding_size
        # Using linux file endings
        with open(vec_path, 'w') as f:
            logging.info("Saving .vec file to {}".format(vec_path))
            f.write("{} {}\n".format(vocab_size, embedding_dimension))
            for word, id in self.dp.w2id.items():
                tensor_id = torch.LongTensor([id])
                if self.use_cuda:
                    tensor_id = tensor_id.cuda()
                embedding = self.u_embeddings(tensor_id).cpu().squeeze(0).detach().numpy()
                f.write("{} {}\n".format(word, ' '.join(map(str, embedding))))


class DataProcessor:

    def __enter__(self):
        return self

    def __init__(self, args):
        self.min_freq = args.min_freq
        self.bytes_to_read = args.bytes_to_read
        self.corpus = args.corpus
        self.vocab_path = args.vocab
        self.batch_size = args.batch_size
        self.window_size = args.window
        self.threshold = args.subsfqwords_tr
        self.randints_to_precalculate = args.random_ints
        self.nsamples = args.nsamples
        self.embedding_size = args.dimension
        self.share_weights = args.shareweights

        self.sanitychecklist = args.sanitychecklist.split()

        self.sanity_check_enabled = args.sanity_check
        self.visdom_enabled = args.visdom
        self.tensorboard_enabled = args.tensorboard

        self.sanity_step = args.sanity_check_step
        self.lossreport_step = args.lossreport_step
        self.eval_aq_step = args.eval_aq_step
        self.eval_intrx_step = args.eval_intrx_step
        self.eval_extrx_step = args.eval_extrx_step
        self.visdom_step = args.visdom_step
        self.tensorboard_step = args.tensorboard_step
        self.epoch_state_step = args.epoch_state_step

        if self.visdom_enabled:
            self.visdom = visdom.Visdom()

        if self.tensorboard_enabled:
            self.writer = SummaryWriter(comment="Skipgram training")

        # Load corpus vocab, and calculate prerequisities
        self.frequency_vocab_with_OOV = self.load_vocab() if args.vocab else self.parse_vocab()
        self.corpus_size = self.calc_corpus_size()
        # Precalculate term used in subsampling of frequent words
        self.t_cs = self.threshold * self.corpus_size

        self.frequency_vocab = self.calc_frequency_vocab()
        self.vocab_size = len(self.frequency_vocab)  # + 1  # +1 For unknown

        self.sample_table = self.init_sample_table()

        # Create id mapping used for fast U embedding matrix indexing
        self.w2id = self.create_w2id()
        self.id2w = {v: k for k, v in self.w2id.items()}

        # Preload eval analogy questions
        if args.eval_aq:
            self.eval_data_aq = args.eval_aq
            self.analogy_questions = read_analogies(file=self.eval_data_aq, w2id=self.w2id)

        self.eval_intrinstric = args.eval_intrinstric

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

    def create_batch_gen(self):
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
            self.bytes_read = wlist_[1]
            self.cnt += 1
            if self.cnt % self.epoch_state_step == 0:
                t = time.time()
                p = t - self.benchmarktime
                # Derive epoch from bytes read
                total_size = fsize * (math.floor(self.bytes_read / fsize) + 1)
                logging.info(
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
                    logging.critical("Encountered unknown word!")
                    logging.critical(e)
                    logging.critical(f"Wlist: {wlist}")
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
                yield word_pairs[:self.batch_size]
                word_pairs = word_pairs[self.batch_size:]

    def load_vocab(self):
        logging.info("Loading vocabulary...")
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
        for i, k in enumerate(self.frequency_vocab, start=0):
            w2id[k] = i
        return w2id

    def __exit__(self, exc_type, exc_value, traceback):
        self.writer.close()


def init_parser(parser):
    # Obligatory arguments
    parser.add_argument("-c", "--corpus", help="input data corpus", required=True)
    parser.add_argument("--vocab", help="precalculated vocabulary")

    # Optional switch arguments
    parser.add_argument("-v", "--verbose", help="increase the model verbosity", action="store_true")
    parser.add_argument("-pc", "--phrase_clustering",
                        help="enable phrase clustering as described by Mikolov (i.e. New York becomes New_York)",
                        action="store_true")
    parser.add_argument("-sc", "--sanity_check", action="store_true")
    parser.add_argument("--tensorboard", help="Visualise training info and embeddings in tensorboard.",
                        action="store_true")
    parser.add_argument("--visdom", help="visualize training via visdom library", action="store_true")
    parser.add_argument("-sw", "--shareweights", help="make both embedding matrices have the same shared weights",
                        action="store_true")
    parser.add_argument("--eval_intrinstric", help="eval embeddings on analogy questions task", action="store_true",
                        default=True)

    # Optional arguments with value
    parser.add_argument("--eval_aq", "--eval_analogy_questions",
                        help="file with analogy questions to do the evaluation on", default=None)
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
    parser.add_argument("-ri", "--random_ints",
                        help="how many random ints for window subsampling to precalculate at once",
                        default=1310720  # 5 megabytes of int32s
                        )
    parser.add_argument("-tr", "--subsfqwords_tr", help="subsample frequent words threshold", default=1e-4)
    parser.add_argument("--sanitychecklist",
                        help='list of words for which the nearest word embeddings are found during training, '
                             'serves as sanity check, i.e. "dog family king eye"',
                        default="dog family king eye")
    parser.add_argument("-l", "--logging", help="external path to save example_logs into",
                        default="./logs")

    ## Step count parameters
    parser.add_argument("--lossreport_step", help="number of steps after which loss value is reported", default=20000)
    parser.add_argument("--epoch_state_step", help="number of steps after epoch progress is reported", default=20000)
    parser.add_argument("--eval_aq_step", help="number of steps after which analogy questions evaluation if performed",
                        default=20000)
    parser.add_argument("--eval_intrx_step", help="number of steps after which intrinstric evaluation if performed",
                        default=20000)
    parser.add_argument("--sanity_check_step", help="number of steps after which sanity check if performed",
                        default=20000)
    parser.add_argument("--eval_extrx_step", help="number of steps after which extrinstric evaluation if performed",
                        default=50000)
    parser.add_argument("--visdom_step", help="number of steps after which data are recorded in visdom",
                        default=1000)
    parser.add_argument("--tensorboard_step", help="number of steps after which embeddings are dumped for tensorboard",
                        default=30000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    init_parser(parser)
    args = parser.parse_args()
    logging = init_logging(os.path.basename(sys.argv[0]).split(".")[0], logpath=args.logging)
    with DataProcessor(args) as data_proc:
        skipgram_model = Skipgram(data_proc)

        # We need to carefully choose optimizer and its parameters to guarantee no global update will be excuted when training.
        # For example, parameters like weight_decay and momentum in torch.optim. SGD require the global calculation
        # on embedding matrix, which is extremely time-consuming.
        bytes_read = 0
        epochs = 10
        for e in range(epochs):
            logging.info(f"Starting epoch: {e}")
            bytes_read = skipgram_model._train(previously_read=bytes_read, epoch=e)
        skipgram_model.save(f"trained/embeddings_e{epochs}.vec")
