import torch

import numpy as np
import torch.nn.functional as F

# Method taken from tensorflow/models skipgram
def read_analogies(file, w2id):
    """Reads through the analogy question file.
    Returns:
      questions: a [n, 4] numpy array containing the analogy question's
                 word ids.
      questions_skipped: questions skipped due to unknown words.
    """
    questions = []
    questions_skipped = 0
    with open(file, "rb") as analogy_f:
        for line in analogy_f:
            if line.startswith(b":"):  # Skip comments.
                continue
            words = line.decode().strip().lower().split()
            ids = [w2id.get(w.strip(), None) for w in words]
            if None in ids or len(ids) != 4:
                questions_skipped += 1
            else:
                questions.append(np.array(ids))

    print("\n###########################################")
    print("Loaded evaluation method: Question analogy")
    print("-------------------------------------------\n")
    print("Eval analogy file: ", file)
    print("Questions: ", len(questions))
    print("Skipped: ", questions_skipped)
    print("###########################################\n")
    return np.array(questions, dtype=np.int32)

# Each analogy task is to predict the 4th word (d) given three
# words: a, b, c.  E.g., a=italy, b=rome, c=france, we should
# predict d=paris
def eval_analogy_questions(data_processor, embeddings, use_cuda, top_k=4,embedding_bag = False):
    """Evaluate analogy questions and reports accuracy."""
    # How many questions we get right at precision@1.
    correct = 0
    aq = data_processor.analogy_questions
    total = aq.shape[0]

    start = 0
    N = 256
    predict_item_index = 3

    # Normalize matrix so we can calculate cosine distances with dot product

    nembs = torch.transpose(F.normalize(embeddings.weight), 0, 1)
    while start < total:
        limit = start + N
        analogy = aq[start:limit, :]

        # Each row of a_emb, b_emb, c_emb is a word's embedding vector.
        # They all have the shape [N, emb_dim]
        a = torch.LongTensor(analogy[:, 0])
        b = torch.LongTensor(analogy[:, 1])
        c = torch.LongTensor(analogy[:, 2])

        if embedding_bag:
            arange = torch.LongTensor(range(len(a)))
            brange = torch.LongTensor(range(len(b)))
            crange = torch.LongTensor(range(len(c)))

        if use_cuda:
            a = a.cuda()
            b = b.cuda()
            c = c.cuda()
            if embedding_bag:
                arange = arange.cuda()
                brange = brange.cuda()
                crange = crange.cuda()

        if embedding_bag:
            a_emb = embeddings(a, arange)
            b_emb = embeddings(b, brange)
            c_emb = embeddings(c, crange)
        else:
            a_emb = embeddings(a)
            b_emb = embeddings(b)
            c_emb = embeddings(c)


        # We expect that d's embedding vectors on the unit hyper-sphere is
        # near: c_emb + (b_emb - a_emb), which has the shape [N, emb_dim].
        d_emb = c_emb + b_emb - a_emb

        # Compute cosine distance of d_emb to each vocab word
        # dist has shape [N, vocab_size]
        dist = torch.matmul(d_emb, nembs)

        # top_k closest EMBEDDINGS
        top_predicted = torch.topk(dist, dim=1, k=top_k)[1].cpu().numpy()

        start = limit
        for question in range(analogy.shape[0]):
            for j in range(top_k):
                if top_predicted[question, j] == analogy[question, predict_item_index]:
                    # Bingo! We predicted correctly. E.g., [italy, rome, france, paris].
                    correct += 1
                    break
                elif top_predicted[question, j] in analogy[question, :predict_item_index]:
                    # We need to skip words already in the question.
                    continue
                else:
                    # The correct label is not the precision@1
                    break
    print("Eval analogy questions %4d/%d accuracy = %4.1f%%" % (correct, total, correct * 100.0 / total))
