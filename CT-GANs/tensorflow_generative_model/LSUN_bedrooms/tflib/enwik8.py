import time
import numpy

def make_generator(corpus, BATCH_SIZE, SHORT_SEQ_LEN, LONG_SEQ_LEN, OVERLAP):
    def generator():
        # Randomly trim the first few chars of the corpus, so that the 
        # subsequence boundaries aren't the same every epoch. Then truncate 
        # what's left so that it fits neatly into batches of the given size.
        if SHORT_SEQ_LEN > 1:
            trim_offset = numpy.random.randint(SHORT_SEQ_LEN)
            trim_amt = (len(corpus)-trim_offset) % (BATCH_SIZE * LONG_SEQ_LEN)
            trimmed_corpus = corpus[trim_offset:-trim_amt].copy()
        else:
            trimmed_corpus = corpus.copy()

        # Split into long sequences, shuffle, and batch
        long_seqs = trimmed_corpus.reshape((-1, LONG_SEQ_LEN))
        if SHORT_SEQ_LEN == 1:
            print "Warning: SHORT_SEQ_LEN=1, not shuffling because otherwise this will take forever"
        else:
            start_time = time.time()
            numpy.random.shuffle(long_seqs)
            print "Shuffling took {}s".format(time.time() - start_time)
        batches = long_seqs.reshape((-1, BATCH_SIZE, LONG_SEQ_LEN))

        if OVERLAP != 0:
            raise Exception('nonzero overlap not supported')
        # # TODO I don't think this will work.
        # # Add some zero-padding
        # batches += 1
        # batches = T.concatenate([
        #     batches, 
        #     numpy.full((-1, BATCH_SIZE, OVERLAP), len(charmap), dtype='int32')
        # ], axis=2)

        for batch in batches:
            # Split each batch of long sequences into batches of shorter
            # subsequences.
            for i in xrange(LONG_SEQ_LEN / SHORT_SEQ_LEN):
                reset = numpy.int32(i == 0)
                inputs = batch[:, SHORT_SEQ_LEN*i : SHORT_SEQ_LEN*(i+1) + OVERLAP]
                yield (inputs, reset)
    return generator

def load(batch_size, short_seq_len, long_seq_len, overlap):
    with open('/home/ishaan/data/enwik8', 'r') as f:
        corpus = f.read()

    charmap = {char:ind for ind, char in enumerate(set(corpus))}
    inv_charmap = {ind:char for ind, char in enumerate(set(corpus))}

    full_corpus = numpy.fromiter(
        (charmap[c] for c in corpus), 
        'int32',
        count=len(corpus)
    )

    len_ = len(full_corpus)
    train_data  = make_generator(full_corpus[:(len_*18/20)], batch_size, short_seq_len, long_seq_len, overlap)
    dev_data    = make_generator(full_corpus[(len_*18/20):(len_*19/20)], batch_size, short_seq_len, long_seq_len, overlap)
    test_data   = make_generator(full_corpus[(len_*19/20):], batch_size, short_seq_len, long_seq_len, overlap)

    return (train_data, dev_data, test_data, charmap, inv_charmap)