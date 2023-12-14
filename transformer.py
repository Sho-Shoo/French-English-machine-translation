
import pickle
from typing import List, Optional, Tuple, Dict

import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib import ticker

import torch
from torch.nn import Module, Linear, Softmax, ReLU, LayerNorm, ModuleList, Dropout, Embedding, CrossEntropyLoss
from torch.optim import Adam

import math

class PositionalEncodingLayer(Module):

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X has shape (batch_size, sequence_length, embedding_dim)

        This function should create the positional encoding matrix
        and return the sum of X and the encoding matrix.

        The positional encoding matrix is defined as follow:

        P_(pos, 2i) = sin(pos / (10000 ^ (2i / d)))
        P_(pos, 2i + 1) = cos(pos / (10000 ^ (2i / d)))

        The output will have shape (batch_size, sequence_length, embedding_dim)
        """
        # b_size, T, d = X.shape
        # P = torch.zeros((T, d))
        # for pos in range(T):
        #     for i in range(d//2):
        #         P[pos, 2*i] = torch.sin(pos / (10000 ** (2*i / d)))
        #         if 2 * i + 1 < d: P[pos, 2*i+1] = torch.cos(pos / (10000 ** (2*i / d)))
        #
        # return X + P

        b_size, seq_len, emb_dim = X.shape

        position = torch.arange(0, seq_len).unsqueeze(1)
        denom = torch.exp(torch.arange(0, emb_dim, 2).float() * -(math.log(10000.0) / emb_dim))
        pos_enc = torch.zeros(1, seq_len, emb_dim)
        pos_enc[0, :, 0::2] = torch.sin(position * denom)
        pos_enc[0, :, 1::2] = torch.cos(position * denom)

        return pos_enc + X


class SelfAttentionLayer(Module):

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()

        self.linear_Q = Linear(in_dim, out_dim)
        self.linear_K = Linear(in_dim, out_dim)
        self.linear_V = Linear(in_dim, out_dim)

        self.softmax = Softmax(-1)

        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, query_X: torch.Tensor, key_X: torch.Tensor, value_X: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        query_X, key_X and value_X have shape (batch_size, sequence_length, in_dim). The sequence length
        may be different for query_X and key_X but must be the same for key_X and value_X.

        This function should return two things:
            - The output of the self-attention, which will have shape (batch_size, sequence_length, out_dim)
            - The attention weights, which will have shape (batch_size, query_sequence_length, key_sequence_length)

        If a mask is passed as input, you should mask the input to the softmax, using `float(-1e32)` instead of -infinity.
        The mask will be a tensor with 1's and 0's, where 0's represent entries that should be masked (set to -1e32).

        Hint: The following functions may be useful
            - torch.bmm (https://pytorch.org/docs/stable/generated/torch.bmm.html)
            - torch.Tensor.masked_fill (https://pytorch.org/docs/stable/generated/torch.Tensor.masked_fill.html)
        """
        q = self.linear_Q(query_X)
        k = self.linear_K(key_X)
        v = self.linear_V(value_X)

        scores = (q @ k.transpose(1, 2)) / torch.sqrt(torch.tensor(self.out_dim))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-1e32'))

        attention_weights = self.softmax(scores)
        attention_output = attention_weights @ v
        return attention_output, attention_weights

class MultiHeadedAttentionLayer(Module):

    def __init__(self, in_dim: int, out_dim: int, n_heads: int) -> None:
        super().__init__()

        self.attention_heads = ModuleList([SelfAttentionLayer(in_dim, out_dim) for _ in range(n_heads)])

        self.linear = Linear(n_heads * out_dim, out_dim)

    def forward(self, query_X: torch.Tensor, key_X: torch.Tensor, value_X: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        This function calls the self-attention layer and returns the output of the multi-headed attention
        and the attention weights of each attention head.

        The attention_weights matrix has dimensions (batch_size, heads, query_sequence_length, key_sequence_length)
        """

        outputs, attention_weights = [], []

        for attention_head in self.attention_heads:
            out, attention = attention_head(query_X, key_X, value_X, mask)
            outputs.append(out)
            attention_weights.append(attention)

        outputs = torch.cat(outputs, dim=-1)
        attention_weights = torch.stack(attention_weights, dim=1)

        return self.linear(outputs), attention_weights
        
class EncoderBlock(Module):

    def __init__(self, embedding_dim: int, n_heads: int) -> None:
        super().__init__()

        self.attention = MultiHeadedAttentionLayer(embedding_dim, embedding_dim, n_heads)

        self.norm1 = LayerNorm(embedding_dim)
        self.norm2 = LayerNorm(embedding_dim)

        self.linear1 = Linear(embedding_dim, 4 * embedding_dim)
        self.linear2 = Linear(4 * embedding_dim, embedding_dim)
        self.relu = ReLU()

        self.dropout1 = Dropout(0.2)
        self.dropout2 = Dropout(0.2)

    def forward(self, X, mask=None):
        """
        Implementation of an encoder block. Both the input and output
        have shape (batch_size, source_sequence_length, embedding_dim).

        The mask is passed to the multi-headed self-attention layer,
        and is usually used for the padding in the encoder.
        """  
        att_out, _ = self.attention(X, X, X, mask)

        residual = X + self.dropout1(att_out)

        X = self.norm1(residual)

        temp = self.linear1(X)
        temp = self.relu(temp)
        temp = self.linear2(temp)

        residual = X + self.dropout2(temp)

        return self.norm2(residual)

class Encoder(Module):

    def __init__(self, vocab_size: int, embedding_dim: int, n_blocks: int, n_heads: int) -> None:
        super().__init__()

        self.embedding_layer = Embedding(vocab_size + 1, embedding_dim, padding_idx=vocab_size)
        self.position_encoding = PositionalEncodingLayer(embedding_dim)
        self.blocks = ModuleList([EncoderBlock(embedding_dim, n_heads) for _ in range(n_blocks)])
        self.vocab_size = vocab_size

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Transformer encoder. The input has dimensions (batch_size, sequence_length)
        and the output has dimensions (batch_size, sequence_length, embedding_dim).

        The encoder returns its output and the location of the padding, which will be
        used by the decoder.
        """

        padding_locations = torch.where(X == self.vocab_size, torch.zeros_like(X, dtype=torch.float64),
                                        torch.ones_like(X, dtype=torch.float64))
        padding_mask = torch.einsum("bi,bj->bij", (padding_locations, padding_locations))

        X = self.embedding_layer(X)
        X = self.position_encoding(X)
        for block in self.blocks:
            X = block(X, padding_mask)
        return X, padding_locations

class DecoderBlock(Module):

    def __init__(self, embedding_dim, n_heads) -> None:
        super().__init__()

        self.attention1 = MultiHeadedAttentionLayer(embedding_dim, embedding_dim, n_heads)
        self.attention2 = MultiHeadedAttentionLayer(embedding_dim, embedding_dim, n_heads)

        self.norm1 = LayerNorm(embedding_dim)
        self.norm2 = LayerNorm(embedding_dim)
        self.norm3 = LayerNorm(embedding_dim)

        self.linear1 = Linear(embedding_dim, 4 * embedding_dim)
        self.linear2 = Linear(4 * embedding_dim, embedding_dim)
        self.relu = ReLU()

        self.dropout1 = Dropout(0.2)
        self.dropout2 = Dropout(0.2)
        self.dropout3 = Dropout(0.2)

    def forward(self, encoded_source: torch.Tensor, target: torch.Tensor,
                mask1: Optional[torch.Tensor]=None, mask2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Implementation of a decoder block. encoded_source has dimensions (batch_size, source_sequence_length, embedding_dim)
        and target has dimensions (batch_size, target_sequence_length, embedding_dim).

        The mask1 is passed to the first multi-headed self-attention layer, and mask2 is passed
        to the second multi-headed self-attention layer.

        Returns its output of shape (batch_size, target_sequence_length, embedding_dim) and
        the attention matrices for each of the heads of the second multi-headed self-attention layer
        (the one where the source and target are "mixed").
        """  
        att_out, _ = self.attention1(target, target, target, mask1)
        residual = target + self.dropout1(att_out)
        
        X = self.norm1(residual)

        att_out, att_weights = self.attention2(X, encoded_source, encoded_source, mask2)

        residual = X + self.dropout2(att_out)
        X = self.norm2(residual)

        temp = self.linear1(X)
        temp = self.relu(temp)
        temp = self.linear2(temp)
        residual = X + self.dropout3(temp)

        return self.norm3(residual), att_weights

class Decoder(Module):

    def __init__(self, vocab_size: int, embedding_dim: int, n_blocks: int, n_heads: int) -> None:
        super().__init__()
        
        self.embedding_layer = Embedding(vocab_size + 1, embedding_dim, padding_idx=vocab_size)
        self.position_encoding = PositionalEncodingLayer(embedding_dim)
        self.blocks = ModuleList([DecoderBlock(embedding_dim, n_heads) for _ in range(n_blocks)])

        self.linear = Linear(embedding_dim, vocab_size + 1)
        self.softmax = Softmax(-1)

        self.vocab_size = vocab_size

    def _lookahead_mask(self, seq_length: int) -> torch.Tensor:
        """
        Compute the mask to prevent the decoder from looking at future target values.
        The mask you return should be a tensor of shape (sequence_length, sequence_length)
        with only 1's and 0's, where a 0 represent an entry that will be masked in the
        multi-headed attention layer.

        Hint: The function torch.tril (https://pytorch.org/docs/stable/generated/torch.tril.html)
        may be useful.
        """
        mask = torch.ones((seq_length, seq_length))
        return torch.tril(mask)


    def forward(self, encoded_source: torch.Tensor, source_padding: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Transformer decoder. encoded_source has dimensions (batch_size, source_sequence_length, embedding),
        source_padding has dimensions (batch_size, source_seuqence_length) and target has dimensions
        (batch_size, target_sequence_length).

        Returns its output of shape (batch_size, target_sequence_length, target_vocab_size) and
        the attention weights from the first decoder block, of shape
        (batch_size, n_heads, source_sequence_length, target_sequence_length)

        Note that the output is not normalized (i.e. we don't use the softmax function).
        """
        
        # Lookahead mask
        seq_length = target.shape[1]
        mask = self._lookahead_mask(seq_length)

        # Padding masks
        target_padding = torch.where(target == self.vocab_size, torch.zeros_like(target, dtype=torch.float64), 
                                     torch.ones_like(target, dtype=torch.float64))
        target_padding_mask = torch.einsum("bi,bj->bij", (target_padding, target_padding))
        mask1 = torch.multiply(mask, target_padding_mask)

        source_target_padding_mask = torch.einsum("bi,bj->bij", (target_padding, source_padding))

        target = self.embedding_layer(target)
        target = self.position_encoding(target)

        att_weights = None
        for block in self.blocks:
            target, att = block(encoded_source, target, mask1, source_target_padding_mask)
            if att_weights is None:
                att_weights = att

        y = self.linear(target)
        return y, att_weights


class Transformer(Module):

    def __init__(self, source_vocab_size: int, target_vocab_size: int, embedding_dim: int, n_encoder_blocks: int,
                 n_decoder_blocks: int, n_heads: int) -> None:
        super().__init__()

        self.encoder = Encoder(source_vocab_size, embedding_dim, n_encoder_blocks, n_heads)
        self.decoder = Decoder(target_vocab_size, embedding_dim, n_decoder_blocks, n_heads)


    def forward(self, source, target):
        encoded_source, source_padding = self.encoder(source)
        return self.decoder(encoded_source, source_padding, target)

    def predict(self, source: List[int], beam_size=1, max_length=12) -> List[int]:
        """
        Given a sentence in the source language, you should output a sentence in the target
        language of length at most `max_length` that you generate using a beam search with
        the given `beam_size`.

        Note that the start of sentence token is 0 and the end of sentence token is 1.

        Return the final top beam (decided using average log-likelihood) and its average
        log-likelihood.

        Hint: The follow functions may be useful:
            - torch.topk (https://pytorch.org/docs/stable/generated/torch.topk.html)
            - torch.softmax (https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html)
        """
        self.eval()  # Set the PyTorch Module to inference mode (this affects things like dropout)

        # Ensure source is a tensor
        source_seq = torch.tensor(source).view(1, -1) if not isinstance(source, torch.Tensor) else source.view(1, -1)

        # Perform encoding using the model's encoder
        encoded_seq, seq_padding_mask = self.encoder(source_seq)
        seq_batch_size = encoded_seq.shape[0]
        init_token_seq = torch.zeros((seq_batch_size, 1)).to(torch.int64)
        search_results = []
        prob_seq_pairs = [[0, init_token_seq]]

        # Perform beam search
        while beam_size > 0:
            candidate_seq_list = []
            for score_seq_pair in prob_seq_pairs:
                last_token_seq = score_seq_pair[1]
                decoder_output, weight = self.decoder(encoded_seq, seq_padding_mask, last_token_seq)
                softmax_output = torch.softmax(decoder_output, dim=2)
                top_tokens = torch.topk(softmax_output[0][-1], beam_size)[1]
                top_scores = torch.topk(softmax_output[0][-1], beam_size)[0]
                for idx in range(len(top_tokens)):
                    base_seq = score_seq_pair[1]
                    extended_seq = torch.cat((base_seq, top_tokens[idx].reshape(1, 1)), dim=1)
                    accumulated_score = np.log(float(top_scores[idx])) + score_seq_pair[0]
                    candidate_seq_list.append([accumulated_score, extended_seq])
            candidate_seq_list.sort(key=lambda x: x[0])
            top_candidates = candidate_seq_list[-beam_size:]
            idx = 0

            while idx < len(top_candidates):
                current_pair = top_candidates[idx]
                current_seq_flat = current_pair[1].tolist()[0]
                if current_seq_flat[-1] == 1 or len(current_seq_flat) >= max_length:
                    current_pair[0] = current_pair[0] / len(current_seq_flat)
                    search_results.append(current_pair)
                    top_candidates.pop(idx)
                    beam_size -= 1
                else:
                    idx += 1
            prob_seq_pairs = top_candidates

        search_results.sort()
        optimal_result = search_results[-1]
        optimal_score = optimal_result[0]
        optimal_seq = optimal_result[1].tolist()[0]

        return optimal_seq, optimal_score

def load_data() -> Tuple[Tuple[List[int], List[int]], Tuple[List[int], List[int]], Dict[int, str], Dict[int, str]]:
    """ Load the dataset.

    :return: (1) train_sentences: list of (source_sentence, target_sentence) pairs, where both source_sentence
                                  and target_sentence are lists of ints
             (2) test_sentences : list of (source_sentence, target_sentence) pairs, where both source_sentence
                                  and target_sentence are lists of ints
             (2) source_vocab   : dictionary which maps from source word index to source word
             (3) target_vocab   : dictionary which maps from target word index to target word
    """
    with open('data/translation_data.bin', 'rb') as f:
        corpus, source_vocab, target_vocab = pickle.load(f)
        test_sentences = corpus[:1000]
        train_sentences = corpus[1000:]
        print("# source vocab: {}\n"
              "# target vocab: {}\n"
              "# train sentences: {}\n"
              "# test sentences: {}\n".format(len(source_vocab), len(target_vocab), len(train_sentences),
                                              len(test_sentences)))
        return train_sentences, test_sentences, source_vocab, target_vocab

def preprocess_data(sentences: Tuple[List[int], List[int]], source_vocab_size,
                    target_vocab_size, max_length):
    
    source_sentences = []
    target_sentences = []

    for source, target in sentences:
        source = [0] + source + ([source_vocab_size] * (max_length - len(source) - 1))
        target = [0] + target + ([target_vocab_size] * (max_length - len(target) - 1))
        source_sentences.append(source)
        target_sentences.append(target)

    return torch.tensor(source_sentences), torch.tensor(target_sentences)

def decode_sentence(encoded_sentence: List[int], vocab: Dict) -> str:
    if isinstance(encoded_sentence, torch.Tensor):
        encoded_sentence = [w.item() for w in encoded_sentence]
    words = [vocab[w] for w in encoded_sentence if w != 0 and w != 1 and w in vocab]
    return " ".join(words)

def visualize_attention(source_sentence: List[int],
                        output_sentence: List[int],
                        source_vocab: Dict[int, str],
                        target_vocab: Dict[int, str],
                        attention_matrix: np.ndarray):
    """
    :param source_sentence_str: the source sentence, as a list of ints
    :param output_sentence_str: the target sentence, as a list of ints
    :param attention_matrix: the attention matrix, of dimension [target_sentence_len x source_sentence_len]
    :param outfile: the file to output to
    """
    source_length = 0
    while source_length < len(source_sentence) and source_sentence[source_length] != 1:
        source_length += 1

    target_length = 0
    while target_length < len(output_sentence) and output_sentence[target_length] != 1:
        target_length += 1

    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attention_matrix[:target_length, :source_length], cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.xaxis.set_major_locator(ticker.FixedLocator(range(source_length)))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(["PAD" if x not in source_vocab else source_vocab[x] for x in source_sentence[:source_length]]))
    ax.yaxis.set_major_locator(ticker.FixedLocator(range(target_length)))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter(["PAD" if x not in target_vocab else target_vocab[x] for x in output_sentence[:target_length]]))

    plt.show()
    plt.close()

def train(model: Transformer, train_source: torch.Tensor, train_target: torch.Tensor,
          test_source: torch.Tensor, test_target: torch.Tensor, target_vocab_size: int,
          epochs: int = 30, batch_size: int = 64, lr: float = 0.0001):

    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss(ignore_index=target_vocab_size)

    epoch_train_loss = np.zeros(epochs)
    epoch_test_loss = np.zeros(epochs)

    for ep in tqdm(range(epochs)):

        train_loss = 0
        test_loss = 0

        permutation = torch.randperm(train_source.shape[0])
        train_source = train_source[permutation]
        train_target = train_target[permutation]

        batches = train_source.shape[0] // batch_size
        model.train()
        for ba in tqdm(range(batches), desc=f"Epoch {ep + 1}"):

            optimizer.zero_grad()

            batch_source = train_source[ba * batch_size: (ba + 1) * batch_size]
            batch_target = train_target[ba * batch_size: (ba + 1) * batch_size]

            target_pred, _ = model(batch_source, batch_target)

            batch_loss = loss_fn(target_pred[:, :-1, :].transpose(1, 2), batch_target[:, 1:])
            batch_loss.backward()
            optimizer.step()

            train_loss += batch_loss.item()

        test_batches = test_source.shape[0] // batch_size
        model.eval()
        for ba in tqdm(range(test_batches), desc="Test", leave=False):

            batch_source = test_source[ba * batch_size: (ba + 1) * batch_size]
            batch_target = test_target[ba * batch_size: (ba + 1) * batch_size]

            target_pred, _ = model(batch_source, batch_target)

            batch_loss = loss_fn(target_pred[:, :-1, :].transpose(1, 2), batch_target[:, 1:])
            test_loss += batch_loss.item()

        epoch_train_loss[ep] = train_loss / batches
        epoch_test_loss[ep] = test_loss / test_batches
        print(f"Epoch {ep + 1}: Train loss = {epoch_train_loss[ep]:.4f}, Test loss = {epoch_test_loss[ep]:.4f}")
    return epoch_train_loss, epoch_test_loss


def count_occurrences(ngram, sentence, n):
    count = 0
    for start in range(len(sentence) - n + 1):
        snippet = tuple(sentence[start:start+n])
        if snippet == ngram: count += 1
    return count


def bleu_score_helper(predicted, target, n):
    ngrams = set()
    for start in range(len(predicted)-n+1):
        ngram = tuple(predicted[start:start+n])
        ngrams.add(ngram)

    enum = 0
    for ngram in ngrams:
        predicted_occurrences = count_occurrences(ngram, predicted, n)
        target_occurrences = count_occurrences(ngram, target, n)
        enum += min(predicted_occurrences, target_occurrences)

    return enum / max(len(predicted) - n + 1, 1)


def strip_sentence(sentence):
    output = []
    for word in sentence:
        if word == 1:
            return output
        elif word == 0:
            continue
        else:
            output.append(word)

    return output

def bleu_score(predicted: List[int], target: List[int], N: int = 4) -> float:
    """
    *** For students in 10-617 only ***
    (Students in 10-417, you can leave `raise NotImplementedError()`)

    Implement a function to compute the BLEU-N score of the predicted
    sentence with a single reference (target) sentence.

    Please refer to the handout for details.

    Make sure you strip the SOS (0), EOS (1), and padding (anything after EOS)
    from the predicted and target sentences.
    
    If the length of the predicted sentence or the target is less than N,
    the BLEU score is 0.
    """
    predicted = strip_sentence(predicted)
    target = strip_sentence(target)

    accum_scores = []
    for n in range(1, N+1):
        accum_scores.append(bleu_score_helper(predicted, target, n))
    accum_scores = np.array(accum_scores)
    geom_mean = accum_scores.prod()**(1.0/len(accum_scores))
    penalty = min(1, np.exp(1 - len(target) / len(predicted)))
    return geom_mean * penalty


def calc_avg_bleu_score(transformer, test_source, test_target, bleu_n):
    scores = []
    for french, english in zip(test_source, test_target):
        french, english = french.tolist(), english.tolist()
        translation, _ = transformer.predict(french, beam_size=3)
        score = bleu_score(translation, english, bleu_n)
        scores.append(score)

    return sum(scores) / len(scores)


if __name__ == "__main__":
    EPOCHS = 30
    QUESTION = '2e'
    N_ENCODERS = 2
    N_DECODERS = 4
    N_HEADS = 4

    train_sentences, test_sentences, source_vocab, target_vocab = load_data()

    train_source, train_target = preprocess_data(train_sentences, len(source_vocab), len(target_vocab), 12)
    test_source, test_target = preprocess_data(test_sentences, len(source_vocab), len(target_vocab), 12)

    # questions = ['2a', '2b', '2c', '2d', '2e']
    # filenames = [f"output/{q}/model.pkl" for q in questions]
    # configs = [[1, 1, 1], [1, 1, 4], [2, 2, 1], [2, 2, 4], [2, 4, 4]]
    #
    # for filename, config, question in zip(filenames, configs, questions):
    #     transformer = Transformer(len(source_vocab), len(target_vocab), 256, *config)
    #     transformer.load_state_dict(torch.load(filename))
    #     transformer.eval()
    #     for bleu_n in range(1, 5):
    #         avg_score = calc_avg_bleu_score(transformer, test_source, test_target, bleu_n)
    #         print(f"Model {question} | BLEU-{bleu_n} : {round(avg_score, 4)}")



    ################### Q5 ###################
    transformer = Transformer(len(source_vocab), len(target_vocab), 256, N_ENCODERS, N_DECODERS, N_HEADS)
    transformer.load_state_dict(torch.load("output/2e/model.pkl"))
    transformer.eval()
    # avg_likelihoods = [-0.24344101760911172, -0.23207810300032686, -0.2273868364227314, -0.2196743762294325, -0.21966718185228618, -0.21944324851449365, -0.21944324851449365, -0.21944324851449365]
    avg_likelihoods = []
    beam_sizes = list(range(1, 9))
    for beam_size in beam_sizes:
        likelihoods = []
        for i in tqdm(range(100), desc=f"Beam size: {beam_size}"):
            french = test_source[i, :].tolist()
            translation, likelihood = transformer.predict(french, beam_size=beam_size)
            likelihoods.append(likelihood)
        avg_likelihoods.append(sum(likelihoods) / len(likelihoods))

    print(f"avg_likelihoods: {avg_likelihoods}")

    plt.title("Q5")
    plt.plot(beam_sizes, avg_likelihoods)
    plt.xlabel("Beam size")
    plt.ylabel("Avg log likelihood")
    plt.show()

    ################### Q4 ###################
    # for i in range(3):
    #     french, english = train_sentences[i][0], train_sentences[i][1]
    #     _, attention_matrices = transformer(train_source[i].reshape(1, -1), train_target[i, :].reshape(1, -1))
    #     attention_matrices = attention_matrices.squeeze()
    #     for attention_matrix in attention_matrices:
    #         attention_matrix = attention_matrix.detach().numpy()
    #         visualize_attention(french, english, source_vocab, target_vocab, attention_matrix)

    ################### Q3 ###################
    # for i in range(8):
    #     french, english = test_sentences[i][0], test_sentences[i][1]
    #     translation, likelihood = transformer.predict(french, beam_size=3)
    #     print(f"Original French is:       {decode_sentence(french, source_vocab)}")
    #     print(f"Corresponding English is: {decode_sentence(english, target_vocab)}")
    #     print(f"Translation is:           {decode_sentence(translation, target_vocab)}")
    #     print(f"Log-likelihood is:        {likelihood}")
    #     print("------------------------------------------------")

    ################### Q2 ###################
    # transformer = Transformer(len(source_vocab), len(target_vocab), 256, N_ENCODERS, N_DECODERS, N_HEADS)
    # train_loss, test_loss = train(transformer, train_source, train_target, test_source, test_target, len(target_vocab),
    #                               epochs=EPOCHS)
    #
    # torch.save(transformer.state_dict(), f'output/{QUESTION}/model.pkl')
    # np.save(f'output/{QUESTION}/train_loss', train_loss)
    # np.save(f'output/{QUESTION}/test_loss', test_loss)
