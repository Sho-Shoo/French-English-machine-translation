
import unittest
import pickle as pk

from numpy.testing import assert_allclose
import torch


import transformer

SEED = 10417617

TOLERANCE = 1e-4

with open("tests.pk", "rb") as f: TESTS = pk.load(f)

class TestPositionalEncoding(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(SEED)
        self.layer = transformer.PositionalEncodingLayer(32)

    def test(self):
        x = torch.arange(3*16*32, dtype=torch.float).reshape((3, 16, 32))
        output = self.layer(x).detach().numpy()
        assert_allclose(output, TESTS[0], atol=TOLERANCE)        

class TestSelfAttention(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(SEED)
        self.layer = transformer.SelfAttentionLayer(12, 16)

    def test(self):
        qx = torch.arange(3*8*12, dtype=torch.float).reshape((3, 8, 12))
        kx = torch.arange(3*10*12, dtype=torch.float).reshape((3, 10, 12)) + 0.5
        vx = torch.arange(3*10*12, dtype=torch.float).reshape((3, 10, 12)) - 0.75

        out, weights = self.layer(qx, kx, vx)

        assert_allclose(out.detach().numpy(), TESTS[1][0], atol=TOLERANCE)
        assert_allclose(weights.detach().numpy(), TESTS[1][1], atol=TOLERANCE)

class TestSelfAttentionWithMask(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(SEED)
        self.layer = transformer.SelfAttentionLayer(12, 16)

    def test(self):
        qx = torch.arange(3*8*12, dtype=torch.float).reshape((3, 8, 12))
        kx = torch.arange(3*10*12, dtype=torch.float).reshape((3, 10, 12)) + 0.5
        vx = torch.arange(3*10*12, dtype=torch.float).reshape((3, 10, 12)) - 0.75

        mask = torch.ones((3, 8, 10), dtype=torch.float)
        mask[0, 4:, :] = 0.
        mask[1, :, 7:] = 0.
        mask[2, 3:, 6:] = 0.

        out, weights = self.layer(qx, kx, vx, mask=mask)

        assert_allclose(out.detach().numpy(), TESTS[2][0], atol=TOLERANCE)
        assert_allclose(weights.detach().numpy(), TESTS[2][1], atol=TOLERANCE)

class TestMask(unittest.TestCase):
    
    def setUp(self):
        torch.manual_seed(SEED)
        self.layer = transformer.Decoder(5, 5, 5, 5)

    def test(self):
        assert_allclose(self.layer._lookahead_mask(10), TESTS[3], atol=TOLERANCE)


class TestBeamSearch(unittest.TestCase):

    def setUp(self):
        self.transformer = transformer.Transformer(3796, 2788, 256, 2, 4, 3)
        self.transformer.load_state_dict(TESTS[4][0])

    def test(self):
        
        x = [0, 25, 26, 3193, 233, 132, 16, 1337, 5, 1, 3796, 3796]

        for b in range(1, 6):
            out, ll = self.transformer.predict(x, beam_size=b)
            assert_allclose(out, TESTS[4][2*b-1], atol=TOLERANCE)
            assert_allclose(ll, TESTS[4][2*b], atol=TOLERANCE)
        
        x = [0, 2, 4, 6, 8, 10, 12, 14, 1, 3796, 3796, 3796]

        for b in range(1, 6):
            out, ll = self.transformer.predict(x, beam_size=b)
            assert_allclose(out, TESTS[4][9+2*b], atol=TOLERANCE)
            assert_allclose(ll, TESTS[4][10+2*b], atol=TOLERANCE)

        x = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 1, 3796]

        for b in range(1, 6):
            out, ll = self.transformer.predict(x, beam_size=b, max_length=6)
            assert_allclose(out, TESTS[4][19+2*b], atol=TOLERANCE)
            assert_allclose(ll, TESTS[4][20+2*b], atol=TOLERANCE)

class TestBleuScore(unittest.TestCase):

    def setUp(self):
        pass

    def test(self):
        target = [0, 2, 3, 4, 5, 6, 7, 8, 9, 1, 3796, 3796]
        predicted = [0, 2, 3, 4, 5, 6, 10, 11, 12, 1, 3796, 3796]

        for n in range(1, 5):
            assert_allclose(transformer.bleu_score(predicted, target, N=n), TESTS[5][n-1], atol=TOLERANCE)

        predicted = [0, 2, 2, 2, 2, 2, 2, 2, 2, 1, 3796, 3796]
        assert_allclose(transformer.bleu_score(predicted, target, N=1), TESTS[5][4], atol=TOLERANCE)

        predicted = [0, 2, 3, 4, 10, 6, 1, 3796, 3796, 3796, 3796, 3796]
        for n in range(1, 5):
            assert_allclose(transformer.bleu_score(predicted, target, N=n), TESTS[5][n+4], atol=TOLERANCE)

        predicted = [0, 2, 3, 4, 5, 6, 10, 11, 12, 7, 8, 1]
        for n in range(1, 5):
            assert_allclose(transformer.bleu_score(predicted, target, N=n), TESTS[5][n+8], atol=TOLERANCE)

        target = [0, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 1]
        predicted = [0, 3, 3, 3, 3, 4, 5, 1]

        for n in range(1, 5):
            assert_allclose(transformer.bleu_score(predicted, target, N=n), TESTS[5][n+12], atol=TOLERANCE)

        predicted = [0, 3, 4, 1]
        assert_allclose(transformer.bleu_score(predicted, target, N=4), TESTS[5][17], atol=TOLERANCE)

        predicted = [0, 2, 3, 4, 5, 6, 7, 1]
        target = [0, 2, 3, 1]
        assert_allclose(transformer.bleu_score(predicted, target, N=4), TESTS[5][18], atol=TOLERANCE)
