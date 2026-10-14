import tempfile
import unittest
from pathlib import Path

import torch

from tinyllm import (CharTokenizer, TinyGPT, count_parameters, evaluate_loss,
                     format_conversation, sample_batch)


def tiny_config(vocab=20):
    return {"n_layer": 2, "n_head": 4, "n_embd": 64, "block_size": 32,
            "dropout": 0.0, "vocab_size": vocab}


def tiny_tokenizer():
    texts = ["<user>\nOi\n<assistant>\nOlá!\n<eos>\n", "abc xyz 123"]
    return CharTokenizer.build(texts)


class TokenizerTests(unittest.TestCase):
    def test_specials_have_reserved_ids(self):
        tok = tiny_tokenizer()
        self.assertEqual(tok.stoi["<pad>"], 0)
        self.assertEqual(tok.stoi["<eos>"], 1)

    def test_roundtrip_preserves_roles_and_accents(self):
        text = "<user>\nO que é Python?\n<assistant>\nÉ uma linguagem.\n<eos>\n"
        tok = CharTokenizer.build([text, "abc xyz 123"])
        self.assertEqual(tok.decode(tok.encode(text)), text)

    def test_unknown_char_maps_to_unk(self):
        tok = tiny_tokenizer()
        self.assertEqual(tok.decode(tok.encode("Ω")), "<unk>")


class ConversationFormatTests(unittest.TestCase):
    def test_roles_become_single_special_tokens(self):
        tok = tiny_tokenizer()
        text = format_conversation([{"role": "user", "content": "Oi"},
                                    {"role": "assistant", "content": "Olá!"}])
        ids = tok.encode(text)
        self.assertIn(tok.stoi["<user>"], ids)
        self.assertIn(tok.stoi["<assistant>"], ids)
        self.assertEqual(ids[-2], tok.stoi["<eos>"])


class ModelTests(unittest.TestCase):
    def test_parameter_count_is_5m_class(self):
        tok = CharTokenizer.build(["<user>\na\n<assistant>\nb\n<eos>\n" + "".join(chr(c) for c in range(97, 130))])
        from tinyllm import DEFAULT_CONFIG
        model = TinyGPT(dict(DEFAULT_CONFIG, vocab_size=tok.vocab_size))
        total = count_parameters(model)
        self.assertGreaterEqual(total, 4_000_000)
        self.assertLessEqual(total, 6_000_000)

    def test_forward_shapes_and_finite_loss(self):
        tok = tiny_tokenizer()
        model = TinyGPT(tiny_config(tok.vocab_size))
        x = torch.randint(0, tok.vocab_size, (2, 16))
        logits, loss = model(x, x)
        self.assertEqual(tuple(logits.shape), (2, 16, tok.vocab_size))
        self.assertTrue(torch.isfinite(loss))

    def test_overfit_single_batch_reduces_loss(self):
        torch.manual_seed(0)
        tok = tiny_tokenizer()
        model = TinyGPT(tiny_config(tok.vocab_size))
        opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
        x = torch.randint(0, tok.vocab_size, (4, 16))
        _, before = model(x, x)
        for _ in range(30):
            _, loss = model(x, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
        _, after = model(x, x)
        self.assertLess(after.item(), before.item())

    def test_generate_never_emits_pad_and_holds_eos(self):
        torch.manual_seed(3)
        tok = tiny_tokenizer()
        model = TinyGPT(tiny_config(tok.vocab_size))
        model.eval()
        idx = torch.tensor([[tok.stoi["<user>"], tok.stoi["<assistant>"]]])
        out = model.generate(idx, max_new=20, temperature=0, min_new_tokens=6)
        fresh = out[0, 2:8].tolist()
        self.assertNotIn(0, out.tolist()[0])  # <pad> nunca
        self.assertNotIn(1, fresh)  # <eos> só após min_new_tokens

    def test_generate_stops_and_stays_in_vocab(self):
        torch.manual_seed(1)
        tok = tiny_tokenizer()
        model = TinyGPT(tiny_config(tok.vocab_size))
        model.eval()
        idx = torch.tensor([[tok.stoi["<user>"], tok.stoi["<assistant>"]]])
        out = model.generate(idx, max_new=20, temperature=0.8, top_k=5)
        self.assertGreaterEqual(out.shape[1], 2)
        self.assertLessEqual(out.shape[1], 2 + 20)
        self.assertTrue(bool(((out >= 0) & (out < tok.vocab_size)).all()))

    def test_resume_state_survives_a_roundtrip(self):
        """O checkpoint precisa guardar o que faz --resume ser equivalente a não parar."""
        import random as _random
        from tinyllm import capture_rng_state, restore_rng_state, save_checkpoint
        torch.manual_seed(4)
        tok = CharTokenizer.build(["<user>\nO que é Python?\n<assistant>\nÉ uma linguagem.\n<eos>\n"])
        config = tiny_config(tok.vocab_size)
        model = TinyGPT(config)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.randint(0, tok.vocab_size, (2, 8))
        _, loss = model(x, x)
        opt.zero_grad(); loss.backward(); opt.step()
        rng = _random.Random(0)
        rng.random()
        _random.seed(7)  # o RNG global, que é o que capture/restore usa
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "r.pt")
            save_checkpoint(path, model, tok, config, step=42, val_loss=1.25,
                            optimizer=opt, rng_state=capture_rng_state(),
                            best_val=1.1, tag="rodada-1")
            blob = torch.load(path, weights_only=False)
            self.assertIn("optimizer", blob)
            self.assertIn("rng", blob)
            self.assertEqual(blob["step"], 42)
            self.assertEqual(blob["best_val"], 1.1)  # melhor que a última loss
            self.assertEqual(blob["tag"], "rodada-1")
            # O RNG global restaurado tem de reproduzir a mesma sequência.
            _random.seed(7)
            expected = _random.random()
            restore_rng_state(blob["rng"])
            self.assertEqual(_random.random(), expected)
            # E precisa sobreviver ao roundtrip do torch.save/load, que é onde
            # o ByteTensor voltava como Tensor genérico.
            _random.seed(7)
            restore_rng_state(torch.load(path, weights_only=False)["rng"])
            self.assertEqual(_random.random(), expected)

    def test_old_checkpoint_without_optimizer_still_loads(self):
        """Checkpoints anteriores têm só os pesos: precisam continuar abrindo."""
        from tinyllm import load_checkpoint, save_checkpoint
        tok = tiny_tokenizer()
        config = tiny_config(tok.vocab_size)
        model = TinyGPT(config)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "old.pt")
            save_checkpoint(path, model, tok, config, step=5, val_loss=2.0)
            blob = torch.load(path, weights_only=False)
            self.assertNotIn("optimizer", blob)
            _model2, tok2, blob2 = load_checkpoint(path, torch.device("cpu"))
            self.assertEqual(blob2["step"], 5)
            self.assertEqual(tok2.decode([tok2.stoi["<assistant>"]]), "<assistant>")

    def test_sample_batch_shapes(self):
        import random
        ids = [[1, 2, 3, 4, 5, 6, 7, 8] * 10]
        x, y = sample_batch(ids, batch=2, block=8, rng=random.Random(0))
        self.assertEqual(tuple(x.shape), (2, 8))
        self.assertTrue(bool(((y[:, :-1] == x[:, 1:]).all())))

    def test_evaluate_loss_is_finite(self):
        tok = tiny_tokenizer()
        model = TinyGPT(tiny_config(tok.vocab_size))
        val = evaluate_loss(model, [[1, 2, 3, 4] * 20], batch=2, block=8,
                            val_batches=3, device=torch.device("cpu"))
        self.assertTrue(val == val and val > 0)

    def test_save_load_roundtrip(self):
        from tinyllm import load_checkpoint, save_checkpoint
        torch.manual_seed(2)
        tok = tiny_tokenizer()
        config = tiny_config(tok.vocab_size)
        model = TinyGPT(config)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "m.pt")
            save_checkpoint(path, model, tok, config, step=7, val_loss=1.5)
            model2, tok2, blob = load_checkpoint(path, torch.device("cpu"))
            self.assertEqual(blob["step"], 7)
            self.assertEqual(tok2.decode([tok2.stoi["<user>"]]), "<user>")
            x = torch.randint(0, tok.vocab_size, (1, 8))
            with torch.no_grad():
                a, _ = model(x)
                b, _ = model2(x)
            self.assertTrue(torch.allclose(a, b))


if __name__ == "__main__":
    unittest.main()
