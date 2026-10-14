import json
import tempfile
import unittest
from pathlib import Path

from miniia import StatisticalModel
from responder import (KnowledgeBase, HybridResponder, build_idf, content_terms, cosine,
                       dedupe_pairs, fold, load_pairs, vectorize)


def make_pairs():
    return [
        {"question": "O que é Python?", "answer": "Python é uma linguagem de programação.",
         "source": "a.json#0", "position": 0},
        {"question": "O que é CSS?", "answer": "CSS estiliza páginas web.",
         "source": "a.json#1", "position": 0},
        {"question": "Como treinar uma IA?", "answer": "Usando dados e gradientes.",
         "source": "b.json#0", "position": 0},
    ]


class TextNormalizationTests(unittest.TestCase):
    def test_fold_strips_accents_and_lowercases(self):
        self.assertEqual(fold("Você é o quê?"), "voce e o que?")

    def test_content_terms_drop_stopwords_and_short_tokens(self):
        self.assertEqual(content_terms("O que é uma LLM?"), ["llm"])

    def test_greetings_collapse_to_a_canonical_token(self):
        self.assertEqual(content_terms("Oi!"), ["ola"])
        self.assertEqual(content_terms("Bom dia"), ["ola"])
        self.assertEqual(content_terms("muito obrigado"), ["obrigado"])

    def test_real_question_terms_survive(self):
        self.assertEqual(content_terms("como treinar uma IA?"), ["treinar", "ia"])


class VectorMathTests(unittest.TestCase):
    def test_idf_favors_rare_terms(self):
        idf = build_idf([["python"], ["css", "python"]])
        self.assertGreater(idf["css"], idf["python"])

    def test_vectors_are_unit_length_and_cosine_is_bounded(self):
        terms = ["python", "ia", "python"]
        vector = vectorize(terms, build_idf([terms, ["css"]]))
        self.assertAlmostEqual(sum(w * w for w in vector.values()), 1.0)
        self.assertAlmostEqual(cosine(vector, vector), 1.0)
        self.assertEqual(cosine(vector, {}), 0.0)


class DatasetLoadingTests(unittest.TestCase):
    def _write(self, directory, name, records):
        path = Path(directory) / name
        path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        return path

    def test_load_pairs_extracts_consecutive_user_assistant(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "c.json", [{"messages": [
                {"role": "user", "content": "Olá"},
                {"role": "assistant", "content": "Olá! Como posso ajudar?"},
                {"role": "user", "content": "O que é IA?"},
                {"role": "assistant", "content": "IA simula capacidades humanas."},
                {"role": "assistant", "content": "resposta órfã"},
            ]}])
            pairs = load_pairs([path])
            self.assertEqual(len(pairs), 2)
            self.assertEqual(pairs[0]["question"], "Olá")
            self.assertEqual(pairs[1]["answer"], "IA simula capacidades humanas.")

    def test_dedupe_keeps_first_and_counts_repeats(self):
        pairs = make_pairs() + [{"question": "O que é python?", "answer": "outra",
                                 "source": "c.json#0", "position": 0}]
        result = dedupe_pairs(pairs)
        self.assertEqual(len(result), 3)
        python = next(p for p in result if p["key"] == "python")
        self.assertEqual(python["repetitions"], 2)
        self.assertEqual(python["answer"], "Python é uma linguagem de programação.")


class KnowledgeBaseTests(unittest.TestCase):
    def setUp(self):
        self.kb = KnowledgeBase(make_pairs())

    def test_stats_reports_size_and_vocabulary(self):
        stats = self.kb.stats()
        self.assertEqual(stats["stored_pairs"], 3)
        self.assertGreater(stats["vocabulary"], 0)

    def test_exact_question_retrieves_its_own_answer(self):
        top = self.kb.search("o que é python?")[0]
        self.assertEqual(top["answer"], "Python é uma linguagem de programação.")
        self.assertAlmostEqual(top["score"], 1.0)

    def test_paraphrase_without_accents_still_matches(self):
        top = self.kb.search("o que e uma linguagem css")[0]
        self.assertEqual(top["question"], "O que é CSS?")

    def test_single_term_query_does_not_boost_to_perfect(self):
        # A lone shared word must not look like an exact match against a longer question.
        results = self.kb.search("ia")
        self.assertLess(results[0]["score"], 1.0)
        self.assertIn("IA", results[0]["question"])

    def test_search_respects_top_k(self):
        self.assertEqual(len(self.kb.search("python e programação", top_k=2)), 2)

    def test_query_with_no_overlap_returns_no_candidates(self):
        # No shared content words means nothing to score; the caller falls back.
        self.assertEqual(self.kb.search("receita de bolo de cenoura com cobertura"), [])


class HybridResponderTests(unittest.TestCase):
    def setUp(self):
        self.kb = KnowledgeBase(make_pairs())
        docs = [("d", "python css ia treinar dados gradientes linguagem programação web")]
        self.model = StatisticalModel(max_order=3).train(docs)

    def test_retrieval_answer_is_verbatim_from_corpus(self):
        result = HybridResponder(self.kb, self.model).answer("o que é css?")
        self.assertEqual(result["method"], "recovery")
        self.assertEqual(result["answer"], "CSS estiliza páginas web.")
        self.assertGreaterEqual(result["score"], 0.35)

    def test_out_of_scope_question_uses_statistical_fallback(self):
        result = HybridResponder(self.kb, self.model, fallback_tokens=5).answer(
            "rezeita bolo cenoura cobertura gourmet")
        self.assertEqual(result["method"], "statistical")
        self.assertIn("note", result)

    def test_fallback_without_model_reports_unavailable(self):
        result = HybridResponder(self.kb, model=None).answer("assunto totalmente novo e desconhecido")
        self.assertEqual(result["method"], "unavailable")

    def test_follow_up_uses_previous_question_as_context(self):
        history = [("user", "o que é python?"), ("assistant", "resposta")]
        # The follow-up alone shares no content word with any stored question.
        result = HybridResponder(self.kb, self.model, threshold=0.35).answer(
            "e ela como funciona", history=history)
        self.assertEqual(result["method"], "recovery-context")
        self.assertEqual(result["answer"], "Python é uma linguagem de programação.")

    def test_empty_question_short_circuits(self):
        self.assertEqual(HybridResponder(self.kb, self.model).answer("   "), {"answer": "", "method": "empty", "score": 0.0})

    def test_threshold_blocks_weak_matches(self):
        result = HybridResponder(self.kb, self.model, threshold=0.99, fallback_tokens=3).answer("treinar")
        self.assertEqual(result["method"], "statistical")


if __name__ == "__main__":
    unittest.main()
