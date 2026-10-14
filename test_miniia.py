import math
import unittest
from unittest.mock import patch
from pathlib import Path

from miniia import (ContextLaws, StatisticalModel, chat_turn, detokenize, evaluate,
                    evaluate_context_window, tokenize, weighted_chat_turn)


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.docs = [("a", "quantização melhora o sinal. quantização preserva dados."),
                     ("b", "o sinal preserva dados e reduz ruído."),
                     ("c", "modelos simples aprendem padrões de texto.")]
        self.model = StatisticalModel(max_order=3).train(self.docs)

    def test_training_frequencies_and_transitions(self):
        self.assertEqual(self.model.counts["quantização"], 2)
        self.assertGreater(self.model.ngrams[2][("quantização",)]["melhora"], 0)
        self.assertGreater(self.model.doc_freq["o"], 0)

    def test_global_frequency_predictor_uses_counts_only(self):
        model = StatisticalModel().train([("a", "lua lua lua sol")])
        distribution = model.global_frequency_distribution()
        self.assertAlmostEqual(sum(distribution.values()), 1.0)
        self.assertAlmostEqual(distribution["lua"], 3/4)
        self.assertEqual(model.predict_global_frequency(), "lua")
        self.assertEqual(model.predict_global_frequency(seed=12, sample=True),
                         model.predict_global_frequency(seed=12, sample=True))

    def test_frequency_generation_conditions_on_prompt_and_updates_context(self):
        model = StatisticalModel(max_order=3).train([
            ("a", "A B C. A B C. A B D."),
            ("b", "X Y Z.")])
        result, report = model.generate_by_frequency("A B", max_tokens=2, top_k=5)
        self.assertEqual(report["steps"][0]["context"], ["A", "B"])
        self.assertEqual(report["steps"][0]["selected"], "C")
        self.assertEqual(report["steps"][0]["candidates"][0]["token"], "C")
        self.assertTrue(result.startswith("A B C"))
        self.assertEqual(report["steps"][1]["context"][-2:], ["B", "C"])

    def test_frequency_generation_backs_off_to_global_counts(self):
        model = StatisticalModel(max_order=3).train([("a", "azul azul verde")])
        _result, report = model.generate_by_frequency("unseen tokens", max_tokens=1)
        self.assertEqual(report["steps"][0]["context"], [])
        self.assertEqual(report["steps"][0]["selected"], "azul")

    def test_chat_turn_carries_user_and_assistant_tokens_forward(self):
        first, history, report = chat_turn(self.model, [], "quantização", max_tokens=5, seed=21,
                                           mode="frequency")
        self.assertTrue(history[:1] == ["quantização"])
        self.assertEqual(len(history), 1 + report["generated_tokens"])
        self.assertEqual(first, first.strip())
        self.assertEqual(report["mode"], "frequency")
        self.assertTrue(report["steps"])
        second, next_history, _ = chat_turn(self.model, history, "e agora?", max_tokens=4, seed=22,
                                            mode="frequency")
        self.assertEqual(next_history[:len(history)], history)
        self.assertIn("e", next_history[len(history):])
        self.assertIsInstance(second, str)

    def test_detokenize_separates_sentences_and_preserves_punctuation(self):
        tokens = ["Olá", "!", "Como", "posso", "ajudar", "você", "hoje", "?"]
        self.assertEqual(detokenize(tokens), "Olá! Como posso ajudar você hoje?")
        self.assertEqual(detokenize(["Não", ",", "pare", ".", "Próxima"]), "Não, pare. Próxima")
        self.assertEqual(detokenize(["3", ".", "14"]), "3.14")

    def test_256_token_window_keeps_old_context_available(self):
        model = StatisticalModel(max_order=2, context_window=256).train([
            ("train", "anchor " + "noise " * 255 + "needle")])
        history = ["anchor"] + ["noise"] * 255
        with_old_token = model.distribution(history)
        without_old_token = model.distribution(history[1:])
        self.assertNotEqual(with_old_token, without_old_token)
        result = evaluate_context_window(model, [("test", "noise " * 257 + "needle")],
                                         window=256, max_tests=5, seed=1)
        self.assertEqual(result["window_tokens"], 256)
        self.assertEqual(result["evaluated_positions"], 2)

    def test_local_and_global_frequency_weights_produce_distinct_predictions(self):
        docs = [("specific", "alpha beta pivot localtarget")]
        docs.extend((f"global-{i}", "other pivot globaltarget") for i in range(5))
        model = StatisticalModel(max_order=2, context_window=256).train(docs)
        history = ["alpha", "beta", "pivot"]
        local = model.frequency_weighted_distribution(history, "window")
        global_ = model.frequency_weighted_distribution(history, "global")
        self.assertNotEqual(local, global_)
        self.assertEqual(max(local, key=local.get), "localtarget")
        self.assertEqual(max(global_, key=global_.get), "globaltarget")

    def test_window_uses_sliding_twenty_token_context_and_keeps_candidate_branches(self):
        prefix = [f"t{i}" for i in range(19)] + ["pivot"]
        text = " ".join(prefix)
        docs = [("branch-a", text + " python") for _ in range(3)]
        docs.append(("branch-b", text + " java"))
        model = StatisticalModel(max_order=21, context_window=256).train(docs)
        branches = model.window_transition_distribution(prefix)
        self.assertIn("python", branches)
        self.assertIn("java", branches)
        self.assertGreater(branches["python"], branches["java"])
        exact = model.ngrams[21][tuple(prefix)]
        self.assertEqual(exact["python"], 3)
        self.assertEqual(exact["java"], 1)
        shifted = tuple(prefix[1:] + ["python"])
        self.assertEqual(model.ngrams[21][shifted]["<EOS>"], 3)
        _text, generated = model.generate_frequency_weighted(
            detokenize(prefix), 1, "window", top_k=10)
        self.assertEqual(generated["steps"][0]["selected"], "python")
        self.assertIn("java", {item["token"] for item in generated["steps"][0]["candidates"]})

    def test_window_indexes_full_history_beyond_twenty_tokens(self):
        context = [f"word{i}" for i in range(30)]
        prefix = " ".join(context)
        model = StatisticalModel(max_order=4, context_window=256).train([
            ("a", prefix + " answer") for _ in range(4)] + [("b", prefix + " alternative")])
        scores = model.frequency_weighted_distribution(context, "window")
        self.assertGreater(scores["answer"], scores["alternative"])
        self.assertEqual(len(model.corpus_stream), model.total_tokens + model.n_docs)

    def test_global_weighted_generation_stops_when_no_observed_transition_exists(self):
        model = StatisticalModel(max_order=2, context_window=256).train([("a", "azul claro")])
        _text, info = model.generate_frequency_weighted("desconhecido", 5, "global")
        self.assertEqual(info["generated_tokens"], 0)

    def test_weighted_chat_requires_256_window_and_tracks_turn_history(self):
        model = StatisticalModel(max_order=2, context_window=256).train([
            ("a", "Olá mundo. Olá pessoa. Como posso ajudar?"),
            ("b", "Olá mundo. Posso ajudar você.")])
        response, history, report = weighted_chat_turn(
            model, [], "Olá", "window", max_tokens=3, seed=5)
        self.assertIsInstance(response, str)
        self.assertEqual(history[:1], ["Olá"])
        self.assertEqual(len(history), 1 + report["prefill_tokens"] + report["weighted"]["generated_tokens"])
        self.assertEqual(report["weighting"], "window")
        _response, global_history, global_report = weighted_chat_turn(
            model, [], "Olá", "global", max_tokens=3, seed=5)
        self.assertEqual(global_report["weighting"], "global")
        self.assertEqual(global_history[:1], ["Olá"])
        with self.assertRaisesRegex(ValueError, "context_window=256"):
            StatisticalModel().train(self.docs).generate_frequency_weighted("Olá", 1)

    def test_weighted_generation_penalizes_repeats_and_activates_loop_chaos(self):
        model = StatisticalModel(context_window=256).train([("a", "O x")])
        with patch.object(model, "frequency_weighted_distribution", return_value={"O": 0.0, "x": -0.1}):
            _text, penalized = model.generate_frequency_weighted(
                "", max_tokens=2, weighting="window", sample=False, frequency_penalty=0.8)
            _text, info = model.generate_frequency_weighted(
                "", max_tokens=8, weighting="window", sample=False,
                frequency_penalty=0.0, loop_chaos=2.0, cycle_limit=3, seed=2)
        self.assertEqual(penalized["steps"][0]["selected"], "O")
        self.assertEqual(penalized["steps"][1]["selected"], "x")
        self.assertTrue(any(step["loop_detected"] for step in info["steps"]))
        self.assertTrue(any(step["selection"] == "sampling" for step in info["steps"]))

    def test_weighted_chat_prefills_with_context_frequency_before_weighted_generation(self):
        model = StatisticalModel(max_order=3, context_window=256).train([
            ("a", "Olá! Como posso ajudar você hoje? Olá! Como posso ajudar você hoje?")])
        response, history, report = weighted_chat_turn(
            model, [], "Olá", "window", max_tokens=6, seed=4, prefill_tokens=3)
        self.assertEqual(report["prefill_tokens"], 3)
        self.assertEqual(report["prefill"]["steps"][0]["selected"], "!")
        self.assertEqual(report["weighted"]["generated_tokens"], 3)
        self.assertTrue(response.startswith("! Como posso"))
        self.assertEqual(history[:1], ["Olá"])

    def test_weighted_chat_stops_prefill_when_frequency_model_reaches_eos(self):
        model = StatisticalModel(max_order=3, context_window=256).train([
            ("a", "Olá! Como posso ajudar você hoje?")])
        response, _history, report = weighted_chat_turn(
            model, [], "Olá", "window", max_tokens=20, seed=4, prefill_tokens=20)
        self.assertTrue(report["prefill_ended_on_eos"])
        self.assertEqual(report["weighted"]["generated_tokens"], 0)
        self.assertIn("Como posso ajudar", response)

    def test_each_context_law_is_independent_and_finite(self):
        history = ["quantização", "o", "quantização"]
        for law in ContextLaws.DEFAULTS:
            flags = {k: False for k in ContextLaws.DEFAULTS}; flags[law] = True
            laws = ContextLaws(flags)
            w = laws.weight("quantização", 1, __import__("collections").Counter(history),
                            self.model.counts, self.model.total_tokens, self.model.doc_freq, self.model.n_docs)
            self.assertTrue(math.isfinite(w) and w > 0)
            scores = self.model.distribution(history, laws)
            self.assertTrue(all(math.isfinite(x) for x in scores.values()))

    def test_probability_distribution_normalized_and_temperature(self):
        scores = {"a": 0.0, "b": -1.0, "c": -2.0}
        cold = self.model.probabilities(scores, 0.2)
        hot = self.model.probabilities(scores, 2.0)
        self.assertAlmostEqual(sum(cold.values()), 1.0)
        self.assertAlmostEqual(sum(hot.values()), 1.0)
        self.assertGreater(cold["a"], hot["a"])

    def test_unknown_context_and_generation_are_safe_and_seeded(self):
        a = self.model.generate("palavra desconhecida", 20, seed=8)
        b = self.model.generate("palavra desconhecida", 20, seed=8)
        self.assertEqual(a, b)
        self.assertTrue(a)
        self.assertTrue(all(math.isfinite(v) for v in self.model.distribution(["x", "y"]).values()))

    def test_cycle_detector_activates_and_reports_event(self):
        model = StatisticalModel(max_order=2).train([("repeat", "ha ha ha ha ha ha")])
        _text, report = model.generate("ha ha", max_tokens=8, seed=2, top_k=1,
                                      cycle_limit=2, diagnostics=True)
        self.assertTrue(any(e["event"] == "cycle_detected" for e in report["anti_loop_events"]))

    def test_full_history_changes_contextual_distribution(self):
        short = self.model.distribution(["dados"])
        long = self.model.distribution(["quantização", "sinal", "dados"])
        self.assertNotEqual(short, long)

    def test_law_switch_changes_contextual_scores(self):
        off = ContextLaws({k: False for k in ContextLaws.DEFAULTS})
        on_flags = {k: False for k in ContextLaws.DEFAULTS}; on_flags["rarity"] = True
        # Compatibility must be enabled for context laws to weight learned evidence.
        on_flags["compatibility"] = True
        off_flags = {k: False for k in ContextLaws.DEFAULTS}; off_flags["compatibility"] = True
        baseline = self.model.distribution(["quantização", "sinal"], ContextLaws(off_flags))
        changed = self.model.distribution(["quantização", "sinal"], ContextLaws(on_flags))
        self.assertNotEqual(baseline, changed)

    def test_save_load_round_trip(self):
        path = Path("_test_model.json")
        compressed_path = Path("_test_model.json.gz")
        try:
            self.model.save(path)
            loaded = StatisticalModel.load(path)
            self.assertEqual(loaded.counts, self.model.counts)
            self.assertEqual(loaded.distribution(["quantização"]), self.model.distribution(["quantização"]))
            self.model.save(compressed_path)
            compressed = StatisticalModel.load(compressed_path)
            self.assertEqual(compressed.counts, self.model.counts)
            self.assertEqual(compressed.corpus_stream, self.model.corpus_stream)
            self.assertEqual(compressed.suffix_array, self.model.suffix_array)
        finally:
            path.unlink(missing_ok=True)
            compressed_path.unlink(missing_ok=True)

    def test_eos_and_eval_metrics_are_finite(self):
        result = evaluate(self.model, [("holdout", "quantização preserva dados.")])
        self.assertTrue(math.isfinite(result["cross_entropy_nats"]))
        self.assertGreater(result["perplexity"], 0)
        self.assertTrue(tokenize("Olá, mundo!") == ["Olá", ",", "mundo", "!"])


if __name__ == "__main__":
    unittest.main()

