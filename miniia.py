"""Small, inspectable statistical autoregressive text model (stdlib only)."""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import math
import random
import re
import sys
import time
from pathlib import Path

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
WORD_RE = re.compile(r"\w", re.UNICODE)
MODEL_VERSION = 1
WINDOW_NGRAM_CONTEXT = 20


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def detokenize(tokens: list[str]) -> str:
    parts: list[str] = []
    quote_open = False
    for i, token in enumerate(tokens):
        previous = tokens[i - 1] if i else ""
        following = tokens[i + 1] if i + 1 < len(tokens) else ""
        if not parts:
            parts.append(token)
        elif token in ('"', "'"):
            if token == "'":
                parts.append(token)
            elif quote_open:
                parts.append(token)
                quote_open = False
            else:
                parts.extend([" ", token])
                quote_open = True
        elif token in "([{\u201c\u2018\u00ab":
            parts.extend([" ", token])
        elif token in ".,!?;:%)]}\u00bb\u201d\u2019":
            parts.append(token)
        elif previous in "([{\u201c\u2018\u00ab'\u2019":
            parts.append(token)
        elif token == "-" and previous and following and WORD_RE.search(previous) and WORD_RE.search(following):
            parts.append(token)
        elif previous == "-" and i >= 2 and WORD_RE.search(tokens[i - 2]) and WORD_RE.search(token):
            parts.append(token)
        elif token.isdigit() and previous in {".", ","} and i >= 2 and tokens[i - 2].isdigit():
            parts.append(token)
        else:
            parts.extend([" ", token])
    return "".join(parts)


def read_documents(paths: list[Path]) -> list[tuple[str, str]]:
    """Read text files or conversation JSON; only visible message `content` is used."""
    result: list[tuple[str, str]] = []
    for path in paths:
        if path.is_dir():
            candidates = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".md", ".json"})
        else:
            candidates = [path]
        for file in candidates:
            if file.suffix.lower() == ".json":
                data = json.loads(file.read_text(encoding="utf-8"))
                for i, record in enumerate(data if isinstance(data, list) else [data]):
                    messages = record.get("messages", []) if isinstance(record, dict) else []
                    parts = [m["content"] for m in messages if isinstance(m, dict) and isinstance(m.get("content"), str)]
                    text = "\n".join(parts)
                    if text.strip():
                        result.append((f"{file}#{i}", text))
            else:
                text = file.read_text(encoding="utf-8", errors="replace")
                if text.strip():
                    result.append((str(file), text))
    return result


class ContextLaws:
    """Independent, configurable influence laws. Their parameters are heuristics."""
    DEFAULTS = {"frequency": True, "rarity": True, "recurrence": True, "distance": True, "compatibility": True}

    def __init__(self, enabled=None, rarity_strength=0.65, frequency_strength=0.65,
                 recurrence_strength=0.45, distance_decay=0.035, max_rarity=2.0):
        self.enabled = dict(self.DEFAULTS if enabled is None else enabled)
        self.rarity_strength = rarity_strength
        self.frequency_strength = frequency_strength
        self.recurrence_strength = recurrence_strength
        self.distance_decay = distance_decay
        self.max_rarity = max_rarity

    def weight(self, token: str, distance: int, history_counts: collections.Counter,
               token_counts: dict[str, int], total_tokens: int, doc_freq: dict[str, int], n_docs: int) -> float:
        count = token_counts.get(token, 0)
        rarity = min(self.max_rarity, math.log1p(total_tokens / max(1, count)) / 5.0)
        common = (math.log1p(count) / max(1.0, math.log1p(total_tokens)))
        w = 1.0
        if self.enabled.get("rarity"):
            w *= 1 + self.rarity_strength * rarity
        if self.enabled.get("frequency"):
            # document prevalence suppresses nonspecific functional words smoothly
            prevalence = doc_freq.get(token, 0) / max(1, n_docs)
            w *= max(0.3, 1 - self.frequency_strength * prevalence * min(1.0, common * 4))
        if self.enabled.get("recurrence"):
            w *= 1 + self.recurrence_strength * min(2.0, math.log1p(max(0, history_counts[token] - 1)))
        if self.enabled.get("distance"):
            w *= math.exp(-self.distance_decay * distance)
        return w


class StatisticalModel:
    def __init__(self, max_order=4, alpha=0.15, context_window=32):
        self.max_order, self.alpha, self.context_window = max_order, alpha, context_window
        self.counts = collections.Counter()
        self.doc_freq = collections.Counter()
        self.ngrams = {n: collections.defaultdict(collections.Counter) for n in range(2, max_order + 1)}
        self.assoc = collections.defaultdict(collections.Counter)
        self.corpus_stream = []
        self.corpus_vocab = []
        self.suffix_array = []
        self.total_tokens = 0
        self.n_docs = 0

    def train(self, documents):
        documents = list(documents)
        for _name, text in documents:
            tokens = tokenize(text)
            if not tokens:
                continue
            self.n_docs += 1
            self.total_tokens += len(tokens)
            self.counts.update(tokens)
            self.doc_freq.update(set(tokens))
            padded = ["<BOS>"] * (self.max_order - 1) + tokens + ["<EOS>"]
            for i in range(self.max_order - 1, len(padded)):
                target = padded[i]
                for order in range(2, self.max_order + 1):
                    prefix = tuple(padded[i-order+1:i])
                    self.ngrams[order][prefix][target] += 1
                start = max(0, i - self.context_window)
                for j in range(start, i):
                    prior = padded[j]
                    if prior not in ("<BOS>", "<EOS>"):
                        self.assoc[prior][target] += 1
        # Keep compact strongest associations; all learned n-grams remain exact.
        for token, targets in list(self.assoc.items()):
            self.assoc[token] = collections.Counter(dict(targets.most_common(256)))
        if not self.counts:
            raise ValueError("No tokens found in training documents")
        self._build_corpus_index(documents)
        return self

    def _build_corpus_index(self, documents):
        """Store the corpus once and index all suffixes without quadratic prefixes."""
        self.corpus_vocab = self.vocab
        token_ids = {token: i + 1 for i, token in enumerate(self.corpus_vocab)}
        stream = []
        for _name, text in documents:
            stream.extend(token_ids[token] for token in tokenize(text))
            stream.append(0)  # document boundary / EOS
        self.corpus_stream = stream
        suffix_array = list(range(len(stream)))
        ranks = stream[:]
        width = 1
        while width < len(stream):
            suffix_array.sort(key=lambda pos: (ranks[pos], ranks[pos+width] if pos+width < len(stream) else -1))
            next_ranks = [0] * len(stream)
            rank = 0
            previous = None
            for pos in suffix_array:
                key = (ranks[pos], ranks[pos+width] if pos+width < len(stream) else -1)
                if previous is not None and key != previous:
                    rank += 1
                next_ranks[pos] = rank
                previous = key
            ranks = next_ranks
            if rank == len(stream) - 1:
                break
            width *= 2
        self.suffix_array = suffix_array

    def _corpus_match_range(self, query):
        """Find suffix-array range whose corpus suffix begins with query IDs."""
        if not query or not self.suffix_array or 0 in query:
            return 0, 0

        def compare(position):
            for offset, target in enumerate(query):
                at = position + offset
                value = self.corpus_stream[at] if at < len(self.corpus_stream) else -1
                if value != target:
                    return -1 if value < target else 1
            return 0

        left, right = 0, len(self.suffix_array)
        while left < right:
            middle = (left + right) // 2
            if compare(self.suffix_array[middle]) < 0:
                left = middle + 1
            else:
                right = middle
        first = left
        right = len(self.suffix_array)
        while left < right:
            middle = (left + right) // 2
            if compare(self.suffix_array[middle]) <= 0:
                left = middle + 1
            else:
                right = middle
        return first, left

    def corpus_suffix_distribution(self, history):
        """Use the longest matching suffix of the complete history in the corpus."""
        if not self.corpus_stream or not self.suffix_array:
            return {"<EOS>": 0.0}
        token_ids = {token: i + 1 for i, token in enumerate(self.corpus_vocab)}
        encoded = [token_ids.get(token, 0) for token in history]
        low, high = 0, min(len(encoded), len(self.corpus_stream))
        while low < high:
            size = (low + high + 1) // 2
            start, end = self._corpus_match_range(encoded[-size:])
            if start < end:
                low = size
            else:
                high = size - 1
        if low == 0:
            return {"<EOS>": 0.0}
        start, end = self._corpus_match_range(encoded[-low:])
        next_counts = collections.Counter()
        for suffix_pos in self.suffix_array[start:end]:
            next_pos = suffix_pos + low
            target_id = self.corpus_stream[next_pos] if next_pos < len(self.corpus_stream) else 0
            candidate = "<EOS>" if target_id == 0 else self.corpus_vocab[target_id - 1]
            next_counts[candidate] += 1
        total = sum(next_counts.values())
        return {candidate: math.log(count / total) for candidate, count in next_counts.items()}

    @property
    def vocab(self):
        return list(self.counts)

    def global_frequency_distribution(self):
        """Unigram probabilities from corpus occurrence counts, without context laws."""
        denominator = sum(self.counts.values())
        if denominator <= 0:
            return {}
        return {token: count / denominator for token, count in self.counts.items()}

    def predict_global_frequency(self, temperature=1.0, seed=None, sample=False):
        """Return the most frequent token, or sample from global frequencies."""
        if not self.counts:
            raise ValueError("Model has no token frequencies")
        if not sample:
            return max(self.counts, key=self.counts.get)
        rng = random.Random(seed)
        scores = {token: math.log(count) for token, count in self.counts.items()}
        probs = self.probabilities(scores, temperature)
        target = rng.random()
        cumulative = 0.0
        for token, probability in probs.items():
            cumulative += probability
            if target <= cumulative:
                return token
        return next(iter(probs))

    def context_frequency_candidates(self, history):
        """Use the longest observed suffix, backing off to shorter suffixes then unigram."""
        for size in range(min(self.max_order - 1, len(history)), 0, -1):
            suffix = tuple(history[-size:])
            counts = self.ngrams[size + 1].get(suffix)
            if counts:
                return counts, list(suffix)
        return self.counts, []

    def frequency_weighted_distribution(self, history, weighting="window"):
        """Return the window-branch or global exact-transition candidate scores."""
        if weighting not in {"window", "global"}:
            raise ValueError("weighting must be 'window' or 'global'")
        if weighting == "window":
            if self.suffix_array:
                return self.corpus_suffix_distribution(history)
            return self.window_transition_distribution(history)
        # Global mode uses the longest exact corpus transition for the current
        # suffix, not broad co-occurrence anywhere in the preceding 256 tokens.
        transition_counts = None
        for size in range(min(self.max_order - 1, len(history)), 0, -1):
            suffix = tuple(history[-size:])
            found = self.ngrams[size + 1].get(suffix)
            if found:
                transition_counts = found
                break
        if not transition_counts:
            # There is no learned next-token transition for this context.
            return {"<EOS>": 0.0}
        prior = {token: count for token, count in self.counts.items()}
        prior["<EOS>"] = self.alpha
        prior_total = sum(prior.values())
        prior = {token: count / max(prior_total, 1e-12) for token, count in prior.items()}
        transition_total = sum(transition_counts.values())
        smoothing = max(self.alpha, 1e-9)
        candidates = set(transition_counts) | set(prior)
        return {
            token: math.log(max(
                (transition_counts.get(token, 0) + smoothing * prior.get(token, 0.0))
                / (transition_total + smoothing), 1e-300))
            for token in candidates
        }

    def window_transition_distribution(self, history, window_size=WINDOW_NGRAM_CONTEXT,
                                       backoff_decay=0.4):
        """Combine observed next-token branches from suffixes up to 20 tokens long."""
        max_context = min(window_size, self.max_order - 1, len(history))
        if max_context <= 0:
            return {"<EOS>": 0.0}
        matches = []
        for size in range(1, max_context + 1):
            suffix = tuple(history[-size:])
            counts = self.ngrams[size + 1].get(suffix)
            if counts:
                matches.append((size, counts))
        if not matches:
            return {"<EOS>": 0.0}

        longest = max(size for size, _counts in matches)
        evidence = collections.Counter()
        for size, counts in matches:
            total = sum(counts.values())
            if not total:
                continue
            # The longest matching suffix has full weight; each shorter suffix
            # contributes alternatives with geometrically decreasing influence.
            level_weight = backoff_decay ** (longest - size)
            for candidate, count in counts.items():
                evidence[candidate] += level_weight * count / total
        evidence_total = sum(evidence.values())
        if not evidence_total:
            return {"<EOS>": 0.0}
        return {candidate: math.log(max(value / evidence_total, 1e-300))
                for candidate, value in evidence.items()}

    def generate_frequency_weighted(self, prompt="", max_tokens=40, weighting="window",
                                    temperature=1.0, top_k=40, seed=None, sample=False,
                                    frequency_penalty=1.5, loop_chaos=2.0, cycle_limit=2):
        """Generate a continuation using one of the 256-token frequency weighting systems."""
        if self.context_window != 256:
            raise ValueError("This test generator requires a model trained with context_window=256")
        if weighting not in {"window", "global"}:
            raise ValueError("weighting must be 'window' or 'global'")
        if max_tokens < 0:
            raise ValueError("max_tokens must be >= 0")
        if frequency_penalty < 0:
            raise ValueError("frequency_penalty must be >= 0")
        if loop_chaos < 1:
            raise ValueError("loop_chaos must be >= 1")
        if cycle_limit < 1:
            raise ValueError("cycle_limit must be >= 1")
        rng = random.Random(seed)
        history = tokenize(prompt)
        prompt_length = len(history)
        generated = []
        steps = []
        for _ in range(max_tokens):
            scores = self.frequency_weighted_distribution(history, weighting)
            generated_counts = collections.Counter(generated)
            for token, count in generated_counts.items():
                if token in scores:
                    scores[token] -= frequency_penalty * count
            cycle = next((period for period in range(1, min(4, len(generated)//(cycle_limit+1))+1)
                          if all(generated[-period:] == generated[-period*(repeat+1):-period*repeat]
                                 for repeat in range(1, cycle_limit+1))), None) if generated else None
            chaotic = cycle is not None
            active_temperature = temperature * (loop_chaos if chaotic else 1.0)
            probs = self.probabilities(scores, active_temperature, top_k)
            if sample or chaotic:
                draw = rng.random()
                cumulative = 0.0
                selected = next(iter(probs))
                for candidate, probability in probs.items():
                    cumulative += probability
                    if draw <= cumulative:
                        selected = candidate
                        break
            else:
                selected = max(probs, key=probs.get)
            if selected == "<EOS>":
                steps.append({"context_tokens": min(len(history), 256), "selected": selected,
                              "frequency_penalty": frequency_penalty,
                              "loop_detected": chaotic, "cycle_period": cycle,
                              "temperature": active_temperature, "selection": "eos"})
                break
            generated.append(selected)
            steps.append({"context_tokens": min(len(history), 256), "selected": selected,
                          "frequency_penalty": frequency_penalty,
                          "loop_detected": chaotic, "cycle_period": cycle,
                          "temperature": active_temperature,
                          "selection": "sampling" if sample or chaotic else "greedy",
                          "candidates": [{"token": token, "probability": probability}
                                         for token, probability in sorted(
                                             probs.items(), key=lambda item: item[1], reverse=True)[:10] ]})
            history.append(selected)
        return detokenize(history), {"weighting": weighting, "window_tokens": 256,
                                    "prompt_tokens": prompt_length,
                                    "generated_tokens": len(history)-prompt_length, "seed": seed,
                                    "steps": steps}

    def generate_by_frequency(self, prompt="", max_tokens=1, top_k=10, temperature=1.0,
                              seed=None, sample=False, frequency_penalty=0.0,
                              loop_chaos=2.0, cycle_limit=2):
        """Generate by empirical next-token counts with longest-context backoff."""
        if max_tokens < 0:
            raise ValueError("max_tokens must be >= 0")
        rng = random.Random(seed)
        history = tokenize(prompt)
        generated = []
        steps = []
        for _ in range(max_tokens):
            counts, matched = self.context_frequency_candidates(history)
            total = sum(counts.values())
            if not total:
                break
            scores = {token: math.log(count) - frequency_penalty * generated.count(token)
                      for token, count in counts.items() if count > 0}
            cycle = next((period for period in range(1, min(4, len(generated)//(cycle_limit+1))+1)
                          if all(generated[-period:] == generated[-period*(repeat+1):-period*repeat]
                                 for repeat in range(1, cycle_limit+1))), None) if generated else None
            chaotic = cycle is not None
            active_temperature = temperature * (loop_chaos if chaotic else 1.0)
            if sample or chaotic:
                probs = self.probabilities(scores, active_temperature)
                draw = rng.random()
                cumulative = 0.0
                selected = next(iter(probs))
                for token, probability in probs.items():
                    cumulative += probability
                    if draw <= cumulative:
                        selected = token
                        break
            else:
                selected = max(scores, key=scores.get)
                probs = {token: count / total for token, count in counts.items()}
            ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:max(0, top_k)]
            steps.append({"context": matched, "context_length": len(matched),
                          "candidates": [{"token": token, "count": count,
                                          "probability": count / total} for token, count in ranked],
                          "selected": selected, "loop_detected": chaotic,
                          "cycle_period": cycle, "selection": "sampling" if sample or chaotic else "greedy"})
            if selected == "<EOS>":
                break
            history.append(selected)
            generated.append(selected)
        return detokenize(history), {"prompt_tokens": len(tokenize(prompt)),
                                     "generated_tokens": len(history) - len(tokenize(prompt)),
                                     "steps": steps, "seed": seed}

    def to_dict(self):
        return {"version": MODEL_VERSION, "max_order": self.max_order, "alpha": self.alpha,
                "context_window": self.context_window, "total_tokens": self.total_tokens, "n_docs": self.n_docs,
                "counts": dict(self.counts), "doc_freq": dict(self.doc_freq),
                "corpus_vocab": self.corpus_vocab, "corpus_stream": self.corpus_stream,
                "suffix_array": self.suffix_array,
                "ngrams": {str(n): [[list(k), dict(v)] for k, v in table.items()] for n, table in self.ngrams.items()},
                "assoc": {k: dict(v) for k, v in self.assoc.items()}}

    @classmethod
    def from_dict(cls, data):
        if data.get("version") != MODEL_VERSION:
            raise ValueError("Unsupported model version")
        model = cls(data["max_order"], data["alpha"], data["context_window"])
        model.total_tokens, model.n_docs = data["total_tokens"], data["n_docs"]
        model.counts = collections.Counter(data["counts"])
        model.doc_freq = collections.Counter(data["doc_freq"])
        model.corpus_vocab = data.get("corpus_vocab", [])
        model.corpus_stream = data.get("corpus_stream", [])
        model.suffix_array = data.get("suffix_array", [])
        for n, rows in data["ngrams"].items():
            model.ngrams[int(n)] = collections.defaultdict(collections.Counter)
            for key, values in rows:
                model.ngrams[int(n)][tuple(key)] = collections.Counter(values)
        model.assoc = collections.defaultdict(collections.Counter, {k: collections.Counter(v) for k, v in data["assoc"].items()})
        return model

    def save(self, path):
        payload = json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))
        if str(path).lower().endswith(".gz"):
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                stream.write(payload)
        else:
            Path(path).write_text(payload, encoding="utf-8")

    @classmethod
    def load(cls, path):
        if str(path).lower().endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                return cls.from_dict(json.load(stream))
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def distribution(self, history, laws=None, repeat_penalty=0.0, recent_window=0, sequence_penalty=0.0):
        laws = laws or ContextLaws()
        vocab = self.vocab
        counts = collections.Counter(history)
        total = self.total_tokens
        # Smoothed unigram log probabilities are the robust backoff floor.
        scores = {t: math.log((self.counts[t] + self.alpha) / (total + self.alpha * len(vocab))) for t in vocab}
        # Interpolated conditional n-grams; every order contributes capped log evidence.
        for order in range(2, self.max_order + 1):
            prefix = tuple((history + ["<BOS>"] * (order - 1))[-(order - 1):])
            table = self.ngrams[order].get(prefix)
            if table:
                denom = sum(table.values()) + self.alpha * (len(vocab) + 1)
                strength = 0.55 / (order - 1)
                for tok, n in table.items():
                    if tok in scores:
                        scores[tok] += strength * math.log1p(n * len(vocab) / max(1, sum(table.values())))
        # Full history remains queryable; only the bounded recent window contributes each step.
        context = history[-self.context_window:]
        if laws.enabled.get("compatibility", True):
            evidence = collections.Counter()
            for distance, prior in enumerate(reversed(context)):
                related = self.assoc.get(prior)
                if not related:
                    continue
                weight = laws.weight(prior, distance, counts, self.counts, total, self.doc_freq, self.n_docs)
                norm = max(1, sum(related.values()))
                for candidate, cooc in related.items():
                    if candidate in scores:
                        evidence[candidate] += weight * cooc / norm
            if evidence:
                # Blend normalized next-token evidence from the full context window
                # with the unigram + n-gram distribution. Context is substantial,
                # while the learned general model remains a stable backoff.
                base = self.probabilities(scores)
                evidence_total = sum(evidence.values())
                smoothing = max(self.alpha, 1e-9)
                blend = 0.65
                denominator = evidence_total + smoothing
                for tok in scores:
                    contextual = (evidence.get(tok, 0.0) + smoothing * base[tok]) / denominator
                    mixed = (1 - blend) * base[tok] + blend * contextual
                    scores[tok] = math.log(max(mixed, 1e-300))
        if recent_window and repeat_penalty:
            for tok in history[-recent_window:]:
                if tok in scores:
                    scores[tok] -= repeat_penalty
        if sequence_penalty and len(history) >= 2:
            # Penalize candidates that would repeat an already seen 2-4-token continuation.
            for size in range(2, min(5, len(history) + 1)):
                prefix = history[-(size - 1):]
                continuations = collections.Counter(history[i + size - 1] for i in range(len(history)-size+1)
                    if history[i:i+size-1] == prefix)
                for candidate, occurrences in continuations.items():
                    if candidate in scores:
                        scores[candidate] -= sequence_penalty * occurrences / (size - 1)
        return scores

    @staticmethod
    def probabilities(scores, temperature=1.0, top_k=0):
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if top_k > 0:
            items = items[:top_k]
        peak = items[0][1]
        vals = [(t, math.exp(max(-745, (s - peak) / temperature))) for t, s in items]
        z = sum(v for _, v in vals)
        if not math.isfinite(z) or z <= 0:
            raise ArithmeticError("Invalid candidate distribution")
        return {t: v / z for t, v in vals}

    def generate(self, prompt="", max_tokens=80, temperature=0.85, seed=None, top_k=40,
                 laws=None, repeat_penalty=0.35, repeat_window=24, cycle_limit=3,
                 adaptive_temperature=0.35, diagnostics=False):
        rng = random.Random(seed)
        generated = tokenize(prompt)
        initial_len = len(generated)
        events = []
        for step in range(max_tokens):
            temp = temperature
            if len(generated) >= 4 and cycle_limit >= 2:
                cycle = next((k for k in (1, 2, 3, 4) if len(generated) >= cycle_limit*k
                              and all(generated[-(i+1)*k:-i*k if i else None] == generated[-k:]
                                      for i in range(1, cycle_limit))), None)
                if cycle:
                    events.append({"step": step, "event": "cycle_detected", "period": cycle})
                    temp = min(2.0, temp + adaptive_temperature)
            scores = self.distribution(generated, laws, repeat_penalty, repeat_window, repeat_penalty * 0.5)
            probs = self.probabilities(scores, temp, top_k)
            target = rng.random()
            cumulative = 0.0
            token = next(iter(probs))
            for candidate, probability in probs.items():
                cumulative += probability
                if target <= cumulative:
                    token = candidate
                    break
            generated.append(token)
            # EOS is represented internally for learned stopping but excluded from visible output.
            if token == "<EOS>":
                generated.pop()
                break
        result = detokenize(generated)
        if diagnostics:
            return result, {"prompt_tokens": initial_len, "generated_tokens": len(generated)-initial_len,
                            "anti_loop_events": events, "seed": seed}
        return result


def evaluate(model, documents, laws=None, max_positions=1000):
    nll = 0.0
    n = 0
    for _name, text in documents:
        tokens = tokenize(text)
        stride = max(1, math.ceil(len(tokens) / max(1, max_positions)))
        history = ["<BOS>"] * (model.max_order - 1)
        for i, token in enumerate(tokens):
            if i % stride == 0:
                scores = model.distribution(history, laws)
                peak = max(scores.values())
                denominator = sum(math.exp(max(-745, value - peak)) for value in scores.values())
                p = math.exp(max(-745, scores.get(token, peak - 35) - peak)) / denominator
                nll -= math.log(max(p, 1e-15))
                n += 1
            history.append(token)
    return {"evaluation_positions": n, "sampling": "uniform token positions, teacher-forced full preceding context",
            "cross_entropy_nats": nll/n if n else float("nan"),
            "perplexity": math.exp(min(700, nll/n)) if n else float("nan")}


def evaluate_context_window(model, documents, window=256, max_tests=250, top_k=5, seed=17,
                            weighting="window"):
    """Evaluate next-token predictions using exactly the preceding `window` tokens."""
    rng = random.Random(seed)
    positions = []
    tokenized_docs = []
    for name, text in documents:
        tokens = tokenize(text)
        tokenized_docs.append((name, tokens))
        positions.extend((len(tokenized_docs)-1, i) for i in range(window, len(tokens)))
    if max_tests > 0 and len(positions) > max_tests:
        positions = sorted(rng.sample(positions, max_tests))
    nll = 0.0
    correct = topk_correct = 0
    examples = []
    for doc_index, i in positions:
        name, tokens = tokenized_docs[doc_index]
        context = tokens[i-window:i]
        actual = tokens[i]
        scores = model.frequency_weighted_distribution(context, weighting)
        peak = max(scores.values())
        denominator = sum(math.exp(max(-745, score-peak)) for score in scores.values())
        probability = math.exp(max(-745, scores.get(actual, peak-35)-peak)) / denominator
        nll -= math.log(max(probability, 1e-15))
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:max(1, top_k)]
        predicted = ranked[0][0]
        correct += predicted == actual
        topk_correct += actual in {token for token, _score in ranked}
        if len(examples) < 10:
            examples.append({"document": name, "context_tail": detokenize(context[-12:]),
                             "actual_next_token": actual, "predicted_next_token": predicted,
                             "top_candidates": [token for token, _score in ranked]})
    count = len(positions)
    return {"window_tokens": window, "weighting": weighting, "evaluated_positions": count,
            "top1_accuracy": correct/count if count else None,
            "top_k": max(1, top_k), "top_k_accuracy": topk_correct/count if count else None,
            "cross_entropy_nats": nll/count if count else None,
            "perplexity": math.exp(min(700, nll/count)) if count else None,
            "selection": "uniform sample among positions with a full preceding window",
            "examples": examples}


def generation_report(model, prompt, laws, seed):
    started = time.perf_counter()
    text, info = model.generate(prompt, max_tokens=60, seed=seed, laws=laws, diagnostics=True)
    elapsed = time.perf_counter() - started
    generated = tokenize(text)[len(tokenize(prompt)):]
    bigrams = list(zip(generated, generated[1:]))
    cycle_count = sum(1 for event in info["anti_loop_events"] if event["event"] == "cycle_detected")
    return {"text": text, "generated_tokens": len(generated),
            "lexical_diversity": len(set(generated))/max(1, len(generated)),
            "token_repeat_rate": 1-len(set(generated))/max(1, len(generated)),
            "repeated_bigram_rate": 1-len(set(bigrams))/max(1, len(bigrams)),
            "cycle_events": cycle_count, "generation_seconds": round(elapsed, 4)}


def chat_turn(model, history, user_text, max_tokens=60, temperature=0.85, seed=None, top_k=40,
              mode="contextual", frequency_sample=False):
    """Append one user turn, generate a continuation, and return updated token history."""
    context = list(history) + tokenize(user_text)
    prompt = detokenize(context)
    if mode == "frequency":
        combined, diagnostics = model.generate_by_frequency(prompt, max_tokens, top_k,
            temperature, seed, frequency_sample)
    else:
        combined, diagnostics = model.generate(prompt=prompt, max_tokens=max_tokens,
            temperature=temperature, seed=seed, top_k=top_k, diagnostics=True)
    diagnostics["mode"] = mode
    combined_tokens = tokenize(combined)
    response_tokens = combined_tokens[len(context):]
    response = detokenize(response_tokens)
    return response, context + response_tokens, diagnostics


def weighted_chat_turn(model, history, user_text, weighting, max_tokens=30,
                       temperature=0.9, seed=None, top_k=40, sample=False,
                       frequency_penalty=1.5, loop_chaos=2.0, cycle_limit=2,
                       prefill_tokens=20):
    context = list(history) + tokenize(user_text)
    prefill_count = min(max(prefill_tokens, 0), max_tokens)
    prefill_text, prefill_report = model.generate_by_frequency(
        detokenize(context), prefill_count, top_k, temperature, seed, False,
        frequency_penalty, loop_chaos, cycle_limit)
    prefill_tokens_generated = tokenize(prefill_text)[len(context):]
    ended_on_eos = bool(prefill_report["steps"] and
                        prefill_report["steps"][-1]["selected"] == "<EOS>")
    remaining = 0 if ended_on_eos else max_tokens - len(prefill_tokens_generated)
    weighted_prompt = detokenize(context + prefill_tokens_generated)
    weighted_text, weighted_report = model.generate_frequency_weighted(
        weighted_prompt, remaining, weighting, temperature, top_k,
        None if seed is None else seed + 1, sample,
        frequency_penalty, loop_chaos, cycle_limit)
    weighted_tokens_generated = tokenize(weighted_text)[len(context) + len(prefill_tokens_generated):]
    response_tokens = prefill_tokens_generated + weighted_tokens_generated
    diagnostics = {"weighting": weighting, "prefill_tokens": len(prefill_tokens_generated),
                   "prefill_ended_on_eos": ended_on_eos,
                   "prefill": prefill_report, "weighted": weighted_report}
    return detokenize(response_tokens), context + response_tokens, diagnostics


def command_train(args):
    docs = read_documents([Path(p) for p in args.inputs])
    start = time.perf_counter()
    model = StatisticalModel(args.order, args.alpha, args.context_window).train(docs)
    model.save(args.output)
    print(json.dumps({"documents": model.n_docs, "tokens": model.total_tokens, "vocabulary": len(model.counts),
                      "model": args.output, "training_seconds": round(time.perf_counter()-start, 3)}, ensure_ascii=False))


def command_generate(args):
    model = StatisticalModel.load(args.model)
    if args.mode == "frequency":
        result, info = model.generate_by_frequency(args.prompt, args.tokens, args.top_k,
                                                  args.temperature, args.seed, args.frequency_sample)
        print(result)
        if args.diagnostics:
            print(json.dumps(info, ensure_ascii=False), file=sys.stderr)
        return
    enabled = {k: not getattr(args, "no_"+k, False) for k in ContextLaws.DEFAULTS}
    laws = ContextLaws(enabled, args.rarity_strength, args.frequency_strength, args.recurrence_strength, args.distance_decay)
    result, info = model.generate(args.prompt, args.tokens, args.temperature, args.seed, args.top_k, laws,
                                  args.repeat_penalty, args.repeat_window, args.cycle_limit,
                                  args.adaptive_temperature, True)
    print(result)
    if args.diagnostics:
        print(json.dumps(info, ensure_ascii=False), file=sys.stderr)


def command_frequency(args):
    model = StatisticalModel.load(args.model)
    result, info = model.generate_by_frequency(args.prompt, args.tokens, args.top_k,
                                              args.temperature, args.seed, args.sample)
    print(json.dumps({"method": "empirical_frequency_with_context_backoff",
                      "starting_prompt": args.prompt,
                      "selection": "sampling" if args.sample else "most_frequent",
                      "result": result, **info}, ensure_ascii=False, indent=2))


def command_chat(args):
    model = StatisticalModel.load(args.model)
    history = []
    turn = 0
    print("Mini chat estatístico. Comandos: /ajuda, /contexto, /limpar, /sair")
    while True:
        try:
            user_text = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSessão encerrada.")
            break
        if not user_text:
            continue
        command = user_text.lower()
        if command in {"/sair", "/exit", "/quit"}:
            print("Sessão encerrada.")
            break
        if command in {"/limpar", "/reset"}:
            history.clear()
            turn = 0
            print("Histórico limpo.")
            continue
        if command in {"/ajuda", "/help"}:
            print("Digite uma mensagem para gerar uma continuação. /contexto mostra o histórico; /limpar reinicia; /sair encerra.")
            continue
        if command in {"/contexto", "/context"}:
            if history:
                print(f"Contexto atual ({len(history)} tokens): {detokenize(history)}")
            else:
                print("Contexto vazio.")
            continue
        turn_seed = None if args.seed is None else args.seed + turn
        response, history, report = chat_turn(model, history, user_text, args.tokens,
                                              args.temperature, turn_seed, args.top_k,
                                              args.mode, args.frequency_sample)
        turn += 1
        print("IA:", response if response else "(nenhum token gerado)")
        if args.diagnostics:
            print(json.dumps(report, ensure_ascii=False))


def command_window_chat(args):
    model = StatisticalModel.load(args.model)
    if model.context_window != 256:
        raise SystemExit(f"O modelo {args.model} foi treinado com janela {model.context_window}; treine com --context-window 256 primeiro.")
    modes = ("window", "global") if args.weight_mode == "both" else (args.weight_mode,)
    if "window" in modes and not model.suffix_array:
        raise SystemExit(f"O modelo {args.model} não tem o índice de contexto completo; treine novamente com `miniia.py train ...`.")
    histories = {mode: [] for mode in modes}
    turn = 0
    print("Chat de teste da janela de 256 tokens. Sistemas: peso local e peso global. Comandos: /contexto, /limpar, /sair")
    while True:
        try:
            user_text = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSessão encerrada.")
            break
        if not user_text:
            continue
        command = user_text.lower()
        if command in {"/sair", "/exit", "/quit"}:
            print("Sessão encerrada.")
            break
        if command in {"/limpar", "/reset"}:
            for history in histories.values():
                history.clear()
            turn = 0
            print("Históricos dos dois sistemas limpos.")
            continue
        if command in {"/ajuda", "/help"}:
            print("Cada mensagem recebe respostas comparáveis dos dois pesos. /contexto mostra os históricos; /limpar reinicia; /sair encerra.")
            continue
        if command in {"/contexto", "/context"}:
            for mode, history in histories.items():
                print(f"Contexto {mode} ({len(history)} tokens): {detokenize(history) if history else '(vazio)'}")
            continue
        for index, mode in enumerate(modes):
            seed = None if args.seed is None else args.seed + turn*2 + index
            response, histories[mode], report = weighted_chat_turn(model, histories[mode], user_text,
                mode, args.tokens, args.temperature, seed, args.top_k, args.sample,
                args.frequency_penalty, args.loop_chaos, args.cycle_limit,
                args.prefill_tokens)
            print(f"IA [{mode}]:", response if response else "(nenhum token gerado)")
            if args.diagnostics:
                print(json.dumps(report, ensure_ascii=False))
        turn += 1


def command_evaluate(args):
    docs = read_documents([Path(p) for p in args.inputs])
    rng = random.Random(args.seed)
    rng.shuffle(docs)
    cut = max(1, int(len(docs) * args.train_fraction)) if len(docs) > 1 else len(docs)
    train, test = docs[:cut], docs[cut:] or docs[:]
    model = StatisticalModel(args.order, args.alpha, args.context_window).train(train)
    configs = {"baseline": {k: False for k in ContextLaws.DEFAULTS},
               "frequency_rarity": {"frequency": True, "rarity": True, "recurrence": False, "distance": False, "compatibility": True},
               "distance": {"frequency": False, "rarity": False, "recurrence": False, "distance": True, "compatibility": True},
               "three_laws": {**ContextLaws.DEFAULTS, "distance": False},
               "all_laws": ContextLaws.DEFAULTS}
    prompt = detokenize(tokenize(test[0][1])[:12]) if test else ""
    results = {name: evaluate(model, test, ContextLaws(flags), args.eval_tokens) for name, flags in configs.items()}
    generations = {name: generation_report(model, prompt, ContextLaws(flags), args.seed)
                   for name, flags in configs.items()}
    print(json.dumps({"train_documents": len(train), "test_documents": len(test), "train_tokens": model.total_tokens,
                      "serialized_model_bytes": len(json.dumps(model.to_dict(), ensure_ascii=False).encode("utf-8")),
                      "shared_generation_prompt": prompt, "results": results, "comparable_generations": generations},
                     ensure_ascii=False, indent=2))


def command_window_test(args):
    docs = read_documents([Path(p) for p in args.inputs])
    rng = random.Random(args.seed)
    rng.shuffle(docs)
    cut = max(1, min(len(docs)-1, int(len(docs) * args.train_fraction))) if len(docs) > 1 else 1
    train, test = docs[:cut], docs[cut:] if len(docs) > 1 else docs
    started = time.perf_counter()
    model = StatisticalModel(args.order, args.alpha, args.window).train(train)
    modes = ("window", "global") if args.weight_mode == "both" else (args.weight_mode,)
    results = {f"{mode}_frequency_weight": evaluate_context_window(
        model, test, args.window, args.max_tests, args.top_k, args.seed, mode) for mode in modes}
    print(json.dumps({"train_documents": len(train), "test_documents": len(test),
                      "train_tokens": model.total_tokens,
                      "training_and_evaluation_seconds": round(time.perf_counter()-started, 3),
                      "results": results}, ensure_ascii=False, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description="Mini IA estatística autoregressiva, sem dependências externas")
    sub = parser.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("train", help="treina ou atualiza modelo a partir de texto/JSON/diretórios")
    tr.add_argument("inputs", nargs="+"); tr.add_argument("-o", "--output", default="modelo.json")
    tr.add_argument("--order", type=int, default=4); tr.add_argument("--alpha", type=float, default=0.15)
    tr.add_argument("--context-window", type=int, default=32); tr.set_defaults(func=command_train)
    ge = sub.add_parser("generate", help="gera texto token a token")
    ge.add_argument("--model", default="modelo.json"); ge.add_argument("--prompt", default="")
    ge.add_argument("--mode", choices=("contextual", "frequency"), default="contextual",
                    help="estratégia de inferência; frequency usa contagens de continuações observadas")
    ge.add_argument("--frequency-sample", action="store_true", help="no modo frequency, amostra em vez de escolher argmax")
    ge.add_argument("--tokens", type=int, default=80); ge.add_argument("--temperature", type=float, default=0.85)
    ge.add_argument("--seed", type=int); ge.add_argument("--top-k", type=int, default=40)
    ge.add_argument("--repeat-penalty", type=float, default=0.35); ge.add_argument("--repeat-window", type=int, default=24)
    ge.add_argument("--cycle-limit", type=int, default=3); ge.add_argument("--adaptive-temperature", type=float, default=0.35)
    ge.add_argument("--diagnostics", action="store_true")
    for law in ContextLaws.DEFAULTS: ge.add_argument("--no-"+law, action="store_true")
    ge.add_argument("--rarity-strength", type=float, default=0.65); ge.add_argument("--frequency-strength", type=float, default=0.65)
    ge.add_argument("--recurrence-strength", type=float, default=0.45); ge.add_argument("--distance-decay", type=float, default=0.035)
    ge.set_defaults(func=command_generate)
    fr = sub.add_parser("frequency", help="prevê próximos tokens pelas ocorrências de contexto no corpus")
    fr.add_argument("--model", default="modelo.json")
    fr.add_argument("--top-k", type=int, default=10, help="quantos candidatos mostrar")
    fr.add_argument("--prompt", default="", help="tokens iniciais para condicionar os próximos tokens")
    fr.add_argument("--tokens", type=int, default=1, help="quantos tokens gerar a partir do prompt")
    fr.add_argument("--sample", action="store_true", help="amostra pela distribuição global em vez de escolher o mais frequente")
    fr.add_argument("--temperature", type=float, default=1.0)
    fr.add_argument("--seed", type=int)
    fr.set_defaults(func=command_frequency)
    ch = sub.add_parser("chat", help="abre uma sessão de conversa interativa no terminal")
    ch.add_argument("--model", default="modelo.json")
    ch.add_argument("--tokens", type=int, default=60, help="limite de tokens por resposta")
    ch.add_argument("--temperature", type=float, default=0.85)
    ch.add_argument("--top-k", type=int, default=40)
    ch.add_argument("--mode", choices=("contextual", "frequency"), default="contextual",
                    help="contextual usa leis e temperatura; frequency usa ocorrências condicionadas ao histórico")
    ch.add_argument("--frequency-sample", action="store_true", help="no modo frequency, amostra as continuações")
    ch.add_argument("--seed", type=int)
    ch.add_argument("--diagnostics", action="store_true")
    ch.set_defaults(func=command_chat)
    wc = sub.add_parser("window-chat", help="compara em chat os pesos locais e globais usando janela 256")
    wc.add_argument("--model", default="modelo_256.json")
    wc.add_argument("--weight-mode", choices=("both", "window", "global"), default="both")
    wc.add_argument("--tokens", type=int, default=30)
    wc.add_argument("--prefill-tokens", type=int, default=20,
                    help="quantos tokens iniciais são gerados pelo modo frequency antes dos pesos window/global")
    wc.add_argument("--temperature", type=float, default=0.9)
    wc.add_argument("--top-k", type=int, default=40)
    wc.add_argument("--frequency-penalty", type=float, default=1.5,
                    help="reduz a pontuação de tokens já repetidos na resposta")
    wc.add_argument("--loop-chaos", type=float, default=2.0,
                    help="multiplicador de temperatura e amostragem após detectar um loop")
    wc.add_argument("--cycle-limit", type=int, default=2,
                    help="repetições do ciclo necessárias para ativar o caos")
    wc.add_argument("--sample", action="store_true", help="amostra em vez de escolher o candidato de maior pontuação")
    wc.add_argument("--seed", type=int)
    wc.add_argument("--diagnostics", action="store_true")
    wc.set_defaults(func=command_window_chat)
    ev = sub.add_parser("evaluate", help="avalia cross-entropy/perplexidade em divisão por documento")
    ev.add_argument("inputs", nargs="+"); ev.add_argument("--seed", type=int, default=17)
    ev.add_argument("--train-fraction", type=float, default=0.8); ev.add_argument("--order", type=int, default=4)
    ev.add_argument("--alpha", type=float, default=0.15); ev.add_argument("--context-window", type=int, default=32)
    ev.add_argument("--eval-tokens", type=int, default=1000, help="máximo de posições avaliadas por documento, amostradas uniformemente")
    ev.set_defaults(func=command_evaluate)
    wt = sub.add_parser("window-test", help="testa previsão com os 256 tokens anteriores (janela configurável)")
    wt.add_argument("inputs", nargs="+")
    wt.add_argument("--window", type=int, default=256)
    wt.add_argument("--max-tests", type=int, default=250, help="máximo de posições de validação; 0 avalia todas")
    wt.add_argument("--top-k", type=int, default=5)
    wt.add_argument("--weight-mode", choices=("both", "window", "global"), default="both",
                    help="compara os dois tipos de peso ou executa apenas um")
    wt.add_argument("--seed", type=int, default=17)
    wt.add_argument("--train-fraction", type=float, default=0.8)
    wt.add_argument("--order", type=int, default=4)
    wt.add_argument("--alpha", type=float, default=0.15)
    wt.set_defaults(func=command_window_test)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
