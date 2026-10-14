"""Hybrid responder: TF-IDF retrieval over corpus Q&A pairs plus statistical fallback (stdlib only).

Recovery is exact and inspectable: the answer returned by retrieval is the answer stored
in the corpus, never invented. When no stored question is similar enough, the statistical
generator of `miniia.py` continues the text conditioned on the question.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from miniia import StatisticalModel, tokenize

WORD_RE = re.compile(r"\w+", re.UNICODE)

STOPWORDS = {
    "a", "o", "os", "as", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das",
    "no", "na", "nos", "nas", "num", "numa", "ao", "aos", "pelo", "pela", "pelos",
    "pelas", "num", "numa", "é", "e", "ea", "em", "com", "sem", "sob", "sobre", "entre",
    "para", "por", "que", "qual", "quais", "quem", "onde", "quando", "como", "porque",
    "se", "mas", "ou", "nem", "ja", "ainda", "tambem", "so", "la", "aqui", "ali", "ai",
    "me", "te", "se", "nos", "vos", "lhe", "lhes", "meu", "minha", "meus", "minhas",
    "seu", "sua", "seus", "suas", "esse", "essa", "esse", "essa", "este", "esta",
    "aquele", "aquela", "isto", "isso", "aquilo", "algo", "nada", "tudo", "todos",
    "todas", "cada", "outro", "outra", "outros", "outras", "mesmo", "mesma", "muito",
    "muita", "pouco", "pouca", "ser", "estar", "ter", "foi", "era", "tem", "tinha",
    "você", "voce", "vc", "ele", "ela", "eles", "elas", "eu", "tu", "sobre",
}

# Whole messages reduced to a canonical token so greetings match each other.
GREETING_ALIASES = {
    "oi": "ola", "ola": "ola", "opa": "ola", "hey": "ola", "hello": "ola",
    "hi": "ola", "salve": "ola", "e ai": "ola", "eai": "ola", "bom dia": "ola",
    "boa tarde": "ola", "boa noite": "ola",
    "obrigado": "obrigado", "obrigada": "obrigado", "valeu": "obrigado",
    "tchau": "tchau", "flw": "tchau", "ate mais": "tchau",
}


def fold(text: str) -> str:
    """Lowercase and strip accents so 'voce' matches 'você'."""
    decomposed = unicodedata.normalize("NFD", text.lower())
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def content_terms(text: str) -> list[str]:
    """Normalized content words used for retrieval."""
    folded = fold(text)
    # Compare greetings on words only, so "Oi!" and "Bom dia," still match an alias.
    bare = " ".join(WORD_RE.findall(folded))
    for greeting, canonical in GREETING_ALIASES.items():
        if bare == greeting:
            return [canonical]
    return [w for w in WORD_RE.findall(folded) if len(w) > 1 and w not in STOPWORDS]


def load_pairs(paths: list[Path]) -> list[dict]:
    """Load consecutive user/assistant message pairs from conversation JSON files."""
    pairs: list[dict] = []
    for path in paths:
        if not path.is_file() or path.suffix.lower() != ".json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for record_index, record in enumerate(data if isinstance(data, list) else [data]):
            messages = record.get("messages", []) if isinstance(record, dict) else []
            for i in range(len(messages) - 1):
                question, answer = messages[i], messages[i + 1]
                if (question.get("role") == "user" and answer.get("role") == "assistant"
                        and isinstance(question.get("content"), str)
                        and isinstance(answer.get("content"), str)
                        and question["content"].strip() and answer["content"].strip()):
                    pairs.append({"question": question["content"].strip(),
                                  "answer": answer["content"].strip(),
                                  "source": f"{path}#{record_index}", "position": i})
    return pairs


def dedupe_pairs(pairs: list[dict]) -> list[dict]:
    """Keep the first pair per normalized question, counting repetitions."""
    result: list[dict] = []
    by_key: dict[str, dict] = {}
    for pair in pairs:
        key = " ".join(content_terms(pair["question"]))
        key = key or fold(pair["question"]).strip()
        if key in by_key:
            by_key[key]["repetitions"] += 1
        else:
            entry = dict(pair, key=key, repetitions=1)
            by_key[key] = entry
            result.append(entry)
    return result


def build_idf(question_terms: list[list[str]]) -> dict[str, float]:
    """Smooth inverse document frequency over the stored questions."""
    document_frequency: Counter = Counter()
    for terms in question_terms:
        document_frequency.update(set(terms))
    total = len(question_terms)
    return {term: math.log((total + 1) / (count + 1)) + 1.0
            for term, count in document_frequency.items()}


def vectorize(terms: list[str], idf: dict[str, float]) -> dict[str, float]:
    """Log-scaled term frequency weighted by idf, normalized to unit length."""
    counts = Counter(terms)
    vector = {term: (1.0 + math.log(count)) * idf[term]
              for term, count in counts.items() if term in idf}
    norm = math.sqrt(sum(weight * weight for weight in vector.values())) or 1.0
    return {term: weight / norm for term, weight in vector.items()}


def cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(weight * right.get(term, 0.0) for term, weight in left.items())


class KnowledgeBase:
    """Question/answer store indexed by TF-IDF vectors of the stored questions."""

    def __init__(self, pairs: list[dict]):
        self.pairs = dedupe_pairs(pairs)
        self.question_terms = [content_terms(pair["question"]) for pair in self.pairs]
        self.idf = build_idf(self.question_terms)
        self.vectors = [vectorize(terms, self.idf) for terms in self.question_terms]

    @classmethod
    def from_paths(cls, paths: list[Path]) -> "KnowledgeBase":
        return cls(load_pairs(paths))

    def stats(self) -> dict:
        return {"stored_pairs": len(self.pairs),
                "repeated_questions_removed": sum(p["repetitions"] - 1 for p in self.pairs),
                "vocabulary": len(self.idf),
                "average_question_terms": round(
                    sum(len(t) for t in self.question_terms) / max(1, len(self.question_terms)), 2)}

    def search(self, query: str, top_k: int = 3, exclude_key: str | None = None) -> list[dict]:
        """Ranked stored questions by cosine similarity; scores are in [0, 1]."""
        query_vector = vectorize(content_terms(query), self.idf)
        if not query_vector:
            return []
        scored = []
        for pair, terms, vector in zip(self.pairs, self.question_terms, self.vectors):
            if exclude_key is not None and pair["key"] == exclude_key:
                continue
            score = cosine(query_vector, vector)
            # Exact content-word overlap on real questions is strong evidence; single-term
            # queries are too weak to boost, since any question containing that term matches.
            query_terms = set(content_terms(query))
            pair_terms = set(terms)
            if len(query_terms) >= 2 and query_terms == pair_terms:
                score = 1.0
            elif len(query_terms) >= 2 and query_terms < pair_terms:
                score = max(score, 0.9)
            scored.append((score, pair))
        scored.sort(key=lambda item: (-item[0], item[1]["question"]))
        return [{"score": round(score, 4), "question": pair["question"], "answer": pair["answer"],
                 "source": pair["source"], "key": pair["key"], "repetitions": pair["repetitions"]}
                for score, pair in scored[:max(1, top_k)]]


class HybridResponder:
    """Answers by retrieval when confident, otherwise falls back to statistical generation."""

    def __init__(self, kb: KnowledgeBase, model: StatisticalModel | None = None,
                 threshold: float = 0.35, fallback_tokens: int = 60,
                 fallback_top_k: int = 10, fallback_temperature: float = 1.0):
        self.kb = kb
        self.model = model
        self.threshold = threshold
        self.fallback_tokens = fallback_tokens
        self.fallback_top_k = fallback_top_k
        self.fallback_temperature = fallback_temperature

    def _retrieval(self, query: str, top_k: int) -> dict | None:
        candidates = self.kb.search(query, top_k=top_k)
        if not candidates or candidates[0]["score"] < self.threshold:
            return None
        return candidates[0]

    def answer(self, question: str, history: list[tuple[str, str]] | None = None,
               seed: int | None = None) -> dict:
        """Return a dict with the answer text, its method and the evidence used."""
        question = question.strip()
        if not question:
            return {"answer": "", "method": "empty", "score": 0.0}
        best = self._retrieval(question, top_k=1)
        # A follow-up with few content words is answered with the previous question attached.
        if best is None and history:
            previous = [text for role, text in history if role == "user"]
            if previous:
                best = self._retrieval(f"{previous[-1]} {question}", top_k=1)
                if best is not None:
                    return {"answer": best["answer"], "method": "recovery-context",
                            "score": best["score"], "matched_question": best["question"],
                            "source": best["source"], "candidates": self.kb.search(question, top_k=3)}
        if best is not None:
            return {"answer": best["answer"], "method": "recovery", "score": best["score"],
                    "matched_question": best["question"], "source": best["source"],
                    "candidates": self.kb.search(question, top_k=3)}
        if self.model is not None:
            generated, _report = self.model.generate_by_frequency(
                prompt=question, max_tokens=self.fallback_tokens, top_k=self.fallback_top_k,
                temperature=self.fallback_temperature, seed=seed, sample=seed is not None,
                frequency_penalty=1.5)
            generated = generated[len(question):].strip() if generated.startswith(question) else generated.strip()
            return {"answer": generated, "method": "statistical", "score": 0.0,
                    "note": "nenhuma pergunta armazenada foi suficientemente parecida"}
        return {"answer": "Não encontrei nada parecido na base e o modelo estatístico não está carregado.",
                "method": "unavailable", "score": 0.0}


def command_ask(args):
    kb = KnowledgeBase.from_paths([Path(p) for p in args.dataset])
    model = _load_model(args.model) if not args.no_fallback else None
    responder = HybridResponder(kb, model, threshold=args.threshold,
                                fallback_tokens=args.tokens, fallback_top_k=args.top_k,
                                fallback_temperature=args.temperature)
    result = responder.answer(args.question, seed=args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def command_chat(args):
    kb = KnowledgeBase.from_paths([Path(p) for p in args.dataset])
    model = _load_model(args.model) if not args.no_fallback else None
    responder = HybridResponder(kb, model, threshold=args.threshold,
                                fallback_tokens=args.tokens, fallback_top_k=args.top_k,
                                fallback_temperature=args.temperature)
    history: list[tuple[str, str]] = []
    print(f"Base: {len(kb.pairs)} perguntas/respostas. Comandos: /contexto /limpar /sair")
    while True:
        try:
            text = input("você> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text in ("/sair", "/quit"):
            break
        if text == "/limpar":
            history.clear()
            print("miniia> Histórico limpo.")
            continue
        if text == "/contexto":
            used = sum(len(tokenize(t)) for _, t in history)
            print(f"miniia> {len(history)} turnos, ~{used} tokens.")
            for role, message in history[-6:]:
                print(f"  {role}> {message[:80]}")
            continue
        result = responder.answer(text, history=history, seed=args.seed)
        history.append(("user", text))
        history.append(("assistant", result["answer"]))
        marker = {"recovery": "recuperada", "recovery-context": "recuperada (com contexto)",
                  "statistical": "estatística", "unavailable": "indisponível"}.get(result["method"], result["method"])
        print(f"miniia> [{marker} {result['score']:.2f}] {result['answer']}")


def command_index(args):
    kb = KnowledgeBase.from_paths([Path(p) for p in args.dataset])
    print(json.dumps(kb.stats(), ensure_ascii=False, indent=2))


def command_eval(args):
    """Leave-one-out retrieval quality: each stored question is queried without itself."""
    pairs = load_pairs([Path(p) for p in args.dataset])
    kb = KnowledgeBase(pairs)
    total = top1 = top3 = related = 0
    correct_scores: list[float] = []
    incorrect_scores: list[float] = []
    for pair in kb.pairs:
        results = kb.search(pair["question"], top_k=3, exclude_key=pair["key"])
        if not results:
            continue
        total += 1
        best = results[0]
        same_record = lambda item: item["source"] == pair["source"]
        if same_record(best):
            top1 += 1
            correct_scores.append(best["score"])
        else:
            incorrect_scores.append(best["score"])
        if any(same_record(r) for r in results):
            top3 += 1
        # A sibling question from the same conversation usually has a relevant answer.
        if best["source"].split("#")[-1] == pair["source"].split("#")[-1]:
            related += 1
    def percentile(values, fraction):
        if not values:
            return 0.0
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]
    report = {"questions_evaluated": total,
              "top1_exact": round(top1 / max(1, total), 4),
              "top3_exact": round(top3 / max(1, total), 4),
              "top1_same_conversation": round(related / max(1, total), 4),
              "score_correct_min": round(min(correct_scores), 4) if correct_scores else None,
              "score_correct_p10": round(percentile(correct_scores, 0.10), 4),
              "score_incorrect_max": round(max(incorrect_scores), 4) if incorrect_scores else None,
              "score_incorrect_p90": round(percentile(incorrect_scores, 0.90), 4),
              "suggested_threshold": round(percentile(incorrect_scores, 0.90), 4)}
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _load_model(path: str) -> StatisticalModel | None:
    file = Path(path)
    if not file.is_file():
        print(f"aviso: modelo estatístico '{path}' não encontrado; respostas ficam limitadas à base",
              file=sys.stderr)
        return None
    return StatisticalModel.load(file)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Resolvedor híbrido: recupera respostas da base e usa geração estatística como fallback")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", nargs="+", default=["dataset_texto_normal.json"],
                        help="arquivos JSON de conversas com roles user/assistant")
    common.add_argument("--model", default="modelo.json", help="modelo estatístico do miniia para o fallback")
    common.add_argument("--no-fallback", action="store_true", help="responde apenas pela base, sem geração estatística")
    common.add_argument("--threshold", type=float, default=0.35,
                        help="similaridade mínima para aceitar uma resposta recuperada")
    common.add_argument("--tokens", type=int, default=60, help="limite de tokens no fallback estatístico")
    common.add_argument("--top-k", type=int, default=10, help="candidatos do fallback e do /contexto")
    common.add_argument("--temperature", type=float, default=1.0)
    common.add_argument("--seed", type=int)
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask", parents=[common], help="responde uma pergunta e mostra a evidência")
    ask.add_argument("question")
    ask.set_defaults(func=command_ask)
    chat = sub.add_parser("chat", parents=[common], help="sessão interativa")
    chat.set_defaults(func=command_chat)
    index = sub.add_parser("index", parents=[common], help="estatísticas da base de conhecimento")
    index.set_defaults(func=command_index)
    evaluate = sub.add_parser("eval", parents=[common], help="validação leave-one-out da recuperação")
    evaluate.set_defaults(func=command_eval)
    return parser


if __name__ == "__main__":
    # The corpus contains symbols (arrows, accents) that the Windows console codec rejects.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass
    args = build_parser().parse_args()
    args.func(args)
