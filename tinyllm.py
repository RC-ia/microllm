"""LLM minúscula (~5M parâmetros) em PyTorch puro: Transformer decoder char-level.

Treina do zero no dataset de conversas, sem dependências além de torch.
Arquitetura padrão GPT: embedding de token + posição, N blocos
(atenção causal multi-head + MLP), LayerNorm final e head de LM com
pesos amarrados ao embedding.

Uso (dentro de C:\\GPT\\dataset\\modelo):
    python tinyllm.py params
    python tinyllm.py train --dataset dataset_texto_normal.json --out tinyllm.pt --steps 3000
    python tinyllm.py generate --checkpoint tinyllm.pt --prompt "O que é uma LLM?"
    python tinyllm.py chat --checkpoint tinyllm.pt
    python tinyllm.py eval --checkpoint tinyllm.pt --dataset dataset_texto_normal.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

try:
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.checkpoint import checkpoint
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyTorch não encontrado. Execute: pip install torch") from exc

# Tokens especiais: ids reservados 0..N-1, o resto são caracteres do dataset.
PAD, EOS, UNK, USER, ASSISTANT = "<pad>", "<eos>", "<unk>", "<user>", "<assistant>"
SPECIALS = [PAD, EOS, UNK, USER, ASSISTANT]

PAD_ID, EOS_ID = 0, 1  # fixos: SPECIALS = [<pad>, <eos>, <unk>, <user>, <assistant>]

DEFAULT_CONFIG = {
    # Janela de 8192 caracteres: cabe a conversa inteira do corpus (resposta
    # máxima = 7.639 chars). O embedding posicional custa block_size*n_embd, então
    # n_layer cai para 4 para o total continuar na classe de 5M.
    "n_layer": 16,
    "n_head": 32,
    "n_embd": 512,
    "block_size": 8192,
    "dropout": 0.1,
    "vocab_size": 0,  # preenchido após construir o tokenizador
}


# ---------------------------------------------------------------- tokenizador
class CharTokenizer:
    """Char-level + tokens especiais (<user>, <assistant>, <eos>...).

    Aprendido 100% do dataset: nenhum vocabulário externo, nada baixado.
    """

    def __init__(self, stoi: dict[str, int], itos: list[str]):
        self.stoi = stoi
        self.itos = itos

    @classmethod
    def build(cls, texts: list[str]) -> "CharTokenizer":
        chars = sorted(set("".join(texts)))
        itos = list(SPECIALS) + [c for c in chars if c not in SPECIALS]
        return cls({s: i for i, s in enumerate(itos)}, itos)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        i = 0
        while i < len(text):
            for special in (USER, ASSISTANT, EOS, PAD):
                if text.startswith(special, i):
                    ids.append(self.stoi[special])
                    i += len(special)
                    break
            else:
                ids.append(self.stoi.get(text[i], self.stoi[UNK]))
                i += 1
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids if 0 <= i < len(self.itos))


# -------------------------------------------------------------------- modelo
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, embd = x.shape
        qkv = self.qkv(x).view(batch, seq, 3, self.n_head, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        # SDPA com is_causal: a máscara causal é aplicada por bloco, então a memória
        # não cresce com seq². Essencial para janelas de milhares de caracteres.
        out = F.scaled_dot_product_attention(
            query, key, value,
            dropout_p=self.drop.p if self.training else 0.0,
            is_causal=True,
        )
        out = out.transpose(1, 2).contiguous().view(batch, seq, embd)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float, use_checkpoint: bool = False):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)
        self.use_checkpoint = use_checkpoint

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x)))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Checkpointing de gradiente: recalcula as ativações no backward em vez de
        # guardá-las. Troca ~memória por ~30% de tempo; essencial em janela longa.
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


class TinyGPT(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = dict(config)
        vocab, embd, block = config["vocab_size"], config["n_embd"], config["block_size"]
        self.tok_emb = nn.Embedding(vocab, embd)
        self.pos_emb = nn.Embedding(block, embd)
        self.drop = nn.Dropout(config["dropout"])
        self.blocks = nn.ModuleList(
            [Block(embd, config["n_head"], config["dropout"],
                   config.get("grad_checkpoint", False))
             for _ in range(config["n_layer"])]
        )
        self.ln_f = nn.LayerNorm(embd)
        self.head = nn.Linear(embd, vocab, bias=False)
        self.head.weight = self.tok_emb.weight  # amarra: economiza V*E parâmetros
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        _, seq = idx.shape
        if seq > self.config["block_size"]:
            raise ValueError(f"sequência {seq} > block_size {self.config['block_size']}")
        positions = torch.arange(seq, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(positions))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            # ignore_index=-100 descarta o padding preenchido em sample_batch.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1),
                                   ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new: int, temperature: float = 0.8,
                 top_k: int = 40, min_new_tokens: int = 10) -> torch.Tensor:
        self.eval()
        for step in range(max_new):
            window = idx[:, -self.config["block_size"]:]
            logits, _ = self(window)
            logits = logits[:, -1, :]
            # <pad> nunca aparece no treino: proibido sempre. <eos> só após
            # min_new_tokens, para checkpoint parcial não devolver vazio.
            logits[:, PAD_ID] = float("-inf")
            if step < min_new_tokens:
                logits[:, EOS_ID] = float("-inf")
            if temperature <= 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / max(1e-6, temperature)
                if top_k and top_k < logits.size(-1):
                    cutoff = torch.topk(logits, top_k).values[:, -1:]
                    logits = torch.where(logits < cutoff, torch.tensor(float("-inf"), device=logits.device), logits)
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
            if int(nxt.item()) == EOS_ID:
                break
        return idx


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------- dados
def load_conversations(path: Path) -> list[list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data if isinstance(data, list) else [data]
    convs = []
    for record in records:
        messages = record.get("messages", []) if isinstance(record, dict) else []
        turns = [(m.get("role", ""), m.get("content", "")) for m in messages
                 if isinstance(m, dict) and isinstance(m.get("content"), str) and m.get("content", "").strip()]
        if turns:
            convs.append([{"role": r, "content": c} for r, c in turns])
    if not convs:
        raise ValueError(f"nenhuma conversa válida em {path}")
    return convs


def format_conversation(messages: list[dict]) -> str:
    """Formato de treino: papéis como tokens especiais de uma peça só."""
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", "")).strip()
        if role == "user":
            parts.append(f"{USER}\n{content}\n")
        elif role == "assistant":
            parts.append(f"{ASSISTANT}\n{content}\n")
        elif content:
            parts.append(content + "\n")
    parts.append(f"{EOS}\n")
    return "".join(parts)


def build_corpus(dataset: Path) -> tuple[CharTokenizer, list[list[int]]]:
    convs = load_conversations(dataset)
    texts = [format_conversation(c) for c in convs]
    tokenizer = CharTokenizer.build(texts)
    return tokenizer, [tokenizer.encode(t) for t in texts]


def sample_batch(ids_list: list[list[int]], batch: int, block: int,
                 rng: random.Random) -> tuple[torch.Tensor, torch.Tensor]:
    """Amostra pedaços causais de até `block` caracteres.

    O comprimento efetivo é o da maior amostra do lote (nunca `block` fixo): com
    block grande e conversas curtas, fixar o bloco desperdiçaria quase todo o
    compute. Alvos preenchidos com -100 são ignorados na loss, então o padding
    à direita não altera o sinal.
    """
    chunks: list[list[int]] = []
    for _ in range(batch):
        ids = rng.choice(ids_list)
        if len(ids) < 2:
            ids = list(ids) + [EOS_ID] * (2 - len(ids))
        length = min(block + 1, len(ids))
        start = rng.randrange(0, len(ids) - length + 1)
        chunks.append(ids[start:start + length])
    width = max(len(chunk) - 1 for chunk in chunks)  # entradas são chunk[:-1]
    inputs = torch.full((batch, width), PAD_ID, dtype=torch.long)
    targets = torch.full((batch, width), -100, dtype=torch.long)
    for i, chunk in enumerate(chunks):
        inputs[i, :len(chunk) - 1] = torch.tensor(chunk[:-1], dtype=torch.long)
        targets[i, :len(chunk) - 1] = torch.tensor(chunk[1:], dtype=torch.long)
    return inputs, targets


@torch.no_grad()
def evaluate_loss(model: TinyGPT, ids_list: list[list[int]], batch: int, block: int,
                 val_batches: int, device: torch.device, seed: int = 0) -> float:
    model.eval()
    rng = random.Random(seed)
    total = 0.0
    for _ in range(val_batches):
        x, y = sample_batch(ids_list, batch, block, rng)
        _, loss = model(x.to(device), y.to(device))
        total += loss.item()
    return total / max(1, val_batches)


# ------------------------------------------------------------------ comandos
def setup_distributed() -> tuple[int, int]:
    if dist.is_available() and not dist.is_initialized():
        world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
        if world_size > 1:
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0") or "0")
            return torch.device(f"cuda:{local_rank}")
        return torch.device("cpu")
    if name == "cuda":
        local_rank = int(os.environ.get("LOCAL_RANK", "0") or "0")
        return torch.device(f"cuda:{local_rank}")
    return torch.device(name)


def command_params(args) -> None:
    tokenizer, _ = build_corpus(Path(args.dataset))
    config = dict(DEFAULT_CONFIG, vocab_size=tokenizer.vocab_size)
    model = TinyGPT(config)
    total = count_parameters(model)
    print(json.dumps({"vocab_size": tokenizer.vocab_size, "config": config,
                      "parameters": total, "millions": round(total / 1e6, 3)},
                     ensure_ascii=False, indent=2))


def command_train(args) -> None:
    rank, world_size = setup_distributed()
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    dataset = Path(args.dataset)
    tokenizer, all_ids = build_corpus(dataset)
    rng = random.Random(args.seed + rank)
    rng.shuffle(all_ids)
    cut = max(1, int(len(all_ids) * 0.9))
    train_ids, val_ids = all_ids[:cut], all_ids[cut:]
    train_chars = sum(len(s) for s in train_ids)

    config = dict(DEFAULT_CONFIG, vocab_size=tokenizer.vocab_size,
                  block_size=args.block, dropout=args.dropout,
                  grad_checkpoint=not args.no_grad_checkpoint)
    model = TinyGPT(config).to(device)
    if dist.is_initialized():
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    total = count_parameters(model)
    if rank == 0:
        print(f"parâmetros: {total} ({total/1e6:.2f}M) | vocab: {tokenizer.vocab_size} | "
              f"treino: {len(train_ids)} conversas (~{train_chars} chars) | val: {len(val_ids)} | device: {device}")
        print(f"janela: {args.block} chars | batch: {args.batch} | "
              f"grad_checkpoint: {config['grad_checkpoint']}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                  weight_decay=0.1)
    warmup = max(1, int(args.steps * 0.05))

    def schedule(step: int) -> float:
        if step < warmup:
            return args.lr * (step + 1) / warmup
        progress = (step - warmup) / max(1, args.steps - warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))

    # --resume retoma de onde parou: mesmo passo, mesmo melhor val_loss, mesmo
    # estado do otimizador e o mesmo RNG. Sem isso o LR voltaria ao topo do
    # cosseno e o modelo reinsistiria no que já tinha aprendido.
    start_step, best_val = 1, float("inf")
    if args.resume:
        if not Path(args.resume).is_file():
            raise SystemExit(f"checkpoint de retomada não encontrado: {args.resume}")
        blob = torch.load(args.resume, map_location=device, weights_only=False)
        saved_config = dict(blob["config"])
        for key in ("n_layer", "n_head", "n_embd", "block_size"):
            if saved_config.get(key) != config[key]:
                print(f"aviso: {key} do checkpoint ({saved_config.get(key)}) difere do "
                      f"argumento ({config[key]}); vale o do checkpoint.", file=sys.stderr)
        model.load_state_dict(blob["model"])
        if "optimizer" in blob:
            optimizer.load_state_dict(blob["optimizer"])
        else:
            print("aviso: checkpoint sem estado do otimizador; os momentos do AdamW "
                  "recomeçam (o LR ainda continua do passo certo).", file=sys.stderr)
        restore_rng_state(blob.get("rng"))
        start_step = int(blob.get("step", 0)) + 1
        best_val = float(blob.get("best_val", blob.get("val_loss", float("inf"))))
        if rank == 0:
            print(f"retomando de '{args.resume}' no passo {start_step}/{args.steps} "
                  f"(melhor val_loss até agora: {best_val:.4f}, tag='{blob.get('tag', '')}')")
        if start_step > args.steps:
            raise SystemExit(f"o checkpoint já está no passo {start_step - 1}, que é >= "
                             f"--steps {args.steps}; aumente --steps para continuar.")

    # Acumulação de gradiente compensa um batch pequeno sem custo de memória.
    micro = max(1, args.batch // args.grad_accum)

    def micro_batch() -> tuple[torch.Tensor, torch.Tensor]:
        x, y = sample_batch(train_ids, micro, args.block, rng)
        return x.to(device), y.to(device)

    model.train()
    for step in range(start_step, args.steps + 1):
        for group in optimizer.param_groups:
            group["lr"] = schedule(step)
        optimizer.zero_grad()
        total_loss = 0.0
        for _ in range(args.grad_accum):
            x, y = micro_batch()
            _, loss = model(x, y)
            # Normaliza por micro-lote para o gradiente independer do accumulate.
            (loss / args.grad_accum).backward()
            total_loss += loss.item() / args.grad_accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_value = total_loss
        if rank == 0 and (step % args.log_every == 0 or step == 1):
            vram = (torch.cuda.max_memory_allocated() / 2**30
                    if device.type == "cuda" else 0.0)
            print(f"step {step}/{args.steps} loss={loss_value:.4f} "
                  f"ppl={math.exp(min(20.0, loss_value)):.1f} vram={vram:.2f}GB", flush=True)
        if rank == 0 and (step % args.eval_every == 0 or step == args.steps):
            val = evaluate_loss(model, val_ids, args.batch, args.block, args.val_batches, device)
            print(f"  [eval] step {step} val_loss={val:.4f} val_ppl={math.exp(min(20.0, val)):.1f}", flush=True)
            if val < best_val:
                best_val = val
                save_checkpoint(args.out, model, tokenizer, config, step, val,
                                optimizer=optimizer, rng_state=capture_rng_state(),
                                best_val=best_val, tag=args.tag)
                print(f"  [eval] novo melhor checkpoint salvo em {args.out}", flush=True)
            model.train()
    if rank == 0:
        save_checkpoint(args.out, model, tokenizer, config, args.steps, best_val,
                        optimizer=optimizer, rng_state=capture_rng_state(),
                        best_val=best_val, tag=args.tag)
        print(f"treino concluído. checkpoint final: {args.out} (melhor val_loss={best_val:.4f})")


def save_checkpoint(path: str, model: TinyGPT, tokenizer: CharTokenizer,
                    config: dict, step: int, val_loss: float, optimizer=None,
                    rng_state: dict | None = None, best_val: float | None = None,
                    tag: str = "") -> None:
    """Salva o estado completo do treino, não só os pesos.

    Guardar o otimizador e o RNG é o que torna `--resume` equivalente a não ter
    parado: sem os momentos do AdamW o LR se reacomoda errado, e sem o RNG a
    ordem dos dados muda. Checkpoints antigos (sem esses campos) continuam
    carregando — apenas retomam com esses dois zerados.
    """
    # Em execução distribuída, só o rank 0 salva para evitar colisões no mesmo arquivo.
    # O modelo em DDP ainda envia state_dict sem wrappers, então o save funciona.
    if isinstance(model, DDP):
        model_to_save = model.module
    else:
        model_to_save = model
    blob = {"config": config, "stoi": tokenizer.stoi, "itos": tokenizer.itos,
            "model": {k: v.cpu() for k, v in model_to_save.state_dict().items()},
            "step": step, "val_loss": val_loss, "tag": tag,
            "best_val": best_val if best_val is not None else val_loss}
    if optimizer is not None:
        blob["optimizer"] = optimizer.state_dict()
    if rng_state is not None:
        blob["rng"] = rng_state
    torch.save(blob, path)


def capture_rng_state() -> dict:
    # Guarda como bytes crus: passar o ByteTensor pelo torch.save/load faz o
    # roundtrip devolvê-lo como Tensor genérico, e set_rng_state exige ByteTensor.
    return {"torch": torch.get_rng_state().numpy().tobytes(),
            "python": random.getstate()}


def restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    if "torch" in state:
        torch.set_rng_state(torch.frombuffer(bytearray(state["torch"]), dtype=torch.uint8))
    if "python" in state:
        random.setstate(state["python"])


def load_checkpoint(path: str, device: torch.device) -> tuple[TinyGPT, CharTokenizer, dict]:
    blob = torch.load(path, map_location=device, weights_only=False)
    tokenizer = CharTokenizer(dict(blob["stoi"]), list(blob["itos"]))
    model = TinyGPT(dict(blob["config"])).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, tokenizer, blob


def answer_question(model: TinyGPT, tokenizer: CharTokenizer, question: str,
                    device: torch.device, max_new: int, temperature: float, top_k: int,
                    min_new_tokens: int = 10) -> str:
    prompt = f"{USER}\n{question.strip()}\n{ASSISTANT}\n"
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(idx, max_new, temperature, top_k, min_new_tokens)
    text = tokenizer.decode(out[0].tolist())
    reply = text[len(prompt):] if text.startswith(prompt) else text
    cut = reply.find(EOS)
    if cut != -1:
        reply = reply[:cut]
    cut = reply.find(USER)
    if cut != -1:
        reply = reply[:cut]
    return reply.strip()


def command_generate(args) -> None:
    device = resolve_device(args.device)
    model, tokenizer, blob = load_checkpoint(args.checkpoint, device)
    reply = answer_question(model, tokenizer, args.prompt, device,
                            args.max_new, args.temperature, args.top_k, args.min_new)
    if args.json:
        print(json.dumps({"answer": reply, "checkpoint": args.checkpoint,
                          "step": blob.get("step"), "val_loss": blob.get("val_loss")},
                         ensure_ascii=False, indent=2))
    else:
        print(reply if reply else "(o modelo não gerou texto; tente outra temperatura.)")


def command_chat(args) -> None:
    device = resolve_device(args.device)
    model, tokenizer, _ = load_checkpoint(args.checkpoint, device)
    print(f"tinyllm em {device}. /sair encerra, /limpar apaga o histórico.")
    history: list[tuple[str, str]] = []
    while True:
        try:
            text = input("\nvocê> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in ("/sair", "/quit", "/exit"):
            break
        if text.lower() == "/limpar":
            history.clear()
            print("miniia> histórico limpo.")
            continue
        context = "".join(f"{USER}\n{u}\n{ASSISTANT}\n{a}\n" for u, a in history[-4:])
        reply = answer_question(model, tokenizer, context + text, device,
                                args.max_new, args.temperature, args.top_k, args.min_new)
        print(f"miniia> {reply if reply else '(sem resposta; tente de novo.)'}")
        history.append((text, reply))


def command_eval(args) -> None:
    device = resolve_device(args.device)
    model, tokenizer, blob = load_checkpoint(args.checkpoint, device)
    _, all_ids = build_corpus(Path(args.dataset))
    rng = random.Random(17)
    rng.shuffle(all_ids)
    val_ids = all_ids[max(1, int(len(all_ids) * 0.9)):]
    val = evaluate_loss(model, val_ids, args.batch, model.config["block_size"],
                        args.val_batches, device)
    print(json.dumps({"val_loss": round(val, 4),
                      "val_perplexity": round(math.exp(min(20.0, val)), 2),
                      "checkpoint_step": blob.get("step"),
                      "checkpoint_val_loss": blob.get("val_loss")}, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM minúscula (~5M) em PyTorch puro")
    parser.add_argument("--device", default="auto", help="auto, cuda ou cpu")
    sub = parser.add_subparsers(dest="command", required=True)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--dataset", default="dataset_texto_normal.json")

    p = sub.add_parser("params", parents=[shared], help="conta os parâmetros do modelo")
    p.set_defaults(func=command_params)

    t = sub.add_parser("train", parents=[shared], help="treina do zero no dataset")
    t.add_argument("--out", default="tinyllm.pt")
    t.add_argument("--resume", default=None,
                   help="retoma deste checkpoint (mesmo passo, otimizador e RNG)")
    t.add_argument("--tag", default="", help="rótulo gravado no checkpoint, para identificar a rodada")
    t.add_argument("--steps", type=int, default=3000)
    t.add_argument("--batch", type=int, default=8, help="batch efetivo total")
    t.add_argument("--grad-accum", type=int, default=2,
                   help="micro-lotes por passo; --batch é dividido entre eles (não aumenta VRAM)")
    t.add_argument("--no-grad-checkpoint", action="store_true",
                   help="desliga o checkpointing de gradiente (mais rápido, muito mais memória)")
    t.add_argument("--block", type=int, default=8192)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--log-every", type=int, default=50)
    t.add_argument("--eval-every", type=int, default=300)
    t.add_argument("--val-batches", type=int, default=40)
    t.set_defaults(func=command_train)

    g = sub.add_parser("generate", help="responde uma pergunta com o checkpoint")
    g.add_argument("--checkpoint", default="tinyllm.pt")
    g.add_argument("--prompt", required=True)
    g.add_argument("--max-new", type=int, default=300)
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--top-k", type=int, default=40)
    g.add_argument("--min-new", type=int, default=10,
                   help="tokens mínimos antes de permitir <eos> (evita resposta vazia)")
    g.add_argument("--json", action="store_true")
    g.set_defaults(func=command_generate)

    c = sub.add_parser("chat", help="conversa interativa no terminal")
    c.add_argument("--checkpoint", default="tinyllm.pt")
    c.add_argument("--max-new", type=int, default=300)
    c.add_argument("--temperature", type=float, default=0.8)
    c.add_argument("--top-k", type=int, default=40)
    c.add_argument("--min-new", type=int, default=10,
                   help="tokens mínimos antes de permitir <eos> (evita resposta vazia)")
    c.set_defaults(func=command_chat)

    e = sub.add_parser("eval", parents=[shared], help="perda/perplexidade na validação")
    e.add_argument("--checkpoint", default="tinyllm.pt")
    e.add_argument("--batch", type=int, default=32)
    e.add_argument("--val-batches", type=int, default=40)
    e.set_defaults(func=command_eval)
    return parser


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass
    args = build_parser().parse_args()
    args.func(args)
