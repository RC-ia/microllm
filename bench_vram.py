import random
import time

import torch
from torch.utils.checkpoint import checkpoint

from tinyllm import DEFAULT_CONFIG, TinyGPT, build_corpus, count_parameters, sample_batch
from pathlib import Path

tok, ids = build_corpus(Path("dataset_texto_normal.json"))
CFG = dict(DEFAULT_CONFIG, vocab_size=tok.vocab_size)
dev = torch.device("cuda")


def run(block, batch, grad_ckpt, amp, steps=3):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = TinyGPT(dict(CFG, grad_checkpoint=grad_ckpt)).to(dev)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp == "fp16")
    rng = random.Random(0)
    peak, dt = 0.0, 0.0
    for _ in range(steps):
        try:
            x, y = sample_batch(ids, batch, block, rng)
            x, y = x.to(dev), y.to(dev)
            t = time.perf_counter()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16 if amp == "bf16" else torch.float16, enabled=bool(amp)):
                _, loss = model(x, y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            torch.cuda.synchronize()
            dt += time.perf_counter() - t
            peak = max(peak, torch.cuda.max_memory_allocated() / 2**30)
        except torch.cuda.OutOfMemoryError:
            print(f"  block={block} batch={batch} ckpt={grad_ckpt} amp={amp or 'fp32'} -> OUT OF MEMORY")
            del model, opt
            torch.cuda.empty_cache()
            return None
    del model, opt
    torch.cuda.empty_cache()
    return dt / steps, peak


print(f"params={count_parameters(TinyGPT(CFG))} | device={torch.cuda.get_device_name(0)}\n")
print(f"{'block':>6} {'batch':>5} {'ckpt':>5} {'amp':>5} {'s/step':>7} {'VRAM GB':>8}  economia")
base = None
for block, batch in ((8192, 8), (8192, 4)):
    for ckpt, amp in ((False, ""), (True, ""), (False, "fp16"), (True, "fp16"), (True, "bf16")):
        r = run(block, batch, ckpt, amp)
        if r is None:
            continue
        dt, peak = r
        if base is None and block == 8192 and batch == 8 and not ckpt and not amp:
            base = peak
        saving = f"-{100*(1-peak/base):.0f}%" if base and peak < base else ""
        print(f"{block:>6} {batch:>5} {str(ckpt):>5} {amp or 'fp32':>5} {dt:>7.2f} {peak:>8.2f}  {saving}")