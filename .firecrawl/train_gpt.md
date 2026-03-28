"""
The \`train\_gpt.py\` and \`train\_gpt\_mlx.py\` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the \`/records\` folder.

Hard stop: To keep readable for newcomers, let's make sure \`train\_gpt.py\` and \`train\_gpt\_mlx.py\` never are longer than 1500 lines.
"""

from \_\_future\_\_ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

\# -----------------------------
\# HYPERPARAMETERS
\# -----------------------------
\# Default Simple Baseline run:
\# - 9 transformer blocks at width 512
\# - 8 attention heads with 4 KV heads (GQA) and 2x MLP expansion
\# - vocab size 1024, sequence length 1024, tied embeddings
\# - 524,288 train tokens per step for 20,000 iterations with a ~10 minute cap

class Hyperparameters:
 # Data paths are shard globs produced by the existing preprocessing pipeline.
 data\_path = os.environ.get("DATA\_PATH", "./data/datasets/fineweb10B\_sp1024")
 train\_files = os.path.join(data\_path, "fineweb\_train\_\*.bin")
 val\_files = os.path.join(data\_path, "fineweb\_val\_\*.bin")
 tokenizer\_path = os.environ.get("TOKENIZER\_PATH", "./data/tokenizers/fineweb\_1024\_bpe.model")
 run\_id = os.environ.get("RUN\_ID", str(uuid.uuid4()))
 seed = int(os.environ.get("SEED", 1337))

 # Validation cadence and batch size. Validation always uses the full fineweb\_val split.
 val\_batch\_size = int(os.environ.get("VAL\_BATCH\_SIZE", 524\_288))
 val\_loss\_every = int(os.environ.get("VAL\_LOSS\_EVERY", 1000))
 train\_log\_every = int(os.environ.get("TRAIN\_LOG\_EVERY", 200))

 # Training length.
 iterations = int(os.environ.get("ITERATIONS", 20000))
 warmdown\_iters = int(os.environ.get("WARMDOWN\_ITERS", 1200))
 warmup\_steps = int(os.environ.get("WARMUP\_STEPS", 20))
 train\_batch\_tokens = int(os.environ.get("TRAIN\_BATCH\_TOKENS", 524\_288))
 train\_seq\_len = int(os.environ.get("TRAIN\_SEQ\_LEN", 1024))
 max\_wallclock\_seconds = float(os.environ.get("MAX\_WALLCLOCK\_SECONDS", 600.0))
 qk\_gain\_init = float(os.environ.get("QK\_GAIN\_INIT", 1.5))

 # Model shape.
 vocab\_size = int(os.environ.get("VOCAB\_SIZE", 1024))
 num\_layers = int(os.environ.get("NUM\_LAYERS", 9))
 num\_kv\_heads = int(os.environ.get("NUM\_KV\_HEADS", 4))
 model\_dim = int(os.environ.get("MODEL\_DIM", 512))
 num\_heads = int(os.environ.get("NUM\_HEADS", 8))
 mlp\_mult = int(os.environ.get("MLP\_MULT", 2))
 tie\_embeddings = bool(int(os.environ.get("TIE\_EMBEDDINGS", "1")))
 rope\_base = float(os.environ.get("ROPE\_BASE", 10000.0))
 logit\_softcap = float(os.environ.get("LOGIT\_SOFTCAP", 30.0))

 # Optimizer hyperparameters.
 embed\_lr = float(os.environ.get("EMBED\_LR", 0.6))
 head\_lr = float(os.environ.get("HEAD\_LR", 0.008))
 tied\_embed\_lr = float(os.environ.get("TIED\_EMBED\_LR", 0.05))
 tied\_embed\_init\_std = float(os.environ.get("TIED\_EMBED\_INIT\_STD", 0.005))
 matrix\_lr = float(os.environ.get("MATRIX\_LR", 0.04))
 scalar\_lr = float(os.environ.get("SCALAR\_LR", 0.04))
 muon\_momentum = float(os.environ.get("MUON\_MOMENTUM", 0.95))
 muon\_backend\_steps = int(os.environ.get("MUON\_BACKEND\_STEPS", 5))
 muon\_momentum\_warmup\_start = float(os.environ.get("MUON\_MOMENTUM\_WARMUP\_START", 0.85))
 muon\_momentum\_warmup\_steps = int(os.environ.get("MUON\_MOMENTUM\_WARMUP\_STEPS", 500))
 beta1 = float(os.environ.get("BETA1", 0.9))
 beta2 = float(os.environ.get("BETA2", 0.95))
 adam\_eps = float(os.environ.get("ADAM\_EPS", 1e-8))
 grad\_clip\_norm = float(os.environ.get("GRAD\_CLIP\_NORM", 0.0))

\# -----------------------------
\# MUON OPTIMIZER
\# -----------------------------
\#
\# As borrowed from modded-nanogpt
\# Background on Muon: https://kellerjordan.github.io/posts/muon/

def zeropower\_via\_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
 # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
 # Muon uses this to normalize matrix-shaped gradients before applying them.
 a, b, c = (3.4445, -4.7750, 2.0315)
 X = G.bfloat16()
 X /= X.norm() + eps
 transposed = G.size(0) > G.size(1)
 if transposed:
 X = X.T
 for \_ in range(steps):
 A = X @ X.T
 B = b \* A + c \* A @ A
 X = a \* X + B @ X
 return X.T if transposed else X

class Muon(torch.optim.Optimizer):
 def \_\_init\_\_(self, params, lr: float, momentum: float, backend\_steps: int, nesterov: bool = True):
 super().\_\_init\_\_(
 params,
 dict(lr=lr, momentum=momentum, backend\_steps=backend\_steps, nesterov=nesterov),
 )

 @torch.no\_grad()
 def step(self, closure=None):
 loss = None
 if closure is not None:
 with torch.enable\_grad():
 loss = closure()

 distributed = dist.is\_available() and dist.is\_initialized()
 world\_size = dist.get\_world\_size() if distributed else 1
 rank = dist.get\_rank() if distributed else 0

 for group in self.param\_groups:
 params = group\["params"\]
 if not params:
 continue
 lr = group\["lr"\]
 momentum = group\["momentum"\]
 backend\_steps = group\["backend\_steps"\]
 nesterov = group\["nesterov"\]

 total\_params = sum(int(p.numel()) for p in params)
 updates\_flat = torch.zeros(total\_params, device=params\[0\].device, dtype=torch.bfloat16)

 curr = 0
 for i, p in enumerate(params):
 if i % world\_size == rank and p.grad is not None:
 g = p.grad
 state = self.state\[p\]
 if "momentum\_buffer" not in state:
 state\["momentum\_buffer"\] = torch.zeros\_like(g)
 buf = state\["momentum\_buffer"\]
 buf.mul\_(momentum).add\_(g)
 if nesterov:
 g = g.add(buf, alpha=momentum)
 g = zeropower\_via\_newtonschulz5(g, steps=backend\_steps)
 # Scale correction from Muon reference implementations.
 g \*= max(1, g.size(0) / g.size(1)) \*\* 0.5
 updates\_flat\[curr : curr + p.numel()\] = g.reshape(-1)
 curr += p.numel()

 if distributed:
 dist.all\_reduce(updates\_flat, op=dist.ReduceOp.SUM)

 curr = 0
 for p in params:
 g = updates\_flat\[curr : curr + p.numel()\].view\_as(p).to(dtype=p.dtype)
 p.add\_(g, alpha=-lr)
 curr += p.numel()

 return loss

\# -----------------------------
\# TOKENIZER-AGNOSTIC EVALUATION SETUP
\# -----------------------------
#
\# It's common for small models have a large fraction of their parameters be embeddings, since the 2 \* d\_model \* d\_vocab vectors can be gigantic.
\# Instead of locking the tokenizer, we let you bring your own and calculate our validation metrics on the average compression of the validation set.
\# We calculate BPB (bits-per-byte) instead of validation loss, so we need methods to count the number of bits per token in the tokenizer.
\# Note: Submissions that edit the tokenizer will be examined more carefully, since screwing this up might unjustly improve your score.

def build\_sentencepiece\_luts(
 sp: spm.SentencePieceProcessor, vocab\_size: int, device: torch.device
) -\> tuple\[Tensor, Tensor, Tensor\]:
 sp\_vocab\_size = int(sp.vocab\_size())
 table\_size = max(sp\_vocab\_size, vocab\_size)
 base\_bytes\_np = np.zeros((table\_size,), dtype=np.int16)
 has\_leading\_space\_np = np.zeros((table\_size,), dtype=np.bool\_)
 is\_boundary\_token\_np = np.ones((table\_size,), dtype=np.bool\_)
 for token\_id in range(sp\_vocab\_size):
 if sp.is\_control(token\_id) or sp.is\_unknown(token\_id) or sp.is\_unused(token\_id):
 continue
 is\_boundary\_token\_np\[token\_id\] = False
 if sp.is\_byte(token\_id):
 base\_bytes\_np\[token\_id\] = 1
 continue
 piece = sp.id\_to\_piece(token\_id)
 if piece.startswith("▁"):
 has\_leading\_space\_np\[token\_id\] = True
 piece = piece\[1:\]
 base\_bytes\_np\[token\_id\] = len(piece.encode("utf-8"))
 return (
 torch.tensor(base\_bytes\_np, dtype=torch.int16, device=device),
 torch.tensor(has\_leading\_space\_np, dtype=torch.bool, device=device),
 torch.tensor(is\_boundary\_token\_np, dtype=torch.bool, device=device),
 )

def load\_validation\_tokens(pattern: str, seq\_len: int) -> Tensor:
 files = \[Path(p) for p in sorted(glob.glob(pattern))\]
 if not files:
 raise FileNotFoundError(f"No files found for pattern: {pattern}")
 # The export pipeline writes the fixed first-50k-doc validation set to fineweb\_val\_\*.
 tokens = torch.cat(\[load\_data\_shard(file) for file in files\]).contiguous()
 usable = ((tokens.numel() - 1) // seq\_len) \* seq\_len
 if usable <= 0:
 raise ValueError(f"Validation split is too short for TRAIN\_SEQ\_LEN={seq\_len}")
 return tokens\[: usable + 1\]

def eval\_val(
 args: Hyperparameters,
 model: nn.Module,
 rank: int,
 world\_size: int,
 device: torch.device,
 grad\_accum\_steps: int,
 val\_tokens: Tensor,
 base\_bytes\_lut: Tensor,
 has\_leading\_space\_lut: Tensor,
 is\_boundary\_token\_lut: Tensor,
) -\> tuple\[float, float\]:
 # Validation computes two metrics:
 # \- val\_loss: token cross-entropy (natural log)
 # \- val\_bpb: tokenizer-agnostic compression metric used by the challenge
 local\_batch\_tokens = args.val\_batch\_size // (world\_size \* grad\_accum\_steps)
 if local\_batch\_tokens < args.train\_seq\_len:
 raise ValueError(
 "VAL\_BATCH\_SIZE must provide at least one sequence per rank; "
 f"got VAL\_BATCH\_SIZE={args.val\_batch\_size}, WORLD\_SIZE={world\_size}, "
 f"GRAD\_ACCUM\_STEPS={grad\_accum\_steps}, TRAIN\_SEQ\_LEN={args.train\_seq\_len}"
 )
 local\_batch\_seqs = local\_batch\_tokens // args.train\_seq\_len
 total\_seqs = (val\_tokens.numel() - 1) // args.train\_seq\_len
 seq\_start = (total\_seqs \* rank) // world\_size
 seq\_end = (total\_seqs \* (rank + 1)) // world\_size
 val\_loss\_sum = torch.zeros((), device=device, dtype=torch.float64)
 val\_token\_count = torch.zeros((), device=device, dtype=torch.float64)
 val\_byte\_count = torch.zeros((), device=device, dtype=torch.float64)

 model.eval()
 with torch.inference\_mode():
 for batch\_seq\_start in range(seq\_start, seq\_end, local\_batch\_seqs):
 batch\_seq\_end = min(batch\_seq\_start + local\_batch\_seqs, seq\_end)
 raw\_start = batch\_seq\_start \* args.train\_seq\_len
 raw\_end = batch\_seq\_end \* args.train\_seq\_len + 1
 local = val\_tokens\[raw\_start:raw\_end\].to(device=device, dtype=torch.int64, non\_blocking=True)
 x = local\[:-1\].reshape(-1, args.train\_seq\_len)
 y = local\[1:\].reshape(-1, args.train\_seq\_len)
 with torch.autocast(device\_type="cuda", dtype=torch.bfloat16, enabled=True):
 batch\_loss = model(x, y).detach()
 batch\_token\_count = float(y.numel())
 val\_loss\_sum += batch\_loss.to(torch.float64) \* batch\_token\_count
 val\_token\_count += batch\_token\_count
 prev\_ids = x.reshape(-1)
 tgt\_ids = y.reshape(-1)
 token\_bytes = base\_bytes\_lut\[tgt\_ids\].to(dtype=torch.int16)
 token\_bytes += (has\_leading\_space\_lut\[tgt\_ids\] & ~is\_boundary\_token\_lut\[prev\_ids\]).to(dtype=torch.int16)
 val\_byte\_count += token\_bytes.to(torch.float64).sum()

 if dist.is\_available() and dist.is\_initialized():
 dist.all\_reduce(val\_loss\_sum, op=dist.ReduceOp.SUM)
 dist.all\_reduce(val\_token\_count, op=dist.ReduceOp.SUM)
 dist.all\_reduce(val\_byte\_count, op=dist.ReduceOp.SUM)

 val\_loss = val\_loss\_sum / val\_token\_count
 bits\_per\_token = val\_loss.item() / math.log(2.0)
 tokens\_per\_byte = val\_token\_count.item() / val\_byte\_count.item()
 model.train()
 return float(val\_loss.item()), float(bits\_per\_token \* tokens\_per\_byte)

\# -----------------------------
\# POST-TRAINING QUANTIZATION
\# -----------------------------
#
\# It's silly to export our model, which is trained in bf16 and fp32, at that same precision.
\# Instead, we get approximately the same model (with a small hit) by quantizing the model to int8 & zlib compressing.
\# We can then decompress the model and run in higher precision for evaluation, after closing in under the size limit.

CONTROL\_TENSOR\_NAME\_PATTERNS = tuple(
 pattern
 for pattern in os.environ.get(
 "CONTROL\_TENSOR\_NAME\_PATTERNS",
 "attn\_scale,attn\_scales,mlp\_scale,mlp\_scales,resid\_mix,resid\_mixes,q\_gain,skip\_weight,skip\_weights",
 ).split(",")
 if pattern
)
INT8\_KEEP\_FLOAT\_FP32\_NAME\_PATTERNS = tuple(
 pattern
 for pattern in os.environ.get(
 "INT8\_KEEP\_FLOAT\_FP32\_NAME\_PATTERNS",
 ",".join(CONTROL\_TENSOR\_NAME\_PATTERNS),
 ).split(",")
 if pattern
)
INT8\_KEEP\_FLOAT\_MAX\_NUMEL = 65\_536
INT8\_KEEP\_FLOAT\_STORE\_DTYPE = torch.float16
INT8\_PER\_ROW\_SCALE\_DTYPE = torch.float16
INT8\_CLIP\_PERCENTILE = 99.99984
INT8\_CLIP\_Q = INT8\_CLIP\_PERCENTILE / 100.0

def tensor\_nbytes(t: Tensor) -> int:
 return int(t.numel()) \* int(t.element\_size())

def keep\_float\_tensor(name: str, t: Tensor, passthrough\_orig\_dtypes: dict\[str, str\]) -> Tensor:
 if any(pattern in name for pattern in INT8\_KEEP\_FLOAT\_FP32\_NAME\_PATTERNS):
 return t.float().contiguous()
 if t.dtype in {torch.float32, torch.bfloat16}:
 passthrough\_orig\_dtypes\[name\] = str(t.dtype).removeprefix("torch.")
 return t.to(dtype=INT8\_KEEP\_FLOAT\_STORE\_DTYPE).contiguous()
 return t

def quantize\_float\_tensor(t: Tensor) -> tuple\[Tensor, Tensor\]:
 t32 = t.float()
 if t32.ndim == 2:
 # Matrices get one scale per row, which usually tracks output-channel
 # ranges much better than a single tensor-wide scale.
 clip\_abs = (
 torch.quantile(t32.abs(), INT8\_CLIP\_Q, dim=1)
 if t32.numel()
 else torch.empty((t32.shape\[0\],), dtype=torch.float32)
 )
 clipped = torch.maximum(torch.minimum(t32, clip\_abs\[:, None\]), -clip\_abs\[:, None\])
 scale = (clip\_abs / 127.0).clamp\_min(1.0 / 127.0)
 q = torch.clamp(torch.round(clipped / scale\[:, None\]), -127, 127).to(torch.int8).contiguous()
 return q, scale.to(dtype=INT8\_PER\_ROW\_SCALE\_DTYPE).contiguous()

 # Vectors / scalars use a simpler per-tensor scale.
 clip\_abs = float(torch.quantile(t32.abs().flatten(), INT8\_CLIP\_Q).item()) if t32.numel() else 0.0
 scale = torch.tensor(clip\_abs / 127.0 if clip\_abs > 0 else 1.0, dtype=torch.float32)
 q = torch.clamp(torch.round(torch.clamp(t32, -clip\_abs, clip\_abs) / scale), -127, 127).to(torch.int8).contiguous()
 return q, scale

def quantize\_state\_dict\_int8(state\_dict: dict\[str, Tensor\]):
 # Single supported clean-script export format:
 # \- per-row int8 for 2D float tensors
 # \- per-tensor int8 for other float tensors
 # \- exact passthrough for non-floats
 # \- passthrough for small float tensors, stored as fp16 to save bytes
 quantized: dict\[str, Tensor\] = {}
 scales: dict\[str, Tensor\] = {}
 dtypes: dict\[str, str\] = {}
 passthrough: dict\[str, Tensor\] = {}
 passthrough\_orig\_dtypes: dict\[str, str\] = {}
 qmeta: dict\[str, dict\[str, object\]\] = {}
 stats = dict.fromkeys(
 ("param\_count", "num\_tensors", "num\_float\_tensors", "num\_nonfloat\_tensors", "baseline\_tensor\_bytes", "int8\_payload\_bytes"),
 0,
 )

 for name, tensor in state\_dict.items():
 t = tensor.detach().to("cpu").contiguous()
 stats\["param\_count"\] += int(t.numel())
 stats\["num\_tensors"\] += 1
 stats\["baseline\_tensor\_bytes"\] += tensor\_nbytes(t)

 if not t.is\_floating\_point():
 stats\["num\_nonfloat\_tensors"\] += 1
 passthrough\[name\] = t
 stats\["int8\_payload\_bytes"\] += tensor\_nbytes(t)
 continue

 # Small float tensors are cheap enough to keep directly. We still downcast
 # fp32/bf16 passthrough tensors to fp16 so metadata does not dominate size.
 if t.numel() <= INT8\_KEEP\_FLOAT\_MAX\_NUMEL:
 kept = keep\_float\_tensor(name, t, passthrough\_orig\_dtypes)
 passthrough\[name\] = kept
 stats\["int8\_payload\_bytes"\] += tensor\_nbytes(kept)
 continue

 stats\["num\_float\_tensors"\] += 1
 q, s = quantize\_float\_tensor(t)
 if s.ndim > 0:
 qmeta\[name\] = {"scheme": "per\_row", "axis": 0}
 quantized\[name\] = q
 scales\[name\] = s
 dtypes\[name\] = str(t.dtype).removeprefix("torch.")
 stats\["int8\_payload\_bytes"\] += tensor\_nbytes(q) + tensor\_nbytes(s)

 obj: dict\[str, object\] = {
 "\_\_quant\_format\_\_": "int8\_clean\_per\_row\_v1",
 "quantized": quantized,
 "scales": scales,
 "dtypes": dtypes,
 "passthrough": passthrough,
 }
 if qmeta:
 obj\["qmeta"\] = qmeta
 if passthrough\_orig\_dtypes:
 obj\["passthrough\_orig\_dtypes"\] = passthrough\_orig\_dtypes
 return obj, stats

def dequantize\_state\_dict\_int8(obj: dict\[str, object\]) -> dict\[str, Tensor\]:
 out: dict\[str, Tensor\] = {}
 qmeta = obj.get("qmeta", {})
 passthrough\_orig\_dtypes = obj.get("passthrough\_orig\_dtypes", {})
 for name, q in obj\["quantized"\].items():
 dtype = getattr(torch, obj\["dtypes"\]\[name\])
 s = obj\["scales"\]\[name\]
 if qmeta.get(name, {}).get("scheme") == "per\_row" or s.ndim > 0:
 s = s.to(dtype=torch.float32)
 # Broadcast the saved row scale back across trailing dimensions.
 out\[name\] = (q.float() \* s.view(q.shape\[0\], \*(\[1\] \* (q.ndim - 1)))).to(dtype=dtype).contiguous()
 else:
 scale = float(s.item())
 out\[name\] = (q.float() \* scale).to(dtype=dtype).contiguous()
 for name, t in obj\["passthrough"\].items():
 # Restore small tensors, undoing the temporary fp16 storage cast if needed.
 out\_t = t.detach().to("cpu").contiguous()
 orig\_dtype = passthrough\_orig\_dtypes.get(name)
 if isinstance(orig\_dtype, str):
 out\_t = out\_t.to(dtype=getattr(torch, orig\_dtype)).contiguous()
 out\[name\] = out\_t
 return out

\# -----------------------------
\# DATA LOADING
\# -----------------------------

def load\_data\_shard(file: Path) -> Tensor:
 header\_bytes = 256 \* np.dtype(" None:
 self.file\_idx = (self.file\_idx + 1) % len(self.files)
 self.tokens = load\_data\_shard(self.files\[self.file\_idx\])
 self.pos = 0

 def take(self, n: int) -> Tensor:
 chunks: list\[Tensor\] = \[\]
 remaining = n
 while remaining > 0:
 avail = self.tokens.numel() - self.pos
 if avail <= 0:
 self.\_advance\_file()
 continue
 k = min(remaining, avail)
 chunks.append(self.tokens\[self.pos : self.pos + k\])
 self.pos += k
 remaining -= k
 return chunks\[0\] if len(chunks) == 1 else torch.cat(chunks)

class DistributedTokenLoader:
 # Each call consumes a contiguous chunk from the shared token stream, then slices out
 # one disjoint span per rank. The extra "+1" token lets us build (x, y) by shifting.
 def \_\_init\_\_(self, pattern: str, rank: int, world\_size: int, device: torch.device):
 self.rank = rank
 self.world\_size = world\_size
 self.device = device
 self.stream = TokenStream(pattern)

 def next\_batch(self, global\_tokens: int, seq\_len: int, grad\_accum\_steps: int) -> tuple\[Tensor, Tensor\]:
 local\_tokens = global\_tokens // (self.world\_size \* grad\_accum\_steps)
 per\_rank\_span = local\_tokens + 1
 chunk = self.stream.take(per\_rank\_span \* self.world\_size)
 start = self.rank \* per\_rank\_span
 local = chunk\[start : start + per\_rank\_span\].to(dtype=torch.int64)
 x = local\[:-1\].reshape(-1, seq\_len)
 y = local\[1:\].reshape(-1, seq\_len)
 return x.to(self.device, non\_blocking=True), y.to(self.device, non\_blocking=True)

\# -----------------------------
\# TRANSFORMER MODULES
\# -----------------------------

class RMSNorm(nn.Module):
 def \_\_init\_\_(self, eps: float \| None = None):
 super().\_\_init\_\_()
 self.eps = eps

 def forward(self, x: Tensor) -> Tensor:
 return F.rms\_norm(x, (x.size(-1),), eps=self.eps)

class CastedLinear(nn.Linear):
 # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.
 def forward(self, x: Tensor) -> Tensor:
 bias = self.bias.to(x.dtype) if self.bias is not None else None
 return F.linear(x, self.weight.to(x.dtype), bias)

def restore\_low\_dim\_params\_to\_fp32(module: nn.Module) -> None:
 # Keep small/control parameters in fp32 even when the model body runs in bf16.
 with torch.no\_grad():
 for name, param in module.named\_parameters():
 if (param.ndim < 2 or any(pattern in name for pattern in CONTROL\_TENSOR\_NAME\_PATTERNS)) and param.dtype != torch.float32:
 param.data = param.data.float()

class Rotary(nn.Module):
 # Caches cos/sin tables per sequence length on the current device.
 def \_\_init\_\_(self, dim: int, base: float = 10000.0):
 super().\_\_init\_\_()
 inv\_freq = 1.0 / (base \*\* (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
 self.register\_buffer("inv\_freq", inv\_freq, persistent=False)
 self.\_seq\_len\_cached = 0
 self.\_cos\_cached: Tensor \| None = None
 self.\_sin\_cached: Tensor \| None = None

 def forward(self, seq\_len: int, device: torch.device, dtype: torch.dtype) -> tuple\[Tensor, Tensor\]:
 if (
 self.\_cos\_cached is None
 or self.\_sin\_cached is None
 or self.\_seq\_len\_cached != seq\_len
 or self.\_cos\_cached.device != device
 ):
 t = torch.arange(seq\_len, device=device, dtype=self.inv\_freq.dtype)
 freqs = torch.outer(t, self.inv\_freq.to(device))
 self.\_cos\_cached = freqs.cos()\[None, None, :, :\]
 self.\_sin\_cached = freqs.sin()\[None, None, :, :\]
 self.\_seq\_len\_cached = seq\_len
 return self.\_cos\_cached.to(dtype=dtype), self.\_sin\_cached.to(dtype=dtype)

def apply\_rotary\_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
 half = x.size(-1) // 2
 x1, x2 = x\[..., :half\], x\[..., half:\]
 return torch.cat((x1 \* cos + x2 \* sin, x1 \* (-sin) + x2 \* cos), dim=-1)

class CausalSelfAttention(nn.Module):
 def \_\_init\_\_(
 self,
 dim: int,
 num\_heads: int,
 num\_kv\_heads: int,
 rope\_base: float,
 qk\_gain\_init: float,
 ):
 super().\_\_init\_\_()
 if dim % num\_heads != 0:
 raise ValueError("model\_dim must be divisible by num\_heads")
 if num\_heads % num\_kv\_heads != 0:
 raise ValueError("num\_heads must be divisible by num\_kv\_heads")
 self.num\_heads = num\_heads
 self.num\_kv\_heads = num\_kv\_heads
 self.head\_dim = dim // num\_heads
 if self.head\_dim % 2 != 0:
 raise ValueError("head\_dim must be even for RoPE")
 kv\_dim = self.num\_kv\_heads \* self.head\_dim
 self.c\_q = CastedLinear(dim, dim, bias=False)
 self.c\_k = CastedLinear(dim, kv\_dim, bias=False)
 self.c\_v = CastedLinear(dim, kv\_dim, bias=False)
 self.proj = CastedLinear(dim, dim, bias=False)
 self.proj.\_zero\_init = True
 self.q\_gain = nn.Parameter(torch.full((num\_heads,), qk\_gain\_init, dtype=torch.float32))
 self.rotary = Rotary(self.head\_dim, base=rope\_base)

 def forward(self, x: Tensor) -> Tensor:
 bsz, seqlen, dim = x.shape
 q = self.c\_q(x).reshape(bsz, seqlen, self.num\_heads, self.head\_dim).transpose(1, 2)
 k = self.c\_k(x).reshape(bsz, seqlen, self.num\_kv\_heads, self.head\_dim).transpose(1, 2)
 v = self.c\_v(x).reshape(bsz, seqlen, self.num\_kv\_heads, self.head\_dim).transpose(1, 2)
 q = F.rms\_norm(q, (q.size(-1),))
 k = F.rms\_norm(k, (k.size(-1),))
 cos, sin = self.rotary(seqlen, x.device, q.dtype)
 q = apply\_rotary\_emb(q, cos, sin)
 k = apply\_rotary\_emb(k, cos, sin)
 q = q \* self.q\_gain.to(dtype=q.dtype)\[None, :, None, None\]
 y = F.scaled\_dot\_product\_attention(
 q,
 k,
 v,
 attn\_mask=None,
 is\_causal=True,
 enable\_gqa=(self.num\_kv\_heads != self.num\_heads),
 )
 y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
 return self.proj(y)

class MLP(nn.Module):
 # relu^2 MLP from the original modded-nanogpt setup
 def \_\_init\_\_(self, dim: int, mlp\_mult: int):
 super().\_\_init\_\_()
 hidden = mlp\_mult \* dim
 self.fc = CastedLinear(dim, hidden, bias=False)
 self.proj = CastedLinear(hidden, dim, bias=False)
 self.proj.\_zero\_init = True

 def forward(self, x: Tensor) -> Tensor:
 x = torch.relu(self.fc(x))
 return self.proj(x.square())

class Block(nn.Module):
 def \_\_init\_\_(
 self,
 dim: int,
 num\_heads: int,
 num\_kv\_heads: int,
 mlp\_mult: int,
 rope\_base: float,
 qk\_gain\_init: float,
 ):
 super().\_\_init\_\_()
 self.attn\_norm = RMSNorm()
 self.mlp\_norm = RMSNorm()
 self.attn = CausalSelfAttention(dim, num\_heads, num\_kv\_heads, rope\_base, qk\_gain\_init)
 self.mlp = MLP(dim, mlp\_mult)
 self.attn\_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
 self.mlp\_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
 self.resid\_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

 def forward(self, x: Tensor, x0: Tensor) -> Tensor:
 mix = self.resid\_mix.to(dtype=x.dtype)
 x = mix\[0\]\[None, None, :\] \* x + mix\[1\]\[None, None, :\] \* x0
 attn\_out = self.attn(self.attn\_norm(x))
 x = x + self.attn\_scale.to(dtype=x.dtype)\[None, None, :\] \* attn\_out
 x = x + self.mlp\_scale.to(dtype=x.dtype)\[None, None, :\] \* self.mlp(self.mlp\_norm(x))
 return x

class GPT(nn.Module):
 def \_\_init\_\_(
 self,
 vocab\_size: int,
 num\_layers: int,
 model\_dim: int,
 num\_heads: int,
 num\_kv\_heads: int,
 mlp\_mult: int,
 tie\_embeddings: bool,
 tied\_embed\_init\_std: float,
 logit\_softcap: float,
 rope\_base: float,
 qk\_gain\_init: float,
 ):
 super().\_\_init\_\_()
 if logit\_softcap <= 0.0:
 raise ValueError(f"logit\_softcap must be positive, got {logit\_softcap}")
 self.tie\_embeddings = tie\_embeddings
 self.tied\_embed\_init\_std = tied\_embed\_init\_std
 self.logit\_softcap = logit\_softcap
 self.tok\_emb = nn.Embedding(vocab\_size, model\_dim)
 self.num\_encoder\_layers = num\_layers // 2
 self.num\_decoder\_layers = num\_layers - self.num\_encoder\_layers
 self.num\_skip\_weights = min(self.num\_encoder\_layers, self.num\_decoder\_layers)
 self.skip\_weights = nn.Parameter(torch.ones(self.num\_skip\_weights, model\_dim, dtype=torch.float32))
 self.blocks = nn.ModuleList(
 \[\
 Block(\
 model\_dim,\
 num\_heads,\
 num\_kv\_heads,\
 mlp\_mult,\
 rope\_base,\
 qk\_gain\_init,\
 )\
 for i in range(num\_layers)\
 \]
 )
 self.final\_norm = RMSNorm()
 self.lm\_head = None if tie\_embeddings else CastedLinear(model\_dim, vocab\_size, bias=False)
 if self.lm\_head is not None:
 self.lm\_head.\_zero\_init = True
 self.\_init\_weights()

 def \_init\_weights(self) -> None:
 if self.tie\_embeddings:
 nn.init.normal\_(self.tok\_emb.weight, mean=0.0, std=self.tied\_embed\_init\_std)
 for module in self.modules():
 if isinstance(module, nn.Linear) and getattr(module, "\_zero\_init", False):
 nn.init.zeros\_(module.weight)

 def forward(self, input\_ids: Tensor, target\_ids: Tensor) -> Tensor:
 x = self.tok\_emb(input\_ids)
 x = F.rms\_norm(x, (x.size(-1),))
 x0 = x
 skips: list\[Tensor\] = \[\]

 # First half stores skips; second half reuses them in reverse order.
 for i in range(self.num\_encoder\_layers):
 x = self.blocks\[i\](x, x0)
 skips.append(x)
 for i in range(self.num\_decoder\_layers):
 if skips:
 x = x + self.skip\_weights\[i\].to(dtype=x.dtype)\[None, None, :\] \* skips.pop()
 x = self.blocks\[self.num\_encoder\_layers + i\](x, x0)

 x = self.final\_norm(x).reshape(-1, x.size(-1))
 targets = target\_ids.reshape(-1)
 if self.tie\_embeddings:
 logits\_proj = F.linear(x, self.tok\_emb.weight)
 else:
 if self.lm\_head is None:
 raise RuntimeError("lm\_head is required when tie\_embeddings=False")
 logits\_proj = self.lm\_head(x)
 logits = self.logit\_softcap \* torch.tanh(logits\_proj / self.logit\_softcap)
 return F.cross\_entropy(logits.float(), targets, reduction="mean")

\# -----------------------------
\# TRAINING
\# -----------------------------

def main() -> None:
 global zeropower\_via\_newtonschulz5

 code = Path(\_\_file\_\_).read\_text(encoding="utf-8")
 args = Hyperparameters()
 zeropower\_via\_newtonschulz5 = torch.compile(zeropower\_via\_newtonschulz5)

 # -----------------------------
 # DISTRIBUTED + CUDA SETUP
 # -----------------------------

 distributed = "RANK" in os.environ and "WORLD\_SIZE" in os.environ
 rank = int(os.environ.get("RANK", "0"))
 world\_size = int(os.environ.get("WORLD\_SIZE", "1"))
 local\_rank = int(os.environ.get("LOCAL\_RANK", "0"))
 if world\_size <= 0:
 raise ValueError(f"WORLD\_SIZE must be positive, got {world\_size}")
 if 8 % world\_size != 0:
 raise ValueError(f"WORLD\_SIZE={world\_size} must divide 8 so grad\_accum\_steps stays integral")
 grad\_accum\_steps = 8 // world\_size
 grad\_scale = 1.0 / grad\_accum\_steps
 if not torch.cuda.is\_available():
 raise RuntimeError("CUDA is required")
 device = torch.device("cuda", local\_rank)
 torch.cuda.set\_device(device)
 if distributed:
 dist.init\_process\_group(backend="nccl", device\_id=device)
 dist.barrier()
 master\_process = rank == 0

 # Fast math knobs
 torch.backends.cuda.matmul.allow\_tf32 = True
 torch.backends.cudnn.allow\_tf32 = True
 from torch.backends.cuda import enable\_cudnn\_sdp, enable\_flash\_sdp, enable\_math\_sdp, enable\_mem\_efficient\_sdp

 enable\_cudnn\_sdp(False)
 enable\_flash\_sdp(True)
 enable\_mem\_efficient\_sdp(False)
 enable\_math\_sdp(False)

 logfile = None
 if master\_process:
 os.makedirs("logs", exist\_ok=True)
 logfile = f"logs/{args.run\_id}.txt"
 print(logfile)

 def log0(msg: str, console: bool = True) -> None:
 if not master\_process:
 return
 if console:
 print(msg)
 if logfile is not None:
 with open(logfile, "a", encoding="utf-8") as f:
 print(msg, file=f)

 log0(code, console=False)
 log0("=" \* 100, console=False)
 log0(f"Running Python {sys.version}", console=False)
 log0(f"Running PyTorch {torch.\_\_version\_\_}", console=False)
 log0(
 subprocess.run(\["nvidia-smi"\], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
 console=False,
 )
 log0("=" \* 100, console=False)

 # -----------------------------
 # TOKENIZER + VALIDATION METRIC SETUP
 # -----------------------------

 random.seed(args.seed)
 np.random.seed(args.seed)
 torch.manual\_seed(args.seed)
 torch.cuda.manual\_seed\_all(args.seed)

 if not args.tokenizer\_path.endswith(".model"):
 raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer\_path}")
 sp = spm.SentencePieceProcessor(model\_file=args.tokenizer\_path)
 if int(sp.vocab\_size()) != args.vocab\_size:
 raise ValueError(
 f"VOCAB\_SIZE={args.vocab\_size} does not match tokenizer vocab\_size={int(sp.vocab\_size())}"
 )
 dataset\_dir = Path(args.data\_path).resolve()
 actual\_train\_files = len(list(dataset\_dir.glob("fineweb\_train\_\*.bin")))
 val\_tokens = load\_validation\_tokens(args.val\_files, args.train\_seq\_len)
 base\_bytes\_lut, has\_leading\_space\_lut, is\_boundary\_token\_lut = build\_sentencepiece\_luts(
 sp, args.vocab\_size, device
 )
 log0(f"val\_bpb:enabled tokenizer\_kind=sentencepiece tokenizer\_path={args.tokenizer\_path}")
 log0(f"train\_loader:dataset:{dataset\_dir.name} train\_shards:{actual\_train\_files}")
 log0(f"val\_loader:shards pattern={args.val\_files} tokens:{val\_tokens.numel() - 1}")

 # -----------------------------
 # MODEL + OPTIMIZER SETUP
 # -----------------------------

 base\_model = GPT(
 vocab\_size=args.vocab\_size,
 num\_layers=args.num\_layers,
 model\_dim=args.model\_dim,
 num\_heads=args.num\_heads,
 num\_kv\_heads=args.num\_kv\_heads,
 mlp\_mult=args.mlp\_mult,
 tie\_embeddings=args.tie\_embeddings,
 tied\_embed\_init\_std=args.tied\_embed\_init\_std,
 logit\_softcap=args.logit\_softcap,
 rope\_base=args.rope\_base,
 qk\_gain\_init=args.qk\_gain\_init,
 ).to(device).bfloat16()
 for module in base\_model.modules():
 if isinstance(module, CastedLinear):
 module.float()
 restore\_low\_dim\_params\_to\_fp32(base\_model)
 compiled\_model = torch.compile(base\_model, dynamic=False, fullgraph=True)
 model: nn.Module = DDP(compiled\_model, device\_ids=\[local\_rank\], broadcast\_buffers=False) if distributed else compiled\_model

 # Optimizer split:
 # \- token embedding (Adam) uses EMBED\_LR
 # \- untied lm\_head (Adam) uses HEAD\_LR
 # \- matrix params in transformer blocks use MATRIX\_LR via Muon
 # \- vectors/scalars use SCALAR\_LR via Adam
 block\_named\_params = list(base\_model.blocks.named\_parameters())
 matrix\_params = \[\
 p\
 for name, p in block\_named\_params\
 if p.ndim == 2 and not any(pattern in name for pattern in CONTROL\_TENSOR\_NAME\_PATTERNS)\
 \]
 scalar\_params = \[\
 p\
 for name, p in block\_named\_params\
 if p.ndim < 2 or any(pattern in name for pattern in CONTROL\_TENSOR\_NAME\_PATTERNS)\
 \]
 if base\_model.skip\_weights.numel() > 0:
 scalar\_params.append(base\_model.skip\_weights)
 token\_lr = args.tied\_embed\_lr if args.tie\_embeddings else args.embed\_lr
 optimizer\_tok = torch.optim.Adam(
 \[{"params": \[base\_model.tok\_emb.weight\], "lr": token\_lr, "base\_lr": token\_lr}\],
 betas=(args.beta1, args.beta2),
 eps=args.adam\_eps,
 fused=True,
 )
 optimizer\_muon = Muon(
 matrix\_params,
 lr=args.matrix\_lr,
 momentum=args.muon\_momentum,
 backend\_steps=args.muon\_backend\_steps,
 )
 for group in optimizer\_muon.param\_groups:
 group\["base\_lr"\] = args.matrix\_lr
 optimizer\_scalar = torch.optim.Adam(
 \[{"params": scalar\_params, "lr": args.scalar\_lr, "base\_lr": args.scalar\_lr}\],
 betas=(args.beta1, args.beta2),
 eps=args.adam\_eps,
 fused=True,
 )
 optimizers: list\[torch.optim.Optimizer\] = \[optimizer\_tok, optimizer\_muon, optimizer\_scalar\]
 if base\_model.lm\_head is not None:
 optimizer\_head = torch.optim.Adam(
 \[{"params": \[base\_model.lm\_head.weight\], "lr": args.head\_lr, "base\_lr": args.head\_lr}\],
 betas=(args.beta1, args.beta2),
 eps=args.adam\_eps,
 fused=True,
 )
 optimizers.insert(1, optimizer\_head)

 n\_params = sum(p.numel() for p in base\_model.parameters())
 log0(f"model\_params:{n\_params}")
 log0(f"world\_size:{world\_size} grad\_accum\_steps:{grad\_accum\_steps}")
 log0("sdp\_backends:cudnn=False flash=True mem\_efficient=False math=False")
 log0(f"attention\_mode:gqa num\_heads:{args.num\_heads} num\_kv\_heads:{args.num\_kv\_heads}")
 log0(
 f"tie\_embeddings:{args.tie\_embeddings} embed\_lr:{token\_lr} "
 f"head\_lr:{args.head\_lr if base\_model.lm\_head is not None else 0.0} "
 f"matrix\_lr:{args.matrix\_lr} scalar\_lr:{args.scalar\_lr}"
 )
 log0(
 f"train\_batch\_tokens:{args.train\_batch\_tokens} train\_seq\_len:{args.train\_seq\_len} "
 f"iterations:{args.iterations} warmup\_steps:{args.warmup\_steps} "
 f"max\_wallclock\_seconds:{args.max\_wallclock\_seconds:.3f}"
 )
 log0(f"seed:{args.seed}")

 # -----------------------------
 # DATA LOADER & MODEL WARMUP
 # -----------------------------

 train\_loader = DistributedTokenLoader(args.train\_files, rank, world\_size, device)

 def zero\_grad\_all() -> None:
 for opt in optimizers:
 opt.zero\_grad(set\_to\_none=True)

 max\_wallclock\_ms = 1000.0 \* args.max\_wallclock\_seconds if args.max\_wallclock\_seconds > 0 else None

 def lr\_mul(step: int, elapsed\_ms: float) -> float:
 if args.warmdown\_iters <= 0:
 return 1.0
 if max\_wallclock\_ms is None:
 warmdown\_start = max(args.iterations - args.warmdown\_iters, 0)
 return max((args.iterations - step) / max(args.warmdown\_iters, 1), 0.0) if warmdown\_start <= step < args.iterations else 1.0
 step\_ms = elapsed\_ms / max(step, 1)
 warmdown\_ms = args.warmdown\_iters \* step\_ms
 remaining\_ms = max(max\_wallclock\_ms - elapsed\_ms, 0.0)
 return remaining\_ms / max(warmdown\_ms, 1e-9) if remaining\_ms <= warmdown\_ms else 1.0

 # Warmup primes the compiled forward/backward/optimizer paths, then we restore the
 # initial weights/optimizer state so measured training starts from the true init.
 if args.warmup\_steps > 0:
 initial\_model\_state = {name: tensor.detach().cpu().clone() for name, tensor in base\_model.state\_dict().items()}
 initial\_optimizer\_states = \[copy.deepcopy(opt.state\_dict()) for opt in optimizers\]
 model.train()
 for warmup\_step in range(args.warmup\_steps):
 zero\_grad\_all()
 for micro\_step in range(grad\_accum\_steps):
 if distributed:
 model.require\_backward\_grad\_sync = micro\_step == grad\_accum\_steps - 1
 x, y = train\_loader.next\_batch(args.train\_batch\_tokens, args.train\_seq\_len, grad\_accum\_steps)
 with torch.autocast(device\_type="cuda", dtype=torch.bfloat16, enabled=True):
 warmup\_loss = model(x, y)
 (warmup\_loss \* grad\_scale).backward()
 for opt in optimizers:
 opt.step()
 zero\_grad\_all()
 if args.warmup\_steps <= 20 or (warmup\_step + 1) % 10 == 0 or warmup\_step + 1 == args.warmup\_steps:
 log0(f"warmup\_step:{warmup\_step + 1}/{args.warmup\_steps}")
 base\_model.load\_state\_dict(initial\_model\_state, strict=True)
 for opt, state in zip(optimizers, initial\_optimizer\_states, strict=True):
 opt.load\_state\_dict(state)
 zero\_grad\_all()
 if distributed:
 model.require\_backward\_grad\_sync = True
 train\_loader = DistributedTokenLoader(args.train\_files, rank, world\_size, device)

 # -----------------------------
 # MAIN TRAINING LOOP
 # -----------------------------

 training\_time\_ms = 0.0
 stop\_after\_step: int \| None = None
 torch.cuda.synchronize()
 t0 = time.perf\_counter()

 step = 0
 while True:
 last\_step = step == args.iterations or (stop\_after\_step is not None and step >= stop\_after\_step)

 should\_validate = last\_step or (args.val\_loss\_every > 0 and step % args.val\_loss\_every == 0)
 if should\_validate:
 torch.cuda.synchronize()
 training\_time\_ms += 1000.0 \* (time.perf\_counter() - t0)
 val\_loss, val\_bpb = eval\_val(
 args,
 model,
 rank,
 world\_size,
 device,
 grad\_accum\_steps,
 val\_tokens,
 base\_bytes\_lut,
 has\_leading\_space\_lut,
 is\_boundary\_token\_lut,
 )
 log0(
 f"step:{step}/{args.iterations} val\_loss:{val\_loss:.4f} val\_bpb:{val\_bpb:.4f} "
 f"train\_time:{training\_time\_ms:.0f}ms step\_avg:{training\_time\_ms / max(step, 1):.2f}ms"
 )
 torch.cuda.synchronize()
 t0 = time.perf\_counter()

 if last\_step:
 if stop\_after\_step is not None and step < args.iterations:
 log0(
 f"stopping\_early: wallclock\_cap train\_time:{training\_time\_ms:.0f}ms "
 f"step:{step}/{args.iterations}"
 )
 break

 elapsed\_ms = training\_time\_ms + 1000.0 \* (time.perf\_counter() - t0)
 scale = lr\_mul(step, elapsed\_ms)
 zero\_grad\_all()
 train\_loss = torch.zeros((), device=device)
 for micro\_step in range(grad\_accum\_steps):
 if distributed:
 model.require\_backward\_grad\_sync = micro\_step == grad\_accum\_steps - 1
 x, y = train\_loader.next\_batch(args.train\_batch\_tokens, args.train\_seq\_len, grad\_accum\_steps)
 with torch.autocast(device\_type="cuda", dtype=torch.bfloat16, enabled=True):
 loss = model(x, y)
 train\_loss += loss.detach()
 (loss \* grad\_scale).backward()
 train\_loss /= grad\_accum\_steps

 frac = min(step / args.muon\_momentum\_warmup\_steps, 1.0) if args.muon\_momentum\_warmup\_steps > 0 else 1.0
 muon\_momentum = (1 - frac) \* args.muon\_momentum\_warmup\_start + frac \* args.muon\_momentum
 for group in optimizer\_muon.param\_groups:
 group\["momentum"\] = muon\_momentum

 for opt in optimizers:
 for group in opt.param\_groups:
 group\["lr"\] = group\["base\_lr"\] \* scale

 if args.grad\_clip\_norm > 0:
 torch.nn.utils.clip\_grad\_norm\_(base\_model.parameters(), args.grad\_clip\_norm)
 for opt in optimizers:
 opt.step()
 zero\_grad\_all()

 step += 1
 approx\_training\_time\_ms = training\_time\_ms + 1000.0 \* (time.perf\_counter() - t0)
 should\_log\_train = (
 args.train\_log\_every > 0
 and (step <= 10 or step % args.train\_log\_every == 0 or stop\_after\_step is not None)
 )
 if should\_log\_train:
 log0(
 f"step:{step}/{args.iterations} train\_loss:{train\_loss.item():.4f} "
 f"train\_time:{approx\_training\_time\_ms:.0f}ms step\_avg:{approx\_training\_time\_ms / step:.2f}ms"
 )

 # Needed to sync whether we've reached the wallclock cap.
 reached\_cap = max\_wallclock\_ms is not None and approx\_training\_time\_ms >= max\_wallclock\_ms
 if distributed and max\_wallclock\_ms is not None:
 reached\_cap\_tensor = torch.tensor(int(reached\_cap), device=device)
 dist.all\_reduce(reached\_cap\_tensor, op=dist.ReduceOp.MAX)
 reached\_cap = bool(reached\_cap\_tensor.item())
 if stop\_after\_step is None and reached\_cap:
 stop\_after\_step = step

 log0(
 f"peak memory allocated: {torch.cuda.max\_memory\_allocated() // 1024 // 1024} MiB "
 f"reserved: {torch.cuda.max\_memory\_reserved() // 1024 // 1024} MiB"
 )

 # -----------------------------
 # SERIALIZATION + ROUNDTRIP VALIDATION
 # -----------------------------
 # Save the raw state (useful for debugging/loading in PyTorch directly), then always produce
 # the compressed int8+zlib artifact and validate the round-tripped weights.

 if master\_process:
 torch.save(base\_model.state\_dict(), "final\_model.pt")
 model\_bytes = os.path.getsize("final\_model.pt")
 code\_bytes = len(code.encode("utf-8"))
 log0(f"Serialized model: {model\_bytes} bytes")
 log0(f"Code size: {code\_bytes} bytes")
 log0(f"Total submission size: {model\_bytes + code\_bytes} bytes")

 quant\_obj, quant\_stats = quantize\_state\_dict\_int8(base\_model.state\_dict())
 quant\_buf = io.BytesIO()
 torch.save(quant\_obj, quant\_buf)
 quant\_raw = quant\_buf.getvalue()
 quant\_blob = zlib.compress(quant\_raw, level=9)
 quant\_raw\_bytes = len(quant\_raw)
 if master\_process:
 with open("final\_model.int8.ptz", "wb") as f:
 f.write(quant\_blob)
 quant\_file\_bytes = os.path.getsize("final\_model.int8.ptz")
 code\_bytes = len(code.encode("utf-8"))
 ratio = quant\_stats\["baseline\_tensor\_bytes"\] / max(quant\_stats\["int8\_payload\_bytes"\], 1)
 log0(
 f"Serialized model int8+zlib: {quant\_file\_bytes} bytes "
 f"(payload:{quant\_stats\['int8\_payload\_bytes'\]} raw\_torch:{quant\_raw\_bytes} payload\_ratio:{ratio:.2f}x)"
 )
 log0(f"Total submission size int8+zlib: {quant\_file\_bytes + code\_bytes} bytes")

 if distributed:
 dist.barrier()
 with open("final\_model.int8.ptz", "rb") as f:
 quant\_blob\_disk = f.read()
 quant\_state = torch.load(io.BytesIO(zlib.decompress(quant\_blob\_disk)), map\_location="cpu")
 base\_model.load\_state\_dict(dequantize\_state\_dict\_int8(quant\_state), strict=True)
 torch.cuda.synchronize()
 t\_qeval = time.perf\_counter()
 q\_val\_loss, q\_val\_bpb = eval\_val(
 args,
 model,
 rank,
 world\_size,
 device,
 grad\_accum\_steps,
 val\_tokens,
 base\_bytes\_lut,
 has\_leading\_space\_lut,
 is\_boundary\_token\_lut,
 )
 torch.cuda.synchronize()
 log0(
 f"final\_int8\_zlib\_roundtrip val\_loss:{q\_val\_loss:.4f} val\_bpb:{q\_val\_bpb:.4f} "
 f"eval\_time:{1000.0 \* (time.perf\_counter() - t\_qeval):.0f}ms"
 )
 log0(f"final\_int8\_zlib\_roundtrip\_exact val\_loss:{q\_val\_loss:.8f} val\_bpb:{q\_val\_bpb:.8f}")

 if distributed:
 dist.destroy\_process\_group()

if \_\_name\_\_ == "\_\_main\_\_":
 main()