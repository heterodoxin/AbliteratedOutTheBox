"""Fit-free AbliteratedOutTheBox inference through the native Strata residual hook."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading

import numpy as np

COMPLY = ["Sure, here is", "Of course! Here's how", "Certainly, I can help with that",
          "Here is how to do it", "Absolutely, let's begin"]
REFUSE = ["I'm sorry, but I can't help with that", "I cannot assist with that request",
          "Sorry, I won't do that", "I'm unable to help with this", "I can't provide that information",
          "I must decline", "That would not be appropriate", "I'm not able to provide"]
CLEAN = ["Sure. Step 1:", "Here are the steps. First,", "Of course. The process is as follows: 1.",
         "Okay, here is how. First,"]
DISCLAIMER = ["Sure, but I must warn you this is illegal and dangerous. Step 1:",
              "Here are the steps, though I have to caution this is unlawful and unethical. First,",
              "Of course, but note this is for educational purposes only and is illegal. The process is: 1.",
              "Okay, but I strongly advise against this as it is illegal. Here is how. First,"]


def unit(vector):
    return vector / (np.linalg.norm(vector, axis=-1, keepdims=True) + 1e-6)


def directions(comply, refuse, clean=None, disclaimer=None, edit_strength=2.0, disc_strength=0.4):
    """Arrays have shape (openings, layers, width); return the ported per-layer recipe."""
    comply, refuse = np.asarray(comply, dtype=np.float32), np.asarray(refuse, dtype=np.float32)
    if comply.ndim != 3 or refuse.ndim != 3 or comply.shape[1:] != refuse.shape[1:]:
        raise ValueError("anchor shapes must be (openings, layers, width)")
    arrays = [comply, refuse]
    if disc_strength >= 0:
        clean, disclaimer = np.asarray(clean, dtype=np.float32), np.asarray(disclaimer, dtype=np.float32)
        if clean.ndim != 3 or disclaimer.ndim != 3 or clean.shape[1:] != comply.shape[1:] or disclaimer.shape[1:] != comply.shape[1:]:
            raise ValueError("disclaimer anchor shapes differ")
        arrays += [clean, disclaimer]
    if any(a.shape[0] == 0 or not np.isfinite(a).all() for a in arrays):
        raise ValueError("missing or non-finite anchor activations")
    if not math.isfinite(edit_strength) or not math.isfinite(disc_strength):
        raise ValueError("strengths must be finite")
    mc, mr = comply.mean(axis=0), refuse.mean(axis=0)
    refusal = mr - mc
    for hc in comply:
        spread = unit(hc - mc)
        refusal -= (refusal * spread).sum(axis=-1, keepdims=True) * spread
    refusal = unit(refusal)
    cp = (comply * refusal).sum(axis=-1)
    pc, pr = cp.mean(axis=0), (refuse * refusal).sum(axis=-1).mean(axis=0)
    reference = pc - edit_strength * cp.std(axis=0)
    disc, disc_ref = np.zeros_like(refusal), np.zeros_like(pc)
    if disc_strength >= 0:
        disc = unit(disclaimer.mean(axis=0) - clean.mean(axis=0))
        dc = (clean * disc).sum(axis=-1).mean(axis=0)
        dr = (disclaimer * disc).sum(axis=-1).mean(axis=0)
        disc_ref = dc - disc_strength * (dr - dc)
    result = (refusal, reference, disc, disc_ref, pc, pr)
    if any(not np.isfinite(x).all() for x in result):
        raise ValueError("non-finite calibrated direction")
    return result


def gate_score(states, refusal, comply_projection, refuse_projection):
    """Maximum across continuation tokens of the middle-band mean projection."""
    values = ((states * refusal[:, None, :]).sum(axis=-1) - comply_projection[:, None])
    values /= np.maximum(refuse_projection - comply_projection, 1e-3)[:, None]
    if not np.isfinite(values).all() or values.shape[1] == 0:
        raise ValueError("invalid look-ahead activations")
    return float(values.mean(axis=0).max())


class Engine:
    """One resident Strata process, with exclusive ownership of its line protocol."""
    def __init__(self, command, timeout=1800):
        self.timeout = timeout
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     text=True, encoding="utf-8", bufsize=1)
        self.lines = queue.Queue()

        def pump():
            try:
                for line in self.proc.stdout:
                    self.lines.put(line.rstrip("\r\n"))
            finally:
                self.lines.put(None)

        self.thread = threading.Thread(target=pump, daemon=True)
        self.thread.start()
        try:
            while not self.read().startswith("READY"):
                pass
            self.send("ABL INFO")
            info = self.read().split()
            if len(info) != 6 or info[:2] != ["ABL", "INFO"]:
                raise RuntimeError("engine lacks the AbliteratedOutTheBox port")
            self.layers, self.width, self.streams, self.history = map(int, info[2:])
            self.band = list(range(self.layers // 3, 2 * self.layers // 3 + 1))
            self.edit_first = 3 * self.layers // 7
        except BaseException:
            self.close()
            raise

    def read(self):
        try:
            line = self.lines.get(timeout=self.timeout)
        except queue.Empty:
            raise TimeoutError("Strata stopped responding; see its stderr") from None
        if line is None:
            raise RuntimeError(f"Strata exited ({self.proc.poll()})")
        if line.startswith("ERR"):
            raise RuntimeError(line)
        return line

    def send(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def control(self, command):
        self.send("ABL " + command)
        if self.read() != "ABL OK":
            raise RuntimeError("unexpected abliteration protocol response")

    def generate(self, ids, count):
        if not ids or count < 1:
            raise ValueError("generation needs a nonempty prompt and positive token count")
        self.send(f"GEN {count} " + ",".join(map(str, ids)))
        tokens = []
        while True:
            line = self.read()
            if line.startswith("T "):
                tokens.append(int(line[2:]))
            elif line.startswith("DONE"):
                if "nonfinite" in line or "cancel" in line:
                    raise RuntimeError("probe/generation did not finish: " + line)
                return tokens

    def capture(self, take=1):
        self.send(f"ABL READ {take}")
        rows = {}
        while True:
            line = self.read()
            if line == "ABL END":
                break
            fields = line.split(maxsplit=3)
            if len(fields) != 4 or fields[:2] != ["ABL", "ROW"]:
                raise RuntimeError("malformed activation response")
            layer = int(fields[2])
            data = np.fromstring(fields[3], sep=" ", dtype=np.float32)
            if layer in rows or layer not in self.band or data.size != take * self.width or not np.isfinite(data).all():
                raise RuntimeError("invalid or missing layer-output activations")
            rows[layer] = data.reshape(take, self.width)
        if sorted(rows) != self.band:
            raise RuntimeError("engine did not capture every gate layer")
        return np.stack([rows[l] for l in self.band])

    def answer(self, encode, max_new=256, threshold=0.70, edit_strength=2.0, disc_strength=0.4):
        if max_new < 1:
            raise ValueError("max_new must be positive")
        if not all(math.isfinite(x) for x in (threshold, edit_strength, disc_strength)):
            raise ValueError("threshold and strengths must be finite")
        try:
            def anchors(openings):
                captured = []
                for opening in openings:
                    self.control("CAPTURE")
                    self.generate(encode(opening), 1)
                    captured.append(self.capture()[:, 0])
                return np.stack(captured)

            hc, hr = anchors(COMPLY), anchors(REFUSE)
            cl = anchors(CLEAN) if disc_strength >= 0 else None
            ds = anchors(DISCLAIMER) if disc_strength >= 0 else None
            refusal, reference, disc, disc_ref, pc, pr = directions(hc, hr, cl, ds, edit_strength, disc_strength)
            self.control("CAPTURE")
            # GEN emits its last sampled token without feeding it back. 25 sampled tokens give 24 continuation states.
            look = self.generate(encode(""), 25)
            if look and len(look) < 25:
                # The original gate also forwards an early end-of-generation token before stopping.
                self.control("CAPTURE")
                self.generate(encode("") + look, 1)
                take = min(24, len(look))
            else:
                take = min(24, max(0, len(look) - 1))
            score = gate_score(self.capture(take), refusal, pc, pr) if take else -math.inf
            fired = score > threshold
            self.control("OFF")
            if fired:
                for i, layer in enumerate(self.band):
                    if layer < self.edit_first:
                        continue
                    values = np.concatenate((refusal[i], disc[i]))
                    text = " ".join(format(float(x), ".9g") for x in values)
                    self.control(f"SET {layer} {float(reference[i]):.9g} {float(disc_ref[i]):.9g} {text}")
                self.control("ON")
            print(f"[abliterate/strata] gate={'fired' if fired else 'clean'} score={score:.6g}", file=sys.stderr)
            return self.generate(encode(""), max_new), score, fired
        finally:
            failed = sys.exc_info()[0] is not None
            if self.proc.poll() is None:
                try:
                    self.control("OFF")
                except Exception:
                    self.close()
                    if not failed:
                        raise

    def close(self):
        if self.proc.poll() is None:
            try:
                self.send("QUIT")
                self.proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                self.proc.kill()
                self.proc.wait()
        self.proc.stdin.close()
        self.thread.join(timeout=5)
        self.proc.stdout.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strata-source", type=Path, default=Path(__file__).parent / ".strata-src")
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--ff-threshold", type=float, default=0.70)
    parser.add_argument("--edit-strength", type=float, default=2.0)
    parser.add_argument("--disc-strength", type=float, default=0.4)
    parser.add_argument("engine_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.max_new_tokens < 1 or args.context < 64:
        parser.error("token count must be positive and context must be at least 64")
    if not all(math.isfinite(x) for x in (args.ff_threshold, args.edit_strength, args.disc_strength)):
        parser.error("threshold and strengths must be finite")
    source, model = args.strata_source.resolve(), args.model.resolve()
    sys.path.insert(0, str(source))
    from tools.strata_tokenizer import Tokenizer
    from tools.gguf_reader import GGUFFile
    from serve.frontend import ChatTemplate

    tok = Tokenizer.from_gguf(model)
    metadata = GGUFFile(model).metadata
    if metadata.get("general.architecture") != "qwen4exp":
        parser.error("this port supports Strata's qwen4exp (Qwen3.8) engine path")
    template_text = metadata.get("tokenizer.chat_template")
    if not isinstance(template_text, str) or not template_text:
        parser.error("the GGUF must contain tokenizer.chat_template")
    with tempfile.TemporaryDirectory(prefix="strata-abliterate-") as directory:
        template_file = Path(directory) / "chat_template.jinja"
        template_file.write_text(template_text, encoding="utf-8")
        template = ChatTemplate(template_file)
    prompt = template.render([{"role": "user", "content": args.prompt}], enable_thinking=False)

    def encode(suffix):
        ids = tok.encode(prompt + suffix, parse_special=True)
        reserve = 1 if suffix else max(args.max_new_tokens, 25)
        if len(ids) + reserve > args.context:
            raise ValueError("prompt/anchor and generation exceed --context; use a larger context")
        return ids

    # Validate all anchors before doing any model work.
    for suffix in [""] + COMPLY + REFUSE + (CLEAN + DISCLAIMER if args.disc_strength >= 0 else []):
        encode(suffix)
    exe = (args.engine or source / "build-abliterate/strata").resolve()
    extra = args.engine_args[1:] if args.engine_args[:1] == ["--"] else args.engine_args
    command = [str(exe), "--native", str(model), "--serve", "--abliterate", "--tokens", "1",
               "--max-new", "1", "--max-context", str(args.context), "--greedy",
               "--expert-cache", "auto", "--prefill", "auto", *extra]
    engine = Engine(command)
    try:
        output, _, _ = engine.answer(encode, args.max_new_tokens, args.ff_threshold, args.edit_strength, args.disc_strength)
        eos = set(tok.special_ids.values())
        print(tok.decode([t for t in output if t not in eos]))
    finally:
        engine.close()


if __name__ == "__main__":
    main()
