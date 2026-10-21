#!/usr/bin/env python3
"""A narrated, syntax-colored JShell session; optionally write asciicast v2."""

import argparse
import json
import os
from pathlib import Path
import re
import selectors
import shutil
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
ESC = "\x1b["
RESET = ESC + "0m"
CYAN = ESC + "1;96m"
GREEN = ESC + "1;92m"
PURPLE = ESC + "1;95m"
WHITE = ESC + "1;97m"
DIM = ESC + "38;5;245m"
TOKEN = re.compile(r'"(?:\\.|[^"\\])*"|\b(?:var|new|float|int|for|try|if)\b|'
                   r'\b(?:Tensors|Tensor|Transforms|Jit|Vmap|System|Math)\b|\b\d+(?:\.\d+)?\b')


class Demo:
    def __init__(self, args, jshell):
        self.args = args
        self.cast = None
        self.started = None
        self.process = subprocess.Popen(
            [jshell, "--class-path", str(ROOT / "jax4j-core/target/classes"),
             "-R--enable-native-access=ALL-UNNAMED", "-R-Djax4j.tensor.backend=cpu",
             "-s", "-"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, bufsize=0, cwd=ROOT)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        if args.record:
            self.cast = open(args.record, "w", encoding="utf-8")
            self.cast.write(json.dumps({
                "version": 2, "width": 96, "height": 28,
                "timestamp": int(time.time()), "title": "JAX4J · Java. Differentiated.",
                "env": {"TERM": "xterm-256color", "SHELL": "/bin/bash"},
            }) + "\n")

    def emit(self, value):
        if self.started is None:
            self.started = time.monotonic()
        sys.stdout.write(value)
        sys.stdout.flush()
        if self.cast:
            # Terminal playback needs CRLF; a bare LF leaves the cursor in its column.
            self.cast.write(json.dumps([round(time.monotonic() - self.started, 4),
                                       "o", value.replace("\n", "\r\n")],
                                      ensure_ascii=False) + "\n")

    def pause(self, seconds):
        if not self.args.fast:
            time.sleep(seconds)

    def execute(self, source):
        marker = "__JAX4J_" + uuid.uuid4().hex + "__"
        self.process.stdin.write((source + '\nSystem.out.println("' + marker + '");\n').encode())
        result = b""
        deadline = time.monotonic() + 60
        while marker.encode() not in result:
            if time.monotonic() > deadline:
                raise RuntimeError("JShell did not finish a snippet within 60 seconds")
            if self.selector.select(timeout=1):
                chunk = os.read(self.process.stdout.fileno(), 65536)
                if not chunk:
                    raise RuntimeError("JShell exited unexpectedly: " + result.decode(errors="replace"))
                result += chunk
        output = result.decode().split(marker)[0].strip()
        if re.search(r"(?m)^\s*\||Exception|Error:", output):
            raise RuntimeError("JShell snippet failed:\n" + source + "\n" + output)
        return output

    def run(self, source, hold=0.7):
        self.emit(GREEN + "jshell> " + RESET)
        offset = 0
        for match in TOKEN.finditer(source):
            self.type(source[offset:match.start()])
            token = match.group()
            color = GREEN if token.startswith('"') else PURPLE
            if token[0].isdigit():
                color = CYAN
            self.emit(color)
            self.type(token)
            self.emit(RESET)
            offset = match.end()
        self.type(source[offset:])
        self.emit("\n")
        output = self.execute(source)
        if output:
            for line in output.splitlines():
                self.emit("  " + CYAN + line + RESET + "\n")
                self.pause(0.25)
        self.pause(hold)

    def type(self, value):
        if self.args.fast:
            self.emit(value.replace("\n", "\n   ...> "))
        else:
            for char in value:
                self.emit("\n   ...> " if char == "\n" else char)
                self.pause(0.009)

    def screen(self, number, title, subtitle):
        self.emit(ESC + "2J" + ESC + "H" + WHITE + "  JAX4J" + DIM +
                  "  /  JAVA 25                                      CPU · F32 · LIVE JSHELL\n")
        self.emit(DIM + "  " + "─" * 90 + "\n\n" + RESET)
        self.emit(PURPLE + f"  0{number} / 04" + WHITE + "   " + title + "\n")
        self.emit(DIM + "  " + subtitle + "\n\n" + RESET)
        self.pause(0.7)

    def note(self, text):
        self.emit("\n  " + GREEN + "→ " + WHITE + text + RESET + "\n")
        self.pause(2.4)

    def close(self):
        try:
            if self.process.poll() is None:
                self.process.stdin.write(b"/exit\n")
                self.process.wait(timeout=5)
        finally:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait()
            self.selector.close()
            if self.cast:
                self.cast.close()


def play(d):
    # Presentation helpers only format actual results. All math runs in JShell.
    d.execute('''
import com.nablatensor.jax4j.tensor.*;
import java.util.Arrays;
void show(Tensor t) { System.out.println(Arrays.toString(t.toFloatArray())); }
void lossRow(int step, Tensor w) {
    double loss = Math.pow(w.item() - 3, 2);
    String bar = "━".repeat(Math.max(1, (int)Math.ceil(loss / 9 * 22)));
    System.out.printf(java.util.Locale.ROOT,
        "%2d    w = %7.4f    loss = %.6f  %s%n", step, w.item(), loss, bar);
}
''')
    d.emit(ESC + "2J" + ESC + "H\n" + CYAN + r"""
       ██╗ █████╗ ██╗  ██╗██╗  ██╗     ██╗
       ██║██╔══██╗╚██╗██╔╝██║  ██║     ██║
       ██║███████║ ╚███╔╝ ███████║     ██║
  ██   ██║██╔══██║ ██╔██╗ ╚════██║██   ██║
  ╚█████╔╝██║  ██║██╔╝ ██╗     ██║╚█████╔╝
   ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝     ╚═╝ ╚════╝
""" + RESET)
    d.emit("\n  " + WHITE + "Java. Differentiated.\n\n" + RESET)
    d.emit("  " + PURPLE + "TENSORS" + DIM + "  /  " + PURPLE + "AUTODIFF" + DIM + "  /  " +
           PURPLE + "JIT" + DIM + "  /  " + PURPLE + "GRADIENT DESCENT\n\n" + RESET)
    d.emit("  " + DIM + "Real Java 25. Real JShell. Every result computed live.\n" + RESET)
    d.pause(3)

    d.screen(1, "Small API. Tensor thinking.", "Start with arrays. Express the computation as Java lambdas.")
    d.run("import com.nablatensor.jax4j.tensor.*;")
    d.run("var x = Tensors.array(new float[] {1, 2, 3});")
    d.run("show(x.mul(x).add(1));")
    d.run("var batch = Tensors.array(new float[][] {{1, 2}, {3, 4}, {5, 6}});")
    d.run("show(Vmap.apply(row -> row.mul(row).sum(), batch));")
    d.note("One function. Every row. That's vmap.")

    d.screen(2, "Write the function. Get the derivatives.", "f(x) = sum(x³)     ∇f = 3x²     diag(H) = 6x")
    d.run("var grad = Transforms.grad(t -> t.mul(t).mul(t).sum());")
    d.run("show(grad.apply(x));")
    d.run("show(Transforms.grad(t -> grad.apply(t).sum(), x));")
    d.note("First derivative → [3, 12, 27]. Differentiate again → [6, 12, 18].")

    d.screen(3, "Compose transforms. Capture once. Replay.", "Compile the gradient itself, then pass new tensor values.")
    d.run("var compiled = Jit.compile(grad);")
    d.run("show(compiled.apply(x));")
    d.run("var next = Tensors.array(new float[] {4, 5, 6});")
    d.run("show(compiled.apply(next));")
    d.run('System.out.println("Cached plans: " + compiled.cachedPlanCount());')
    d.note("New inputs. Same shape. One cached execution plan.")

    d.screen(4, "Let the gradient do the work.", "Minimize (w − 3)² with a compiled gradient and plain Java.")
    d.run("var slope = Jit.compile(Transforms.grad(w -> w.sub(3).mul(w.sub(3)).sum()));")
    d.run("var w = Tensors.array(new float[] {0});")
    d.run("lossRow(0, w);")
    d.run('''for (int i = 1; i <= 12; i++) {
    try (var g = slope.apply(w); var step = g.mul(0.2)) {
        var updated = w.sub(step);
        w.close(); w = updated;
    }
    if (i == 2 || i == 4 || i == 8 || i == 12) lossRow(i, w);
}''', hold=1.5)
    d.execute('''
if (Math.abs(w.item() - 3) > 0.007
    || compiled.cachedPlanCount() != 1 || slope.cachedPlanCount() != 1) {
    throw new IllegalStateException("Demo validation failed");
}
''')
    d.note("Loss: 9 → below 0.00005. Twelve updates. One compiled gradient.")
    d.emit("\n  " + CYAN + "github.com/nablatensor-dev/jax4j" + RESET + "\n")
    d.pause(4)
    d.execute("x.close(); batch.close(); next.close(); w.close();")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="skip typing delays and pauses")
    parser.add_argument("--record", type=Path, help="write an asciicast v2 recording")
    args = parser.parse_args()
    java_home = os.environ.get("JAVA_HOME")
    jshell = str(Path(java_home) / "bin/jshell") if java_home else shutil.which("jshell")
    if not jshell or not Path(jshell).is_file():
        parser.error("JShell not found; set JAVA_HOME to a JDK 25+ installation")
    if not (ROOT / "jax4j-core/target/classes/com/nablatensor/jax4j/tensor/Tensors.class").is_file():
        parser.error("compile the core first: mvn -pl jax4j-core -am -DskipTests compile")
    d = Demo(args, jshell)
    try:
        play(d)
    finally:
        d.close()


if __name__ == "__main__":
    main()
