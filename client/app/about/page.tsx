import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "About · LLM From Scratch",
  description: "How a transformer language model was built in pure NumPy: forward pass, hand-written backpropagation, attention, LoRA and deployment.",
};

const BUILD_LOG = [
  { day: "Day 1", title: "Embeddings and the forward pass", body: "Character vocabulary, token and position embedding matrices, and getting a prediction out for an input character." },
  { day: "Day 2", title: "Manual backpropagation", body: "Deriving and coding the gradient of every operation, then updating weights with stochastic gradient descent." },
  { day: "Day 3", title: "Self-attention", body: "Query, key and value projections, causal masking, softmax and its Jacobian, wrapped into multi-head attention." },
  { day: "Day 4", title: "Assembling the transformer", body: "Stacking blocks with residual connections, LayerNorm and a feed-forward layer into a model that actually trains." },
  { day: "Day 5", title: "The interface", body: "A FastAPI streaming endpoint and this Next.js front end, with temperature, length and an attention heatmap." },
];

const SPECS: [string, string][] = [
  ["Parameters", "368,832"],
  ["Tokenization", "character-level, 131 symbols"],
  ["Transformer blocks", "6"],
  ["Attention heads", "8 per block (head size 8)"],
  ["Embedding size", "64"],
  ["Feed-forward", "64 → 256 → 64, ReLU"],
  ["Context window", "1,000 characters"],
  ["Normalization", "post-norm LayerNorm, 2 per block"],
  ["Output layer", "tied to the token embedding"],
  ["Dependencies", "NumPy"],
];

const TIMELINE = [
  { date: "2025", title: "fast.ai Practical Deep Learning", body: "Course notebooks: image models, neural nets from scratch, random forests." },
  { date: "Jun 2, 2025", title: "duck_or_desert", body: "First trained model: a ResNet-18 image classifier using fastai." },
  { date: "Jun 2025", title: "LeanAI", body: "Scraped and LLM-labelled my own dataset; deployed an 11-model regression ensemble." },
  { date: "Jun 17, 2025", title: "lora-from-scratch", body: "Implemented the LoRA paper on a pre-trained Hugging Face model." },
  { date: "Jun–Jul 2025", title: "llm-from-scratch", body: "This project. Removed the frameworks and built the transformer myself.", current: true },
];

export default function About() {
  return (
    <main className="min-h-screen px-4 py-12 sm:py-20">
      <article className="mx-auto w-full max-w-2xl">
        <Link href="/" className="text-sm text-zinc-500 transition hover:text-zinc-200">
          ← Back to the generator
        </Link>

        <header className="mt-8">
          <p className="text-sm text-zinc-500">About</p>
          <h1 className="mt-1 text-3xl font-semibold tracking-tight text-zinc-50">LLM From Scratch</h1>
          <p className="mt-4 leading-relaxed text-zinc-400">
            I&apos;m a Computing and Mathematics (COMA) student at Queen&apos;s University. This project is a GPT-style language model I wrote without PyTorch or TensorFlow. Every layer, every
            gradient and the training loop are implemented in NumPy.
          </p>
        </header>

        <Section title="Why from scratch">
          <p>
            In my earlier projects, fastai and PyTorch handled the hard parts for me. That made me productive, but it also hid how
            the models worked. I wanted to understand language models in a practical way, including the complexity that libraries
            abstract away. Implementing the forward pass, backpropagation and optimizer myself meant I had to understand each piece
            well enough to derive its gradient.
          </p>
        </Section>

        <Section title="How I built it">
          <p>I worked through the model one component at a time and kept a log as I went.</p>
          <ol className="mt-2 divide-y divide-white/[0.06] overflow-hidden rounded-xl border border-white/[0.08]">
            {BUILD_LOG.map((d) => (
              <li key={d.day} className="flex gap-4 px-4 py-3.5">
                <span className="w-12 shrink-0 font-mono text-xs leading-6 text-zinc-500">{d.day}</span>
                <div>
                  <p className="text-sm font-medium text-zinc-200">{d.title}</p>
                  <p className="mt-0.5 text-sm text-zinc-400">{d.body}</p>
                </div>
              </li>
            ))}
          </ol>
        </Section>

        <Section title="Architecture">
          <p>
            The model follows the decoder-only transformer from <em>Attention Is All You Need</em> (2017). Each block applies
            multi-head causal self-attention, a residual connection and LayerNorm, then a ReLU feed-forward layer with a second
            residual connection and LayerNorm. The output projection reuses the embedding matrix, and sampling divides the logits by
            a temperature before the softmax.
          </p>
          <dl className="mt-4 overflow-hidden rounded-xl border border-white/[0.08] text-sm">
            {SPECS.map(([k, v], i) => (
              <div key={k} className={`flex justify-between gap-4 px-4 py-2.5 ${i % 2 ? "bg-white/[0.02]" : ""}`}>
                <dt className="text-zinc-500">{k}</dt>
                <dd className="text-right font-mono text-zinc-200">{v}</dd>
              </div>
            ))}
          </dl>
          <p>
            Writing the backward passes was the core of the work. Attention needs the gradient through the softmax Jacobian and
            the masked score matrix. LayerNorm needs the gradient through the mean and variance. Each residual connection sends
            gradients down two paths that have to be summed. Each layer caches its activations on the forward pass so the backward
            pass can use them.
          </p>
        </Section>

        <Section title="Training">
          <p>
            I trained the base model on the source code of the Python <span className="font-mono text-zinc-300">requests</span>{" "}
            library, about 377,000 characters, using random 128-character windows in batches of 32. Training ran on a CPU for over
            24 hours. Cross-entropy loss fell from about 18 at initialization to 0.66, which means the model puts an average
            probability of roughly e<sup>−0.66</sup> ≈ 52% on the correct next character.
          </p>
          <p>
            The training loop uses a linear warm-up followed by either cosine or linear learning-rate decay, both implemented by
            hand.
          </p>
        </Section>

        <Section title="LoRA fine-tuning">
          <p>
            Just before this project, I implemented the LoRA paper (<em>Low-Rank Adaptation of Large Language Models</em>) on a
            pre-trained Hugging Face model. Here I rebuilt it for my own architecture: rank-8 adapters (α = 1) wrap the query and
            value projections in all 48 attention heads. The base weights stay frozen and only the adapters train, which is
            55,296 parameters or about 15% of the model.
          </p>
          <p>
            I fine-tuned on a 3.6 MB corpus of general Python code, with 100 warm-up steps and a learning rate decaying from
            3×10<sup>−4</sup> to 3×10<sup>−5</sup>. The demo runs the base model with these adapters applied.
          </p>
        </Section>

        <Section title="Serving the model">
          <p>
            The NumPy model runs in a FastAPI service in a Docker container on Hugging Face Spaces. Because the model generates one
            character at a time, the API streams each character to this page as soon as it&apos;s sampled, using Server-Sent
            Events. After the last character it runs one forward pass over the full text and returns the attention weights averaged
            across all 48 heads, which is what the heatmap shows.
          </p>
        </Section>

        <Section title="What I learned">
          <ul className="list-disc space-y-2 pl-5 marker:text-zinc-600">
            <li>How attention, normalization and residual paths behave numerically, not just conceptually.</li>
            <li>How to read a research paper and turn it into working code, twice: the transformer and LoRA.</li>
            <li>Numerical stability is part of the math in practice: subtracting the max before a softmax and adding an epsilon inside the log.</li>
            <li>Why frameworks make the design choices they do, such as weight tying and caching activations for the backward pass.</li>
            <li>The cost of training without a GPU, and how much model size and context length matter in practice.</li>
          </ul>
        </Section>

        <Section title="Limitations">
          <p>
            At 369K parameters and character-level tokenization, the model learns Python&apos;s look (indentation, keywords,
            identifiers and punctuation) but not its meaning. Its output is plausible-looking code, not working code. A next step
            I&apos;d planned was moving to a more modern design, starting with rotary positional embeddings (RoPE).
          </p>
        </Section>

        <Section title="Where it fits">
          <ol className="mt-2">
            {TIMELINE.map((t, i) => (
              <li key={t.title} className="relative pb-6 pl-6 last:pb-0">
                {i < TIMELINE.length - 1 && <span className="absolute left-[5px] top-3 h-full w-px bg-white/10" />}
                <span className={`absolute left-0 top-1.5 h-[11px] w-[11px] rounded-full border-2 ${t.current ? "border-zinc-100 bg-zinc-100" : "border-zinc-600 bg-zinc-950"}`} />
                <p className="font-mono text-xs text-zinc-500">{t.date}</p>
                <p className={`mt-0.5 font-medium ${t.current ? "text-zinc-50" : "text-zinc-300"}`}>{t.title}</p>
                <p className="mt-0.5 text-sm text-zinc-400">{t.body}</p>
              </li>
            ))}
          </ol>
        </Section>

        <footer className="mt-14 flex flex-wrap gap-x-6 gap-y-2 border-t border-white/[0.08] pt-6 text-sm text-zinc-500">
          <a href="https://github.com/kalandjl/llm-from-scratch" className="transition hover:text-zinc-200">Source on GitHub</a>
          <a href="https://github.com/kalandjl/llm-from-scratch/blob/main/core/notebooks/model.ipynb" className="transition hover:text-zinc-200">Training notebook</a>
          <a href="https://github.com/kalandjl/llm-from-scratch/blob/main/core/notebooks/lora.ipynb" className="transition hover:text-zinc-200">LoRA notebook</a>
        </footer>
      </article>
    </main>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="mt-12">
      <h2 className="text-lg font-semibold tracking-tight text-zinc-100">{title}</h2>
      <div className="mt-3 space-y-4 leading-relaxed text-zinc-400 [&>p]:text-[15px]">{children}</div>
    </section>
  );
}
