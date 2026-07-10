"use client";

import { useState } from "react";
import { Sparkles, Send, Loader2, ThumbsUp, ThumbsDown, HelpCircle } from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface ClassifyResult {
  label: string;
  confidence: number;
  raw_output: string;
}

export default function Home() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ClassifyResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`${API_BASE}/classify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(`${res.status}: ${txt.slice(0, 200)}`);
      }
      const data: ClassifyResult = await res.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  };

  const labelConfig: Record<string, { icon: typeof ThumbsUp; color: string; bg: string }> = {
    positive: { icon: ThumbsUp, color: "text-success", bg: "bg-success/10 border-success/30" },
    negative: { icon: ThumbsDown, color: "text-error", bg: "bg-error/10 border-error/30" },
    unknown: { icon: HelpCircle, color: "text-foreground-subtle", bg: "bg-surface-2 border-border" },
  };

  const lc = result ? labelConfig[result.label] || labelConfig.unknown : null;
  const Icon = lc?.icon;

  return (
    <div className="flex min-h-full flex-col">
      <header className="flex items-center justify-between border-b border-border bg-surface/80 px-6 py-3 backdrop-blur-sm">
        <div className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent-subtle">
            <Sparkles className="h-4 w-4 text-accent" />
          </div>
          <div>
            <h1 className="text-sm font-semibold">sentirise</h1>
            <p className="text-[10px] text-foreground-subtle">QLoRA Sentiment Classifier</p>
          </div>
        </div>
        <div className="text-xs text-foreground-subtle">Qwen3-0.6B · LoRA fine-tuned</div>
      </header>

      <div className="mx-auto w-full max-w-2xl flex-1 px-4 py-8">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={5}
          placeholder="Type or paste text to classify sentiment..."
          className="w-full resize-none rounded-xl border border-border bg-surface-2 px-4 py-3 text-sm text-foreground placeholder:text-foreground-subtle focus:border-accent/50 focus:outline-none"
        />

        <button
          onClick={handleSubmit}
          disabled={!text.trim() || loading}
          className="mt-4 flex w-full items-center justify-center gap-2 rounded-xl bg-accent py-3 text-sm font-medium text-white transition-colors hover:bg-accent-hover disabled:opacity-40"
        >
          {loading ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              Classifying...
            </>
          ) : (
            <>
              <Send className="h-4 w-4" />
              Classify Sentiment
            </>
          )}
        </button>

        {error && (
          <div className="mt-4 rounded-xl border border-error/30 bg-error/10 px-4 py-3 text-sm text-error">
            {error}
          </div>
        )}

        {result && Icon && lc && (
          <div className="mt-6 animate-fade-in-up">
            <div className={`rounded-xl border p-6 ${lc.bg}`}>
              <div className="flex items-center gap-3">
                <Icon className={`h-8 w-8 ${lc.color}`} />
                <div>
                  <div className={`text-2xl font-bold capitalize ${lc.color}`}>
                    {result.label}
                  </div>
                  <div className="text-xs text-foreground-subtle">
                    confidence: {Math.round(result.confidence * 100)}%
                  </div>
                </div>
              </div>
              <div className="mt-4 border-t border-border/50 pt-3">
                <div className="text-xs text-foreground-subtle">raw model output</div>
                <div className="mt-1 font-mono text-sm text-foreground-muted">
                  {result.raw_output || "(empty)"}
                </div>
              </div>
            </div>
          </div>
        )}

        <div className="mt-8 text-center text-xs text-foreground-subtle">
          Fine-tuned with QLoRA on IMDB · 4-bit quantized · runs on local GPU
        </div>
      </div>
    </div>
  );
}
