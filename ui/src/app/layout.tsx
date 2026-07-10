import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "sentirise - QLoRA Sentiment Classifier",
  description: "Fine-tuned Qwen3-0.6B sentiment classifier with LoRA. 100% local GPU.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="flex h-full flex-col bg-background text-foreground">{children}</body>
    </html>
  );
}
