"use client"
import HeatMap from "@/components/HeatMap";
import { getGenerationStream } from "@/lib/model";
import Link from "next/link";
import { FormEvent, useState } from "react";

export default function Home() {

    const [generation, setGeneration] = useState<string>(``);
    const [prompt, setPrompt] = useState<string>("");
    const [temperature, setTemperature] = useState<number>(0.7);
    const [heatMap, setHeatMap] = useState<boolean>(false);
    const [length, setLength] = useState<number>(100);
    const [error, setError] = useState<string | undefined>();
    const [heatmap, setHeatmap] = useState<number[][]>([[]])
    const [dataLoading, setDataLoading] = useState<boolean>(false)

    const handleData = (chunk: string) => {
        const cleaned = chunk.replace(/\\n/g, "\n");
        setGeneration(prevGeneration => `${prevGeneration}${cleaned}`);
    }

    const handleHeatMap = (data: number[][]) => {
        setHeatmap(data)
    }

    const handleCompletion = () => {
        setDataLoading(false)
    };

    const handleError = (error: Error) => {
        console.error("Stream failed:", error);
        setError(error.message);
        setDataLoading(false);
    };

    const handleSubmit = async (e: FormEvent) => {
        e.preventDefault();

        if (dataLoading || !prompt) return;

        setError(undefined);
        setGeneration(``);
        setHeatmap([[]]);

        try {
            setDataLoading(true)
            getGenerationStream(
                { prompt: prompt, temperature: temperature, length: length },
                handleData,
                handleHeatMap,
                handleCompletion,
                handleError
            );
        } catch (err) {
            console.error("Failed to get generation:", err);
            setError("Sorry, something went wrong. Please try again.");
            setDataLoading(false);
        }
    };

    const labelClass = "flex items-baseline justify-between text-sm text-zinc-400";
    const rangeClass = "mt-3 w-full accent-zinc-100 cursor-pointer";

    return (
        <main className="min-h-screen px-4 py-12 sm:py-20">
            <div className="mx-auto w-full max-w-2xl">
                <div className="mb-8 flex items-center justify-between">
                    <h1 className="text-2xl font-semibold tracking-tight text-zinc-50">AI Python Code Generator</h1>
                    <div className="flex items-center gap-2">
                        <Link
                            href="/about"
                            className="rounded-full px-3 py-1.5 text-xs font-medium text-zinc-400 transition hover:text-zinc-50"
                        >
                            About
                        </Link>
                        <Link
                            href="https://www.github.com/kalandjl/llm-from-scratch"
                            className="rounded-full border border-white/10 px-3 py-1.5 text-xs font-medium text-zinc-300 transition hover:border-white/25 hover:text-zinc-50"
                        >
                            Check out the git repo!
                        </Link>
                    </div>
                </div>

                <div className="rounded-2xl border border-white/[0.08] bg-zinc-900/60 p-6 sm:p-8">
                    <form onSubmit={handleSubmit} className="space-y-6">
                        {/* Prompt Input */}
                        <div>
                            <label htmlFor="prompt-input" className="mb-2 block text-sm text-zinc-400">
                                Your Prompt
                            </label>
                            <textarea
                                id="prompt-input"
                                rows={4}
                                spellCheck={false}
                                className="w-full resize-none rounded-xl border border-white/10 bg-black/30 p-3.5 font-mono text-sm text-zinc-100 placeholder-zinc-600 outline-none transition focus:border-white/25"
                                placeholder="e.g. def app():"
                                value={prompt}
                                onChange={e => setPrompt(e.target.value)}
                            />
                        </div>

                        {/* Configuration Options */}
                        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
                            <div>
                                <label htmlFor="temperature-input" className={labelClass}>
                                    Temperature
                                    <span className="font-mono text-zinc-100">{temperature.toFixed(1)}</span>
                                </label>
                                <input
                                    type="range"
                                    id="temperature-input"
                                    min="0.1"
                                    max="2"
                                    step="0.1"
                                    className={rangeClass}
                                    value={temperature}
                                    onChange={e => setTemperature(parseFloat(e.currentTarget.value))}
                                />
                            </div>

                            <div>
                                <label htmlFor="length-input" className={labelClass}>
                                    Max Length
                                    <span className="font-mono text-zinc-100">{length}</span>
                                </label>
                                <input
                                    type="range"
                                    id="length-input"
                                    min="10"
                                    max="250"
                                    step="10"
                                    className={rangeClass}
                                    value={length}
                                    onChange={e => setLength(parseInt(e.currentTarget.value))}
                                />
                            </div>
                        </div>

                        {/* HeatMap Toggle */}
                        <label htmlFor="heatMap-input" className="flex cursor-pointer items-center justify-between rounded-xl border border-white/[0.08] px-4 py-3">
                            <span className="text-sm text-zinc-300">Include Heatmap</span>
                            <input
                                type="checkbox"
                                id="heatMap-input"
                                className="peer sr-only"
                                checked={heatMap}
                                onChange={e => setHeatMap(e.currentTarget.checked)}
                            />
                            <span className="relative h-5 w-9 rounded-full bg-zinc-700 transition peer-checked:bg-zinc-100 peer-focus-visible:ring-2 peer-focus-visible:ring-zinc-400 after:absolute after:left-0.5 after:top-0.5 after:h-4 after:w-4 after:rounded-full after:bg-zinc-300 after:transition peer-checked:after:translate-x-4 peer-checked:after:bg-zinc-900" />
                        </label>

                        {/* Submit Button */}
                        <button
                            type="submit"
                            disabled={dataLoading || !prompt}
                            className="flex h-11 w-full items-center justify-center gap-2 rounded-xl bg-zinc-50 text-sm font-medium text-zinc-900 transition hover:bg-white active:scale-[0.99] disabled:cursor-not-allowed disabled:bg-white/[0.06] disabled:text-zinc-500"
                        >
                            {dataLoading && <span className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />}
                            {dataLoading ? "Generating..." : "Get Generation"}
                        </button>
                    </form>
                </div>

                {/* Output Section */}
                {error && (
                    <div className="mt-6 rounded-xl border border-red-500/20 bg-red-500/[0.06] px-4 py-3 text-sm text-red-300">
                        {error}
                    </div>
                )}
                {generation && (
                    <div className="mt-6 overflow-hidden rounded-2xl border border-white/[0.08] bg-zinc-900/60">
                        <div className="border-b border-white/[0.08] px-5 py-3">
                            <h2 className="text-sm font-medium text-zinc-300">Result:</h2>
                        </div>
                        <pre className="overflow-x-auto p-5 font-mono text-sm leading-relaxed text-zinc-200 whitespace-pre-wrap">{generation}</pre>
                    </div>
                )}

                {heatMap && (
                    <div className="mt-6">
                        <HeatMap heatmapData={heatmap} tokens={generation.split("")} loading={dataLoading && generation.length > 0} />
                    </div>
                )}
            </div>
        </main>
    );
}
