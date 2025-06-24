"use client"
import { getGeneration } from "@/lib/model";
import { FormEvent, useState } from "react";

export default function Home() {

    const [generation, setGeneration] = useState<string | undefined>();
    const [prompt, setPrompt] = useState<string>("");
    const [temperature, setTemperature] = useState<number>(0.7);
    const [heatMap, setHeatMap] = useState<boolean>(false);
    const [length, setLength] = useState<number>(100);
    const [loading, setLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | undefined>();

    // Handle form submission asynchronously
    const handleSubmit = async (e: FormEvent) => {
        e.preventDefault(); // Prevent page from reloading
        if (loading || !prompt) return;

        setLoading(true);
        setError(undefined);
        setGeneration(undefined);

        try {
            const result = await getGeneration(prompt, temperature, heatMap, length);
            setGeneration(result);
        } catch (err) {
            console.error("Failed to get generation:", err);
            setError("Sorry, something went wrong. Please try again.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <main className="flex min-h-screen items-center justify-center bg-gray-900 p-4 text-white">
            <div className="w-full max-w-2xl">
                <div className="rounded-xl bg-gray-800 p-8 shadow-lg">
                    <h1 className="mb-6 text-center text-3xl font-bold">AI Python Code Generator</h1>

                    <form onSubmit={handleSubmit} className="space-y-6">
                        {/* Prompt Input */}
                        <div>
                            <label htmlFor="prompt-input" className="mb-2 block font-medium text-gray-300">
                                Your Prompt
                            </label>
                            <textarea
                                id="prompt-input"
                                rows={3}
                                className="w-full rounded-md border border-gray-600 bg-gray-700 p-3 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
                                placeholder="e.g. def app():"
                                value={prompt}
                                onChange={e => setPrompt(e.target.value)}
                            />
                        </div>

                        {/* Configuration Options */}
                        <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
                            {/* Temperature Input */}
                            <div>
                                <label htmlFor="temperature-input" className="mb-2 block font-medium text-gray-300">
                                    Temperature ({temperature})
                                </label>
                                <input
                                    type="range"
                                    id="temperature-input"
                                    min="0.1"
                                    max="2"
                                    step="0.1"
                                    className="w-full"
                                    value={temperature}
                                    onChange={e => setTemperature(parseFloat(e.currentTarget.value))}
                                />
                            </div>

                            {/* Length Input */}
                            <div>
                                <label htmlFor="length-input" className="mb-2 block font-medium text-gray-300">
                                    Max Length ({length})
                                </label>
                                <input
                                    type="range"
                                    id="length-input"
                                    min="10"
                                    max="1000"
                                    step="10"
                                    className="w-full"
                                    value={length}
                                    onChange={e => setLength(parseInt(e.currentTarget.value))}
                                />
                            </div>
                        </div>

                        {/* HeatMap Checkbox */}
                        <div className="flex items-center gap-3">
                            <input
                                type="checkbox"
                                id="heatMap-input"
                                className="h-5 w-5 rounded border-gray-600 bg-gray-700 text-blue-600 focus:ring-blue-500"
                                // FIX: Use `checked` for checkbox state, not `value`
                                checked={heatMap}
                                onChange={e => setHeatMap(e.currentTarget.checked)}
                            />
                            <label htmlFor="heatMap-input" className="font-medium text-gray-300">
                                Include Heatmap
                            </label>
                        </div>
                        
                        {/* Submit Button */}
                        <button
                            type="submit"
                            disabled={loading || !prompt}
                            className="w-full rounded-md bg-blue-600 py-3 font-semibold text-white transition-colors hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-800 disabled:cursor-not-allowed disabled:bg-gray-500"
                        >
                            {loading ? "Generating..." : "Get Generation"}
                        </button>
                    </form>
                </div>

                {/* Output Section */}
                {error && (
                    <div className="mt-8 rounded-lg bg-red-900/50 p-6 text-center text-red-300">
                        {error}
                    </div>
                )}
                {generation && (
                    <div className="mt-8 rounded-lg bg-gray-950/50 p-6">
                        <h2 className="text-xl font-semibold text-gray-300">Result:</h2>
                        <p className="mt-4 whitespace-pre-wrap text-gray-200">{generation}</p>
                    </div>
                )}
            </div>
        </main>
    );
}