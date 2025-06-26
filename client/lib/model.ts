
export const getGeneration = async (prompt: string, temperature: number, heatMap: boolean, length: number) => {

    let url = "https://kalandjl-llm-from-scratch-api.hf.space/generate";
    url = 'http://localhost:7860/generate'

    try {
        const response = await fetch(url, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                'prompt': prompt, 
                'temperature': temperature, 
                'heatMap': heatMap, 
                'length': length
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        
        // Gradio returns result in data array
        return result.generation // The generated text
    } catch (error) {

        console.error("Error calling generation API:", error)
        throw error
    }
}

interface Params {
    prompt: string
    temperature: number
    length: number
}

// Using an http stream, make a constant generation from the llm
export const getGenerationStream = (params: Params, onData: (chunk: string) => void, onComplete: () => void, onError: (error: Event) => void) => {

    const { prompt, temperature, length } = params

    const url = new URL("http://localhost:7860/generate-stream")

    url.searchParams.append('prompt', prompt)
    url.searchParams.append('temperature', temperature.toString())
    url.searchParams.append('length', length.toString())

    const eventSource = new EventSource(url)

    eventSource.onmessage = (event) => {

        if (event.data === "[DONE]") { 
            eventSource.close()
            onComplete()
        } else {
            onData(event.data)
        }
    }

    eventSource.onerror = (err) => {
        console.error("EventSource ecountered an error:", err)
        eventSource.close()
        onError(err)
    }

    return eventSource
}