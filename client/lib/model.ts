
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
export const getGenerationStream = (
    params: Params, 
    onData: (chunk: string) => void, 
    onHeatMap: (data: number[][]) => void, 
    onComplete: () => void, 
    onError: (error: Error) => void
) => {
    const { prompt, temperature, length } = params;

    // Construct the URL with query parameters
    let url = new URL("https://kalandjl-llm-from-scratch-api.hf.space/generate")
    // url = new URL("http://localhost:7860/generate")

    url.searchParams.append('prompt', prompt);
    url.searchParams.append('temperature', temperature.toString());
    url.searchParams.append('length', length.toString());

    // Create a new EventSource to connect to the stream
    const eventSource = new EventSource(url.toString());

    // Handle incoming messages from the server
    eventSource.onmessage = (event) => {
        // Parse the JSON data from the event
        const message = JSON.parse(event.data);

        // Process the message based on its 'event' type
        switch (message.event) {
            case 'prompt':
                // This event contains the initial prompt, you can use it if needed
                console.log("Received prompt:", message.data.prompt);
                // We will use onData to display the initial prompt tokens
                onData(message.data.tokens.join(""));
                break;
            
            case 'token':
                // This event contains a single generated character
                onData(message.data);
                break;

            case 'end':
                // This is the final event, containing the heatmap
                console.log("Stream finished.");
                onHeatMap(message.heatmap);
                onComplete();
                eventSource.close(); // Close the connection
                break;
        }
    };

    // Handle any errors with the connection
    eventSource.onerror = (err) => {
        console.error("EventSource encountered an error:", err);
        eventSource.close();
    };

    // Return the eventSource instance so it can be managed (e.g., closed manually)
    return eventSource;
};