
export const getGeneration = async (prompt: string, temperature: number, heatMap: boolean, length: number) => {

    const url = "http://localhost:4000/generate";

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

