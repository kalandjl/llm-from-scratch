
export const getGeneration = async (prompt: string, temperature: number, heatMap: boolean, length: number) => {

    const url = "http://localhost:4000/api/generate"

    try {
        const response = await fetch(url, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                data: [prompt, temperature, length] 
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        
        // Gradio returns result in data array
        return result.data[0]; // The generated text
    } catch (error) {

        console.error("Error calling generation API:", error)
        throw error
    }
}