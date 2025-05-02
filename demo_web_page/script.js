/* 
        to use the webpage: 
        open a terminal, go to the web_page_all3 folder, start a http.server on port 8000 (reason is CORS - see notes.txt):

        python -m http.server 8000

        Then open the index.html in the browser:

        http://localhost:8000/index.html 

*/

async function runModel() {
    const model_name = 'models/model_Sachkonto_stratified_All3.onnx';
    const Sachkonten_file = 'mappings/target_dict.json';
    const Steuerkennzeichen_file = 'mappings/Steuerkennzeichen_dict_invers.json';

    async function loadMapping(url) {
        try {
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error("Error loading JSON mapping:", error);
            throw error;
        }
    }

    // Jetzt warten wir korrekt auf beide Mappings
    const [sachkontoDict, steuerDict] = await Promise.all([
        loadMapping(Sachkonten_file),
        loadMapping(Steuerkennzeichen_file)
    ]);

    // Jetzt sind die Inputs dran
    let input1 = Number.parseInt(document.getElementById("input1").value, 10);
    let input2 = Number.parseInt(document.getElementById("input2").value, 10);
    let input3_text = document.getElementById("input3").value.trim().toUpperCase();
    if (input3_text === "") input3_text = "nan";

    const input3 = steuerDict[input3_text];

    if (isNaN(input1) || isNaN(input2) || isNaN(input3)) {
        document.getElementById("output").textContent = "Invalid input. Enter numbers and valid text.";
        return;
    }

    async function runInference(input1, input2, input3) {
        try {
            const session = await ort.InferenceSession.create(model_name);
            const inputTensor = new ort.Tensor("float32", new Float32Array([input1, input2, input3]), [1, 3]);
            const results = await session.run({ [session.inputNames[0]]: inputTensor });
            return results.probabilities.cpuData;
        } catch (error) {
            console.error("Error running ONNX model:", error);
            document.getElementById("output").textContent = "Error running the model.";
            return null;
        }
    }

    const outputData = await runInference(input1, input2, input3);
    if (!outputData) return;

    const predictedClass = outputData.indexOf(Math.max(...outputData));
    const sachkonto_prediction = sachkontoDict[predictedClass];

    function getTopThreeIndices(arr) {
        return Array.from(arr)
            .map((value, index) => ({ value, index }))
            .sort((a, b) => b.value - a.value)
            .slice(0, 3)
            .map(item => item.index);
    }

    const top_3_indeces = getTopThreeIndices(outputData);
    const top_3_sachkontos = top_3_indeces.map(idx => sachkontoDict[idx]);

    document.getElementById("output").textContent = sachkonto_prediction;
    document.getElementById("top-values").innerHTML = `
        <p>Highest probability: ${top_3_sachkontos[0]}</p>
        <p>Second: ${top_3_sachkontos[1]}</p>
        <p>Third: ${top_3_sachkontos[2]}</p>
    `;
}
