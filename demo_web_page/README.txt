web_page_all3 Folder:

- "all3" refers to the 3 datasets imported and concatenated into one dataset in the jupyter-notebook,
     that is used to build the model that is finally exported to an onnx-model, used here.
- contains the index.html that expects "Buchungskreis", "Lieferant", "Steuerkennzeichen" and outputs "Sachkonto"
- the onnx-model used is referenced in the script.js
- the script.js is referenced/used in the index.html
- to use an onnx-model it has to be copied to the web_page_all3 folder
- an onnx-model should be tested to be sure, that it has the same performance as the original model: use the testOnnx.ipynb
- when an onnx-model is created, the name and the Sachkonto-Array is written to the cell-output in the notebook: 
    name and array have to be used in the script.js - and also in the testOnnx.ipnb of coures
- to use the webpage: 
    open a terminal, go to the \sakto_SAP_BSAK\web_page_all3 folder, start a http.server on port 8000 (reason is CORS - see notes.txt):
    
    python -m http.server 8000

    Then open the index.html in the browser:

    http://localhost:8000/index.html

- the "Test_Onnx_Model.ipynb" contains the code to test the onnx-model - after all, the onnx model is a re-make of the original model with onnx means...
