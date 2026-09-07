const canvas = document.getElementById("drawing-canvas");
const context = canvas.getContext("2d");

const clearButton = document.getElementById("clear-button");

const predictionText = document.getElementById("prediction");
const confidenceText = document.getElementById("confidence");

// White background
context.fillStyle = "white";
context.fillRect(0, 0, canvas.width, canvas.height);


// Brush settings
context.strokeStyle = "black";
context.lineWidth = 20;
context.lineCap = "round";
context.lineJoin = "round";

let drawing = false;
let predictionTimer;


function schedulePrediction() {
    clearTimeout(predictionTimer);

    predictionTimer = setTimeout(() => {
        predictCanvas();
    }, 200);
}


canvas.addEventListener("pointerdown", (event) => {
    clearTimeout(predictionTimer);

    drawing = true;

    canvas.setPointerCapture(event.pointerId);

    context.beginPath();
    context.moveTo(event.offsetX, event.offsetY);
});


canvas.addEventListener("pointermove", (event) => {
    if (!drawing) {
        return;
    }

    context.lineTo(event.offsetX, event.offsetY);
    context.stroke();
});


canvas.addEventListener("pointerup", () => {
    drawing = false;

    schedulePrediction();
});


clearButton.addEventListener("click", () => {

    clearTimeout(predictionTimer);

    context.fillStyle = "white";
    context.fillRect(
        0,
        0,
        canvas.width,
        canvas.height
    );

    context.strokeStyle = "black";

    predictionText.textContent = "Prediction: -";
    confidenceText.textContent = "Confidence: -";
});

async function predictCanvas() {
    canvas.toBlob(async (blob) => {

        const formData = new FormData();

        formData.append(
            "image",
            blob,
            "drawing.png"
        );

        const response = await fetch(
            "https://number-classification-api-ywzz.onrender.com/predict",
            {
                method: "POST",
                body: formData
            }
        );

        const result = await response.json();

        predictionText.textContent =
            `Prediction: ${result["Predicted Number"]}`;

        confidenceText.textContent =
            `Confidence: ${(result["Confidence"] * 100).toFixed(2)}%`;

    }, "image/png");
}
