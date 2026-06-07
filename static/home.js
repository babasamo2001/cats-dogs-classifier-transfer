document.addEventListener("DOMContentLoaded", () => {

    const fileInput = document.getElementById("fileInput");
    const predictBtn = document.getElementById("predictBtn");
    const originalImg = document.getElementById("originalImg");

    // IMAGE PREVIEW
    fileInput.addEventListener("change", (e) => {

        const file = e.target.files[0];
        if (!file) return;

        const reader = new FileReader();

        reader.onload = (event) => {
            originalImg.src = event.target.result;
        };

        reader.readAsDataURL(file);
    });

    predictBtn.addEventListener("click", predict);
});


// PREDICT FUNCTION
async function predict() {

    const fileInput = document.getElementById("fileInput");

    if (!fileInput.files[0]) {
        alert("Upload an image first");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    const res = await fetch("/predict", {
        method: "POST",
        body: formData
    });

    const data = await res.json();

    document.getElementById("result").innerText =
        "Prediction: " + (data.prediction || data.class || data.result || "N/A");

    document.getElementById("confidence").innerText =
        "Confidence: " + (
            ((data.confidence ?? data.score ?? data.prob ?? 0) * 100).toFixed(2)
        ) + "%";
}


