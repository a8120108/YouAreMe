

const FILE_URL = "./photos/faces.jpeg";
const MODEL_URL = "./weights";

(async () => {
    // Load model
    await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);
    await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
    await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);

    // Detect Face
    const input = document.getElementById(FILE_URL);
    const result = await faceapi
        .detectSingleFace(input, new faceapi.SsdMobilenetv1Options())
        .withFaceLandmarks()
        .withFaceDescriptor();
    const displaySize = { width: input.width, height: input.height };
    // resize the overlay canvas to the input dimensions
    const canvas = document.getElementById("myCanvas");
    faceapi.matchDimensions(canvas, displaySize);
    const resizedDetections = faceapi.resizeResults(result, displaySize);
    console.log(resizedDetections);

    // Recognize Face
    const labeledFaceDescriptors = await detectFace();
    const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.7);
    if (result) {
        const bestMatch = faceMatcher.findBestMatch(result.descriptor);
        const box = resizedDetections.detection.box;
        const drawBox = new faceapi.draw.DrawBox(box, { label: bestMatch.label });
        drawBox.draw(canvas);
    }
})();

// detect for one people
async function detectFace() {
    const label = "Huu";
    const numberImage = 5;
    const descriptions = [];
    // for (let i = 1; i <= numberImage; i++) {
        const img = await faceapi.fetchImage(
            // `/data/Huu/${i}.jpg`
            "./photos/faces.jpeg"
        );
        const detection = await faceapi
            .detectSingleFace(img)
            .withFaceLandmarks()
            .withFaceDescriptor();
        descriptions.push(detection.descriptor);
    // }
    return new faceapi.LabeledFaceDescriptors(label, descriptions);
}


// detect for more people
async function detectAllLabeledFaces() {
    const labels = ["An Nhien", "Huu"];
    return Promise.all(
        labels.map(async label => {
            const descriptions = [];
            for (let i = 1; i <= 2; i++) {
                const img = await faceapi.fetchImage(
                    // `/data/${label}/${i}.jpg`
                    
                );
                const detection = await faceapi
                    .detectSingleFace(img)
                    .withFaceLandmarks()
                    .withFaceDescriptor();
                descriptions.push(detection.descriptor);
            }
            return new faceapi.LabeledFaceDescriptors(label, descriptions);
        })
    );
}
