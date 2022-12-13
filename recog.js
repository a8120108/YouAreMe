function usepics(num, folderList) {
    const FILE_URL = "./photos";
    const MODEL_URL = "./weights";

    (async () => {
        // Load model
        await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);
        await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
        await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);

        // Detect Face
        const input = document.getElementById(`myImg${num}`);
        const analyse = await imageToVector(input);
        console.log(analyse)
        console.log(analyse[0].descriptor)
        
        const result = await faceapi
            .detectAllFaces(input, new faceapi.SsdMobilenetv1Options())
            .withFaceLandmarks()
            .withFaceDescriptors();
        const displaySize = { width: input.width, height: input.height };
        // resize the overlay canvas to the input dimensions
        const canvas = document.getElementById(`myCanvas${num}`);
        faceapi.matchDimensions(canvas, displaySize);
        for (var i = 0; i < result.length; i++) {
            const resizedDetections = faceapi.resizeResults(result[i], displaySize);


            // Recognize Face
            const labeledFaceDescriptors = await detectAllLabeledFaces(folderList);
            const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.7);
            if (result[i]) {
                const bestMatch = faceMatcher.findBestMatch(result[i].descriptor);
                const box = resizedDetections.detection.box;
                const drawBox = new faceapi.draw.DrawBox(box, { label: bestMatch.label });
                drawBox.draw(canvas);
            }
        }
    })();

    // detect for one people
    async function detectFace() {
        const label = "Ohtani";
        const numberImage = 5;
        const descriptions = [];
        for (let i = 1; i <= numberImage; i++) {
            const img = await faceapi.fetchImage(
                `./datas/Ohtani/${i}.jpeg`
            );
            const detection = await faceapi
                .detectSingleFace(img)
                .withFaceLandmarks()
                .withFaceDescriptor();
            descriptions.push(detection.descriptor);
        }
        return new faceapi.LabeledFaceDescriptors(label, descriptions);
    }


    // detect for more people
    async function detectAllLabeledFaces(folderList) {
        const labels = folderList;
        return Promise.all(
            labels.map(async label => {
                const descriptions = [];


                for (let i = 1; i <= 2; i++) {
                    const img = await faceapi.fetchImage(
                        `./datas/${label}/${i}.jpeg`
                    
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

    async function imageToVector(blob, inputSize = 512) {

        var scoreThreshold = 0.8;
        const OPTION = new faceapi.SsdMobilenetv1Options({
            inputSize,
            scoreThreshold,
        });

        let fullDesc = await faceapi.detectAllFaces(blob, OPTION)
            .withFaceLandmarks()
            .withFaceDescriptors();
        
        return fullDesc;
    }
};