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

};

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

async function make128Dvectors(image) {
    const MODEL_URL = "./weights";
    await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);
    await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
    await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);

    const answer = await imageToVector(image)

    for (var i = 0; i < 128; i++) {
        // console.log(answer[0].descriptor[i])

    }
    console.log("----------------")
    return answer[0].descriptor;
    // console.log(answer[0].descriptor)
}


async function receiveAndRun(names) {
    var vector;
    var vectors = [];
    for (var i = 0; i < names.length; i++) {
        var name = names[i]
        vector = await make128Dvectors(name);
        vectors.push(vector);

        if (i == names.length - 1) {
            var sum = 0;
            var cosum = 0;
            var first = 0;
            var second = 0;

            //処理部分

            for (var j = 0; j < 128; j++) {
                //euclidean
                sum += (vectors[0][j] - vectors[1][j]) ** 2

                //cosSim
                cosum += vectors[0][j] * vectors[1][j]
                first += vectors[0][j] ** 2
                second += vectors[1][j] ** 2
            }
            first = Math.sqrt(first)
            second = Math.sqrt(second)
            sum = Math.sqrt(sum)


            //cosum
            var answer = cosum / (first * second);

            if (answer > 0.948 && sum < 0.575) {
                console.log("ｵﾅｼﾞﾋﾄﾀﾞﾖｰ!‼︎")
                console.log("cosSimilarity = " + answer)
                console.log("euclideanDistance = " + sum)
            } else if (answer > 0.948) {
                console.log("cosはヨシ！")
                console.log("euclideanDistance = " + sum)
                console.log("一致しないみたいです。")
            } else if (sum < 0.575) {
                console.log("euclidはヨシ！")
                console.log("cosSimilarity = " + answer)
                console.log("一致しないみたいです。")
            } else {
                console.log("cosSimilarity = " + answer)
                console.log("euclideanDistance = " + sum)
                console.log("一致しないみたいです。")
                console.log(vectors)
            }

        }
    }
}
