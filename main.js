// 省略



const FILE_URL = "./photos/faces.jpeg";
const MODEL_URL = "./weights";

let img, canvas, context;

window.onload = (event) => {
    console.log("onload!");
    loadModels();
}

async function loadModels() {
    console.log("loadModels");
    Promise.all([
        faceapi.loadSsdMobilenetv1Model(MODEL_URL),
        faceapi.loadFaceLandmarkModel(MODEL_URL),
        faceapi.loadFaceRecognitionModel(MODEL_URL)
    ]).then(detectAllFaces);
}

async function detectAllFaces() {
    console.log("detectAllFaces");
    // ここで画像解析を行います。
}


async function detectAllFaces() {
    console.log("detectAllFaces");

    // 1, 画像の読み込み
    img = await faceapi.fetchImage("photos/0010171846705174-web.jpg");

    // 2, HTMLからキャンバスを取得し画像を描画
    canvas = document.getElementById("myCanvas");
    canvas.width = img.width;
    canvas.height = img.height;
    context = canvas.getContext("2d");
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.drawImage(img, 0, 0);// 画像の描画

    // 3, 顔認識の実行と認識結果の取得
    const iSize = { width: img.width, height: img.height };
    const fData = await faceapi.detectAllFaces(img).withFaceLandmarks();

    // 4, 認識結果のリサイズ
    const rData = await faceapi.resizeResults(fData, iSize);
    rData.forEach(data => { drawResult(data); });
}

function drawResult(data) {
    console.log("drawResult!!");
    //console.log(data);

    const box = data.detection.box;// 長方形のデータ
    const mrks = data.landmarks.positions;

    context.fillStyle = "red";
    context.strokeStyle = "red";
    context.lineWidth = 4;
    context.strokeRect(box.x, box.y, box.width, box.height);// 長方形の描画
}