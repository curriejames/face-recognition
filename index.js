function openCvReady() {
	function onloadCallback() {
		const utils = new Utils('errorMessage');
		let video = document.getElementById("cam_input"); // video is the id of video tag
		navigator.mediaDevices.getUserMedia({ video: true, audio: false })
		.then(function(stream) {
			video.srcObject = stream;
			video.play();
		})
		.catch(function(err) {
			console.log("An error occurred! " + err);
		});
		let classifier = new cv.CascadeClassifier();
		let faceCascadeFile = 'haarcascade_frontalface_default.xml'
		utils.createFileFromUrl(faceCascadeFile, faceCascadeFile, () => {
				// load pre-trained classifiers
				classifier.load(faceCascadeFile);
				const FPS = 30;
				function processVideo() {
						let src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
						let dst = new cv.Mat(video.height, video.width, cv.CV_8UC4);
						let gray = new cv.Mat();
						let cap = new cv.VideoCapture(video);
						let faces = new cv.RectVector();
						let begin = Date.now();
						// start processing.
						cap.read(src);
						src.copyTo(dst);
						cv.cvtColor(dst, gray, cv.COLOR_RGBA2GRAY, 0);
						// detect faces.
						try {
							classifier.detectMultiScale(gray, faces, 1.1, 3, 0);
						} catch (err) {
							utils.printError(err);
						}

						// draw faces.
						for (let i = 0; i < faces.size(); ++i) {
							let face = faces.get(i);
							let point1 = new cv.Point(face.x, face.y);
							let point2 = new cv.Point(face.x + face.width, face.y + face.height);
							cv.rectangle(dst, point1, point2, [255, 0, 0, 255]);
						}
						cv.imshow('canvas_output', dst);
						// schedule the next one.
						let delay = 1000/FPS - (Date.now() - begin);
						setTimeout(processVideo, delay);
				};

				// schedule the first one.
				setTimeout(processVideo, 0);
		});
	}
	if (cv.getBuildInformation)
	{
		console.log(cv.getBuildInformation());
		onloadCallback();
	}
	else
	{
		// WASM
		cv['onRuntimeInitialized']=()=>{
			console.log(cv.getBuildInformation());
			onloadCallback();
		}
	}
}