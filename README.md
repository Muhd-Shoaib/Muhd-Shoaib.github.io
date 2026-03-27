<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Camera AI Detector</title>
    <!-- Tailwind CSS for Styling -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- TensorFlow.js Core -->
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
    <!-- COCO-SSD Model (for animals/objects) -->
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd"></script>
    <!-- Face-API.js (for facial detection & recognition) -->
    <script src="https://cdn.jsdelivr.net/npm/@vladmandic/face-api/dist/face-api.js"></script>

    <style>
        body {
            background-color: #0f172a; /* Tailwind slate-900 */
            color: #f8fafc;
        }
        #feed-container {
            position: relative;
            width: 100%;
            max-width: 1000px;
            margin: 0 auto;
            background-color: #1e293b;
            border-radius: 0.5rem;
            overflow: hidden;
            min-height: 400px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        #camera-feed {
            width: 100%;
            height: auto;
            display: block;
            object-fit: contain;
            max-height: 70vh;
        }
        #overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none; /* Let clicks pass through to the image if needed */
        }
        .loader {
            border: 4px solid #334155;
            border-top: 4px solid #3b82f6;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body class="min-h-screen p-4 md:p-8 flex flex-col font-sans">

    <div class="max-w-5xl mx-auto w-full flex-grow flex flex-col gap-6">
        
        <!-- Header -->
        <header class="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-slate-700 pb-4">
            <div>
                <h1 class="text-3xl font-bold text-white flex items-center gap-3">
                    <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" fill="currentColor" viewBox="0 0 256 256"><path d="M248,64V192a16,16,0,0,1-16,16H24A16,16,0,0,1,8,192V64A16,16,0,0,1,24,48H232A16,16,0,0,1,248,64Zm-16,0H24V192H232V64Zm-56,64a48,48,0,1,1-48-48A48.05,48.05,0,0,1,176,128Zm-16,0a32,32,0,1,0-32,32A32,32,0,0,0,160,128Z"></path></svg>
                    AI Vision Hub
                </h1>
                <p class="text-slate-400 mt-1">Live Facial Recognition & Animal Tracking via Web Browser</p>
            </div>
            
            <div id="status-badge" class="px-4 py-2 rounded-full bg-slate-800 text-slate-300 border border-slate-600 flex items-center gap-2 text-sm font-medium">
                <div id="status-dot" class="w-2.5 h-2.5 rounded-full bg-yellow-500 animate-pulse"></div>
                <span id="status-text">Loading AI Models...</span>
            </div>
        </header>

        <!-- Controls -->
        <div class="bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-lg">
            <div class="flex flex-col md:flex-row gap-4">
                <div class="flex-grow">
                    <label for="stream-url" class="block text-sm font-medium text-slate-300 mb-2">Camera Stream or Snapshot URL</label>
                    <input type="text" id="stream-url" placeholder="http://192.168.1.100:8080/video" 
                        class="w-full bg-slate-900 border border-slate-600 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-colors">
                </div>
                <div class="flex items-end gap-2">
                    <button id="btn-connect" disabled class="bg-blue-600 hover:bg-blue-700 disabled:bg-slate-600 disabled:cursor-not-allowed text-white px-6 py-3 rounded-lg font-medium transition-colors shadow-lg shadow-blue-900/20">
                        Connect & Detect
                    </button>
                    <button id="btn-demo" disabled class="bg-slate-700 hover:bg-slate-600 disabled:bg-slate-800 disabled:cursor-not-allowed text-white px-6 py-3 rounded-lg font-medium transition-colors">
                        Test Demo Image
                    </button>
                </div>
            </div>

            <!-- Refresh Mode Toggle -->
            <div class="mt-4 p-3 bg-slate-750 rounded-lg border border-slate-600">
                <label class="flex items-center gap-3 text-sm text-slate-300 cursor-pointer w-fit">
                    <input type="checkbox" id="refresh-mode" class="w-4 h-4 rounded bg-slate-900 border-slate-600 text-blue-500 focus:ring-blue-500">
                    <div>
                        <strong class="text-white block mb-0.5">Auto-Refresh Snapshot Mode</strong>
                        <span class="text-slate-400 text-xs">Enable this if your URL is a static image (.jpg) and not a continuous video stream.</span>
                    </div>
                </label>
            </div>
            
            <!-- Messages / Alerts -->
            <div id="message-box" class="mt-4 hidden p-4 rounded-lg bg-red-900/30 border border-red-800 text-red-200 text-sm">
                <!-- Error messages will appear here -->
            </div>

            <!-- Face Recognition Training -->
            <div class="mt-4 p-4 bg-slate-750 rounded-lg border border-slate-600">
                <h3 class="text-white font-medium mb-2 flex items-center gap-2">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="currentColor" viewBox="0 0 256 256"><path d="M128,24A104,104,0,1,0,232,128,104.11,104.11,0,0,0,128,24ZM74.08,197.5a64,64,0,0,1,107.84,0,87.83,87.83,0,0,1-107.84,0ZM96,120a32,32,0,1,1,32,32A32,32,0,0,1,96,120Zm97.76,66.41a79.66,79.66,0,0,0-36.06-28.75,48,48,0,1,0-59.4,0,79.66,79.66,0,0,0-36.06,28.75,88,88,0,1,1,131.52,0Z"></path></svg>
                    Face Recognition Training
                </h3>
                <p class="text-xs text-slate-400 mb-3">Upload a clear photo of a person to teach the AI to recognize them.</p>
                <div class="flex flex-col md:flex-row gap-3 items-start md:items-center">
                    <input type="text" id="reg-name" placeholder="Person's Name" class="bg-slate-900 border border-slate-600 rounded px-3 py-2 text-white text-sm w-full md:w-48 focus:border-indigo-500 focus:outline-none">
                    <input type="file" id="reg-image" accept="image/*" class="text-sm text-slate-400 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-indigo-600 file:text-white hover:file:bg-indigo-700 cursor-pointer w-full md:w-auto">
                    <button id="btn-register" class="bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded text-sm font-medium transition-colors w-full md:w-auto shadow-lg shadow-indigo-900/20 whitespace-nowrap">Teach AI</button>
                </div>
                <p id="reg-status" class="text-xs mt-2 h-4 empty:hidden"></p>
                <div id="registered-list" class="flex flex-wrap gap-2 mt-2 empty:hidden">
                    <!-- Registered faces will appear here -->
                </div>
            </div>
        </div>

        <!-- Video/Image Container -->
        <div class="flex flex-col gap-4">
            <div id="feed-container" class="shadow-2xl shadow-black/50 border border-slate-700">
                <!-- Loading overlay for video -->
                <div id="feed-loader" class="absolute inset-0 flex flex-col items-center justify-center bg-slate-900 z-10 hidden">
                    <div class="loader mb-4"></div>
                    <span class="text-slate-400 font-medium">Connecting to stream...</span>
                </div>
                
                <!-- Default Placeholder -->
                <div id="placeholder" class="text-slate-500 flex flex-col items-center">
                    <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" fill="currentColor" viewBox="0 0 256 256" class="mb-2 opacity-50"><path d="M216,40H40A16,16,0,0,0,24,56V200a16,16,0,0,0,16,16H216a16,16,0,0,0,16-16V56A16,16,0,0,0,216,40Zm0,16V158.75l-26.07-26.06a16,16,0,0,0-22.63,0l-20,20-44-44a16,16,0,0,0-22.62,0L40,149.37V56ZM40,172l52-52,80,80H40Zm176,28H194.63l-36-36,20-20L216,181.38V200ZM144,96a24,24,0,1,1-24-24A24,24,0,0,1,144,96Z"></path></svg>
                    <p>Waiting for video feed...</p>
                </div>

                <!-- The crucial img tag for the MJPEG stream. crossorigin="anonymous" is required for TF.js to read the pixels -->
                <img id="camera-feed" crossorigin="anonymous" class="hidden" alt="Live Camera Feed">
                
                <!-- Canvas where we draw the bounding boxes -->
                <canvas id="overlay" class="hidden"></canvas>
            </div>

            <!-- Real-time Detection Result Text -->
            <div class="bg-slate-800/50 border border-slate-700 rounded-lg p-4 flex items-center justify-between">
                <div class="flex items-center gap-3">
                    <div class="bg-blue-500/20 p-2 rounded-lg">
                        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="#3b82f6" viewBox="0 0 256 256"><path d="M208,32H48A16,16,0,0,0,32,48V208a16,16,0,0,0,16,16H208a16,16,0,0,0,16-16V48A16,16,0,0,0,208,32Zm0,176H48V48H208V208ZM160,128a32,32,0,1,1-32-32A32,32,0,0,1,160,128Z"></path></svg>
                    </div>
                    <div>
                        <p class="text-xs font-semibold text-slate-500 uppercase tracking-wider">Detection Log</p>
                        <p id="detection-summary" class="text-slate-200 font-medium italic">Scanning for objects...</p>
                    </div>
                </div>
            </div>
        </div>
        
    </div>

    <script>
        // --- State and DOM Elements ---
        let cocoSsdModel = null;
        let isDetecting = false;
        let detectionTimeoutId = null;
        let currentStreamUrl = '';
        
        // Face Recognition State
        let faceMatcher = null;
        let labeledFaceDescriptors = [];

        const elements = {
            urlInput: document.getElementById('stream-url'),
            btnConnect: document.getElementById('btn-connect'),
            btnDemo: document.getElementById('btn-demo'),
            refreshMode: document.getElementById('refresh-mode'),
            feedContainer: document.getElementById('feed-container'),
            cameraFeed: document.getElementById('camera-feed'),
            overlay: document.getElementById('overlay'),
            ctx: document.getElementById('overlay').getContext('2d'),
            statusText: document.getElementById('status-text'),
            statusDot: document.getElementById('status-dot'),
            placeholder: document.getElementById('placeholder'),
            feedLoader: document.getElementById('feed-loader'),
            messageBox: document.getElementById('message-box'),
            detectionSummary: document.getElementById('detection-summary'),
            
            // Registration elements
            regName: document.getElementById('reg-name'),
            regImage: document.getElementById('reg-image'),
            btnRegister: document.getElementById('btn-register'),
            regStatus: document.getElementById('reg-status'),
            registeredList: document.getElementById('registered-list')
        };

        // --- Target Classes (What we want to detect) ---
        const TARGET_ANIMALS = [
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
            'elephant', 'bear', 'zebra', 'giraffe'
        ];

        // --- Initialization ---
        async function loadAIModels() {
            try {
                elements.statusText.textContent = 'Downloading Models (Wait)...';
                const FACE_MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/';

                // Load object detection and ALL required face-api models concurrently
                const results = await Promise.all([
                    cocoSsd.load(),
                    faceapi.nets.ssdMobilenetv1.loadFromUri(FACE_MODEL_URL),
                    faceapi.nets.faceLandmark68Net.loadFromUri(FACE_MODEL_URL),
                    faceapi.nets.faceRecognitionNet.loadFromUri(FACE_MODEL_URL)
                ]);

                cocoSsdModel = results[0];

                // Update UI
                elements.statusText.textContent = 'AI Ready';
                elements.statusDot.classList.replace('bg-yellow-500', 'bg-green-500');
                elements.statusDot.classList.remove('animate-pulse');
                
                // Enable buttons
                elements.btnConnect.disabled = false;
                elements.btnDemo.disabled = false;
            } catch (error) {
                console.error("Failed to load models:", error);
                elements.statusText.textContent = 'Model Load Failed';
                elements.statusDot.classList.replace('bg-yellow-500', 'bg-red-500');
                showMessage('Failed to download AI models. Check your internet connection.');
            }
        }

        // --- Face Training / Registration ---
        async function registerFace(imageElement, name) {
            // Extract the face descriptor (fingerprint) of the single most prominent face
            const detection = await faceapi.detectSingleFace(imageElement)
                                           .withFaceLandmarks()
                                           .withFaceDescriptor();
            if (!detection) {
                throw new Error('No face detected in the uploaded image. Try a clearer photo.');
            }
            
            // Save it
            const lfd = new faceapi.LabeledFaceDescriptors(name, [detection.descriptor]);
            labeledFaceDescriptors.push(lfd);
            
            // Re-initialize the global matcher (0.6 is a standard distance threshold)
            faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);
        }

        elements.btnRegister.addEventListener('click', async () => {
            const file = elements.regImage.files[0];
            const name = elements.regName.value.trim();
            
            if (!file || !name) {
                elements.regStatus.textContent = "Error: Provide both a name and an image.";
                elements.regStatus.className = "text-xs text-red-400 mt-2 h-4 block";
                return;
            }
            
            elements.regStatus.textContent = "Extracting facial features...";
            elements.regStatus.className = "text-xs text-yellow-400 mt-2 h-4 block animate-pulse";
            
            const img = new Image();
            img.src = URL.createObjectURL(file);
            img.onload = async () => {
                try {
                    await registerFace(img, name);
                    
                    elements.regStatus.textContent = `Success: AI learned "${name}"!`;
                    elements.regStatus.className = "text-xs text-green-400 mt-2 h-4 block";
                    
                    // Add badge to UI
                    const badge = document.createElement('span');
                    badge.className = "bg-indigo-900 text-indigo-200 text-xs px-2 py-1 rounded border border-indigo-700 flex items-center gap-1";
                    badge.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" fill="currentColor" viewBox="0 0 256 256"><path d="M229.66,77.66l-128,128a8,8,0,0,1-11.32,0l-56-56a8,8,0,0,1,11.32-11.32L96,188.69,218.34,66.34a8,8,0,0,1,11.32,11.32Z"></path></svg> ${name}`;
                    elements.registeredList.appendChild(badge);
                    
                    // Clear inputs
                    elements.regName.value = '';
                    elements.regImage.value = '';
                } catch (error) {
                    elements.regStatus.textContent = `Error: ${error.message}`;
                    elements.regStatus.className = "text-xs text-red-400 mt-2 h-4 block";
                }
            };
        });

        // --- Stream Handling ---
        function startStream(url) {
            stopDetection();
            hideMessage();
            currentStreamUrl = url;
            
            elements.placeholder.classList.add('hidden');
            elements.feedLoader.classList.remove('hidden');
            elements.cameraFeed.classList.add('hidden');
            elements.overlay.classList.add('hidden');

            // Set up image load listener
            elements.cameraFeed.onload = () => {
                elements.feedLoader.classList.add('hidden');
                elements.cameraFeed.classList.remove('hidden');
                elements.overlay.classList.remove('hidden');
                
                // Start detection loop once image is loaded/streaming
                if (!isDetecting) {
                    isDetecting = true;
                    detectFrame();
                }
            };

            elements.cameraFeed.onerror = () => {
                elements.feedLoader.classList.add('hidden');
                elements.placeholder.classList.remove('hidden');
                showMessage('Failed to load image/video stream. Check the URL and ensure the camera is reachable.');
            };

            // Trigger the first load
            triggerImageLoad();
        }

        function triggerImageLoad() {
            if (elements.refreshMode.checked) {
                // Add timestamp to bypass browser cache for snapshot URLs
                const separator = currentStreamUrl.includes('?') ? '&' : '?';
                elements.cameraFeed.src = `${currentStreamUrl}${separator}_t=${Date.now()}`;
            } else {
                elements.cameraFeed.src = currentStreamUrl;
            }
        }

        // --- Core Detection Loop ---
        async function detectFrame() {
            // Need both models loaded to run
            if (!isDetecting || !cocoSsdModel || !faceapi.nets.ssdMobilenetv1.isLoaded) return;

            // 1. Sync Canvas Size to the intrinsic size of the image stream
            const naturalWidth = elements.cameraFeed.naturalWidth;
            const naturalHeight = elements.cameraFeed.naturalHeight;
            
            if (naturalWidth && naturalHeight && 
               (elements.overlay.width !== naturalWidth || elements.overlay.height !== naturalHeight)) {
                elements.overlay.width = naturalWidth;
                elements.overlay.height = naturalHeight;
            }

            // 2. Clear previous frame drawings
            elements.ctx.clearRect(0, 0, elements.overlay.width, elements.overlay.height);
            
            // Keep track of what we find for the text summary
            let detectedItems = [];

            try {
                // Ensure image has dimensions before predicting
                if (naturalWidth > 0 && naturalHeight > 0) {
                    
                    // Run both models concurrently for better performance
                    const [facePredictions, objectPredictions] = await Promise.all([
                        faceapi.detectAllFaces(elements.cameraFeed).withFaceLandmarks().withFaceDescriptors(),
                        cocoSsdModel.detect(elements.cameraFeed)
                    ]);

                    // 3. Draw Faces & Recognize
                    facePredictions.forEach(fd => {
                        const box = fd.detection.box;
                        let label = 'Unknown Person';
                        let color = '#ef4444'; // Red for unknown
                        let displayScore = Math.round(fd.detection.score * 100);

                        if (faceMatcher) {
                            // Find the best match amongst registered faces
                            const bestMatch = faceMatcher.findBestMatch(fd.descriptor);
                            
                            if (bestMatch.label !== 'unknown') {
                                // Convert distance to a confidence percentage
                                const confidence = Math.max(0, Math.round((1 - bestMatch.distance) * 100));
                                label = `${bestMatch.label} (${confidence}%)`;
                                color = '#3b82f6'; // Blue for recognized
                                detectedItems.push(`Recognized: ${bestMatch.label}`);
                            } else {
                                label = `Unknown (${displayScore}%)`;
                                detectedItems.push(`Unknown Face`);
                            }
                        } else {
                            label = `Face (${displayScore}%)`; // Fallback if no training data is set
                            color = '#8b5cf6'; // Purple for generic face
                            detectedItems.push(`Face`);
                        }

                        drawBoundingBox(box.x, box.y, box.width, box.height, label, color, fd.detection.score);
                    });

                    // 4. Draw Animals and People (Body tracking from COCO-SSD)
                    objectPredictions.forEach(prediction => {
                        const [x, y, width, height] = prediction.bbox;
                        const className = prediction.class.toLowerCase();
                        
                        if (className === 'person') {
                            // Draw person body box
                            drawBoundingBox(x, y, width, height, `Person (${Math.round(prediction.score * 100)}%)`, '#10b981', prediction.score); // Green
                            detectedItems.push('Person');
                        } else if (TARGET_ANIMALS.includes(className)) {
                            drawBoundingBox(x, y, width, height, `${prediction.class} (${Math.round(prediction.score * 100)}%)`, '#f59e0b', prediction.score); // Orange/Yellow
                            detectedItems.push(prediction.class.charAt(0).toUpperCase() + prediction.class.slice(1));
                        }
                    });
                }
            } catch (error) {
                console.error("Detection Error:", error);
            }

            // Update the text summary
            if (detectedItems.length > 0) {
                // Remove duplicates and join
                const uniqueDetections = [...new Set(detectedItems)];
                elements.detectionSummary.textContent = `Detected: ${uniqueDetections.join(', ')}`;
                elements.detectionSummary.classList.remove('italic');
                elements.detectionSummary.classList.add('text-blue-400');
            } else {
                elements.detectionSummary.textContent = `No targets detected in current frame.`;
                elements.detectionSummary.classList.add('italic');
                elements.detectionSummary.classList.remove('text-blue-400');
            }

            // 5. Loop with Yielding
            if (isDetecting) {
                if (elements.refreshMode.checked) {
                    // Snapshot Mode
                    detectionTimeoutId = setTimeout(() => {
                        triggerImageLoad();
                        requestAnimationFrame(detectFrame);
                    }, 800);
                } else {
                    // MJPEG Mode
                    detectionTimeoutId = setTimeout(() => {
                        requestAnimationFrame(detectFrame);
                    }, 80); 
                }
            }
        }

        // --- Visuals Helper ---
        function drawBoundingBox(x, y, width, height, label, color, score) {
            const ctx = elements.ctx;
            if (score < 0.4) return;

            // Box
            ctx.beginPath();
            ctx.rect(x, y, width, height);
            ctx.lineWidth = 4;
            ctx.strokeStyle = color;
            ctx.stroke();

            // Label
            ctx.font = '16px Arial';
            const textWidth = ctx.measureText(label).width;
            ctx.fillStyle = color;
            ctx.fillRect(x, y - 24, textWidth + 16, 24);
            ctx.fillStyle = '#ffffff';
            ctx.fillText(label, x + 8, y - 6);
        }

        // --- Utility Functions ---
        function stopDetection() {
            isDetecting = false;
            if (detectionTimeoutId) {
                clearTimeout(detectionTimeoutId);
            }
            elements.ctx.clearRect(0, 0, elements.overlay.width, elements.overlay.height);
            elements.detectionSummary.textContent = "Scanner idle.";
        }

        function showMessage(msg) {
            elements.messageBox.innerHTML = `<strong>Attention:</strong> ${msg}`;
            elements.messageBox.classList.remove('hidden');
        }
        
        function hideMessage() {
            elements.messageBox.classList.add('hidden');
        }

        // --- Event Listeners ---
        elements.btnConnect.addEventListener('click', () => {
            const url = elements.urlInput.value.trim();
            if (url) {
                startStream(url);
            } else {
                showMessage("Please enter a valid camera URL.");
            }
        });

        // Demo mode
        elements.btnDemo.addEventListener('click', () => {
            elements.urlInput.value = ''; 
            elements.refreshMode.checked = false;
            const demoImageUrl = 'https://images.unsplash.com/photo-1548199973-03cce0bbc87b?q=80&w=1000&auto=format&fit=crop';
            startStream(demoImageUrl);
        });

        // --- Start Application ---
        loadAIModels();

    </script>
</body>
</html>
