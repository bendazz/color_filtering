// CNN 3D Kernel Visualization - Three Layer Version with Numbers
class CNNVisualization {
    constructor() {
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        
        // Image layers properties
        this.imageLayers = []; // Array of 3 input layers (red, green, blue)
        this.outputLayer = []; // Single output layer (grayscale)
        this.allCubes = []; // All cubes across input layers
        this.allOutputCubes = []; // All cubes in output layer
        this.originalMaterials = []; // Store original materials for input layers
        this.originalOutputMaterials = []; // Store original materials for output layer
        this.cubeSize = 0.8;
        this.spacing = 1.0;
        this.layerSize = 8; // 8x8 per input layer
        this.layerSpacing = 3.0; // Distance between input layers
        this.outputSize = 6; // 6x6 output (8-3+1 = 6 for valid convolution)
        
        // Kernel properties
        this.kernelSize = 3; // 3x3 kernel (no Z movement)
        this.kernelPosition = { x: 2, y: 2 };
        
        // Animation
        this.autoMove = false;
        this.animationSpeed = 0.02;
        this.animationTime = 0;
        
        // Visibility toggles
        this.kernelHighlightVisible = true;
        
        // Numerical data for calculations
        this.inputValues = []; // 3D array: [layer][x][y] = integer 0-9
        this.kernelWeights = []; // 3D array: [layer][x][y] = float 0.0-1.0
        this.outputValues = []; // 2D array: [x][y] = calculated result
        
        // Materials
        this.highlightMaterial = new THREE.MeshLambertMaterial({ 
            color: 0xffff00, 
            transparent: false, 
            opacity: 1.0 
        });
        
        this.init();
        this.setupControls();
        this.animate();
    }
    
    initializeNumericalData() {
        // Initialize input values (0-9 integers)
        this.inputValues = [];
        for (let layer = 0; layer < 3; layer++) {
            this.inputValues[layer] = [];
            for (let x = 0; x < this.layerSize; x++) {
                this.inputValues[layer][x] = [];
                for (let y = 0; y < this.layerSize; y++) {
                    this.inputValues[layer][x][y] = Math.floor(Math.random() * 10);
                }
            }
        }
        
        // Initialize kernel weights (0.0-1.0 with 1 decimal place)
        this.kernelWeights = [];
        for (let layer = 0; layer < 3; layer++) {
            this.kernelWeights[layer] = [];
            for (let x = 0; x < this.kernelSize; x++) {
                this.kernelWeights[layer][x] = [];
                for (let y = 0; y < this.kernelSize; y++) {
                    this.kernelWeights[layer][x][y] = Math.round(Math.random() * 10) / 10;
                }
            }
        }
        
        // Initialize output values
        this.outputValues = [];
        for (let x = 0; x < this.outputSize; x++) {
            this.outputValues[x] = [];
            for (let y = 0; y < this.outputSize; y++) {
                this.outputValues[x][y] = 0;
            }
        }
        
        this.create3DKernel();
        this.calculateAllOutputValues();
    }
    
    calculateAllOutputValues() {
        // Calculate convolution results for all valid kernel positions
        for (let kernelX = 0; kernelX < this.outputSize; kernelX++) {
            for (let kernelY = 0; kernelY < this.outputSize; kernelY++) {
                let sum = 0;
                
                // Calculate convolution for this kernel position
                for (let layer = 0; layer < 3; layer++) {
                    for (let kx = 0; kx < this.kernelSize; kx++) {
                        for (let ky = 0; ky < this.kernelSize; ky++) {
                            const inputX = kernelX + kx;
                            const inputY = kernelY + ky;
                            
                            if (inputX < this.layerSize && inputY < this.layerSize) {
                                const inputVal = this.inputValues[layer][inputX][inputY];
                                const kernelVal = this.kernelWeights[layer][kx][ky];
                                sum += inputVal * kernelVal;
                            }
                        }
                    }
                }
                
                // Store the result
                this.outputValues[kernelX][kernelY] = sum;
            }
        }
        
        // Update the current calculation display for the initial kernel position
        this.calculateCurrentOutput();
    }
    
    create3DKernel() {
        // Create a 3D kernel visualization to the left of the input layers
        this.kernelGroup = new THREE.Group();
        this.kernelCubes = [];
        
        const kernelCubeSize = 1.2; // Larger cubes for better readability
        const kernelSpacing = 1.4;  // Spacing within each layer
        const layerSpacing = 2.5;   // Increased spacing between layers to see all faces
        
        for (let layer = 0; layer < 3; layer++) {
            this.kernelCubes[layer] = [];
            for (let x = 0; x < this.kernelSize; x++) {
                this.kernelCubes[layer][x] = [];
                for (let y = 0; y < this.kernelSize; y++) {
                    // Create cube geometry
                    const geometry = new THREE.BoxGeometry(kernelCubeSize, kernelCubeSize, kernelCubeSize);
                    
                    // Create material - yellow for kernel visualization
                    const material = new THREE.MeshLambertMaterial({ 
                        color: 0xffdd44,
                        transparent: true,
                        opacity: 0.9
                    });
                    
                    // Create cube
                    const cube = new THREE.Mesh(geometry, material);
                    
                    // Position the cube with increased layer spacing
                    cube.position.set(
                        (x - 1) * kernelSpacing,
                        (y - 1) * kernelSpacing,
                        layer * layerSpacing
                    );
                    
                    // Add only edge wireframe (no diagonal lines)
                    const edges = new THREE.EdgesGeometry(geometry);
                    const edgeMaterial = new THREE.LineBasicMaterial({ 
                        color: 0x000000,
                        linewidth: 2 
                    });
                    const wireframe = new THREE.LineSegments(edges, edgeMaterial);
                    cube.add(wireframe);
                    
                    // Add text showing the kernel weight with larger font
                    const text = this.createTextMesh(this.kernelWeights[layer][x][y].toFixed(1), 0.8);
                    text.position.set(0, 0, kernelCubeSize/2 + 0.1);
                    cube.add(text);
                    
                    this.kernelCubes[layer][x][y] = cube;
                    this.kernelGroup.add(cube);
                }
            }
        }
        
        // Position the entire kernel group closer to the input layers
        this.kernelGroup.position.set(-8, 0, 0);
        this.scene.add(this.kernelGroup);
    }
    
    calculateCurrentOutput() {
        const kernelX = this.kernelPosition.x;
        const kernelY = this.kernelPosition.y;
        
        let sum = 0;
        let steps = [];
        
        // Calculate convolution for current kernel position
        for (let layer = 0; layer < 3; layer++) {
            const layerName = ['Red', 'Green', 'Blue'][layer];
            steps.push(`${layerName} Channel:`);
            
            for (let kx = 0; kx < this.kernelSize; kx++) {
                for (let ky = 0; ky < this.kernelSize; ky++) {
                    const inputX = kernelX + kx;
                    const inputY = kernelY + ky;
                    
                    if (inputX < this.layerSize && inputY < this.layerSize) {
                        const inputVal = this.inputValues[layer][inputX][inputY];
                        const kernelVal = this.kernelWeights[layer][kx][ky];
                        const product = inputVal * kernelVal;
                        sum += product;
                        
                        steps.push(`  ${inputVal} × ${kernelVal.toFixed(1)} = ${product.toFixed(1)}`);
                    }
                }
            }
            steps.push('');
        }
        
        // Store the result
        this.outputValues[kernelX][kernelY] = sum;
        
        // Update the calculation display
        this.updateCalculationDisplay(steps, sum);
    }
    
    updateCalculationDisplay(steps, result) {
        const stepsElement = document.getElementById('calculation-steps');
        const resultElement = document.getElementById('calculation-result');
        
        if (stepsElement) {
            stepsElement.innerHTML = steps.map(step => 
                `<div class="calculation-step">${step}</div>`
            ).join('');
        }
        
        if (resultElement) {
            resultElement.textContent = `Result: ${result.toFixed(1)}`;
        }
    }
    
    createTextTexture(text, fontSize = 128, color = '#000000') {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        // Use higher resolution for better text quality, especially for smaller text
        canvas.width = 256;
        canvas.height = 256;
        
        context.fillStyle = 'rgba(255, 255, 255, 0)'; // Transparent background
        context.fillRect(0, 0, canvas.width, canvas.height);
        
        context.fillStyle = color;
        context.font = `bold ${fontSize}px Arial`;
        context.textAlign = 'center';
        context.textBaseline = 'middle';
        context.fillText(text, canvas.width / 2, canvas.height / 2);
        
        const texture = new THREE.CanvasTexture(canvas);
        texture.needsUpdate = true;
        return texture;
    }
    
    addTextToCube(cube, text, scale = 1.0) {
        const textTexture = this.createTextTexture(text.toString());
        const textMaterial = new THREE.MeshBasicMaterial({
            map: textTexture,
            transparent: true,
            depthWrite: false
        });
        
        // Create a small plane for the text with adjustable scale
        const textGeometry = new THREE.PlaneGeometry(0.6 * scale, 0.6 * scale);
        const textMesh = new THREE.Mesh(textGeometry, textMaterial);
        
        // Position the text slightly in front of the cube face
        textMesh.position.z = 0.41;
        cube.add(textMesh);
        
        // Store reference to update later
        cube.textMesh = textMesh;
        cube.textMaterial = textMaterial;
        
        return textMesh;
    }
    
    createTextMesh(text, scale = 1.0) {
        const textTexture = this.createTextTexture(text.toString());
        const textMaterial = new THREE.MeshBasicMaterial({
            map: textTexture,
            transparent: true,
            depthWrite: false
        });
        
        // Create a small plane for the text
        const textGeometry = new THREE.PlaneGeometry(0.6 * scale, 0.6 * scale);
        const textMesh = new THREE.Mesh(textGeometry, textMaterial);
        
        return textMesh;
    }
    
    init() {
        // Create scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x1a1a2e);
        
        // Create camera with closer initial position for better zoom
        this.camera = new THREE.PerspectiveCamera(
            75,
            window.innerWidth / window.innerHeight,
            0.1,
            1000
        );
        this.camera.position.set(10, 10, 10); // Moved closer from (15, 15, 15)
        
        // Create renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.shadowMap.enabled = false; // Disable shadows for uniform lighting
        document.getElementById('container').appendChild(this.renderer.domElement);
        
        // Create controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.maxDistance = 50;
        this.controls.minDistance = 5;
        
        // Add lights
        this.setupLights();
        
        // Initialize numerical data first
        this.initializeNumericalData();
        
        // Create the three image layers (8x8 each)
        this.createImageLayers();
        
        // Create the output layer (6x6 grayscale)
        this.createOutputLayer();
        
        // Initial kernel highlight
        this.updateKernelHighlight();
        
        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());
    }
    
    setupLights() {
        // High ambient light for uniform illumination from all directions
        const ambientLight = new THREE.AmbientLight(0xffffff, 1.0);
        this.scene.add(ambientLight);
        
        // Add a very subtle directional light just for basic depth perception
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.2);
        directionalLight.position.set(1, 1, 1);
        directionalLight.castShadow = false; // No shadows
        this.scene.add(directionalLight);
    }
    
    createImageLayers() {
        const geometry = new THREE.BoxGeometry(this.cubeSize, this.cubeSize, this.cubeSize);
        
        // Colors for each layer
        const layerColors = [
            0xff4444, // Red layer
            0x44ff44, // Green layer  
            0x4444ff  // Blue layer
        ];
        
        // Create 3 layers, each 8x8
        for (let layer = 0; layer < 3; layer++) {
            const layerGroup = new THREE.Group();
            const layerCubes = [];
            
            const material = new THREE.MeshLambertMaterial({
                color: layerColors[layer],
                transparent: false,
                opacity: 1.0
            });
            
            // Create 8x8 grid for this layer
            for (let x = 0; x < this.layerSize; x++) {
                for (let y = 0; y < this.layerSize; y++) {
                    const cube = new THREE.Mesh(geometry, material.clone());
                    
                    // Add wireframe edges for better definition
                    const edges = new THREE.EdgesGeometry(geometry);
                    const edgeMaterial = new THREE.LineBasicMaterial({ 
                        color: 0x000000, // Black edges
                        linewidth: 2 
                    });
                    const wireframe = new THREE.LineSegments(edges, edgeMaterial);
                    cube.add(wireframe);
                    
                    // Add number text to the cube
                    const value = this.inputValues[layer][x][y];
                    this.addTextToCube(cube, value);
                    
                    // Position the cube in the layer
                    cube.position.set(
                        (x - 3.5) * this.spacing,
                        (y - 3.5) * this.spacing,
                        0
                    );
                    
                    // Store references
                    cube.layerIndex = layer;
                    cube.gridX = x;
                    cube.gridY = y;
                    
                    layerGroup.add(cube);
                    layerCubes.push(cube);
                    this.allCubes.push(cube);
                    this.originalMaterials.push(cube.material.clone());
                }
            }
            
            // Position the entire layer
            layerGroup.position.z = (layer - 1) * this.layerSpacing;
            
            this.imageLayers.push(layerCubes);
            this.scene.add(layerGroup);
        }
    }
    
    createOutputLayer() {
        const geometry = new THREE.BoxGeometry(this.cubeSize, this.cubeSize, this.cubeSize);
        const outputGroup = new THREE.Group();
        
        // Gray material for output layer
        const material = new THREE.MeshLambertMaterial({
            color: 0x888888, // Gray
            transparent: false,
            opacity: 1.0
        });
        
        // Create 6x6 grid for output layer (result of valid convolution: 8-3+1=6)
        for (let x = 0; x < this.outputSize; x++) {
            for (let y = 0; y < this.outputSize; y++) {
                const cube = new THREE.Mesh(geometry, material.clone());
                
                // Add wireframe edges for better definition
                const edges = new THREE.EdgesGeometry(geometry);
                const edgeMaterial = new THREE.LineBasicMaterial({ 
                    color: 0x000000, // Black edges
                    linewidth: 2 
                });
                const wireframe = new THREE.LineSegments(edges, edgeMaterial);
                cube.add(wireframe);
                
                // Add calculated result text with larger font size for better visibility
                const calculatedValue = this.outputValues[x][y];
                this.addTextToCube(cube, calculatedValue.toFixed(1), 0.7);
                
                // Position the cube in the output layer
                cube.position.set(
                    (x - 2.5) * this.spacing, // Center the 6x6 grid
                    (y - 2.5) * this.spacing,
                    0
                );
                
                // Store references
                cube.outputX = x;
                cube.outputY = y;
                
                outputGroup.add(cube);
                this.outputLayer.push(cube);
                this.allOutputCubes.push(cube);
                this.originalOutputMaterials.push(cube.material.clone());
            }
        }
        
        // Position the output layer closer to the input layers
        outputGroup.position.set(8, 0, 0); // Move closer from 12 to 8
        
        this.scene.add(outputGroup);
    }
    
    updateKernelHighlight() {
        if (!this.kernelHighlightVisible) return;
        
        // Reset all input layer cubes to original materials
        this.allCubes.forEach((cube, index) => {
            cube.material = this.originalMaterials[index].clone();
        });
        
        // Reset output layer cubes to original materials
        this.allOutputCubes.forEach((cube, index) => {
            cube.material = this.originalOutputMaterials[index].clone();
        });
        
        // Highlight cubes covered by kernel (3x3 area across all 3 layers)
        const startX = this.kernelPosition.x;
        const startY = this.kernelPosition.y;
        
        // Apply to all 3 input layers simultaneously
        for (let layer = 0; layer < 3; layer++) {
            const layerCubes = this.imageLayers[layer];
            
            // Highlight 3x3 area in this layer
            for (let x = 0; x < this.kernelSize; x++) {
                for (let y = 0; y < this.kernelSize; y++) {
                    const cubeX = startX + x;
                    const cubeY = startY + y;
                    
                    if (cubeX >= 0 && cubeX < this.layerSize &&
                        cubeY >= 0 && cubeY < this.layerSize) {
                        
                        const cubeIndex = cubeX * this.layerSize + cubeY;
                        if (layerCubes[cubeIndex]) {
                            layerCubes[cubeIndex].material = this.highlightMaterial.clone();
                        }
                    }
                }
            }
        }
        
        // Highlight corresponding output position
        const outputIndex = startX * this.outputSize + startY;
        if (this.outputLayer[outputIndex]) {
            this.outputLayer[outputIndex].material = this.highlightMaterial.clone();
        }
        
        // Update explanation text and calculation
        this.updateExplanationText();
        this.calculateCurrentOutput();
        this.updateOutputCubeText();
    }
    
    updateOutputCubeText() {
        const kernelX = this.kernelPosition.x;
        const kernelY = this.kernelPosition.y;
        const cubeIndex = kernelX * this.outputSize + kernelY;
        const cube = this.outputLayer[cubeIndex];
        
        if (cube && cube.textMesh) {
            const value = this.outputValues[kernelX][kernelY];
            const newTexture = this.createTextTexture(value.toFixed(1));
            cube.textMaterial.map = newTexture;
            cube.textMaterial.needsUpdate = true;
        }
    }
    
    updateExplanationText() {
        const currentOpElement = document.getElementById('currentOperation');
        if (currentOpElement) {
            const x = this.kernelPosition.x;
            const y = this.kernelPosition.y;
            const inputValues = 27; // 3x3x3 kernel covers 27 input values
            
            currentOpElement.textContent = 
                `Position (${x},${y}): Processing ${inputValues} input values (9 from each RGB layer) to produce 1 output value.`;
        }
    }
    
    setupControls() {
        // Kernel position sliders
        const kernelXSlider = document.getElementById('kernelX');
        const kernelYSlider = document.getElementById('kernelY');
        
        const kernelXValue = document.getElementById('kernelXValue');
        const kernelYValue = document.getElementById('kernelYValue');
        
        kernelXSlider.addEventListener('input', (e) => {
            this.kernelPosition.x = parseInt(e.target.value);
            kernelXValue.textContent = e.target.value;
            this.updateKernelHighlight();
        });
        
        kernelYSlider.addEventListener('input', (e) => {
            this.kernelPosition.y = parseInt(e.target.value);
            kernelYValue.textContent = e.target.value;
            this.updateKernelHighlight();
        });
        
        // Auto move button
        const autoMoveBtn = document.getElementById('autoMoveBtn');
        autoMoveBtn.addEventListener('click', () => {
            this.autoMove = !this.autoMove;
            autoMoveBtn.textContent = this.autoMove ? 'Stop Auto Move' : 'Auto Move';
            autoMoveBtn.classList.toggle('paused', this.autoMove);
        });
        
        // Reset button
        const resetBtn = document.getElementById('resetBtn');
        resetBtn.addEventListener('click', () => {
            this.kernelPosition = { x: 2, y: 2 };
            kernelXSlider.value = 2;
            kernelYSlider.value = 2;
            kernelXValue.textContent = '2';
            kernelYValue.textContent = '2';
            this.updateKernelHighlight();
        });
        
        // Kernel highlight toggle
        const kernelToggle = document.getElementById('kernelToggle');
        kernelToggle.addEventListener('click', () => {
            this.kernelHighlightVisible = !this.kernelHighlightVisible;
            if (this.kernelHighlightVisible) {
                this.updateKernelHighlight();
                kernelToggle.textContent = 'Hide Kernel Area';
            } else {
                // Reset all cubes to original materials
                this.allCubes.forEach((cube, index) => {
                    cube.material = this.originalMaterials[index].clone();
                });
                this.allOutputCubes.forEach((cube, index) => {
                    cube.material = this.originalOutputMaterials[index].clone();
                });
                kernelToggle.textContent = 'Show Kernel Area';
            }
        });
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        
        // Auto movement animation
        if (this.autoMove) {
            this.animationTime += this.animationSpeed;
            
            // Create a smooth path through all valid kernel positions (6x6 grid)
            const totalPositions = 36; // 6x6 positions
            const cycle = Math.floor(this.animationTime) % totalPositions;
            
            // Convert linear cycle to x,y coordinates
            this.kernelPosition.x = cycle % 6;
            this.kernelPosition.y = Math.floor(cycle / 6);
            
            // Update sliders to reflect automatic movement
            document.getElementById('kernelX').value = this.kernelPosition.x;
            document.getElementById('kernelY').value = this.kernelPosition.y;
            document.getElementById('kernelXValue').textContent = this.kernelPosition.x;
            document.getElementById('kernelYValue').textContent = this.kernelPosition.y;
            
            this.updateKernelHighlight();
        }
        
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
    
    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }
}

// Initialize the visualization when the page loads
window.addEventListener('load', () => {
    new CNNVisualization();
});
