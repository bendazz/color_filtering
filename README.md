# 3D CNN Kernel Visualization

An interactive web-based visualization tool for teaching Convolutional Neural Networks (CNNs) and demonstrating how 3D kernels work on color images with RGB channels.

## Features

### 🎯 Interactive 3D Visualization
- **5×5×5 Image Cube**: Represents a color image with three distinct layers
  - Red layer (front)
  - Green layer (middle)  
  - Blue layer (back)
- **3×3×3 Kernel**: Yellow sliding window that demonstrates convolution operation
- **Real-time Movement**: Interactive controls to position the kernel anywhere within valid bounds

### 🎮 User Controls
- **Manual Positioning**: X, Y, Z sliders to manually position the kernel
- **Auto Movement**: Automated kernel traversal for demonstration
- **Visibility Toggles**: Show/hide kernel, switch between solid and wireframe modes
- **3D Navigation**: Mouse controls for rotating, zooming, and panning the view

### 📚 Educational Benefits
- Visualizes the relationship between 2D images and their 3D representation with color channels
- Demonstrates how CNN kernels slide across multi-dimensional input data
- Shows the spatial relationship between kernel size and valid convolution positions
- Highlights which input pixels are covered by the kernel at any position

## Usage

1. Open `index.html` in a modern web browser
2. Use the control panel to:
   - Move the kernel manually with the X, Y, Z sliders
   - Enable "Auto Move" to see automated kernel traversal
   - Toggle kernel visibility or wireframe mode
   - Reset kernel to center position
3. Use mouse to rotate and zoom the 3D scene
4. Observe how the kernel covers different parts of the RGB channels

## Technical Implementation

- **Three.js**: 3D graphics rendering and scene management
- **OrbitControls**: Interactive camera controls
- **WebGL**: Hardware-accelerated 3D rendering
- **Responsive Design**: Adapts to different screen sizes

## Educational Context

This visualization helps students understand:
- How color images are represented as 3D tensors (width × height × channels)
- The concept of 3D convolution operations in CNNs
- Why kernel size affects the output dimensions
- The relationship between input channels and convolution operations

Perfect for computer vision courses, deep learning workshops, and CNN tutorials!

## Browser Requirements

- Modern browser with WebGL support
- Recommended: Chrome, Firefox, Safari, or Edge (latest versions)