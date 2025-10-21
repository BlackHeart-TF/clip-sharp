# ClipSharp

A .NET implementation of CLIP (Contrastive Language-Image Pre-training) with support for multiple numeric types and image processing backends.

Based on https://huggingface.co/mlunar/clip-variants and https://github.com/dansav/clip-sharp

## Features

- **Multiple Numeric Types**: Support for `float`, `Half`, `sbyte`, and other numeric types via generics
- **Dual Image Processing Backends**: Choose between SkiaSharp or ImageSharp for image processing
- **Cross-Platform**: Runs on Windows, Linux, macOS (untested), and Android
- **GPU Acceleration**: Automatic CUDA detection and fallback to CPU
- **Self-Contained**: Each backend produces a single DLL with all dependencies included

## Installation

Choose one of the available packages based on your image processing preference:

### SkiaSharp Backend (Recommended)
```xml
<PackageReference Include="ClipSharp.SkiaSharp" Version="1.0.0" />
```

### ImageSharp Backend
```xml
<PackageReference Include="ClipSharp.ImageSharp" Version="1.0.0" />
```

## Usage

### Basic Usage

```csharp
using ClipSharp;

// Load models (automatic processor detection)
var visualModel = VisualModel<float>.Load("clip-vit-base-patch16-visual-float32.onnx");
var textualModel = TextualModel<float>.Load("clip-vit-base-patch16-textual-float32.onnx");

// Encode images
var imageEmbeddings = visualModel.Encode(new[] { "path/to/image.jpg" });

// Encode text
var textEmbeddings = textualModel.Encode(new[] { "a photo of a cat" });

// Dispose when done
visualModel.Dispose();
textualModel.Dispose();
```

### Using Different Numeric Types

```csharp
// Half precision (smaller memory footprint)
var visualModel = VisualModel<Half>.Load("clip-vit-base-patch16-visual-float16.onnx");

// Int8 quantization (fastest inference)
var visualModel = VisualModel<sbyte>.Load("clip-vit-base-patch16-visual-int8.onnx");
```

## Architecture

### Dual Backend System

ClipSharp provides two self-contained packages:

- **`ClipSharp.SkiaSharp.dll`**: Core functionality + SkiaSharp image processing
- **`ClipSharp.ImageSharp.dll`**: Core functionality + ImageSharp image processing

Choose the one that best suits your application.

### Type Safety

All models are generic over numeric types:
- `VisualModel<T>` where `T : struct, INumber<T>`
- `TextualModel<T>` where `T : struct, INumber<T>`
- `IImageProcessor<T>` where `T : struct, INumber<T>`

## Dependencies

### Core Dependencies
- Microsoft.ML.OnnxRuntime.Gpu

### Backend-Specific Dependencies
- SkiaSharp
- SixLabors.ImageSharp

## Platform Support

- **Windows**: CPU + CUDA + DirectML
- **Linux**: CPU + CUDA
- **macOS**: CPU + CoreML (via ONNX Runtime)
- **Android**: CPU + NNAPI

## Model Compatibility

Tested with vit-base-patch16 models from https://huggingface.co/mlunar/clip-variants:
- Float32 models: Use with `VisualModel<float>` and `TextualModel<float>`
- Float16 models: Use with `VisualModel<Half>` and `TextualModel<Half>`
- Int8 models: Use with `VisualModel<sbyte>` and `TextualModel<sbyte>` 
