using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using System.Runtime.InteropServices;

namespace ClipSharp;

public class VisualModel
{
    private readonly InferenceSession _session;
    private readonly int _inputSize;
    private readonly string _inputName;
    private readonly string _outputName;

    public static VisualModel Load(string modelPath)
    {
        var options = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            // EnableMemoryPattern = false,
            // LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING,

        };

        // Always use CPU execution provider
        options.AppendExecutionProvider_CPU();
        
        // Only try CUDA on desktop platforms (not mobile)
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Create("Android")) && 
            !RuntimeInformation.IsOSPlatform(OSPlatform.Create("iOS")) &&
            !RuntimeInformation.IsOSPlatform(OSPlatform.Create("Tizen")))
        {
            try
            {
                options.AppendExecutionProvider_CUDA();
            }
            catch
            {
                // CUDA not available, continue with CPU only
            }
        }
        // options.RegisterOrtExtensions();

        var session = new InferenceSession(modelPath, options);

        return new VisualModel(session);
    }

    public VisualModel(InferenceSession session)
    {
        _session = session;

        var input = _session.InputMetadata.First();
        if (input.Value.Dimensions.Length != 4 || input.Value.Dimensions[2] != input.Value.Dimensions[3])
        {
            throw new ArgumentException($"Unexpected input dimensions (expected height and width to be equal)");
        }

        _inputSize = input.Value.Dimensions[2];
        _inputName = input.Key;

        var output = _session.OutputMetadata.First();
        _outputName = output.Key;
    }


    public IReadOnlyCollection<float[]> Encode(string[] images)
    {
        var imgFs = images.Select(img => ImageToVector(img)).ToArray();

        Memory<float> tokens = imgFs.SelectMany(l => l.ToFlatArray()).ToArray();
        var inputTensor = new DenseTensor<float>(tokens, new[] { images.Length, 3, 224, 224 });

        using var results = _session.Run(new[] { NamedOnnxValue.CreateFromTensor(_inputName, inputTensor) });

        using var result = results.First();
        var embeddings = (DenseTensor<float>)result.Value;

        var output = new float[embeddings.Dimensions[0]][];
        for (int i = 0; i < embeddings.Dimensions[0]; i++)
        {
            output[i] = new float[embeddings.Dimensions[1]];
            for (int j = 0; j < embeddings.Dimensions[1]; j++)
            {
                output[i][j] = embeddings[i, j];
            }
        }

        return output;
    }

    private static float[,,] ImageToVector(string imagePath)
    {
        using var originalBitmap = SKBitmap.Decode(imagePath);
        if (originalBitmap == null)
        {
            throw new InvalidOperationException("Could not decode image data");
        }
        
        // Resize to 224x224
        using var resizedBitmap = originalBitmap.Resize(new SKImageInfo(224, 224), SKFilterQuality.High);
        
        // Convert to normalized CHW format
        return ToNormalizedColorHeightWidthArray(resizedBitmap);
    }

    public static float[,,] ToNormalizedColorHeightWidthArray(SKBitmap bitmap)
    {
        var img = new float[3, bitmap.Height, bitmap.Width];

        for (var y = 0; y < bitmap.Height; y++)
        {
            for (var x = 0; x < bitmap.Width; x++)
            {
                var pixel = bitmap.GetPixel(x, y);
                
                // CLIP normalization (ImageNet stats)
                img[0, y, x] = (pixel.Red / 255f - 0.48145466f) / 0.26862954f;   // Red
                img[1, y, x] = (pixel.Green / 255f - 0.4578275f) / 0.26130258f; // Green  
                img[2, y, x] = (pixel.Blue / 255f - 0.40821073f) / 0.27577711f;  // Blue
            }
        }

        return img;
    }
}
