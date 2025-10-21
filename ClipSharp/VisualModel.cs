using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Numerics;
using System.Runtime.InteropServices;
#if SKIASHARP
using ClipSharp.SkiaSharp;
#elif IMAGESHARP
using ClipSharp.ImageSharp;
#endif

namespace ClipSharp;

public class VisualModel<T> : IDisposable where T : struct, INumber<T>
{
    private readonly InferenceSession _session;
    private readonly IImageProcessor<T> _imageProcessor;
    private readonly int _inputSize;
    private readonly string _inputName;
    private readonly string _outputName;

    public static VisualModel<T> Load(string modelPath, IImageProcessor<T> imageProcessor)
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
                // Try CUDA with explicit device ID
                options.AppendExecutionProvider_CUDA(0);
                Console.WriteLine("CUDA provider enabled");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"CUDA not available: {ex.Message}");
            }
        }
        // options.RegisterOrtExtensions();

        var session = new InferenceSession(modelPath, options);

        return new VisualModel<T>(session, imageProcessor);
    }

    /// <summary>
    /// Creates a VisualModel with automatic processor selection based on available libraries
    /// </summary>
    public static VisualModel<T> Load(string modelPath)
    {
        IImageProcessor<T> processor = CreateDefaultProcessor<T>();
        
        var options = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
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
                // Try CUDA with explicit device ID
                options.AppendExecutionProvider_CUDA(0);
                Console.WriteLine("CUDA provider enabled");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"CUDA not available: {ex.Message}");
            }
        }

        var session = new InferenceSession(modelPath, options);

        return new VisualModel<T>(session, processor);
    }

    private static IImageProcessor<T> CreateDefaultProcessor<T>() where T : struct, INumber<T>
    {
#if SKIASHARP
        return new SkiaSharpImageProcessor<T>();
#elif IMAGESHARP
        return new ImageSharpImageProcessor<T>();
#else
        throw new InvalidOperationException(
            "No image processor specified. Install ClipSharp.SkiaSharp or ClipSharp.ImageSharp package.");
#endif
    }

    public VisualModel(InferenceSession session, IImageProcessor<T> imageProcessor)
    {
        _session = session;
        _imageProcessor = imageProcessor;

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

    public IReadOnlyCollection<T[]> Encode(string[] images)
    {
        var imgFs = images.Select(img => _imageProcessor.LoadAndProcessImage(img)).ToArray();

        // Convert to the appropriate tensor type
        var tokens = imgFs.SelectMany(l => l.ToFlatArray()).ToArray();
        
        // Create tensor based on type
        var inputTensor = new DenseTensor<T>(tokens, new[] { images.Length, 3, 224, 224 });

        using var results = _session.Run(new[] { NamedOnnxValue.CreateFromTensor(_inputName, inputTensor) });

        using var result = results.First();
        var embeddings = (DenseTensor<T>)result.Value;

        var output = new T[embeddings.Dimensions[0]][];
        for (int i = 0; i < embeddings.Dimensions[0]; i++)
        {
            output[i] = new T[embeddings.Dimensions[1]];
            for (int j = 0; j < embeddings.Dimensions[1]; j++)
            {
                output[i][j] = embeddings[i, j];
            }
        }

        return output;
    }

    public void Dispose()
    {
        _session?.Dispose();
    }
}
