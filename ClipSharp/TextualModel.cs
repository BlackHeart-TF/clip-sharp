using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;

namespace ClipSharp;

public class TextualModel<T> : IDisposable where T : struct, INumber<T>
{
    private readonly InferenceSession _session;
    private readonly ITextTokenizer _tokenizer;
    private readonly int[] _padding;
    private readonly int _inputSize;
    private readonly string _inputName;
    private readonly string _outputName;

    public static TextualModel<T> Load(string modelPath, ITextTokenizer tokenizer)
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

        return new TextualModel<T>(session, tokenizer);
    }

    public TextualModel(InferenceSession session, ITextTokenizer tokenizer)
    {
        _session = session;
        _tokenizer = tokenizer;

        var input = _session.InputMetadata.First().Value;

        if (input.Dimensions.Length != 2 || input.Dimensions[1] != 77)
        {
            throw new ArgumentException($"Unexpected input dimensions (expected [-1, 77])");
        }
        _inputSize = input.Dimensions[1];
        _inputName = session.InputNames.First();

        Debug.WriteLine($"Textual inference ready, input size {_inputSize}, type {input.OnnxValueType}");

        _outputName = session.OutputNames.First();

        _padding = Enumerable.Range(0, 77).Select(i => tokenizer.EotToken).ToArray(); // todo
    }

    public IReadOnlyCollection<T[]> Encode(IReadOnlyCollection<string> texts)
    {
        var textTokens = new List<IEnumerable<int>>();

        foreach (var text in texts) 
        {
            var tmp = new List<int> { _tokenizer.SotToken };
            tmp.AddRange(_tokenizer.Encode(text));
            tmp.Add(_tokenizer.EotToken);
            textTokens.Add(Pad(tmp));
        }

        Memory<int> tokens = textTokens.SelectMany(l => l).ToArray();
        var inputTensor = new DenseTensor<int>(tokens, new[] { texts.Count, _inputSize });

        using var results = _session.Run(new[] { NamedOnnxValue.CreateFromTensor(_inputName, inputTensor) });
        using var result = results.First();
        var embeddings = (DenseTensor<T>)result.Value;

        var output = new T[embeddings.Dimensions[0]][];
        for (int i = 0; i< embeddings.Dimensions[0]; i++)
        {
            output[i] = new T[embeddings.Dimensions[1]];
            for (int j = 0; j < embeddings.Dimensions[1]; j++)
            {
                output[i][j] = embeddings[i,j];
            }
        }

        return output;
    }

    private IEnumerable<int> Pad(IEnumerable<int> input)
    {
        return input.Concat(_padding).Take(_inputSize);
    }

    public void Dispose()
    {
        _session?.Dispose();
    }
}
