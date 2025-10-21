using SkiaSharp;
using System.Numerics;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;

namespace ClipSharp.SkiaSharp;

public class SkiaSharpImageProcessor<T> : IImageProcessor<T> where T : struct, INumber<T>
{
    public T[,,] LoadAndProcessImage(string imagePath)
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

    public T[,,] ProcessImageData(byte[] imageData, int width, int height)
    {
        using var bitmap = SKBitmap.Decode(imageData);
        if (bitmap == null)
        {
            throw new InvalidOperationException("Could not decode image data");
        }

        // Resize to 224x224
        using var resizedBitmap = bitmap.Resize(new SKImageInfo(224, 224), SKFilterQuality.High);
        
        return ToNormalizedColorHeightWidthArray(resizedBitmap);
    }

    private T[,,] ToNormalizedColorHeightWidthArray(SKBitmap bitmap)
    {
        var img = new T[3, bitmap.Height, bitmap.Width];

        for (var y = 0; y < bitmap.Height; y++)
        {
            for (var x = 0; x < bitmap.Width; x++)
            {
                var pixel = bitmap.GetPixel(x, y);
                
                // CLIP normalization (ImageNet stats) - convert to target type
                img[0, y, x] = ConvertToT((pixel.Red / 255f - 0.48145466f) / 0.26862954f);   // Red
                img[1, y, x] = ConvertToT((pixel.Green / 255f - 0.4578275f) / 0.26130258f); // Green  
                img[2, y, x] = ConvertToT((pixel.Blue / 255f - 0.40821073f) / 0.27577711f);  // Blue
            }
        }

        return img;
    }

    private static T ConvertToT(float value)
    {
        // Handle different numeric types
        if (typeof(T) == typeof(float))
            return (T)(object)value;
        if (typeof(T) == typeof(Half))
            return (T)(object)(Half)value;
        if (typeof(T) == typeof(sbyte))
            return (T)(object)(sbyte)Math.Round(Math.Max(-128, Math.Min(127, value)));
        
        // Default conversion
        return (T)Convert.ChangeType(value, typeof(T));
    }
}
