using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System.Numerics;
using ClipSharp;

namespace ClipSharp.ImageSharp;

public class ImageSharpImageProcessor<T> : ClipSharp.IImageProcessor<T> where T : struct, INumber<T>
{
    public T[,,] LoadAndProcessImage(string imagePath)
    {
        using var image = SixLabors.ImageSharp.Image.Load<Rgb24>(imagePath);
        
        // Resize to 224x224
        image.Mutate(x => x.Resize(224, 224, KnownResamplers.Lanczos3));
        
        // Convert to normalized CHW format
        return ToNormalizedColorHeightWidthArray(image);
    }

    public T[,,] ProcessImageData(byte[] imageData, int width, int height)
    {
        using var image = SixLabors.ImageSharp.Image.Load<Rgb24>(imageData);
        
        // Resize to 224x224
        image.Mutate(x => x.Resize(224, 224, KnownResamplers.Lanczos3));
        
        return ToNormalizedColorHeightWidthArray(image);
    }

    private T[,,] ToNormalizedColorHeightWidthArray(Image<Rgb24> image)
    {
        var img = new T[3, image.Height, image.Width];

        for (var y = 0; y < image.Height; y++)
        {
            for (var x = 0; x < image.Width; x++)
            {
                var pixel = image[x, y];
                
                // CLIP normalization (ImageNet stats) - convert to target type
                img[0, y, x] = ConvertToT((pixel.R / 255f - 0.48145466f) / 0.26862954f);   // Red
                img[1, y, x] = ConvertToT((pixel.G / 255f - 0.4578275f) / 0.26130258f);  // Green  
                img[2, y, x] = ConvertToT((pixel.B / 255f - 0.40821073f) / 0.27577711f);  // Blue
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
