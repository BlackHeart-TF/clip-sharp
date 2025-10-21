using System.Numerics;

namespace ClipSharp;

/// <summary>
/// Interface for image processing operations required by CLIP visual model
/// </summary>
/// <typeparam name="T">The numeric type for image data</typeparam>
public interface IImageProcessor<T> where T : struct, INumber<T>
{
    /// <summary>
    /// Loads an image from a file path and converts it to normalized CHW format
    /// </summary>
    /// <param name="imagePath">Path to the image file</param>
    /// <returns>Normalized image data in CHW format (3, height, width)</returns>
    T[,,] LoadAndProcessImage(string imagePath);
    
    /// <summary>
    /// Converts image data to normalized CHW format
    /// </summary>
    /// <param name="imageData">Raw image data</param>
    /// <param name="width">Image width</param>
    /// <param name="height">Image height</param>
    /// <returns>Normalized image data in CHW format (3, height, width)</returns>
    T[,,] ProcessImageData(byte[] imageData, int width, int height);
}
