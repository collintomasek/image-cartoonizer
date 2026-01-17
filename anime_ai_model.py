"""
AnimeGANv2 - Real AI Model Implementation using ONNX
Uses pre-trained neural network to transform photos into anime art
"""
import onnxruntime as ort
import cv2
import numpy as np
import os
import time

def process_image_for_model(img):
    """Prepare image for AI model input"""
    h, w = img.shape[:2]
    
    # Resize to multiple of 32 for model compatibility
    def to_32s(x):
        return 256 if x < 256 else x - x % 32
    
    resized = cv2.resize(img, (to_32s(w), to_32s(h)))
    
    # Convert BGR to RGB and normalize to [-1, 1]
    processed = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    
    # Add batch dimension
    processed = np.expand_dims(processed, axis=0)
    
    return processed, (h, w)

def postprocess_output(output, original_size):
    """Convert model output back to displayable image"""
    # Remove batch dimension and denormalize
    img = (np.squeeze(output) + 1.0) / 2.0 * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    # Resize to original size
    h, w = original_size
    img = cv2.resize(img, (w, h))
    
    # Convert RGB back to BGR for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    return img

def load_onnx_model(model_path):
    """Load ONNX model for inference"""
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return None
    
    print(f"Loading AI model: {os.path.basename(model_path)}")
    
    # Create inference session
    session = ort.InferenceSession(model_path, None)
    
    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    print(f"Model loaded successfully!")
    return session, input_name, output_name

def generate_anime_with_ai(input_path, output_path='anime_ai_model_result.jpg', model_path='models/Shinkai.onnx'):
    """
    Transform photo to anime using real AI model
    
    Args:
        input_path: Path to input photo
        output_path: Path to save anime result
        model_path: Path to ONNX model file
    """
    print("=" * 70)
    print("AnimeGANv2 - REAL AI MODEL")
    print("=" * 70)
    print(f"Style: Makoto Shinkai (Your Name / Weathering With You)")
    print(f"Input: {input_path}")
    print()
    
    # Check input file
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found!")
        return None
    
    # Load model
    model_data = load_onnx_model(model_path)
    if model_data is None:
        print("Failed to load model.")
        return None
    
    session, input_name, output_name = model_data
    
    # Read image
    print(f"\nReading image...")
    original = cv2.imread(input_path)
    if original is None:
        print(f"Error: Could not read image")
        return None
    
    print(f"Image size: {original.shape[1]}x{original.shape[0]}")
    
    # Prepare image for model
    print("Preprocessing image...")
    model_input, original_size = process_image_for_model(original)
    
    # Run AI inference
    print("Generating anime with AI neural network...")
    start_time = time.time()
    
    output = session.run([output_name], {input_name: model_input})
    
    inference_time = time.time() - start_time
    print(f"AI inference completed in {inference_time:.2f} seconds")
    
    # Post-process output
    print("Post-processing result...")
    result = postprocess_output(output[0], original_size)
    
    # Save result
    cv2.imwrite(output_path, result)
    print(f"\nAnime image saved: {output_path}")
    
    # Create comparison (larger display)
    scale = 0.9  # 90% of original size for better viewing
    h, w = original.shape[:2]
    preview_size = (int(w * scale), int(h * scale))
    
    print(f"Creating comparison at {preview_size[0]}x{preview_size[1]}...")
    orig_resized = cv2.resize(original, preview_size, interpolation=cv2.INTER_LANCZOS4)
    result_resized = cv2.resize(result, preview_size, interpolation=cv2.INTER_LANCZOS4)
    
    comparison = np.hstack((orig_resized, result_resized))
    comparison_path = 'anime_ai_model_comparison.jpg'
    cv2.imwrite(comparison_path, comparison)
    print(f"Comparison saved: {comparison_path}")
    
    # Display
    print("\nDisplaying result... (Press any key to close)")
    cv2.imshow("Original vs AI Anime (Real Neural Network)", comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 70)
    print("SUCCESS! Your AI-generated anime image is ready!")
    print("=" * 70)
    print(f"\nThis was generated using a real deep learning model")
    print(f"trained on {1445} frames from Makoto Shinkai films!")
    
    return result

if __name__ == "__main__":
    generate_anime_with_ai(
        input_path='input2.jpg',
        output_path='anime_ai_model_result.jpg',
        model_path='models/Shinkai.onnx'
    )