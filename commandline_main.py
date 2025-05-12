import numpy as np
import cv2
from test import *
import argparse

def denoise_and_show(input_path):
    """
    Denoises an image from the given file path and displays the input and output side by side.

    Args:
        input_path (str): The path to the input noisy image.
    """

    try:
        # 1. Load the image using OpenCV
        input_img = cv2.imread(input_path)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # 2. Denoise the image
        output_img = denoise(input_img)  # Pass the numpy array directly

        # 3. Handle output image range (IMPORTANT)
        if np.min(output_img) < 0 or np.max(output_img) > 1:
            print("Warning: Output image values outside the expected range [0, 1]. Clipping.")
            output_img = np.clip(output_img, 0, 1)

        output_img = (output_img * 255).astype(np.uint8)  # Scale to 0-255 range

        # 4. Create side-by-side comparison
        # Get dimensions
        height1, width1 = input_img.shape[:2]
        height2, width2 = output_img.shape[:2]
        
        # Create a new image with combined width
        combined_width = width1 + width2
        combined_height = max(height1, height2)
        combined_img = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
        
        # Copy images side by side
        combined_img[:height1, :width1] = input_img
        combined_img[:height2, width1:width1+width2] = output_img
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined_img, "Original", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(combined_img, "Denoised", (width1 + 10, 30), font, 1, (255, 255, 255), 2)
        
        # Save the comparison
        cv2.imwrite('comparison.png', cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR))
        
        # Show the combined image
        cv2.imshow("Original vs Denoised", cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error processing image: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoise an image from the command line.")
    parser.add_argument("input_image", type=str, help="Path to the input noisy image.")
    args = parser.parse_args()

    denoise_and_show(args.input_image)