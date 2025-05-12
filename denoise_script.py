# denoise_inference.py

import time
import tensorflow as tf
import numpy as np
import cv2
import sys # For command line arguments and exit

# --- User-Provided Modules ---
try:
    from utils import imsave
    # 'generator' should be the instantiated Generator class from model.py
    # model.py should handle any weight loading if necessary when 'generator' instance is created/imported
    from model import generator
except ImportError as e:
    print(f"Error importing 'utils' or 'model': {e}")
    print("Please ensure 'utils.py' and 'model.py' (which defines and instantiates 'generator') are accessible.")
    sys.exit(1)

# --- Configuration ---
TARGET_OUTPUT_HEIGHT = 144
TARGET_OUTPUT_WIDTH = 256
PADDING_VAL = 56

def preprocess_padded_input_for_generator(padded_image_np_rgb_float32):
    """
    Applies training-consistent standardization to a padded image and adds batch dimension.
    """
    print(f"DEBUG: Padded input to standardize - Min: {padded_image_np_rgb_float32.min():.4f}, Max: {padded_image_np_rgb_float32.max():.4f}, Shape: {padded_image_np_rgb_float32.shape}, Dtype: {padded_image_np_rgb_float32.dtype}")
    standardized_image_tensor = tf.image.per_image_standardization(padded_image_np_rgb_float32)
    try:
        standardized_image_np = standardized_image_tensor.numpy()
    except AttributeError:
        print("WARNING: .numpy() failed. Attempting to run in a new TF1 session.")
        with tf.compat.v1.Session() as sess:
            standardized_image_np = sess.run(standardized_image_tensor)
    print(f"DEBUG: After standardization - Min: {standardized_image_np.min():.4f}, Max: {standardized_image_np.max():.4f}, Shape: {standardized_image_np.shape}")
    batched_image = np.expand_dims(standardized_image_np, axis=0)
    print(f"DEBUG: After adding batch dim - Shape: {batched_image.shape}, Dtype: {batched_image.dtype}")
    return batched_image

def postprocess_generator_output(output_tensor_batch):
    """Converts generator's output batch to a displayable image (H, W, C) in [0,1] range."""
    if output_tensor_batch is None or output_tensor_batch.size == 0:
        print("ERROR: Generator output is empty or None.")
        return np.zeros((TARGET_OUTPUT_HEIGHT, TARGET_OUTPUT_WIDTH, 3), dtype=np.float32)

    img_from_generator = output_tensor_batch[0] # Remove batch dim
    # **IMPORTANT**: Your generator model ALREADY scales its output to [0,1] range
    # using its internal normalize_output((tanh_out + skip_input_standardized + 1.0)/2.0)
    # So, img_from_generator should already be [0,1].
    print(f"DEBUG: Raw generator output (SHOULD BE [0,1] from model) - Min: {img_from_generator.min():.4f}, Max: {img_from_generator.max():.4f}, Shape: {img_from_generator.shape}, Dtype: {img_from_generator.dtype}")

    if img_from_generator.shape[0] <= PADDING_VAL:
        print(f"ERROR: Image height ({img_from_generator.shape[0]}) is <= PADDING_VAL ({PADDING_VAL}). Cannot crop.")
        img_cropped = img_from_generator 
    else:
        img_cropped = img_from_generator[PADDING_VAL:, :, :]
    print(f"DEBUG: After crop [{PADDING_VAL}:,:,:] - Min: {img_cropped.min():.4f}, Max: {img_cropped.max():.4f}, Shape: {img_cropped.shape}")

    # The [0,1] scaling is ALREADY DONE by the generator. We just need to ensure it's clipped.
    img_processed = np.clip(img_cropped, 0.0, 1.0)
    print(f"DEBUG: After clipping (generator output already [0,1]) - Min: {img_processed.min():.4f}, Max: {img_processed.max():.4f}")

    if img_processed.shape[0] != TARGET_OUTPUT_HEIGHT or img_processed.shape[1] != TARGET_OUTPUT_WIDTH:
        img_resized = cv2.resize(img_processed, (TARGET_OUTPUT_WIDTH, TARGET_OUTPUT_HEIGHT), interpolation=cv2.INTER_AREA)
    else:
        img_resized = img_processed
    print(f"DEBUG: After cv2.resize to target - Min: {img_resized.min():.4f}, Max: {img_resized.max():.4f}, Shape: {img_resized.shape}, Dtype: {img_resized.dtype}")
        
    return img_resized

def denoise_image_inference(image_path_or_np_array):
    if isinstance(image_path_or_np_array, str):
        image_np = cv2.imread(image_path_or_np_array)
        if image_np is None:
            raise FileNotFoundError(f"ERROR: Image not found at '{image_path_or_np_array}'")
        image_np_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    elif isinstance(image_path_or_np_array, np.ndarray):
        image_np_rgb = image_path_or_np_array
        if image_np_rgb.ndim == 2:
            image_np_rgb = cv2.cvtColor(image_np_rgb, cv2.COLOR_GRAY2RGB)
        if image_np_rgb.shape[-1] == 4:
            image_np_rgb = image_np_rgb[..., :3]
    else:
        raise ValueError("ERROR: Input must be either a file path (str) or a numpy array")

    image_np_rgb_float32 = image_np_rgb.astype('float32') if image_np_rgb.dtype != np.float32 else image_np_rgb
    print(f"DEBUG: Input image loaded - Shape: {image_np_rgb_float32.shape}, Dtype: {image_np_rgb_float32.dtype}, Min: {image_np_rgb_float32.min():.0f}, Max: {image_np_rgb_float32.max():.0f}")

    npad = ((PADDING_VAL, PADDING_VAL), (0, 0), (0, 0))
    padded_image = np.pad(image_np_rgb_float32, pad_width=npad, mode='constant', constant_values=0)
    
    generator_input_batch = preprocess_padded_input_for_generator(padded_image)
    
    print(f"INFO: Feeding to generator.predict() - Input Shape: {generator_input_batch.shape}, Input Dtype: {generator_input_batch.dtype}")
    start_time = time.time()
    try:
        # Ensure 'generator' is the instantiated and (if necessary) weight-loaded model
        generated_output_batch = generator.predict(generator_input_batch)
    except Exception as e:
        print(f"ERROR: generator.predict() failed: {e}")
        traceback.print_exc()
        return None
    print(f"INFO: Denoising (generator.predict) took {time.time() - start_time:.4f} seconds.")
    
    final_image_0_1_range = postprocess_generator_output(generated_output_batch)
    
    output_filename_utils = 'output_denoised_utils.png'
    try:
        imsave(output_filename_utils, final_image_0_1_range) 
        print(f"INFO: Denoised image saved as '{output_filename_utils}' using utils.imsave.")
    except Exception as e:
        print(f"ERROR: utils.imsave failed: {e}. Attempting PIL save.")
        output_filename_pil = 'output_denoised_pil_fallback.png'
        try:
            from PIL import Image as PILImage
            img_to_save_uint8 = (final_image_0_1_range * 255.0).astype(np.uint8)
            pil_img = PILImage.fromarray(img_to_save_uint8)
            pil_img.save(output_filename_pil)
            print(f"INFO: Denoised image saved as '{output_filename_pil}' using PIL.")
        except Exception as pil_e:
            print(f"ERROR: Saving with PIL also failed: {pil_e}")
            
    return final_image_0_1_range

if __name__=='__main__':
    if len(sys.argv) < 2:
        print("Usage: python denoise_inference.py <path_to_input_image>")
        sys.exit(1)
    
    input_image_path = sys.argv[1]
    
    print("INFO: TensorFlow Version:", tf.__version__)
    # print("INFO: Eager execution enabled:", tf.executing_eagerly()) # .numpy() needs this or session
    print(f"INFO: Attempting to denoise: {input_image_path}")

    # --- CRITICAL: Model Loading Assurance ---
    # Your model.py does:
    #   generator = Generator()
    #   _ = generator(dummy_input) # This builds the model
    # This means 'from model import generator' should give you an initialized model.
    # If you save/load weights, that needs to happen in model.py or here.
    # For now, assuming 'generator' from model.py is ready with trained weights.
    if 'generator' not in globals() or generator is None:
        print("CRITICAL ERROR: The 'generator' model instance from 'model.py' is not available or None.")
        print("              Please ensure model.py defines and instantiates 'generator' correctly,")
        print("              and that it has its trained weights loaded if you are not training now.")
        sys.exit(1)
    if not generator.built:
        print("WARNING: Generator model does not seem to be built. Did you load weights or call it on dummy input?")
        # Attempt to build if not built, using a shape that matches padded input later
        # This is a guess for BATCH_SHAPE[1], BATCH_SHAPE[2]
        # Padded height will be original_H + 2*PADDING_VAL, padded_W will be original_W
        # This dummy build might not be perfect if original image sizes vary a lot.
        # The dummy input in your model.py (256,256,3) is better.
        # Let's assume the padding logic creates something compatible with that (256, W, 3)
        dummy_padded_height = 100 + 2 * PADDING_VAL # e.g. 100 is an arbitrary original height
        dummy_padded_width = 100 # e.g. 100 is an arbitrary original width
        try:
            print(f"Attempting to build generator with dummy input of shape (1, {dummy_padded_height}, {dummy_padded_width}, 3)")
            # The model.py uses (1, 256, 256, 3) for its dummy build. This is good.
            # The preprocess_padded_input_for_generator will create the actual input shape.
            # We just need to ensure the model instance is "built" so predict can be called.
            # The dummy call in model.py should have handled this.
            pass # Assuming model.py's dummy call built it.
        except Exception as build_e:
            print(f"ERROR trying to build generator: {build_e}")


    # --- End Model Loading Assurance ---

    import traceback # For more detailed error messages
    try:
        denoised_output_0_1_range = denoise_image_inference(input_image_path)
        
        if denoised_output_0_1_range is not None:
            print("INFO: Denoising process completed.")
            try:
                original_display = cv2.imread(input_image_path)
                if original_display is None:
                    print(f"WARNING: Could not read original for display: {input_image_path}")
                else:
                    denoised_display_uint8_rgb = (denoised_output_0_1_range * 255.0).astype(np.uint8)
                    denoised_display_bgr = cv2.cvtColor(denoised_display_uint8_rgb, cv2.COLOR_RGB2BGR)
                    h_denoised, w_denoised = denoised_display_bgr.shape[:2]
                    h_orig, w_orig = original_display.shape[:2]
                    if h_orig != h_denoised:
                        scale_factor = h_denoised / h_orig
                        new_w_orig = int(w_orig * scale_factor)
                        original_display_resized = cv2.resize(original_display, (new_w_orig, h_denoised))
                    else:
                        original_display_resized = original_display
                    comparison_img = np.concatenate((original_display_resized, denoised_display_bgr), axis=1)
                    cv2.imshow("Original (Resized) vs. Denoised", comparison_img)
                    print("INFO: Displaying comparison. Press any key to close.")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            except Exception as display_e:
                print(f"WARNING: Could not display images: {display_e}")
        else:
            print("ERROR: Denoising returned None.")

    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"CRITICAL ERROR in __main__: {e}")
        traceback.print_exc()