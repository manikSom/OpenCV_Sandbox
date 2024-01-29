import cv2
from skimage import filters, metrics
import numpy as np


def estimate_focus_area(image, text_position=(10, 30)):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the gradient magnitude using Sobel
    gradient_magnitude = filters.sobel(gray)

    # Normalize the gradient values to range [0, 255]
    normalized_gradient = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to 3-channel image for concatenation
    focus_area_image = cv2.cvtColor(normalized_gradient.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Quantify focus and add text
    focus_percentage = quantify_focus(image)
    sharpness_score = quantify_image_sharpness_ssim(image)

    text_focus = f"Focus: {focus_percentage:.2f}%"
    #text_sharpness = f"Sharpness: {sharpness_score:.2f}"

    cv2.putText(focus_area_image, text_focus, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                cv2.LINE_AA)
    #cv2.putText(focus_area_image, text_sharpness, (text_position[0], text_position[1] + 30), cv2.FONT_HERSHEY_SIMPLEX,
    #            1, (255, 255, 255), 2, cv2.LINE_AA)

    return focus_area_image


def highlight_focus_area(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the gradient magnitude using Sobel
    gradient_magnitude = filters.sobel(gray)

    # Normalize the gradient values to range [0, 255]
    normalized_gradient = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to 3-channel image for overlay
    overlay = cv2.cvtColor(normalized_gradient.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Define a color for the overlay (e.g., green)
    color = (255, 0, 0)

    # Create a binary mask for the focus area
    focus_mask = normalized_gradient > 80  # Adjust the threshold as needed

    # Apply the color overlay only to the focus area
    output_image = image.copy()
    output_image[focus_mask] = color

    return output_image


def quantify_focus(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the gradient magnitude using Sobel
    gradient_magnitude = filters.sobel(gray)

    # Normalize the gradient values to range [0, 1]
    normalized_gradient = cv2.normalize(gradient_magnitude, None, 0, 1, cv2.NORM_MINMAX)

    # Threshold the gradient magnitude
    threshold = 0.2  # Adjust the threshold as needed
    focus_mask = normalized_gradient > threshold

    # Calculate the percentage of in-focus pixels
    in_focus_percentage = (np.sum(focus_mask) / focus_mask.size) * 100

    return in_focus_percentage


def quantify_image_sharpness(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the gradient magnitude using Sobel
    gradient_magnitude = filters.sobel(gray)

    # Normalize the gradient values to range [0, 1]
    normalized_gradient = cv2.normalize(gradient_magnitude, None, 0, 1, cv2.NORM_MINMAX)

    # Calculate the average gradient magnitude within the depth of field
    depth_of_field_mask = normalized_gradient > 0.2  # Adjust the threshold as needed
    average_sharpness = np.mean(normalized_gradient[depth_of_field_mask])

    return average_sharpness


def quantify_image_sharpness_ssim(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the SSIM (Structural Similarity Index)
    ssim_score, _ = metrics.structural_similarity(gray, gray, full=True)

    return ssim_score


def quantify_image_sharpness_laplacian(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply the Laplacian filter
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Calculate the variance of the Laplacian
    sharpness_score = laplacian.var()

    return sharpness_score


def quantify_image_sharpness_tenengrad(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the gradient using Sobel
    gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate Tenengrad as the square root of the sum of squared gradients
    tenengrad = np.sqrt(gradient_x**2 + gradient_y**2)

    # Calculate the variance of Tenengrad
    sharpness_score = tenengrad.var()

    return sharpness_score


def quantify_image_sharpness_blur_index(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the blur index
    blur_index = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Invert the blur index to get a sharpness score
    sharpness_score = 1 / blur_index if blur_index != 0 else 0

    return sharpness_score


# Example usage
image_path = "img/test5.jpg"
original_image = cv2.imread(image_path)

# Get the images
focus_area_image = estimate_focus_area(original_image)
highlighted_focus_area_image = highlight_focus_area(original_image)

# Resize the original image while maintaining the aspect ratio
desired_width = 1800
aspect_ratio = original_image.shape[1] / original_image.shape[0]
desired_height = int(desired_width / (3 * aspect_ratio))

# Resize the original image to match the size of focus_area_image
original_image_resized = cv2.resize(original_image, (focus_area_image.shape[1], focus_area_image.shape[0]))

# Create a blank canvas
canvas = np.zeros((original_image_resized.shape[0], 3 * original_image_resized.shape[1], 3), dtype=np.uint8)

# Paste the images onto the canvas
canvas[:, :original_image_resized.shape[1]] = original_image_resized
canvas[:, original_image_resized.shape[1]:2*original_image_resized.shape[1]] = focus_area_image
canvas[:, 2*original_image_resized.shape[1]:] = highlighted_focus_area_image

# Resize the canvas to a reasonable width
canvas_resized = cv2.resize(canvas, (desired_width, desired_height))

print(f"Sharpness Score laplacian: {quantify_image_sharpness_laplacian(original_image):.2f}")
print(f"Sharpness Score tenengrad: {quantify_image_sharpness_tenengrad(original_image):.2f}")
print(f"Sharpness Score blur index: {quantify_image_sharpness_blur_index(original_image):.2f}")
print(f"Sharpness Score ssim: {quantify_image_sharpness_ssim(original_image):.2f}")

# Display the resized result
cv2.imshow("Combined Image", canvas_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()