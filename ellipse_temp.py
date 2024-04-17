import cv2
import numpy as np

def hough_shape_detection(image):
  """
  Detects lines and ellipses (approximated) in an image using Hough Transform.

  Args:
      image: Grayscale image.

  Returns:
      image: Original image with detected shapes overlaid.
  """
  # Apply edge detection (adjust parameters as needed)
  edges = cv2.Canny(image, 50, 150)

  # Define Hough transform parameters
  rho = 1  # Distance resolution
  theta = np.pi / 180  # Angle resolution
  threshold_lines = 100  # Minimum votes for lines
  # Adjust parameters for ellipses (increase minDist to avoid circle merging)
  param1_ellipses = 50  # Gradient accumulation threshold
  param2_ellipses = 30  # Idk
  min_radius = 0  # Minimum radius (set to 0 for ellipses)
  max_radius = 0  # Maximum radius (set to 0 for ellipses)

  # Perform Hough Line Transform
  lines = cv2.HoughLinesP(edges, rho, theta, threshold_lines, minLineLength=30, maxLineGap=10)

  # Perform Hough Ellipse Transform (approximate)
  circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20,
                             param1=param1_ellipses, param2=param2_ellipses, minRadius=min_radius, maxRadius=max_radius)

  # Draw detected lines and ellipses on the original image
  if lines is not None:
    for line in lines:
      x1, y1, x2, y2 = line[0]
      cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
  if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
      # Access only the first 3 elements (x, y, radius)
      center = (i[0], i[1])
      radius = i[2]
      cv2.ellipse(image, center, (radius, radius), 0, 0, 360, (255, 0, 0), 2)

  return image

# Load image
image_path = 'F:/Test Input Images/250x166/3.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Detect shapes
detected_image = hough_shape_detection(image)

# Display results
cv2.imshow("Detected Shapes", detected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
