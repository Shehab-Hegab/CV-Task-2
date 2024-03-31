import cv2
import numpy as np

image_path = 'F:/Test Input Images/250x166/3.jpg'


# Read the image
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detector
edges = cv2.Canny(gray, 100, 200)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through contours
for contour in contours:
    # Fit ellipse to contour
    if len(contour) >= 5: # Ellipse fitting requires at least 5 points
        ellipse = cv2.fitEllipse(contour)
        
        # Filter ellipses based on area, aspect ratio, etc.
        if 10 < cv2.contourArea(contour) < 50:
            # Draw ellipse on original image
            cv2.ellipse(image, ellipse, (0, 255, 0), 2)

# Display the result
cv2.imshow('Ellipses', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
