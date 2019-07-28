# Prototype code to detect red color
# Use cases within an IT environment:
# - Analysing in-rack live camera feeds to detect servers with error conditions (red or amber LEDs)
# - Cross-checking whether server monitoring tools are reporting error conditions 
# - Helping operators with difficulty interpreting color coded information 

import cv2
import numpy as np

#Import the image
source = cv2.imread('C:/prg/python/red-detect-3.jpg')

#Remove noise from the source image
median = cv2.medianBlur(source, 15)

# Convert BGR to HSV
hsv_source = cv2.cvtColor(median, cv2.COLOR_BGR2HSV)

# Lower red mask
lower_red = np.array([0,100,100])
upper_red = np.array([10,255,255])
red_mask_1 = cv2.inRange(hsv_source, lower_red, upper_red)

# Upper red mask
lower_red = np.array([160,100,100])
upper_red = np.array([180,255,255])
red_mask_2 = cv2.inRange(hsv_source, lower_red, upper_red)

# Join the masks
red_mask_final = red_mask_1 + red_mask_2

# Bitwise-AND mask on the hsv_source image
output_hsv = cv2.bitwise_and(hsv_source, hsv_source, mask = red_mask_final)

# Blur the image to smoothen out bright areas
blurred = cv2.GaussianBlur(output_hsv, (15, 15), 0)

# Threshold the image to reveal bright areas
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

# Split the image into components
h_chan, s_chan, v_chan = cv2.split(thresh)

# Erode and dilate the image to remove small unwanted areas
kernel = np.ones((3,3), np.uint8)
v_chan = cv2.erode(v_chan, kernel, iterations=2)
v_chan = cv2.dilate(v_chan, kernel, iterations=4)

# Perform a connected component analysis
nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(v_chan)

# Create a mask to store the points of interest
poi_mask = np.zeros(v_chan.shape, dtype="uint8")

# Create a variable to store the number of points
poi = 0

# Loop over the labels
for label in np.unique(labels):
    # Ignore the background label
    if label == 0:
        continue

    # Build the label mask and count the number of pixels in the area
    label_mask = np.zeros(v_chan.shape, dtype="uint8")
    label_mask[labels == label] = 255
    num_pixels = cv2.countNonZero(label_mask)

    # if the number of pixels in the component are sufficient in number then add the area to the point of interest mask
    if num_pixels > 50:
        poi_mask = cv2.add(poi_mask, label_mask)
        poi += 1

print ('Points of interest:', poi)

#Convert the POI mask back to RGB before overlaying the source image
poi_mask_back_to_rgb = cv2.cvtColor(poi_mask, cv2.COLOR_GRAY2RGB)
overlay = cv2.addWeighted(source, 0.4, poi_mask_back_to_rgb, 0.6, 0)

cv2.imshow('Source Image', source)
cv2.imshow('Points of Interest Mask', poi_mask)
cv2.imshow('Overlayed Image', overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()