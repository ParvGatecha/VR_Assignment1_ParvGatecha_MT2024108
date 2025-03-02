import cv2
import matplotlib.pyplot as plt
import numpy as np

# PART1 -- Task a: Detect coins using edge detection.
# --------------------
image = cv2.imread("./imgs/coins.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (15, 15), 1)

edges = cv2.Canny(blurred, 30, 150)

output_image = image.copy()

contours1, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(output_image, contours1, -1, (0, 255, 0), 2)
plt.imshow(output_image)
plt.title("Detected coins by Canny")
plt.savefig(f"./imgs/canny.png", bbox_inches="tight", dpi=300)
plt.axis("off")
plt.show()

# -----------------------------------------------------------------------------------------------------
# Task b: Region based segmentation of coins
# -----------------------------------------------------------------------------------------------------

min_coin_area = 350  # Minimum area to be considered a coin
valid_contours = [cnt for cnt in contours1 if cv2.contourArea(cnt) > min_coin_area]

output_image = image.copy()
cv2.drawContours(output_image, valid_contours, -1, (0, 255, 0), 2)

plt.imshow(output_image)
plt.title("Detected coins based on contour area")
plt.savefig(f"./imgs/Contour Area.png", bbox_inches="tight", dpi=300)
plt.axis("off")
plt.show()


for i, contour in enumerate(valid_contours):
    # mask for the current coin
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contour], -1, 255, -1)

    coin = cv2.bitwise_and(image, image, mask=mask)

    # bounding box
    x, y, w, h = cv2.boundingRect(contour)
    cropped_coin = coin[y : y + h, x : x + w]

    plt.imshow(cropped_coin)
    plt.title(f"Segmented Coin {i+1}")
    plt.savefig(f"./imgs/Segmented Coin{i+1}.png", bbox_inches="tight", dpi=300)
    plt.axis("off")
    plt.show()

# ------------------------------------------------------------------------------------
# Task c: Detect Number of coins
# ------------------------------------------------------------------------------------

# Here dilation of edges adds pixels to the edges thereby reducing the countours
# dilation merges the nearby edges, fills the small gaps

dilated = cv2.dilate(edges, (1, 1), iterations=2)
plt.imshow(dilated)
plt.savefig(f"./imgs/dilation.png", bbox_inches="tight", dpi=300)
plt.show()

output_image = image.copy()
contours2, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(output_image, contours2, -1, (0, 255, 0), 2)

print("Number of coins in the image: ", len(contours2))

plt.imshow(output_image)
plt.title("Detected coins after dilation")
plt.savefig(f"./imgs/After Dilation.png", bbox_inches="tight", dpi=300)
plt.axis("off")
plt.show()
