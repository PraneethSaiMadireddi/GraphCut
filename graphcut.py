import cv2
import imageio
import matplotlib.pyplot as plt
from utils import quilt_image

input_image = cv2.imread('images/input_images/green.jpeg')
refined_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
plt.imshow(refined_image)
final_image = quilt_image(refined_image, 25, (6, 6), "cut")
plt.imshow(final_image)
# rst = final_image.astype('uint8')
imageio.imwrite('images/generated_images/green_new.jpeg', final_image)
