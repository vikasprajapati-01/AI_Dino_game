import cv2
import numpy as np
import pyautogui

# game_location={'top': 230, 'left': 680, 'width': 500, 'height': 50}
game_location = {'top': 150, 'left': 400, 'width': 600, 'height': 400}

screenshot = pyautogui.screenshot(region=(
    game_location['left'], 
    game_location['top'], 
    game_location['width'], 
    game_location['height']
))

image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

cv2.imshow("Captured Area", image)

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Grayscale Image", gray)

cv2.waitKey(0)
cv2.destroyAllWindows()
