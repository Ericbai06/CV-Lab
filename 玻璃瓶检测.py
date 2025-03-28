import cv2
img=cv2.imread('检测 玻璃瓶/水滴形截面的玻璃瓶/Image_20191006232054610.jpg')
blurred_image = cv2.GaussianBlur(img, (5, 5), 0)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)

cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()


