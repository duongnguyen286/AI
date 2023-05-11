import cv2
import os
import numpy as np
from resolve import solve_sudoku
from scipy.spatial import distance_matrix
in_file = os.path.join("data", "page.png")

img = cv2.imread(in_file)
gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,thresh1 = cv2.threshold(gray,225,255,cv2.THRESH_BINARY_INV)
cv2.imwrite("thresh1.jpg",thresh1)
contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_LIST , cv2.CHAIN_APPROX_NONE)
contours = sorted(contours, key=cv2.contourArea,reverse=True )[1:82]
contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1] * img.shape[1])
img0 = img.copy()
cv2.drawContours(img0,contours, -1, (0,255,0), 3)
cv2.imwrite("img0.jpg",img0)
cv2.imshow('img0', img0)
arr = []
for j, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    arr.append([x,y])

arr_matrix = np.zeros((9,9))
img1 = img.copy()
for j in range(1, 10):
    template = cv2.imread(os.path.join("data", f"match{j}.png"),0)
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(thresh1,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        # print(pt)
        # cv2.rectangle(img1, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        cv2.putText(img1, "{}".format(j), (pt[0], pt[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
            (0, 0, 255), 2)
        distance = distance_matrix([[pt[0], pt[1]]],arr)
        # print(np.argmin(distance),len(arr))
        x,y = np.unravel_index(np.argmin(distance),arr_matrix.shape)
        arr_matrix[x][y] = j

print(arr_matrix)

cv2.imshow('img1', img1)
cv2.imwrite("img1.jpg",img1)
arr_matrix = solve_sudoku(arr_matrix)
for i in range(0, 9):
    for j in range(0, 9):
        idx = i * 9 + j
        pt = arr[idx]
        cv2.putText(img, "{}".format(int(arr_matrix[i][j])), (pt[0], pt[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
            (0, 0, 255), 2)
cv2.imshow('img', img)
cv2.imwrite("img.jpg",img)
cv2.waitKey()