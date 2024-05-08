import streamlit as st
import numpy as np
from PIL import Image
import cv2
import easyocr as oc
import pandas as pd


def detect(image):
    # image=cv2.imread(image)
    img2=image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray",gray)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    # cv2.imshow("blur",blur)
    canny=cv2.Canny(blur,0,150)
    # cv2.imshow("canny",canny)
    # print(canny.dtype)
    hough_lines_image2 = np.zeros_like(image)
    linesP = cv2.HoughLinesP(canny, 1, np.pi / 90, 50, None, 100, 20)
    if linesP is not None:
      for i in range(0, len(linesP)):
       l = linesP[i][0]
       cv2.line(hough_lines_image2, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
    # cv2.imshow("hou",hough_lines_image2)
    hough_lines_image2=cv2.cvtColor(hough_lines_image2,cv2.COLOR_BGR2GRAY)
    # print(hough_lines_image2.dtype)
    # rho_resolution = 1
    # theta_resolution = np.pi/90
    # threshold = 230

    # hough_lines = cv2.HoughLines(canny, rho_resolution , theta_resolution , threshold)

    # hough_lines_image = np.zeros_like(image)
    # for line in hough_lines:
    #         for rho,theta in line:
    #             a = np.cos(theta)
    #             b = np.sin(theta)
    #             x0 = a*rho
    #             y0 = b*rho
    #             x1 = int(x0 + 1000*(-b))
    #             y1 = int(y0 + 1000*(a))
    #             x2 = int(x0 - 1000*(-b))
    #             y2 = int(y0 - 1000*(a))
    
    #             cv2.line(hough_lines_image,(x1,y1),(x2,y2),(0,255,0),2)

    # cv2.imshow("sdf",hough_lines_image)
    # original_image_with_hough_lines = cv2.addWeighted(hough_lines_image, 0.8, image, 1., 0.)




    # threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)

    # invert = cv2.bitwise_not(threshold)

    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    # dilate = cv2.dilate(invert, kernel, iterations = 1)
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contours))

    approx_contours = [cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True) for cnt in contours]
    # print(approx_contours)
    table_contours = [cnt for cnt in approx_contours if len(cnt) == 4]
    # print(table_contours)
    # df=pd.DataFrame(narr,index=area)
    # print(df)



    max_area=0
    max_cont=0
    for cnt in table_contours:        
                # print(cv2.contourArea(cnt))
                image=cv2.drawContours(image,[cnt],-1,(255,0,0),2)
                if(max_area<cv2.contourArea(cnt)):
                    max_area=cv2.contourArea(cnt)
                    max_cont=cnt
    large_box=cv2.boundingRect(max_cont)
    # cv2.imshow("image",image)

    x,y,w,h=large_box
    # print(x,y,w,h)
    newimg=img2[y:y+h,x:x+w]
    # cv2.imshow("cropped image",newimg)
    # cv2.imshow("3.jpg", blur)
    # cv2.imshow("4.jpg", threshold)
    # cv2.imshow("5.jpg", invert)
    # cv2.imshow("6.jpg", dilate)
    # height, width = newimg.shape[:2]

    # new_height = 2*height
    # new_width = 2*width
    # area2=new_height*new_width
    # # Resize the image using the calculated dimensions
    # newimg = cv2.resize(newimg, (new_width, new_height))

    reader = oc.Reader(['en'], gpu=True)
    # result = reader.readtext('image copy 9.png', detail=0) #paragraph= True,

    result = reader.readtext(newimg) #paragraph=True
    # print("Detected Test Results:", result)

    for i in range(len(result)):
        top_left = tuple(map(int, result[i][0][0]))
        bottom_right = tuple(map(int, result[i][0][2]))
        # print(top_left, bottom_right)
        image = cv2.rectangle(newimg,top_left,bottom_right,(0,255,0),2)

    table = []
    cols = []
    col_ref = []
    x1 = []
    y1 = []
    for tup in range(len(result)):
        x, y = result[tup][0][0]
        x1.append(x)
        y1.append(y)
    for i in range(y1[0]-10, y1[0]+10):
        for j in range(len(x1)):
            if y1[j] == i:
                cols.append(result[j][1])
                col_ref.append(x1[j])
    print(cols)
    for i in range(len(cols)):
        table.append([])
    # print(table)
    for tup in range(len(cols), len(result)): 
        for col_range in range(int(x1[tup])-30, int(x1[tup])+30):
            if x1[tup] == col_range:
                for i in range(len(cols)):
                    if (col_range > (col_ref[i] - 100)) and (col_range < (col_ref[i] + 100)):
                        table[i].append(result[tup][1])

    df = pd.DataFrame(table).transpose()

    df = df.rename(columns = {i:cols[i] for i in range(len(cols))})


    st.write(df)
    # cv2.imshow("2.jpg",image)






    return image

st.title("Text Extraction")
try:
    img=st.sidebar.file_uploader("file",type=["jpeg","jpg","webp","png"],accept_multiple_files=False)

    img1=Image.open(img)

    st.image(img1,"Original image")
    if(st.button("detect")):
        st.image(detect(np.array(img1)))
except:
    st.write("Insert Image!!!")
