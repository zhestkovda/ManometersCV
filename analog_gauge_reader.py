'''  
Copyright (c) 2019 ElMetro LLC.
'''


import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import random

##########################################
#### GLOBAL PARAMETERS OF ALGORITHM
##########################################
angle_blank = 40 # absolute value of blank angle. E.g. if angle_blank=40 in this areas 0-40 and 320-360 degrees MIN aand MAX scale lines will not be searched
Arrow_r1 = 0.6 # Arrow_r1*ArrowL - is radius of first arc for arrow center calculation
Arrow_r2 = 0.9 # Arrow_r2*ArrowL - is radius of second arc for arrow center calculation
minCtrRadius = 0.85 # 0.85*ArrowL is internal radius of tor where scale lines are searched
maxCtrRadius = 1.15 # 1.15*ArrowL is internal radius of tor where scale lines are searched
##threshold_bin = 100 # порог, по которому происходит бинаризация
##########################################
##########################################
''' average circle calculation '''
def avg_circles(circles, b):
    avg_x=0
    avg_y=0
    avg_r=0
    if b == 1:
        return circles[0][0], circles[0][1], circles[0][2]
    else:
        for i in range(b):
            #optional - average for multiple circles (can happen when a gauge is at a slight angle)
            avg_x = avg_x + circles[i][0]
            avg_y = avg_y + circles[i][1]
            avg_r = avg_r + circles[i][2]
    avg_x = int(avg_x/(b))
    avg_y = int(avg_y/(b))
    avg_r = int(avg_r/(b))
    return avg_x, avg_y, avg_r

''' calculate distance between 2 points'''
def dist_2_pts(x1, y1, x2, y2):
    #print (np.sqrt((x2 - x1)**2 + (y2 - y1)**2))
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def draw_markers(path,x,y,r,scale_min_angle,scale_max_angle, final_angle, value, time):
    img = cv2.imread(path)
    img_output = img.copy()
    # draw center and circle
    cv2.circle(img_output, (x, y), r, (0, 255, 0), 2)  # draw circle
    cv2.circle(img_output, (x, y), 5, (0, 255, 0), thickness=-1)  # draw center of circle
    #draw min and max angles
    cv2.line(img_output, (x, y), (x - int(r * np.sin(scale_min_angle * 3.14 / 180)), y + int(r * np.cos(scale_min_angle * 3.14 / 180))), (0, 0, 255), 2)
    cv2.line(img_output, (x, y), (x - int(r * np.sin(scale_max_angle * 3.14 / 180)), y + int(r * np.cos(scale_max_angle * 3.14 / 180))), (0, 0, 255), 2)
    # draw final angle
    cv2.line(img_output, (x, y), (x - int(r * np.sin(final_angle * 3.14 / 180)), y + int(r * np.cos(final_angle * 3.14 / 180))), (0, 0, 255), 2)
    # draw value
    cv2.putText(img_output, "Value: " + str(value), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255),2)
    # draw calculation time
    cv2.putText(img_output, "Calculation time: " + str( np.round_(time,2) ) + " sec.", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    cv2.imwrite(os.path.dirname(path) + '/' + os.path.splitext(os.path.basename(path))[0] + '-markers.jpg', img_output)


def get_current_value(path):
    t1 = cv2.getTickCount()
    img = cv2.imread(path)
    '''
        This function should be run using a test image in order to calibrate the range available to the dial as well as the
        units.  It works by first finding the center point and radius of the gauge.  Then it draws lines at hard coded intervals
        (separation) in degrees.  It then prompts the user to enter position in degrees of the lowest possible value of the gauge,
        as well as the starting value (which is probably zero in most cases but it won't assume that).  It will then ask for the
        position in degrees of the largest possible value of the gauge. Finally, it will ask for the units.  This assumes that
        the gauge is linear (as most probably are).
        It will return the min value with angle in degrees (as a tuple), the max value with angle in degrees (as a tuple),
        and the units (as a string).
    '''
    #img = cv2.imread('gauge-%s.%s' %(gauge_number, file_type))
    #cv2.imshow("Image", img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    print("img shape = %s" %  (img.shape,))
    height, width = img.shape[:2]
    img_output = img.copy()

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #convert to gray scale
    #cv2.imwrite(os.path.dirname(path) + '/' + os.path.splitext(os.path.basename(path))[0] + '-gray.jpg', img_gray)

    # Set threshold
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    bright = np.average(v)
    threshold_bin = 0.24 * bright + 68
    img_black = np.zeros(img.shape[0:2], dtype=np.uint8)
    img_white = np.full(img.shape[0:2], 255, dtype=np.uint8)
    th, img_bin = cv2.threshold(img_gray, threshold_bin, 255, cv2.THRESH_BINARY_INV);
    cv2.imwrite(os.path.dirname(path) + '/' + os.path.splitext(os.path.basename(path))[0] + '-thresholded.jpg', img_bin)

    #detect circles
    #restricting the search from 35-48% of the possible radius gives fairly good results across different samples.  Remember that
    #these are pixel values which correspond to the possible radius search range.
    # cv2.HoughCircles(img, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) → circles

    circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, int(width * 0.3), np.array([]), 100, 50, int(width * 0.2), int(width * 0.5))
    #circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, int(height*0.1), np.array([]), 100, 50, int(height*0.35), int(height*0.45))
    ###circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 20, np.array([]), 100, 50, int(height * 0.35), int(height * 0.48))
    # average found circles, found it to be more accurate than trying to tune HoughCircles parameters to get just the right one
    print ("circles: %s" % (circles,))

    # ensure at least some circles were found
    if len(circles):
        a, b, c = circles.shape
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")

        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(img_output, (x, y), r, (0, 0, 255), 1)
            cv2.rectangle(img_output, (x - 1, y - 1), (x + 1, y + 1), (0, 0, 255), cv2.FILLED)
        #x,y,r = avg_circles(circles, b)
        # b=1 => выбираем первую окружность !!!"
        x = circles[0][0]
        y = circles[0][1]
        r = circles[0][2]
        # draw circle with center
        cv2.circle(img_output, (x, y), r, (0, 0, 255), 4)  # draw circle
        cv2.circle(img_output, (x, y), 2, (0, 0, 255), cv2.FILLED)  # draw center of circle

    # уточняем координаты центра манометра путем поиска окружности малого радиуса вблизи уже найденного центра
    x_min = x-150
    y_min = y-150
    img_crop = img_gray[y-150:y+150, x-150:x+150]
    #cv2.imwrite(os.path.dirname(path) + '/' + os.path.splitext(os.path.basename(path))[0] + '-crop.jpg', img_crop)
    small_circles = cv2.HoughCircles(img_crop, cv2.HOUGH_GRADIENT, 1, 100, np.array([]), 50, 20, 20,150)
    # ensure at least some small circles were found
    if len(small_circles):
        # convert the (x, y) coordinates and radius of the circles to integers
        small_circles = np.round(small_circles[0, :]).astype("int")

        # loop over the (x, y) coordinates and radius of the circles
        for (x0, y0, r0) in small_circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(img_output, (x_min + x0, y_min + y0), r0, (0, 255, 0), 1)
            cv2.rectangle(img_output, (x_min + x0 - 1, y_min + y0 - 1), (x_min + x0 + 1, y_min + y0 + 1), (0, 255, 0), cv2.FILLED)
        # b=1 => выбираем первую окружность !!!"
        x0 = small_circles[0][0]
        y0 = small_circles[0][1]
        r0 = small_circles[0][2]

        # draw small circle with center
        x0 = x_min + x0
        y0 = y_min + y0
        cv2.circle(img_output, (x0, y0), r0, (0, 255, 0), 4)  # draw circle
        cv2.circle(img_output, (x0, y0), 3, (0, 255, 0), cv2.FILLED)  # draw center of circle

    #cv2.imwrite(os.path.dirname(path) + '/' + os.path.splitext(os.path.basename(path))[0] + '-circles.jpg', img_output)



    ##################################
    ### ПОИСК СТРЕЛКИ
    ##################################
    ArrowL = 0
    ArrowAngle1 = 0
    Arrow_x1=0
    Arrow_y1 = 0
    Arrow_x2 = 0
    Arrow_y2 = 0
    img_Arrow = img.copy() # цветное изображение
    angle_step = 0.5  # step 1 degree
    sector_r1 = []
    sector_r2 = []
    for alfa in np.arange(0, 360, angle_step):
        temp = np.array( # формируем сектор
            [[x0 + 3 * np.cos(alfa * np.pi / 180) - r * np.sin(alfa * np.pi / 180),y0 + 3 * np.sin(alfa * np.pi / 180) + r * np.cos(alfa * np.pi / 180)],
             [x0 - 3 * np.cos(alfa * np.pi / 180) - r * np.sin(alfa * np.pi / 180),y0 - 3 * np.sin(alfa * np.pi / 180) + r * np.cos(alfa * np.pi / 180)],
             [x0 - 3 * np.cos(alfa * np.pi / 180), y0 - 3 * np.sin(alfa * np.pi / 180)],
             [x0 + 3 * np.cos(alfa * np.pi / 180), y0 + 3 * np.sin(alfa * np.pi / 180)]]
            # [[x0 - r * np.sin((alfa + angle_step) * np.pi / 180),y0 + r*np.cos((alfa + angle_step) * np.pi / 180)],
            #  [x0 - r * np.sin(alfa*np.pi/180),y0 + r*np.cos(alfa*np.pi/180)],
            #  [x0, y0]]
        ).reshape((-1, 1, 2)).astype(np.int32)
        img_temp = cv2.drawContours(img_black.copy(), [temp], 0, (255, 255, 255), thickness=cv2.FILLED) # black image with only 1 white sector
        img_intersection = cv2.bitwise_and(img_bin, img_temp.astype(np.uint8))
        lines = cv2.HoughLinesP(img_intersection, 1, np.pi/180, 127, np.array([]), minLineLength=20, maxLineGap=1)
        if (lines is not None and len(lines) > 0):  # если найдена хоть одна линия в пересечении
            for line in np.array(lines):
                x1, y1, x2, y2 = line[0]
                dist01 = dist_2_pts(x0, y0, x1, y1)  # dist from center of circle to point 1
                dist02 = dist_2_pts(x0, y0, x2, y2)  # dist from center of circle to point 2
                dist12 = dist_2_pts(x1, y1, x2, y2)  # line length
                # set diff1 to be the smaller (closest to the center) of the two), makes the math easier
                if (dist01 > dist02):
                    temp = dist01
                    dist01 = dist02
                    dist02 = temp
                    tempx = x1
                    tempy = y1
                    x1 = x2
                    y1 = y2
                    x2 = tempx
                    y2 = tempy
                # check if line is within an acceptable range
                if (dist02 - dist01 >= 0.90 * dist12):
                    if (dist12 > ArrowL):
                        ArrowL = dist12
                        ArrowAngle1 = alfa
                        Arrow_x1 = x0
                        Arrow_y1 = y0
                        Arrow_x2 = x2
                        Arrow_y2 = y2
    ArrowL = dist_2_pts(x0, y0, Arrow_x2, Arrow_y2)
    #cv2.line(img_Arrow, (Arrow_x1, Arrow_y1), (Arrow_x2, Arrow_y2), (0, 0, 255), 2)
    #take the arc tan of y/x to find the angle
    res = np.arctan(np.divide(np.abs(float(Arrow_x2 - x0)), np.abs(float(Arrow_y2 - y0))))

    res = np.rad2deg(res)
    if (Arrow_x2 - x0) < 0 and (Arrow_y2 - y0) > 0:  #in quadrant I
        angle_1 = res
    if (Arrow_x2 - x0) < 0 and (Arrow_y2 - y0) < 0:  #in quadrant II
        angle_1 = 180 - res
    if (Arrow_x2 - x0) > 0 and (Arrow_y2 - y0) < 0:  #in quadrant III
        angle_1 = 180 + res
    if (Arrow_x2 - x0) > 0 and (Arrow_y2 - y0) > 0:  #in quadrant IV
        angle_1 = 360 - res
    cv2.line(img_Arrow, (x0, y0),(x0 - int(ArrowL * np.sin(angle_1 * 3.14 / 180)), y0 + int(ArrowL * np.cos(angle_1 * 3.14 / 180))),(0, 0, 255), 1)

    # у точняем угол angle_1
    angle_step_presize = 0.1
    for i in np.arange(angle_1 - 5.0, angle_1 + 5.0, angle_step_presize):
        # строим дугу с радиусов 0,6*ArrowL
        r1 = img_bin[int(y0 + Arrow_r1 * ArrowL * np.cos(i * np.pi / 180)), int(x0 - Arrow_r1 * ArrowL * np.sin(i * np.pi / 180))]
        sector_r1.append(r1)
        img_Arrow[int(y0 + Arrow_r1 * ArrowL * np.cos(i * np.pi / 180)), int(x0 - Arrow_r1 * ArrowL * np.sin(i * np.pi / 180))] = [0,0,255]
        # строим дугу с радиусов 0,85*ArrowL
        r2 = img_bin[int(y0 + Arrow_r2 * ArrowL * np.cos(i * np.pi / 180)), int(x0 - Arrow_r2 * ArrowL * np.sin(i * np.pi / 180))]
        sector_r2.append(r2)
        img_Arrow[int(y0 + Arrow_r2 * ArrowL * np.cos(i * np.pi / 180)), int(x0 - Arrow_r2 * ArrowL * np.sin(i * np.pi / 180))] = [0,0,255]
    #plt.plot(sector_r1)
    #plt.plot(sector_r2)

    # поиск пиксела со  значением 0 в списке sector_r1
    maxL_r1,maxL_r2 = (0,0)
    maxstartL_r1, maxstartL_r2 = (0,0)
    maxstopL_r1, maxstopL_r2 = (0,0)
    curL_r1,curL_r2 = (0,0)
    curstartL_r1, curstartL_r2 = (0,0)
    curstopL_r1, curstopL_r2 = (0,0)
    for k in range(1, len(sector_r1), 1): # len(sector_r1) = len(sector_r2)
        if sector_r1[k] == 255 and sector_r1[k - 1] == 0: # начало области
            curstartL_r1 = k
            curL_r1 = 1
        if sector_r1[k] == 255 and sector_r1[k - 1] == 255:  # продолжение области
            curL_r1 = curL_r1 + 1
        if sector_r1[k] == 0 and sector_r1[k - 1] == 255:  # конец области
            curstopL_r1 = k - 1
            if (curL_r1 > maxL_r1):
                maxL_r1 = curL_r1
                maxstartL_r1 = curstartL_r1
                maxstopL_r1 = curstopL_r1
        if sector_r2[k] == 255 and sector_r2[k - 1] == 0: # начало области
            curstartL_r2 = k
            curL_r2 = 1
        if sector_r2[k] == 255 and sector_r2[k - 1] == 255:  # продолжение области
            curL_r2 = curL_r2 + 1
        if sector_r2[k] == 0 and sector_r2[k - 1] == 255:  # конец области
            curstopL_r2 = k - 1
            if (curL_r2 > maxL_r2):
                maxL_r2 = curL_r2
                maxstartL_r2 = curstartL_r2
                maxstopL_r2 = curstopL_r2

    angle_r1 = angle_1 - 5.0 + (maxstartL_r1 + maxstopL_r1) * 0.5 * angle_step_presize
    cv2.line(img_Arrow, (x0, y0), (x0 - int(ArrowL * np.sin(angle_r1 * 3.14 / 180)), y0 + int(ArrowL * np.cos(angle_r1 * 3.14 / 180))), (0, 255, 0), 1)

    angle_r2 = angle_1 - 5.0 + (maxstartL_r2 + maxstopL_r2) * 0.5 * angle_step_presize
    cv2.line(img_Arrow, (x0, y0),(x0 - int(ArrowL * np.sin(angle_r2 * 3.14 / 180)), y0 + int(ArrowL * np.cos(angle_r2 * 3.14 / 180))),(255, 0, 0), 1)

    if (np.abs(angle_r1 - angle_r2) >= 0.5):
        # angle_r1 is true
        if (np.abs(angle_r1 - angle_1) < 0.5):
            final_angle = 0.5 * (angle_r1 + angle_1)
        else:
            final_angle = angle_r1
    else:
        # angle_r2 is true
        if (np.abs(angle_r2 - angle_1) < 0.5):
            final_angle = 0.5 * (angle_r2 + angle_1)
        else:
            final_angle = angle_r2


    cv2.line(img_Arrow, (x0, y0),(x0 - int(r * np.sin(final_angle * 3.14 / 180)), y0 + int(r * np.cos(final_angle * 3.14 / 180))),(255, 0, 255), 1)
    cv2.imwrite(os.path.dirname(path) + '/' + os.path.splitext(os.path.basename(path))[0] + '-Arrow.jpg', img_Arrow)



    ##################################
    ### ПОИСК МИНИМУМА И МАКСИМУМА ШКАЛЫ
    ##################################
    # form sectors and triangles
    img_sectors = img_bin.copy()
    angle_step = 0.3
    sectors = []
    for alfa in np.arange(0,360,angle_step):
        temp1 = np.array([[x0-minCtrRadius*ArrowL*np.sin(alfa*np.pi/180),y0 + minCtrRadius*ArrowL*np.cos(alfa*np.pi/180)],
                          [x0-minCtrRadius*ArrowL*np.sin((alfa+angle_step)*np.pi/180),y0 + minCtrRadius*ArrowL*np.cos((alfa+angle_step)*np.pi/180)],
                          [x0-maxCtrRadius*ArrowL*np.sin((alfa+angle_step)*np.pi / 180),y0 + maxCtrRadius*ArrowL*np.cos((alfa+angle_step) * np.pi / 180)],
                          [x0-maxCtrRadius*ArrowL*np.sin(alfa*np.pi/180),y0 + maxCtrRadius*ArrowL*np.cos(alfa*np.pi/180)]]).reshape((-1, 1, 2)).astype(np.int32)
        # temp2 = np.array(
        #                 [[x0+3*np.cos(alfa*np.pi/180)-maxCtrRadius*r*np.sin(alfa*np.pi/180),y0+3*np.sin(alfa*np.pi/180)+maxCtrRadius*r*np.cos(alfa*np.pi/180)],
        #                 [x0-3*np.cos(alfa*np.pi/180)-maxCtrRadius*r*np.sin(alfa*np.pi/180),y0-3*np.sin(alfa*np.pi/180)+maxCtrRadius*r*np.cos(alfa*np.pi/180)],
        #                 [x0-3*np.cos(alfa*np.pi/180),y0-3*np.sin(alfa*np.pi/180)],
        #                 [x0+3*np.cos(alfa*np.pi/180),y0+3*np.sin(alfa*np.pi/180)]]
                        # [[x0, y0],
                        # [x0-maxCtrRadius*r*np.sin((alfa+angle_step)*np.pi / 180),y0+maxCtrRadius*r*np.cos((alfa+angle_step)*np.pi / 180)],
                        # [x0-maxCtrRadius*r*np.sin(alfa*np.pi/180),y0+maxCtrRadius*r*np.cos(alfa*np.pi/180)]]
                        # ).reshape((-1, 1, 2)).astype(np.int32)
        sectors.append(temp1)
        # triangles.append(temp2)
        cv2.drawContours(img_sectors, [temp1], 0, (255, 0, 0), 1)
        # cv2.drawContours(img_triangles, [temp2], 0, (255, 0, 0), 1)
    cv2.imwrite(os.path.dirname(path) + '/' + os.path.splitext(os.path.basename(path))[0] + '-sectors.jpg', img_sectors)
    # cv2.imwrite(os.path.dirname(path) + '/' + os.path.splitext(os.path.basename(path))[0] + '-triangles.jpg', img_triangles)

    # find lines in tor between radius minCtrRadius and maxCtrRadius
    # make tor mask
    img_circle1 = np.zeros(img.shape[0:2], dtype=np.uint8)
    img_circle2 = np.zeros(img.shape[0:2], dtype=np.uint8)

    cv2.circle(img_circle1, (x0, y0), int(ArrowL * maxCtrRadius) , (255, 255, 255), cv2.FILLED) # outer white circle
    cv2.circle(img_circle2, (x0, y0), int(ArrowL * minCtrRadius) , (255, 255, 255), cv2.FILLED) # inner white circle
    img_tor = cv2.bitwise_xor(img_circle1, img_circle2)
    img_tor = cv2.bitwise_and(img_bin, img_tor)
    #cv2.imwrite(os.path.dirname(path) + '/' + os.path.splitext(os.path.basename(path))[0] + '-tor.jpg', img_tor)


    ###########################
    ## поиск всех линий шкалы в торе и вычисление их средней длины
    ###########################
    minLineLength = 20
    maxLineGap = 1
    img_Lines = img.copy()
    lines = cv2.HoughLinesP(img_tor, 3, np.pi / 180, 127, np.array([]), minLineLength, maxLineGap)
    if (lines is None or len(lines) == 0):
        return
    for line in np.array(lines):
        x1, y1, x2, y2 = line[0]
        cv2.line(img_Lines, (x1, y1), (x2, y2), (0, 0, 255), 1)
    #cv2.imwrite(os.path.dirname(path) + '/' + os.path.splitext(os.path.basename(path))[0] + '-lines.jpg', img_Lines)
    scale_lines = []
    scale_lengths = []
    # поиск во всем массиве линий тех, которые относятся к линиям шкалы
    for i in range(0, len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            dist01 = dist_2_pts(x0, y0, x1, y1)  # dist from center of circle to point 1
            dist02 = dist_2_pts(x0, y0, x2, y2)  # dist from center of circle to point 2
            dist12 = dist_2_pts(x1, y1, x2, y2)  # line length
            # set diff1 to be the smaller (closest to the center) of the two), makes the math easier
            if (dist01 > dist02):
                temp = dist01
                dist01 = dist02
                dist02 = temp
            # check if line is within an acceptable range
            if (dist02 - dist01 >= 0.90 * dist12):
                scale_lines.append([x1, y1, x2, y2])
                scale_lengths.append(dist12)
    img_ScaleLines = img.copy()
    for i in range(0, len(scale_lines)):
        x1,y1,x2,y2 = scale_lines[i]
        cv2.line(img_ScaleLines, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #cv2.imwrite(os.path.dirname(path) + '/' + os.path.splitext(os.path.basename(path))[0] + '-ScaleLines.jpg', img_ScaleLines)
    ScaleLineLengthAv = np.average(scale_lengths)

    # перебор пересечений секторов и треугольников тора с бинарным исходным изображением
    scale_min_angle = 0
    scale_max_angle = 0
    img_All_intersections = np.zeros(img.shape[0:2], dtype=np.uint8)
    ListNumberOfSectorWP = []
    sector_min_number_of_wp = 80 # нижний порог количества белых пикселей в секторе
    sector_max_number_of_wp = 0  # максимальное значение белых пикселей в одном из секторов
    for j, s, in enumerate(sectors):
        if j>(angle_blank/angle_step) and j<((360-angle_blank)/angle_step):
            # ПОИСК минимума и максимума ШКАЛЫ
            img_temp = cv2.drawContours(img_black.copy(),sectors,j,(255,255,255),thickness=cv2.FILLED)  # black image with only 1 white sector
            #cv2.imwrite(os.path.dirname(path) + '/' + os.path.splitext(os.path.basename(path))[0] + '-temp.jpg',img_temp)
            img_intersection = cv2.bitwise_and(img_bin,img_temp.astype(np.uint8)) # find intersection between binary image and black image with only 1 white sector
            #cv2.imwrite(os.path.dirname(path) + '/' + os.path.splitext(os.path.basename(path))[0] + '-intersection.jpg', img_intersection)

            lines = cv2.HoughLinesP(img_intersection, 3, np.pi / 180, 127, np.array([]), minLineLength, maxLineGap)
            if (lines is not None and len(lines) > 0):  # если найдена хоть одна линия в пересечении
                #line_lengths = []
                maxL = 0
                for line in np.array(lines):
                    x1, y1, x2, y2 = line[0]
                    #line_lengths.append(dist_2_pts(x1, y1, x2, y2))
                    tempL = dist_2_pts(x1, y1, x2, y2)
                    if tempL > maxL:
                        maxL = tempL
                #LineLengthAv = np.average(line_lengths)
                if maxL > ScaleLineLengthAv: # if this sector contains scale lines
                    number_wp = np.sum(img_intersection == 255)
                    if number_wp > sector_min_number_of_wp: # check number of white pixels
                        img_All_intersections = cv2.bitwise_or(img_All_intersections, img_intersection)
                        ListNumberOfSectorWP.append(number_wp)
                        if number_wp > sector_max_number_of_wp:
                            sector_max_number_of_wp = number_wp
                else:
                    ListNumberOfSectorWP.append(0)
            else:
                ListNumberOfSectorWP.append(0)

        else:
            ListNumberOfSectorWP.append(0)

    #plt.plot(range(0, len(ListNumberOfSectorWP)), ListNumberOfSectorWP)
    #plt.plot(range(0, len(ListNumberOfSectorWP)), [0.5 * sector_max_number_of_wp] * len(ListNumberOfSectorWP))

    # вычисление мин и макс шкалы  = первого и последнего пика в массиве ListNumberOfSectorWP с амплитудой > 0,3*sector_max_number_of_wp
    for i in range(2,len(ListNumberOfSectorWP)-3,1):
        if  ListNumberOfSectorWP[i] >= ListNumberOfSectorWP[i-1] and ListNumberOfSectorWP[i] >= ListNumberOfSectorWP[i-2] and ListNumberOfSectorWP[i] >= ListNumberOfSectorWP[i+1] \
                and ListNumberOfSectorWP[i] > ListNumberOfSectorWP[i+2] and ListNumberOfSectorWP[i] > 0.5*sector_max_number_of_wp:
            if ListNumberOfSectorWP[i]-ListNumberOfSectorWP[i-1]<sector_min_number_of_wp and ListNumberOfSectorWP[i]-ListNumberOfSectorWP[i+1]<sector_min_number_of_wp: # если тупой пик слева и справа
                scale_min_angle = i * angle_step + (angle_step / 2)
            elif ListNumberOfSectorWP[i]-ListNumberOfSectorWP[i-1]<sector_min_number_of_wp and ListNumberOfSectorWP[i]-ListNumberOfSectorWP[i+1]>sector_min_number_of_wp: # если тупой пик слева
                scale_min_angle = i * angle_step
            elif ListNumberOfSectorWP[i]-ListNumberOfSectorWP[i-1]>sector_min_number_of_wp and ListNumberOfSectorWP[i]-ListNumberOfSectorWP[i+1]<sector_min_number_of_wp: # если тупой пик справа
                scale_min_angle = i * angle_step + angle_step
            else: # если острый пик
                scale_min_angle = i * angle_step + (angle_step / 2)
            break

    for i in range(len(ListNumberOfSectorWP)-3,2, -1):
        if ListNumberOfSectorWP[i] >= ListNumberOfSectorWP[i - 1] and ListNumberOfSectorWP[i] >= ListNumberOfSectorWP[i - 2] and ListNumberOfSectorWP[i] >= ListNumberOfSectorWP[i + 1] \
                and ListNumberOfSectorWP[i] >= ListNumberOfSectorWP[i + 2] and ListNumberOfSectorWP[i] > 0.5 * sector_max_number_of_wp:
            if ListNumberOfSectorWP[i]-ListNumberOfSectorWP[i-1]<sector_min_number_of_wp and ListNumberOfSectorWP[i]-ListNumberOfSectorWP[i+1]<sector_min_number_of_wp: # если тупой пик слева и справа
                scale_max_angle = i * angle_step + (angle_step / 2)
            elif ListNumberOfSectorWP[i]-ListNumberOfSectorWP[i-1]<sector_min_number_of_wp and ListNumberOfSectorWP[i]-ListNumberOfSectorWP[i+1]>sector_min_number_of_wp: # если тупой пик слева
                scale_max_angle = i * angle_step
            elif ListNumberOfSectorWP[i]-ListNumberOfSectorWP[i-1]>sector_min_number_of_wp and ListNumberOfSectorWP[i]-ListNumberOfSectorWP[i+1]<sector_min_number_of_wp: # если тупой пик справа
                scale_max_angle = i * angle_step + angle_step
            else: # если острый пик
                scale_max_angle = i * angle_step + angle_step
            break

    cv2.imwrite(os.path.dirname(path) + '/' + os.path.splitext(os.path.basename(path))[0] + '-AllIntersections.jpg', img_All_intersections)


    t2 = cv2.getTickCount()
    time = (t2 - t1) / cv2.getTickFrequency()
    print('Время работы алгоритма: ' + str(time) + " секунд")
    return x0, y0, r, scale_min_angle, scale_max_angle, final_angle, time

def main():
    path = 'dataset2/gauge-36.jpg'

    x0, y0, r, min_angle, max_angle, angle, time = get_current_value(path)
    # get user input on min, max, values, and units
    min_value = input('Min value: ')  # usually zero
    max_value = input('Max value: ')  # maximum reading of the gauge
    units = input('Enter units: ')
    degree_range = (float(max_angle) - float(min_angle))
    value_range = (float(max_value) - float(min_value))
    if degree_range != 0:
        val = np.round(  (( (angle - float(min_angle)) * value_range) / degree_range) + float(min_value), 4)
        draw_markers(path, x0, y0, r, min_angle, max_angle, angle, val, time)
        print ('Текущее измерение: '  + str(val) + " " + str(units))
    else:
        print('Ошибка определения диапазона шкалы!')
if __name__=='__main__':
    main()
   	
