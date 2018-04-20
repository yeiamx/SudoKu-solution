import cv2
import numpy as np

debug = True

def draw_str(dst, (x, y), s):
    """
    Draw a string with a dark contour
    """
    cv2.putText(dst, s, (x + 1, y + 1),
                cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0),
                thickness=2, lineType=cv2.CV_AA)
    cv2.putText(dst, s, (x, y),
                cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255),
                lineType=cv2.CV_AA)

def sort_grid_points(points):
    """
    Given a flat list of points (x, y), this function returns the list of
    points sorted from top to bottom, then groupwise from left to right.

    We assume that the points are nearly equidistant and have the form of a
    square.
    """
    w, _ = points.shape
    sqrt_w = int(np.sqrt(w))
    # sort by y
    points = points[np.argsort(points[:, 1])]
    # put the points in groups (rows)
    points = np.reshape(points, (sqrt_w, sqrt_w, 2))
    # sort rows by x
    points = np.vstack([row[np.argsort(row[:, 0])] for row in points])
    # undo shape transformation
    points = np.reshape(points, (w, 1, 2))
    return points

def cmp_height(x, y):
    """used for sorting by height"""
    _, _, _, hx = cv2.boundingRect(x)
    _, _, _, hy = cv2.boundingRect(y)
    return hy - hx


def cmp_width(x, y):
    """used for sorting by width"""
    _, _, wx, _ = cv2.boundingRect(x)
    _, _, wy, _ = cv2.boundingRect(y)
    return wy - wx

def process(frame):
    #
    # 1. preprocessing
    #
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        src=gray, maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY, blockSize=11, C=2)
    blurred = cv2.medianBlur(binary, ksize=3)
    print 1
    #
    # 2. try to find the sudoku
    #
    contours, _ = cv2.findContours(image=cv2.bitwise_not(blurred),
                                   mode=cv2.RETR_LIST,
                                   method=cv2.CHAIN_APPROX_SIMPLE)
    sudoku_area = 0
    sudoku_contour = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        if (0.7 < float(w) / h < 1.3     # aspect ratio
                and area > 150 * 150     # minimal area
                and area > sudoku_area   # biggest area on screen
                and area > .5 * w * h):  # fills bounding rect
            print 'area: '+ str(area)
            print 'w:'+str(w) + ' h: '+str(h)
            sudoku_area = area
            sudoku_contour = cnt
    print 2
    #
    # 3. separate sudoku from background
    #
    if sudoku_contour is not None:

        # approximate the contour with connected lines
        perimeter = cv2.arcLength(curve=sudoku_contour, closed=True)
        approx = cv2.approxPolyDP(curve=sudoku_contour,
                                  epsilon=0.1 * perimeter,
                                  closed=True)

        if len(approx) == 4:
            # successfully approximated
            # we now transform the sudoku to a fixed size 450x450
            # plus 50 pixel border and remove the background

            # create empty mask image
            mask = np.zeros(gray.shape, np.uint8)
            # fill a the sudoku-contour with white
            cv2.drawContours(mask, [sudoku_contour], 0, 255, -1)
            # invert the mask
            mask_inv = cv2.bitwise_not(mask)
            # the blurred picture is already thresholded so this step shows
            # only the black areas in the sudoku
            separated = cv2.bitwise_or(mask_inv, blurred)
            if debug:
                cv2.imshow('separated', separated)

            # create a perspective transformation matrix. "square" defines the
            # target dimensions (450x450). The image we warp "separated" in
            # has bigger dimensions than that (550x550) to assure that no
            # pixels are cut off accidentially on twisted images
            square = np.float32([[50, 50], [500, 50], [50, 500], [500, 500]])
            approx = np.float32([i[0] for i in approx])  # api needs conversion
            # sort the approx points to match the points defined in square
            approx = sort_grid_points(approx)

            m = cv2.getPerspectiveTransform(approx, square)
            transformed = cv2.warpPerspective(separated, m, (550, 550))
            if debug:
                cv2.imshow('transformed', transformed)
            print 3
            #
            # 4. get crossing points to determine grid buckling
            #

            # 4.1 vertical lines
            #

            # sobel x-axis
            sobel_x = cv2.Sobel(transformed, ddepth=-1, dx=1, dy=0)

            # closing x-axis
            kernel_x = np.array([[1]] * 20, dtype='uint8')  # vertical kernel
            dilated_x = cv2.dilate(sobel_x, kernel_x)
            closed_x = cv2.erode(dilated_x, kernel_x)
            _, threshed_x = cv2.threshold(closed_x, thresh=250, maxval=255,
                                          type=cv2.THRESH_BINARY)

            # generate mask for x
            contours, _ = cv2.findContours(image=threshed_x,
                                           mode=cv2.RETR_LIST,
                                           method=cv2.CHAIN_APPROX_SIMPLE)
            # sort contours by height
            sorted_contours = sorted(contours, cmp=cmp_height)

            # fill biggest 10 contours on mask (white)
            mask_x = np.zeros(transformed.shape, np.uint8)
            cv2.drawContours(mask_x, sorted_contours[:10], -1, 255, -1)
            if debug:
                cv2.imshow('mask_x', mask_x)

            # 4.2 horizontal lines
            #

            # this is essentially the same procedure as for the x-axis
            # sobel y-axis
            sobel_y = cv2.Sobel(transformed, ddepth=-1, dx=0, dy=1)

            # closing y-axis
            kernel_y = np.array([[[1]] * 20], dtype='uint8')  # horizontal krnl
            dilated_y = cv2.dilate(sobel_y, kernel_y)
            closed_y = cv2.erode(dilated_y, kernel_y)
            _, threshed_y = cv2.threshold(closed_y, 250, 255,
                                          cv2.THRESH_BINARY)

            # generate mask for y
            contours, _ = cv2.findContours(image=threshed_y,
                                           mode=cv2.RETR_LIST,
                                           method=cv2.CHAIN_APPROX_SIMPLE)
            sorted_contours = sorted(contours, cmp=cmp_width)

            # fill biggest 10 on mask
            mask_y = np.zeros(transformed.shape, np.uint8)
            cv2.drawContours(mask_y, sorted_contours[:10], -1, 255, -1)

            #
            # 4.3 close the grid
            #
            dilated_ver = cv2.dilate(mask_x, kernel_x)
            dilated_hor = cv2.dilate(mask_y, kernel_y)
            # now we have the single crossing points as well as the complete
            # grid
            grid = cv2.bitwise_or(dilated_hor, dilated_ver)
            crossing = cv2.bitwise_and(dilated_hor, dilated_ver)
            print 4
            #
            # 5. sort crossing points
            #
            contours, _ = cv2.findContours(image=crossing,
                                           mode=cv2.RETR_LIST,
                                           method=cv2.CHAIN_APPROX_SIMPLE)

            # a complete sudoku must have exactly 100 crossing points
            if len(contours) == 100:
                # take the center points of the bounding rects of the crossing
                # points. This should be precise enough, calculating the
                # moments is not necessary.
                crossing_points = np.empty(shape=(100, 2))
                for n, cnt in enumerate(contours):
                    x, y, w, h = cv2.boundingRect(cnt)
                    cx, cy = (x + .5 * w, y + .5 * h)
                    crossing_points[n] = [int(cx), int(cy)]
                sorted_cross_points = sort_grid_points(crossing_points)
                # show the numbers next to the points
                for n, p in enumerate(sorted_cross_points):
                    draw_str(grid, map(int, p[0]), str(n))
                if debug:
                    cv2.imshow('sorted grid', grid)
            print 5
    cv2.drawContours(frame, [sudoku_contour], 0, 255)
    cv2.imshow('Input', frame)

img = cv2.imread('test_pictures//sudoku44.jpg')
process(img)
cv2.waitKey(0)
