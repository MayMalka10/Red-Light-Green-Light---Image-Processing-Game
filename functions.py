import numpy as np
import cv2 as cv2
import time
import turtle
import threading
import math
import copy

def set_background(which_cam,TAKE_BG = False, SEARCH_LINE = True):
    cap = cv2.VideoCapture(which_cam)
    while (True & TAKE_BG):
        ret, back = cap.read()
        back = cv2.resize(back, (640, 480))
        cv2.putText(back, "Press Y To Save Clean Background Image", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2,
                    cv2.LINE_AA)
        if cv2.waitKey(1) & 0xFF == ord('y'):
            ret2, back2 = cap.read()
            # lab_image = cv2.cvtColor(back2, cv2.COLOR_BGR2LAB) #new
            # l_channel, a_channel, b_channel = cv2.split(lab_image) #new
            cv2.imwrite('clean_background.png', back2) #a_channel instead of 'back2'
            cv2.destroyAllWindows()
            break
        cv2.imshow('click Y to save image', back)
        cv2.moveWindow('click Y to save image', 360, 120)
    img = cv2.imread('clean_background.png')
    background = cv2.resize(img, (640, 480))
    line_found = False
    if SEARCH_LINE:
        line_found, line1, line2 = find_line(background)
    if not line_found:            #line havn't found
        line1 = (0, 450)
        line2 = (639, 450)

    # cv2.line(background, line1, line2, (0, 0, 0), 3)
    background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY) #new
    return background, line1, line2


def mark_shoes(cap,PLAYERS):
    # Take left feet locations:
    while True:
        ret, all_players_frame = cap.read()
        cv2.resize(all_players_frame, (640, 480))
        cv2.putText(all_players_frame, "Press s To Save Start Position", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2,
                    cv2.LINE_AA)
        # cv2.line(all_players_frame, (0, 450), (639, 450), (0, 0, 0), 3)  # manually add line
        cv2.imshow('All players', all_players_frame)
        cv2.moveWindow('All players', 360, 120)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            ret, all_players_frame = cap.read()
            cv2.resize(all_players_frame, (640, 480))
            cv2.destroyAllWindows()
            break

    points = []
    for i in range(PLAYERS):
        c, r, w, h = cv2.selectROI('Mark Shoe',all_players_frame)
        cv2.moveWindow('Mark Shoe', 360, 120)
        tmp_list = [c, r, w, h]
        points.append(tmp_list)
        cv2.destroyAllWindows()  # I don't know if it should be here

    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    old_gray = cv2.cvtColor(all_players_frame, cv2.COLOR_BGR2GRAY)

    p0_list = []
    for i in range(PLAYERS):
        [c, r, w, h] = points[i]
        p0 = [[((c + w / 2), (r + h / 2))]]
        p0 = np.float32(np.asarray(p0))
        p0_list.append(p0)

    # TODO - remove:
    x1 = 0
    y1 = 450
    x2 = 639
    y2 = 450
    l1 = [x1, y1]
    l2 = [x2, y2]
    # end remove
    return p0_list, lk_params, old_gray, l1, l2

def post_loser_mode(current_player_dict,backround,new_frame,PLAYERS,thresh=0.5):
    game_over = False
    active_players = 0
    for player in current_player_dict:
        player_dic = current_player_dict[player]
        if player_dic['Active'] == True:
            active_players += 1

    if active_players == 0:
        game_over = True
        return [] , game_over

    BB, _ = boundingbox(new_frame, backround, users=active_players)
    old_dict = copy.deepcopy(current_player_dict)
    next_players_dict = bb_to_player(current_player_dict, BB)
    loser_list = BB_overlap(old_dict,next_players_dict,PLAYERS,thresh)
    return loser_list ,game_over


def loser_mode(loser_list):
    str = ''
    for i in range(len(loser_list)):
        name = loser_list[i]
        if i == 0:
            str += name  # player_0
        else:
            str += (' & ' + name)  # & Player_x
    str += ' lost! '
    cv2.destroyAllWindows()
    screen = cv2.imread('lost_bg.jpeg')
    screen = cv2.resize(screen, (1100, 600))
    textsize = cv2.getTextSize(str, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
    textX = int((screen.shape[1] - textsize[0]) / 2)
    textY = int((screen.shape[0] + textsize[1]) / 2)
    cv2.putText(screen, str, (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3, 30)
    get_out_text = "Please get out.. press c to continue"
    cv2.putText(screen, get_out_text, (textX - 65, textY + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, 5)
    cv2.imshow("Control", screen)
    cv2.moveWindow("Control",80,50)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('c'):
            cv2.destroyAllWindows()
            break
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def winner_mode(moving_players, player):
    game_over = False
    active_players = 0
    for player in moving_players:
        player_dic = moving_players[player]
        if player_dic['Active'] == True:
            active_players += 1

    if active_players == 0:
        game_over = True

    str = player + " finished!! "
    cv2.destroyAllWindows()
    screen = cv2.imread('lost_bg.jpeg')
    screen = cv2.resize(screen, (1100, 700))
    textsize = cv2.getTextSize(str, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    textX = int((screen.shape[1] - textsize[0]) / 2)
    textY = int((screen.shape[0] + textsize[1]) / 2)
    cv2.putText(screen, str, (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, 30)
    get_out_text = "Please get out.. press c to continue"
    cv2.putText(screen, get_out_text, (textX - 30, textY + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, 5)
    cv2.imshow("Control", screen)
    cv2.moveWindow("Control",80,50)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('c'):
            cv2.destroyAllWindows()
            return(game_over)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return(game_over)

def find_line(img):
    edges = cv2.Canny(img, 50, 200)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)  # , minLineLength=100, maxLineGap=20)
    try:
        for r, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * r
            y0 = b * r
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            a = -(y2 - y1) / (x2 - x1)
            b = 1
            c = -(y1 - a * x1)
            line = True
            # assert ((x1 > 0) & (x2 > 0) & (y1 > 0) & (y2 > 0))
    except:
        line = False
        x1 = 0
        y1 = 0
        x2 = 0
        y2 = 0
        print("No line")
    return line, (x1, y1), (x2, y2)        #TODO - see what we need

def shortest_distance(point, p1, p2):
    m = (p1[1] - p2[1])/(p1[0] - p2[0])
    b = p1[1] - m*p1[0]
    DD = abs((m*point[0] - point[1] + b)/(math.sqrt((m**2) + 1)))
    return DD

def boundingbox(frame, backround_img, users=2, factor=0.1,MIN_BB_AREA=12000):
    diff = cv2.absdiff(frame, backround_img)  # subtract the frame from the original backround
    # gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) # Covert to gray scale
    blur = cv2.GaussianBlur(diff, (7, 7), 1)  # Gaussian filter
    mid = cv2.medianBlur(blur,7)
    thresh = cv2.threshold(mid, 50, 255, cv2.THRESH_BINARY)[1]  # Binary Thershold using otsu
    # cv2.imshow('tr',thresh)
    # cv2.waitKey(0)
    # Applying Morphological transformations
    kernel = np.ones((9, 9), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    kernel = np.ones((9, 9), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # cv2.imshow('tr', thresh)
    # cv2.waitKey(0)

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, None, None, None, 8, cv2.CV_32S)
    areas = stats[1:, cv2.CC_STAT_AREA]
    result = np.zeros((labels.shape), np.uint8)
    for i in range(0, nlabels - 1):
        if areas[i] >= 150:  # keep
            result[labels == i + 1] = 255
    # Finding the bounderys of the image
    kernel = np.array([[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]],
                      dtype=np.uint8)
    # img = data.astype(np.uint8)
    thresh = cv2.dilate(result, kernel, iterations=20)
    # cv2.imshow('tr', thresh)
    # cv2.waitKey(0)

    # Finding the bounderys of the image
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0]

    BBs = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        bb_area = w * h
        if bb_area < MIN_BB_AREA:
            continue
        # x = int(x - factor * x)
        # y = int(y - factor * y)
        # w = int(w + 2 * factor * x)
        # h = int(h + 2 * factor * y)
        tup_of_bb = ([x, y, w, h], bb_area)
        BBs.append(tup_of_bb)
    BBs = sorted(BBs, key=lambda tup: tup[1])
    Top_users_bbs = BBs[-users:]
    BBs = sorted(Top_users_bbs, key=lambda tup: tup[0][0])
    BBs = [elem[0] for elem in BBs]
    return BBs, thresh

def new_boundingbox(frame, backround_img, users=2, factor=0.05,MIN_BB_AREA=12000):
    diff = cv2.absdiff(frame, backround_img)  # subtract the frame from the original backround
    # gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) # Covert to gray scale
    blur = cv2.GaussianBlur(diff, (5, 5), 0.8)  # Gaussian filter
    thresh = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)[1]  # Binary Thershold using otsu
    # Applying Morphological transformations
    kernel = np.ones((7, 7), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, None, None, None, 8, cv2.CV_32S)
    #get CC_STAT_AREA component as stats[label, COLUMN]
    areas = stats[1:,cv2.CC_STAT_AREA]
    result = np.zeros((labels.shape), np.uint8)
    for i in range(0, nlabels - 1):
        if areas[i] >= 150:   #keep
            result[labels == i + 1] = 255
    # Finding the bounderys of the image
    kernel = np.array([[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0]], dtype=np.uint8)
    # img = data.astype(np.uint8)
    result = cv2.dilate(result, kernel, iterations=10)               # top and bottom only
    # cv2.imshow('BB', result)
    # cv2.waitKey(0)
    contours = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0]

    # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    ROI_number = 0
    h_max = 0
    BBs = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        bb_area = w * h
        if bb_area < MIN_BB_AREA:
            continue
        x = int(x - factor * x)
        y = int(y - factor * y)
        w = int(w + 2 * factor * x)
        h = int(h + 2 * factor * y)
        tup_of_bb = ([x, y, w, h], bb_area)
        BBs.append(tup_of_bb)
    BBs = sorted(BBs, key=lambda tup: tup[1])
    Top_users_bbs = BBs[-users:]
    BBs = sorted(Top_users_bbs, key=lambda tup: tup[0][0])
    BBs = [elem[0] for elem in BBs]
    BBs = check_overlap_bbs(BBs)
    return BBs, result


def ROI(img, BB):
    roi_arr = []
    for bb in BB:
        x, y, w, h = bb
        ROI = img[y:y + h, x:x + w]
        roi_arr.append(ROI)
    # if len(BB) == 1:
    #     return roi_arr[0]
    return roi_arr


def motion_detect(flow, thresh):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]

    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)

    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v * 4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    sub_bgr = np.linalg.norm(bgr)
    moved = True
    if sub_bgr < thresh:
        moved = False
    else:
        moved = True

    return moved


def bb_to_player(dic, bbs):
    try:
        idx = 0
        for player in dic:
            player_dic = dic[player]
            if player_dic['Active'] is True:
                player_dic['BB'] = bbs[idx]
                idx += 1
        return dic
    except AttributeError:
        pass
    except Exception as error:
        pass
    except UnboundLocalError:
        pass
    return dic


def calc_x_y_min(BB):
    min_x = BB[0]
    min_y = BB[1]
    max_x = BB[0] + BB[2]     #check
    max_y = BB[1] + BB[3]     #check
    return min_x, min_y, max_x, max_y

def BB_moved(old_BB, new_BB, th_factor):
    # th_factor - should be float between [0 1] closer to 1
    new_area = new_BB[2] * new_BB[3]
    old_area = old_BB[2] * old_BB[3]
    BB1_min_x, BB1_min_y, BB1_max_x, BB1_max_y = calc_x_y_min(old_BB)
    BB2_min_x, BB2_min_y, BB2_max_x, BB2_max_y = calc_x_y_min(new_BB)
    dx = min(BB1_max_x, BB2_max_x) - max(BB1_min_x, BB2_min_x)
    dy = min(BB1_max_y, BB2_max_y) - max(BB1_min_y, BB2_min_y)
    area = 0
    if (dx>=0) and (dy>=0):
        area = dx*dy
    union_area = new_area + old_area - area
    relative_area = area/union_area              # (0 <= relative_area <= 1) we want it to be close to 1
    if relative_area > th_factor:

        return False
    else:
        return True


def BB_overlap(old_players_dict, new_players_dict, PLAYERS, th_factor):
    losers = []
    for i in range(PLAYERS):
        name = "Player_" + str(i)
        player_new_dict = new_players_dict[name]
        if player_new_dict['Active']:                       #check if active
            player_old_dict = old_players_dict[name]
            if BB_moved(player_old_dict['BB'], player_new_dict['BB'], th_factor):           #compare BB
                losers.append(name)                                                         # add loser name
    return losers


def check_overlap_bbs(bbs):
    bbs_fixed = bbs.copy()
    length = len(bbs)
    for num in range(length-1):
        bb = bbs[num]
        width = bb[0] + bb[2]
        next_bb = bbs[num+1]
        next_bb_start = next_bb[0]
        if width > next_bb_start:
            mid = int(0.5*width - 0.5*next_bb_start)
            bb[2] = bb[2] - mid-1
            next_bb[0] = next_bb[0] + mid+1
            bbs_fixed[num] = bb
            bbs_fixed[num+1] = next_bb
        else:
            continue

    return bbs_fixed