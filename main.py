import cv2
import numpy as np
import cv2 as cv2
import time
import turtle
import threading
cv2.startWindowThread()
import random
from functions import *
from sound import *

theme_sound()

## GAME PARAMETERS

SEARCH_LINE = True
TAKE_BG = True

PLAYERS = int(input('Please Enter The number of players: '))
TM_THRESH = 1
BBOX_THRESH = 0.005 # bb area factor
JACCARD_THRESH = 200
MIN_BB_AREA = 7000 # result from testing
Move_time = 5
Stop_time = 8
PLAY = True # while playing the game
STOP = False
GREEN = True
RED = True
Enter_loser_mode = False
my_timer = 3
i = 2  # frame counter
winer = None

#Take backround image
background, line_x1, line_x2 = set_background(TAKE_BG=TAKE_BG, SEARCH_LINE=SEARCH_LINE)
cap = cv2.VideoCapture(0)
p0_list, lk_params, old_gray, l1, l2 = mark_shoes(cap,PLAYERS)


#### Set up Screen
width = 600
height = 600
s = turtle.Screen()
s.setup(1200,650,startx=70,starty=30)

s.bgcolor("red")
s.title("RED light GREEN Light")

init_counter = 4
pen= turtle.Pen()
pen.hideturtle()
pen.write("Welcome to Squid games.",align="center",font=(None,30))
time.sleep(1)
pen.clear()
pen.write("The game will begin in...",align="center",font=(None,30))
time.sleep(1)
for timer in range(init_counter,-1,-1):
    pen.clear()
    pen.write(timer,align="center",font=(None,80))
    time.sleep(1)
pen.clear()
pen.color("black")
s.bgcolor("lightgreen")
stop_music()
pen.write("Game On ! ",align="center",font=(None,50))
time.sleep(1)
pen.clear()
s.setup(width,height,startx=700,starty=60)
####

def green_screen():
    global my_timer
    num = random.randint(4,6)
    my_timer = num
    for x in range(num):
        my_timer = my_timer - 1
        time.sleep(1)

def red_screen():
    global my_timer_red
    num2 = random.randint(8,9)
    my_timer_red = num2
    for x in range(num2):
        my_timer_red = my_timer_red - 1
        time.sleep(1)

def suspend():
    global suspend_var
    suspend_var = 'STOP'
    time.sleep(1)
    suspend_var = 'GO'

# initializing a moving player dictionary
moving_players = {}
for i in range(PLAYERS):
    name = "Player_" + str(i)
    moving_players[name] = {'Active':True,'BB': None, 'p0':p0_list[i], 'D': None, 'TM': 0} # {Active , Boundry_box, time_moved}

while (PLAY):

#######################################
############# Green MODE ##############
#######################################
    red_light_sound()
    Green_count = threading.Thread(target=green_screen)
    s.bgcolor("lightgreen")
    Green_count.start()
    # my_timer = 10
    my_timer_updated = 0
    timer_changed = 1
    game_over = 0
    while(my_timer > 0):
        if my_timer != my_timer_updated:
            timer_changed = 1
            my_timer_updated = my_timer
        else:
            timer_changed = 0
        ret, frame = cap.read()
        cv2.resize(frame, (640, 480))
        if ret == False:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for player in moving_players:
            player_dic = moving_players[player]
            if player_dic['Active'] == False:
                continue
            else:
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, player_dic['p0'], None, **lk_params)
                # good_new = p1[st == 1]
                # good_old = p1[st == 1]
                player_dic['p0'] = p1
                center = (int(p1[0][0][0]), int(p1[0][0][1]))
                cv2.circle(frame, center, 15, (0,0,255), 3)
                D = shortest_distance(list(center), line_x1, line_x2)
                player_dic['D'] = D
                cv2.putText(frame, str(D), (center[0], center[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                if D < 15:                          # winning - check the 15 th
                    stop_music()
                    winer = player
                    print(str(player) + ' won !')
                    player_dic['Active'] = False
                    game_over = True
                    if game_over:
                        break
            moving_players[player] = player_dic
        if game_over:
            break
        cv2.line(frame, line_x1, line_x2, (0,0,0), 3)             # manually add line (remove)
        cv2.imshow("Mode", frame)
        cv2.moveWindow('Mode', 20, 120)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        old_gray = frame_gray.copy()
        if timer_changed:
            pen.clear()
            pen.write(str(my_timer),align="center",font=(None,50))

    if game_over:
        s.bye()
        screen = cv2.imread('lost_bg.jpeg')
        screen = cv2.resize(screen, (1100, 600))
        str = winer + " won!"
        textsize = cv2.getTextSize(str, cv2.FONT_HERSHEY_SIMPLEX, 2, 4)[0]
        textX = int((screen.shape[1] - textsize[0]) / 2)
        textY = int((screen.shape[0] + textsize[1]) / 2)
        cv2.putText(screen, str, (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(screen, "click r to exit the game", (textX, textY+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, 30)
        cv2.imshow("Control", screen)
        cv2.moveWindow('Control', 80, 50)
        print("GAME OVER =[")
        if cv2.waitKey(0) & 0xFF == ord('r'):
            cv2.destroyAllWindows()
            break  ## breaks Main PLAY while loop



    pen.clear()
    pen.write(str(0),align="center",font=(None,50))
    time.sleep(0.2)
    # # time.sleep(0.4)    #to follow
    pen.clear()
    s.bgcolor("red")

#####################################
############ Pre RED MODE ###########
#####################################

    turning_sound()
    pen.write("STOP ! ", align="center", font=(None, 50))
    time.sleep(0.5)
    cv2.destroyAllWindows()
    threading.Thread(target=red_screen).start()
    ret, frame1 = cap.read()
    frame1 = cv2.resize(frame1, (640, 480))
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    # Take previous frame
    BB, _ = boundingbox(frame1, background, users=PLAYERS)
    if len(BB) == 0:
        game_over=True
        break

    moving_players = bb_to_player(moving_players, BB)
    ret, prev = cap.read()
    gray_prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    gray_prev = cv2.resize(gray_prev, (640, 480))
    prev_player_list = ROI(gray_prev,BB)

#####################################
############# RED MODE ##############
#####################################
    dist_dict = {}
    for key in moving_players:
        if moving_players[key]['Active'] == True:
            print(key + ' - distance from the line:', moving_players[key]['D'])
            dist_dict[key] = moving_players[key]['D']
        else:
            continue
    min_dist = min(dist_dict, key=dist_dict.get)
    print('The closest player to the finish line: ' + min_dist)
    print("================================================")
    scanning_sound()
    loser_list = []
    my_timer_red_updated = 0
    timer_red_changed = 1
    while(my_timer_red > 0):
        if my_timer_red != my_timer_red_updated:
            timer_red_changed = 1
            my_timer_red_updated = my_timer_red
        else:
            timer_red_changed = 0
        # Take Next frame
        ret, next = cap.read()
        gray_next = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)
        gray_next = cv2.resize(gray_next, (640, 480))
        next_player_list = ROI(gray_next, BB)
        i = 0
        Moved = []

        # Active player for loop ---------------------------------------------------------------------------------------
        for player in moving_players:
            player_dic = moving_players[player]
            if player_dic['Active'] == False:
                continue
            else:
                # Calculate opticl flow :
                flow = cv2.calcOpticalFlowFarneback(prev_player_list[i], next_player_list[i], None, 0.5, 3, 15, 3, 5, 1.2, 0)
                my_dic_BB = player_dic['BB']
                cv2.rectangle(next, (my_dic_BB[0], my_dic_BB[1] - 15), (my_dic_BB[0]+my_dic_BB[2], my_dic_BB[1]+my_dic_BB[3]), (0, 0, 200), 5)
                apadtive_threshhold = 280 + BBOX_THRESH*my_dic_BB[2]*my_dic_BB[3] ##my_dic_BB[2]*my_dic_BB[3] /BBOX_THRESH #  0 + 0.025*my_dic_BB[2]*my_dic_BB[3]       ###/ BBOX_THRESH # let the threshold change acording to the size of the roi. [Area / 30]
                Move = motion_detect(flow,apadtive_threshhold)
                if Move:
                    player_dic['TM'] += 1
                i += 1


        # End Active player for loop -----------------------------------------------------------------------------------
        prev_player_list = next_player_list


        for player in moving_players:
            player_dic = moving_players[player]
            if player_dic['Active'] == False:
                continue
            else:
                if player_dic['TM'] > TM_THRESH:
                    stop_music()
                    cv2.putText(next, str(player) + " Moved ", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,cv2.LINE_AA)
                    loser_list.append(player)
                    player_dic['Active'] = False

        cv2.imshow('Mode', next)
        cv2.moveWindow('Mode',20,120)
        cv2.waitKey(10)

        if loser_list != []:
            stop_music()
            Enter_loser_mode = True
            break

    cv2.destroyAllWindows()
    stop_music()
    if Enter_loser_mode:
        gun_shoot_sound()
        while (loser_list != []):
            loser_mode(loser_list)
            ret, frame = cap.read()
            frame = cv2.resize(frame, (640, 480))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            loser_list, game_over = post_loser_mode(moving_players, background, frame, 0.5)
            if game_over:
                break  # breaks the loser_list != [] while loop
            for loser in loser_list:
                loser_dict = moving_players[loser]
                loser_dict['Active'] = False
        Enter_loser_mode = False

        if game_over:
            s.bye()
            cv2.destroyAllWindows()
            screen = cv2.imread('lost_bg.jpeg')
            screen = cv2.resize(screen, (1100, 600))
            str = "GAME OVER =["
            textsize = cv2.getTextSize(str, cv2.FONT_HERSHEY_SIMPLEX, 2, 4)[0]
            textX = int((screen.shape[1] - textsize[0]) / 2)
            textY = int((screen.shape[0] + textsize[1]) / 2)
            cv2.putText(screen, str, (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.putText(screen, "click r to exit the game", (textX, textY+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, 30)
            cv2.imshow("Control", screen)
            cv2.moveWindow("Control",80,50)
            print("GAME OVER =[")
            if cv2.waitKey(0) & 0xFF == ord('r'):
                cv2.destroyAllWindows()
                break  ## breaks Main PLAY while loop

        cv2.putText(next, player + " Moved ", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("Mode", next)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    pen.clear()


print("end")
cv2.destroyAllWindows()
# When everything done, release the capture
cap.release()
# and release the output

# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)