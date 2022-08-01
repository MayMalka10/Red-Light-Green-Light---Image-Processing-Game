from pygame import mixer

mixer.init()

def theme_sound():
    mixer.music.load('sounds/theme.wav')
    mixer.music.play(-1)

def click_sound():
    mixer.music.load('sounds/click.wav')
    mixer.music.play()

def red_light_sound():
    mixer.music.load('sounds/red-light.wav')
    mixer.music.play()

def gun_shoot_sound():
    mixer.music.load('sounds/gun-shoot.wav')
    mixer.music.play()

def turning_sound():
    mixer.music.load('sounds/turning.wav')
    mixer.music.play()

def scanning_sound():
    mixer.music.load('sounds/scanning.wav')
    mixer.music.play()

def movement_detected_sound():
    mixer.music.load('sounds/beep.wav')
    mixer.music.play()

def stop_music():
    mixer.music.stop()