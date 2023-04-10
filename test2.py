from pynput import keyboard
import sys
import threading
import time

class KeyListener:
    def __init__(self):
        self.listener = None
        self.running = True

    def on_key_press(self, key):
        try:
            print(f'Key {key.char} pressed')
        except AttributeError:
            print(f'Special key {key} pressed')

    def on_key_release(self, key):
        print(f'Key {key} released')
        if key == keyboard.Key.esc:
            self.running = False
            return False

    def start(self):
        self.listener = keyboard.Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release)
        self.listener.start()


if __name__ == '__main__':
    key_listener = KeyListener()
    key_listener.start()

    # while key_listener.running:
    #     time.sleep(1)

    while True:
        time.sleep(1)
        print('ello')

    print('Exiting...')