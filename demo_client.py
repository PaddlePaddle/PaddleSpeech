from pynput import keyboard
import struct
import socket
import sys
import pyaudio

HOST, PORT = "10.104.18.14", 8086

is_recording = False
enable_trigger_record = True


def on_press(key):
    """On-press keyboard callback function."""
    global is_recording, enable_trigger_record
    if key == keyboard.Key.space:
        if (not is_recording) and enable_trigger_record:
            sys.stdout.write("Start Recording ... ")
            sys.stdout.flush()
            is_recording = True


def on_release(key):
    """On-release keyboard callback function."""
    global is_recording, enable_trigger_record
    if key == keyboard.Key.esc:
        return False
    elif key == keyboard.Key.space:
        if is_recording == True:
            is_recording = False


data_list = []


def callback(in_data, frame_count, time_info, status):
    """Audio recorder's stream callback function."""
    global data_list, is_recording, enable_trigger_record
    if is_recording:
        data_list.append(in_data)
        enable_trigger_record = False
    elif len(data_list) > 0:
        # Connect to server and send data
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((HOST, PORT))
        sent = ''.join(data_list)
        sock.sendall(struct.pack('>i', len(sent)) + sent)
        print('Speech[length=%d] Sent.' % len(sent))
        # Receive data from the server and shut down
        received = sock.recv(1024)
        print "Recognition Results: {}".format(received)
        sock.close()
        data_list = []
    enable_trigger_record = True
    return (in_data, pyaudio.paContinue)


def main():
    # prepare audio recorder
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt32,
        channels=1,
        rate=16000,
        input=True,
        stream_callback=callback)
    stream.start_stream()

    # prepare keyboard listener
    with keyboard.Listener(
            on_press=on_press, on_release=on_release) as listener:
        listener.join()

    # close up
    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == "__main__":
    main()
