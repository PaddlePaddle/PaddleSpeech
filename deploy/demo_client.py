"""Client-end for the ASR demo."""
import keyboard
import struct
import socket
import sys
import argparse
import pyaudio

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--host_ip",
    default="localhost",
    type=str,
    help="Server IP address. (default: %(default)s)")
parser.add_argument(
    "--host_port",
    default=8086,
    type=int,
    help="Server Port. (default: %(default)s)")
args = parser.parse_args()

is_recording = False
enable_trigger_record = True


def on_press_release(x):
    """Keyboard callback function."""
    global is_recording, enable_trigger_record
    press = keyboard.KeyboardEvent('down', 28, 'space')
    release = keyboard.KeyboardEvent('up', 28, 'space')
    if x.event_type == 'down' and x.name == press.name:
        if (not is_recording) and enable_trigger_record:
            sys.stdout.write("Start Recording ... ")
            sys.stdout.flush()
            is_recording = True
    if x.event_type == 'up' and x.name == release.name:
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
        sock.connect((args.host_ip, args.host_port))
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
    while (1):
        keyboard.hook(on_press_release)
        if keyboard.record('esc'):
            break

    # close up
    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == "__main__":
    main()
