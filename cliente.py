import cv2
import subprocess
import socket

SERVER_IP = '127.0.0.1'  # Cambia a IP del servidor real
SERVER_PORT = 9999

def main():
    # Abrimos la webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error al abrir webcam")
        return

    # Abrimos socket TCP hacia servidor
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((SERVER_IP, SERVER_PORT))

    # Configuramos FFmpeg para codificar en H264 y enviar por stdout
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}",
        '-r', str(int(cap.get(cv2.CAP_PROP_FPS)) or 30),
        '-i', '-',
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-tune', 'zerolatency',
        '-f', 'mpegts',
        'pipe:1'
    ]

    ffmpeg = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Enviamos frame raw a ffmpeg
            ffmpeg.stdin.write(frame.tobytes())

            # Leemos datos codificados y enviamos por socket al servidor
            data = ffmpeg.stdout.read(1316)  # Leer chunks tipo TS
            if not data:
                break
            sock.sendall(data)

    except BrokenPipeError:
        print("Conexión cerrada por servidor")
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        ffmpeg.stdin.close()
        ffmpeg.stdout.close()
        ffmpeg.terminate()
        sock.close()

if __name__ == '__main__':
    main()
