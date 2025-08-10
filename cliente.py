import cv2
import socket
import struct

SERVER_IP = '127.0.0.1'  # Cambia a la IP del servidor si es necesario
SERVER_PORT = 9999

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        return

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((SERVER_IP, SERVER_PORT))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error al capturar frame")
                break

            # Codificar frame a JPEG para enviar
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                print("Error al codificar frame")
                break

            data = buffer.tobytes()
            # Enviar tamaño del frame
            sock.sendall(struct.pack(">L", len(data)) + data)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        sock.close()

if __name__ == '__main__':
    main()
