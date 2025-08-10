import cv2
import socket
import threading
import numpy as np
import struct

HOST = '0.0.0.0'
PORT = 9999

# Inicializa el detector HOG con el descriptor pre-entrenado para personas
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def client_thread(conn, addr, client_id):
    print(f"[SERVIDOR] Cliente conectado {addr}, ID: {client_id}")
    data_buffer = b''
    payload_size = struct.calcsize(">L")

    try:
        while True:
            while len(data_buffer) < payload_size:
                packet = conn.recv(4096)
                if not packet:
                    print(f"[SERVIDOR] Cliente {client_id} desconectado")
                    return
                data_buffer += packet

            packed_msg_size = data_buffer[:payload_size]
            data_buffer = data_buffer[payload_size:]
            msg_size = struct.unpack(">L", packed_msg_size)[0]

            while len(data_buffer) < msg_size:
                packet = conn.recv(4096)
                if not packet:
                    print(f"[SERVIDOR] Cliente {client_id} desconectado")
                    return
                data_buffer += packet

            frame_data = data_buffer[:msg_size]
            data_buffer = data_buffer[msg_size:]

            frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                print(f"[SERVIDOR] Cliente {client_id} frame vacío o corrupto")
                continue

            # Detección de personas
            rects, weights = hog.detectMultiScale(frame, winStride=(8,8), padding=(16,16), scale=1.05)
            
            # Dibujar rectángulos en las personas detectadas
            for (x, y, w, h) in rects:
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

            cv2.imshow(f'Cliente {client_id}', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[SERVIDOR] Cierre solicitado")
                break
    finally:
        conn.close()
        cv2.destroyWindow(f'Cliente {client_id}')
        print(f"[SERVIDOR] Cliente {client_id} desconectado")

def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(5)
    print(f"[SERVIDOR] Escuchando en {HOST}:{PORT}")

    client_id = 0
    try:
        while True:
            conn, addr = server.accept()
            client_id += 1
            threading.Thread(target=client_thread, args=(conn, addr, client_id), daemon=True).start()
    except KeyboardInterrupt:
        print("[SERVIDOR] Servidor detenido")
    finally:
        server.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
