import socket
import sys
from _thread import start_new_thread

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

if len(sys.argv) != 3:
    print("Correct usage: script, IP address, port number")
    exit()

IP_address = str(sys.argv[1])
Port = int(sys.argv[2])

server.bind((IP_address, Port))
server.listen(100)

list_of_clients = []
client_names = {}


def clientthread(conn, addr):
    conn.send("Welcome to this chatroom! Please enter your username:".encode())
    username = conn.recv(2048).decode().strip()
    if not username:
        username = addr[0]
    client_names[conn] = username

    welcome_msg = f"{username} ({addr[0]}) has joined the chat!"
    print(welcome_msg)
    broadcast(welcome_msg.encode(), conn)

    while True:
        try:
            message = conn.recv(2048).decode()
            if message:
                formatted_message = f"<{username} ({addr[0]})> {message}"
                print(formatted_message)
                broadcast(formatted_message.encode(), conn)
            else:
                remove(conn)
                break
        except Exception as e:
            print(f"Error with {username} ({addr[0]}): {e}")
            remove(conn)
            break


def broadcast(message, connection):
    for client in list_of_clients:
        if client != connection:
            try:
                client.send(message)
            except:
                client.close()
                remove(client)


def remove(connection):
    if connection in list_of_clients:
        username = client_names.get(connection, "Unknown")
        list_of_clients.remove(connection)
        del client_names[connection]
        disconnect_msg = f"{username} has left the chat."
        print(disconnect_msg)
        broadcast(disconnect_msg.encode(), connection)
        connection.close()


def broadcast_user_list():
    user_list = ", ".join(client_names.values()) or "No users"
    msg = f"Active users: {user_list}".encode()
    for client in list_of_clients:
        try:
            client.send(msg)
        except:
            remove(client)


print(f"Server running on {IP_address}:{Port}")
while True:
    try:
        conn, addr = server.accept()
        list_of_clients.append(conn)
        print(f"{addr[0]} connected")
        start_new_thread(clientthread, (conn, addr))
        broadcast_user_list()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        break
    except Exception as e:
        print(f"Server error: {e}")
        continue

for conn in list_of_clients:
    conn.close()
server.close()