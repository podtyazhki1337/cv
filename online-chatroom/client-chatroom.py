import socket

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
IP_address = "127.0.0.1"
Port = 5000
client.connect((IP_address, Port))

print(client.recv(2048).decode(), end="")
username = input()
client.send(username.encode())

while True:
    message = input(f"{username}> ")
    if message.lower() == "exit":
        break
    client.send(message.encode())
    response = client.recv(2048).decode()
    print(response)
client.close()