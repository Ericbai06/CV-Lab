from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import FTPServer
import os
import socket

def run_ftp_server():
    # 创建一个允许匿名访问的用户
    authorizer = DummyAuthorizer()
    
    # 当前目录作为FTP根目录，用户可以下载文件
    authorizer.add_anonymous(os.getcwd(), perm="elr")  # e:进入目录, l:列表, r:读取文件
    
    handler = FTPHandler
    handler.authorizer = authorizer
    
    # 尝试获取本地IP地址
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except:
        local_ip = "127.0.0.1"
    
    # 使用标准FTP端口21（如果被占用可以换其他端口）
    server = FTPServer((local_ip, 2121), handler)
    
    print(f"\nFTP服务器已启动在 ftp://{local_ip}:2121")
    print("可用以下命令下载文件：")
    print(f"curl ftp://{local_ip}:2121/文件名 -o 保存文件名")
    print(f"wget ftp://{local_ip}:2121/文件名")
    print("\n注意：这是一个匿名FTP服务器，任何人都可以访问")
    print("需要确保计算机可以从互联网访问，可能需要配置路由器端口转发")
    print("\n按 Ctrl+C 停止服务器\n")
    
    server.serve_forever()

if __name__ == "__main__":
    run_ftp_server() 