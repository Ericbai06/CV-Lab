import http.server
import socketserver
import os
import urllib.parse
import socket

PORT = 8000

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        print(f"收到请求: {self.path}")
        # 处理中文文件名
        path = urllib.parse.unquote(self.path)
        if path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            
            # 获取当前目录下的所有文件
            files = [f for f in os.listdir('.') if os.path.isfile(f)]
            
            html = f"""
            <html>
            <head>
                <title>文件下载服务器</title>
                <meta charset="utf-8">
                <style>
                    body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
                    h1 {{ color: #333; }}
                    .file-list {{ margin-top: 20px; }}
                    .file-item {{ padding: 10px; border-bottom: 1px solid #eee; }}
                    .file-name {{ font-weight: bold; }}
                    .download-cmd {{ background-color: #f5f5f5; padding: 10px; margin-top: 5px; font-family: monospace; }}
                </style>
            </head>
            <body>
                <h1>文件下载服务器</h1>
                <p>可用以下命令下载文件：</p>
                <div class="file-list">
            """
            
            # 获取本地IP地址
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                s.close()
            except:
                local_ip = "您的IP地址"
            
            for file in files:
                encoded_filename = urllib.parse.quote(file)
                size_mb = os.path.getsize(file) / (1024 * 1024)
                html += f"""
                <div class="file-item">
                    <div class="file-name">{file} ({size_mb:.2f} MB)</div>
                    <div class="download-cmd">curl http://{local_ip}:{PORT}/{encoded_filename} -o {file}</div>
                    <div class="download-cmd">wget http://{local_ip}:{PORT}/{encoded_filename}</div>
                </div>
                """
            
            html += """
                </div>
                <p>注意：需要确保您的计算机可以从互联网访问，可能需要配置路由器端口转发。</p>
            </body>
            </html>
            """
            
            self.wfile.write(html.encode('utf-8'))
        else:
            return http.server.SimpleHTTPRequestHandler.do_GET(self)

def run_server():
    handler = CustomHTTPRequestHandler
    
    # 尝试获取本地IP地址
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except:
        local_ip = "127.0.0.1"
    
    print(f"\n文件下载服务器启动在 http://{local_ip}:{PORT}/")
    print("其他人可以使用以下命令下载文件：")
    print(f"curl http://{local_ip}:{PORT}/文件名 -o 保存的文件名")
    print(f"wget http://{local_ip}:{PORT}/文件名")
    print("\n按 Ctrl+C 停止服务器\n")
    
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n服务器已停止")

if __name__ == "__main__":
    run_server() 