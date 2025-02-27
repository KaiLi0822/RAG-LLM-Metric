from http.server import HTTPServer, BaseHTTPRequestHandler
import requests
import logging
logger = logging.getLogger(__name__)

class ProxyHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Remote server to forward requests to
        remote_server = "http://10.99.254.24:8000"
        
        # Forward the request to the remote server
        forwarded_url = remote_server + self.path
        response = requests.get(forwarded_url)
        
        # Send the response back to the client
        self.send_response(response.status_code)
        for header, value in response.headers.items():
            if header.lower() not in ['content-encoding', 'transfer-encoding', 'content-length']:
                self.send_header(header, value)
        self.end_headers()
        
        self.wfile.write(response.content)

    def do_POST(self):
        # Remote server to forward requests to
        remote_server = "http://10.99.254.24:8000"

        # Read and forward the request body
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        headers = dict(self.headers)

        response = requests.post(remote_server + self.path, data=body, headers=headers)
        
        # Send the response back to the client
        self.send_response(response.status_code)
        for header, value in response.headers.items():
            if header.lower() not in ['content-encoding', 'transfer-encoding', 'content-length']:
                self.send_header(header, value)
        self.end_headers()

        self.wfile.write(response.content)

# Start the proxy server
def run(server_class=HTTPServer, handler_class=ProxyHTTPRequestHandler, port=8000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    logger.info(f"Starting proxy server on port {port}")
    httpd.serve_forever()

if __name__ == "__main__":
    run()

