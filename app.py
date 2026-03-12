from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/')
def hello():
    return jsonify(message='Hello, World!')

@app.route('/api/test', methods=['POST'])
def test():
    client_ip = request.headers.get('X-Forwarded-For', request.remote_addr).split(',')[0].strip()
    print(f'Request from: {client_ip}')
    return jsonify(message='server up!')

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=6000)
