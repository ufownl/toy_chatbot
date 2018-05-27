import sys
import re
import argparse
import http.server
import urllib.parse
import mxnet as mx
from dataset import load_conversations, dataset_filter, make_vocab
from seq2seq_lstm import Seq2seqLSTM

parser = argparse.ArgumentParser(description='Start a test http server.')
parser.add_argument("--addr", help="set address of chatbot server (default: 0.0.0.0)", type=str, default="0.0.0.0")
parser.add_argument("--port", help="set port of chatbot server (default: 80)", type=int, default=80)
parser.add_argument("--device_id", help="select device that the model using (default: 0)", type=int, default=0)
parser.add_argument("--gpu", help="using gpu acceleration", action="store_true")
args = parser.parse_args()

if args.gpu:
    context = mx.gpu(args.device_id)
else:
    context = mx.cpu(args.device_id)
num_embed = 128
num_hidden = 1024
num_layers = 2
sequence_length = 16

print("Loading dataset...", flush=True)
dataset = dataset_filter(load_conversations("data/xiaohuangji50w_nofenci.conv"), sequence_length)
vocab = make_vocab(dataset)

print("Loading model...", flush=True)
model = Seq2seqLSTM(vocab.size(), num_embed, num_hidden, num_layers)
model.load_params("model/seq2seq_lstm.params", ctx=context)

print("Done.", flush=True)


class ChatbotHandler(http.server.BaseHTTPRequestHandler):
    _path_pattern = re.compile("^(/[^?\s]*)(\?\S*)?$")
    _param_pattern = re.compile("^([A-Za-z0-9_]+)=(.*)$")

    def do_GET(self):
        self._handle_request()
        sys.stdout.flush()
        sys.stderr.flush()

    def do_POST(self):
        self.do_GET()

    def _handle_request(self):
        m = self._path_pattern.match(self.path)
        if not m or m.group(0) != self.path:
            self.send_response(http.HTTPStatus.BAD_REQUEST)
            self.end_headers()
            return

        if m.group(1) == "/chatbot/say":
            params = {}
            if m.group(2):
                for param in urllib.parse.unquote(m.group(2)[1:]).split("&"):
                    kv = self._param_pattern.match(param)
                    if kv:
                        params[kv.group(1)] = kv.group(2)

            content = params["content"]
            if not content:
                self.send_response(http.HTTPStatus.BAD_REQUEST)
                self.end_headers()
                return

            print(args.device_id, "say:", content)
            source = [vocab.char2idx(ch) for ch in content]
            if sequence_length > len(source):
                source += [vocab.char2idx("<PAD>")] * (sequence_length - len(source))
            print(args.device_id, "tokenize:", source)
            source = mx.nd.array(source, ctx=context)
            hidden = model.begin_state(func=mx.nd.zeros, batch_size=1, ctx=context)
            hidden = model.encode(source.reshape((1, -1)).T, hidden)
            target = mx.nd.array([vocab.char2idx("<GO>")], ctx=context)
            reply = ""
            while True:
                output, hidden = model.decode(target.reshape((1, -1)).T, hidden)
                index = mx.nd.argmax(output, axis=1)
                if index[-1].asscalar() == vocab.char2idx("<EOS>"):
                    break;
                target = mx.nd.array([index[-1].asscalar()], ctx=context)
                reply += vocab.idx2char(index[-1].asscalar())
            print(args.device_id, "reply:", reply)

            self.send_response(http.HTTPStatus.OK)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET,POST")
            self.send_header("Access-Control-Allow-Headers", "Keep-Alive,User-Agent,Authorization,Content-Type")
            self.end_headers()
            self.wfile.write(reply.encode())
        else:
            self.send_response(http.HTTPStatus.NOT_FOUND)
            self.end_headers()
            return


httpd = http.server.HTTPServer((args.addr, args.port), ChatbotHandler)
httpd.serve_forever()
