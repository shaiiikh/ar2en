import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import pickle
from nltk.tokenize.treebank import TreebankWordDetokenizer
import pyarabic.araby as araby
import os

# Ensure all required files exist
REQUIRED_FILES = ["model.pt", "src_vocab.pkl", "trg_vocab.pkl"]
for file in REQUIRED_FILES:
    assert os.path.exists(file), f"❌ Error: {file} is missing. Please upload it."

# Load vocabularies
@st.cache_resource
def load_vocab():
    with open("src_vocab.pkl", "rb") as f:
        src_vocab = pickle.load(f)
    with open("trg_vocab.pkl", "rb") as f:
        trg_vocab = pickle.load(f)
    return src_vocab, trg_vocab

src_vocab, trg_vocab = load_vocab()

# Define Model Components
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        a = self.attention(hidden, encoder_outputs).unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs).permute(1, 0, 2)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        prediction = self.fc_out(torch.cat((output.squeeze(0), weighted.squeeze(0), embedded.squeeze(0)), dim=1))
        return prediction, hidden.squeeze(0)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.0):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)
        input = trg[0,:]
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output
            input = output.argmax(1)
        return outputs

# Load Model
device = torch.device("cpu")  # Force CPU for Streamlit deployment

@st.cache_resource
def load_model():
    INPUT_DIM, OUTPUT_DIM = len(src_vocab), len(trg_vocab)
    EMB_DIM, HID_DIM, DROPOUT = 512, 1024, 0.3

    attn = Attention(HID_DIM, HID_DIM)
    enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, HID_DIM, DROPOUT).to(device)
    dec = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, HID_DIM, DROPOUT, attn).to(device)

    model = Seq2Seq(enc, dec, device).to(device)
    model.load_state_dict(torch.load("model.pt", map_location=device))
    model.eval()
    return model

model = load_model()

# Preprocessing
def preprocess_ar(text):
    text = araby.strip_diacritics(text).strip()
    text = re.sub(r'[a-zA-Z]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Translation Function
def translate_sentence(model, sentence, src_vocab, trg_vocab, device):
    tokens = preprocess_ar(sentence).split()
    src_indexes = [src_vocab[token] if token in src_vocab else src_vocab["<unk>"] for token in tokens]
    src_tensor = torch.LongTensor([src_vocab["<sos>"]] + src_indexes + [src_vocab["<eos>"]]).unsqueeze(1).to(device)
    
    target_length = len(tokens) + 3
    target = torch.zeros(target_length, 1).to(torch.int64).to(device)

    with torch.no_grad():
        model.eval()
        output = model(src_tensor, target, 0)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)

    prediction = [torch.argmax(i).item() for i in output]
    tokens = trg_vocab.lookup_tokens(prediction)

    translated_sentence = TreebankWordDetokenizer().detokenize(tokens).replace('', "").replace('"', "").strip()

    return translated_sentence

# Streamlit UI
st.title("🌍 Arabic to English Translator")
input_text = st.text_area("✍️ Enter Arabic Sentence:", "", height=150)

if st.button("🎯 Translate"):
    if input_text.strip():
        translated_text = translate_sentence(model, input_text, src_vocab, trg_vocab, device)
        st.success(f"✅ Translation: {translated_text}")
    else:
        st.warning("⚠️ Please enter a valid Arabic sentence.")
