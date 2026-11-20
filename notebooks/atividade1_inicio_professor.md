# Atividade 1 — Notebook inicial: tokenizador e dataloaders

*Material do professor, enviado em notebook e renderizado aqui em Markdown. O conteúdo das células está na ordem original; as saídas de execução aparecem em blocos próprios.*

## Índice

O notebook tem 29 células, 17 delas de código.

- 1. Instalação e configuração
- 2. Localizar o corpus
- 3. O tokenizador
- 4. Pré-tokenização (uma vez só)
- 5. Dataset: a janela deslizante do livro, sem estourar a memória
- 6. DataLoaders
- 7. Conferência com o `GPTModel` do livro
- 8. Checkpoint e retomada
- 9. Bits por byte na validação
- 10. Próximos passos

---

Este notebook deixa pronta a parte de infraestrutura da Atividade 1, seguindo a estrutura do capítulo 2 do livro *Build a Large Language Model (From Scratch)* e do repositório [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch):

1. carregar o **tokenizador de 16k** do corpus e deixá-lo com a mesma interface do `tiktoken` usada no livro;
2. **pré-tokenizar** o corpus uma única vez e salvar em disco;
3. criar o **Dataset** e os **DataLoaders** de treino e validação (equivalentes ao `GPTDatasetV1` e ao `create_dataloader_v1` do livro);
4. conferir que o `GPTModel` do livro roda com esses dados;
5. funções auxiliares de **checkpoint e retomada** e de **bits por byte**.

O treino em si, o loop do Apêndice D e as modificações são com vocês.

> **GPU:** no Kaggle, ative *Settings › Accelerator › GPU T4* e *Internet on*. No Colab, *Ambiente de execução › Alterar o tipo de ambiente › T4 GPU*.

## 1. Instalação e configuração

**Código 1**

```python
# tokenizers: biblioteca do tokenizador BPE
# llms-from-scratch: o código do livro como pacote (--no-deps evita instalar TensorFlow e outras dependências pesadas)
!pip install -q tokenizers
!pip install -q --no-deps llms-from-scratch
```

**Código 2**

```python
import os, math, time, glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo:", device, "|", torch.cuda.get_device_name(0) if device.type == "cuda" else "sem GPU")
```

Saída:

```text
Dispositivo: cpu | sem GPU
```

## 2. Localizar o corpus

O pacote `corpus_ufcg_introducoes_alunos.zip` traz `train.txt`, `val.txt`, os `.parquet` com metadados e a pasta `tokenizer_16k/`.

- **Colab:** preencha `ID_DRIVE` com o ID do link do Google Drive que o professor compartilhou.
- **Local:** descompacte numa pasta `corpus/` ao lado do notebook.

**Código 3**

```python
from google.colab import drive
import os
import glob
import subprocess
import zipfile
import shutil

# 1. O estudante deve montar o seu próprio Google Drive para gravar os artefatos
drive.mount('/content/drive')

# 2. ID do Google Drive do Professor (de onde o corpus será baixado)
ID_DRIVE_PROFESSOR = "1bwMhXvZK0QXUrbeXQLq3cmyGuQGfZ2Ch"

def localizar_corpus():
    # Kaggle: procura train.txt dentro de /kaggle/input
    achados = glob.glob("/kaggle/input/**/train.txt", recursive=True)
    if achados:
        return os.path.dirname(achados[0])

    # Local ou já descompactado
    for pasta in ["corpus", "/content/corpus"]:
        if os.path.exists(os.path.join(pasta, "train.txt")):
            return pasta

    # Colab: baixa usando o ID do drive compartilhado pelo professor
    if ID_DRIVE_PROFESSOR:
        # Limpa resíduos anteriores caso o download ou extração tenham falhado no meio
        if os.path.exists("corpus.zip"):
            os.remove("corpus.zip")
        if os.path.exists("corpus"):
            shutil.rmtree("corpus")

        print("Instalando gdown...")
        subprocess.run("pip install -q gdown", shell=True, check=True)

        print("Baixando corpus do Google Drive...")
        subprocess.run(f"gdown --id {ID_DRIVE_PROFESSOR} -O corpus.zip", shell=True, check=True)

        print("Descompactando corpus...")
        with zipfile.ZipFile("corpus.zip", "r") as zip_ref:
            zip_ref.extractall("corpus")

        if os.path.exists(os.path.join("corpus", "train.txt")):
            return "corpus"

    raise FileNotFoundError("Corpus não encontrado. Verifique se o ID_DRIVE_PROFESSOR é válido e se o arquivo corpus.zip foi baixado corretamente.")

DATA_DIR = localizar_corpus()

# 3. Caminho no Drive do estudante onde serão gravados os checkpoints e artefatos
WORK_DIR = "/content/drive/MyDrive/Atividade1_checkpoints"  # Alunos podem alterar esta pasta se desejarem
os.makedirs(WORK_DIR, exist_ok=True)

print("\nCorpus baixado e extraído com sucesso em:", DATA_DIR)
print(sorted(os.listdir(DATA_DIR)))
print("Os checkpoints e artefatos de saída serão salvos em:", WORK_DIR)
```

Saída:

```text
Mounted at /content/drive
Instalando gdown...
Baixando corpus do Google Drive...
Descompactando corpus...

Corpus baixado e extraído com sucesso em: corpus
['README.md', 'tokenizer_16k', 'train.parquet', 'train.txt', 'val.parquet', 'val.txt']
Os checkpoints e artefatos de saída serão salvos em: /content/drive/MyDrive/Atividade1_checkpoints
```

## 3. O tokenizador

O tokenizador é um BPE em nível de byte, com vocabulário de 16.000, treinado só no `train.txt`. O token especial `<|endoftext|>` (id 0) separa os documentos.

O código do livro chama `tokenizer.encode(texto, allowed_special={...})` e `tokenizer.decode(ids)`, como no `tiktoken`. A classe abaixo dá essa mesma interface ao nosso tokenizador, para o resto do código do livro funcionar sem mudanças.

**Código 4**

```python
from tokenizers import Tokenizer

class TokenizadorUFCG:
    """Envolve o tokenizador da Hugging Face com a interface do tiktoken usada no livro."""
    def __init__(self, caminho):
        self.tok = Tokenizer.from_file(caminho)
        self.eot_id = self.tok.token_to_id("<|endoftext|>")
        self.n_vocab = self.tok.get_vocab_size()

    def encode(self, texto, allowed_special=None):
        # <|endoftext|> já é reconhecido como token especial pelo tokenizer.json
        return self.tok.encode(texto).ids

    def encode_batch(self, textos):
        return [e.ids for e in self.tok.encode_batch(textos)]

    def decode(self, ids):
        return self.tok.decode(list(ids), skip_special_tokens=False)

tokenizer = TokenizadorUFCG(os.path.join(DATA_DIR, "tokenizer_16k", "tokenizer.json"))
print("Vocabulário:", tokenizer.n_vocab, "| id de <|endoftext|>:", tokenizer.eot_id)
```

Saída:

```text
Vocabulário: 16000 | id de <|endoftext|>: 0
```

**Código 5**

```python
exemplo = "Este trabalho tem como objetivo analisá-lo à luz da literatura."
ids = tokenizer.encode(exemplo)
print(ids)
print([tokenizer.decode([i]) for i in ids])   # cada token decodificado separadamente
print(tokenizer.decode(ids))                  # volta ao texto original
```

Saída:

```text
[2449, 556, 602, 361, 873, 991, 300, 13, 419, 442, 3909, 313, 1494, 14]
['Este', ' trabalho', ' tem', ' como', ' objetivo', ' analis', 'á', '-', 'lo', ' à', ' luz', ' da', ' literatura', '.']
Este trabalho tem como objetivo analisá-lo à luz da literatura.
```

Repare que palavras frequentes no corpus (como *trabalho* e *objetivo*) viram um token só, e palavras menos comuns são quebradas em pedaços. A **fertilidade** é o número médio de tokens por palavra; com este tokenizador, fica em torno de 1,37 no treino.

**Código 6**

```python
with open(os.path.join(DATA_DIR, "val.txt"), encoding="utf-8") as f:
    amostra = f.read(200_000)
n_palavras = len(amostra.split())
n_tokens = len(tokenizer.encode(amostra))
print(f"Fertilidade numa amostra da validação: {n_tokens / n_palavras:.2f} tokens por palavra")
```

Saída:

```text
Fertilidade numa amostra da validação: 1.42 tokens por palavra
```

## 4. Pré-tokenização (uma vez só)

O livro tokeniza o texto inteiro dentro do `GPTDatasetV1`. Com ~27 milhões de tokens, vale tokenizar **uma vez**, salvar os ids num arquivo `.npy` e reaproveitar nas próximas sessões. Isso economiza minutos de GPU a cada sessão do Kaggle.

Cada documento é codificado separadamente (em lote, o que é bem mais rápido) e recebe o `<|endoftext|>` no fim. Como o vocabulário tem 16.000 tokens, os ids cabem em `uint16`, metade da memória de `int32`.

**Código 7**

```python
SEP = "\n<|endoftext|>\n"   # separador de documentos nos arquivos .txt

def pre_tokenizar(nome, tamanho_lote=1000):
    destino = os.path.join(WORK_DIR, f"{nome}_ids.npy")
    if os.path.exists(destino):
        return np.load(destino)
    with open(os.path.join(DATA_DIR, f"{nome}.txt"), encoding="utf-8") as f:
        docs = [d for d in f.read().split(SEP) if d.strip()]
    pedacos = []
    for i in range(0, len(docs), tamanho_lote):
        for ids in tokenizer.encode_batch(docs[i:i + tamanho_lote]):
            pedacos.append(np.asarray(ids + [tokenizer.eot_id], dtype=np.uint16))
    tokens = np.concatenate(pedacos)
    np.save(destino, tokens)
    return tokens

t0 = time.time()
train_ids = pre_tokenizar("train")
val_ids = pre_tokenizar("val")
print(f"Treino: {len(train_ids):,} tokens | Validação: {len(val_ids):,} tokens | {time.time() - t0:.0f} s")
```

Saída:

```text
Treino: 27,451,491 tokens | Validação: 1,576,142 tokens | 3 s
```

## 5. Dataset: a janela deslizante do livro, sem estourar a memória

O `GPTDatasetV1` do capítulo 2 corta o texto em janelas de `max_length` tokens; o alvo de cada janela é a mesma janela deslocada uma posição (prever o próximo token). O `stride` define de quanto em quanto começa uma nova janela.

O original guarda **um tensor por janela** já no `__init__`. Para 27M tokens isso ocupa vários GB. A versão abaixo produz **exatamente as mesmas janelas**, mas recorta cada uma do array de ids só quando é pedida.

**Código 8**

```python
class GPTDatasetV1(Dataset):
    """Cópia do GPTDatasetV1 do livro (cap. 2), usada só para conferência num trecho pequeno."""
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids, self.target_ids = [], []
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(torch.tensor(token_ids[i:i + max_length]))
            self.target_ids.append(torch.tensor(token_ids[i + 1:i + max_length + 1]))
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


class GPTDatasetTokens(Dataset):
    """Mesmas janelas do GPTDatasetV1, recortadas sob demanda de um array de ids pré-tokenizado."""
    def __init__(self, token_ids, max_length, stride):
        self.ids = torch.from_numpy(token_ids.astype(np.int64))
        self.max_length, self.stride = max_length, stride
        self.n = len(range(0, len(token_ids) - max_length, stride))
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        i = idx * self.stride
        x = self.ids[i:i + self.max_length]
        y = self.ids[i + 1:i + self.max_length + 1]
        return x, y
```

**Código 9**

```python
# Conferência: num trecho pequeno, as duas classes geram as mesmas janelas
trecho = tokenizer.decode(val_ids[:5000])
ds_livro = GPTDatasetV1(trecho, tokenizer, max_length=64, stride=32)
ds_nosso = GPTDatasetTokens(np.asarray(tokenizer.encode(trecho), dtype=np.uint16), max_length=64, stride=32)
assert len(ds_livro) == len(ds_nosso)
assert all(torch.equal(ds_livro[k][0], ds_nosso[k][0]) and torch.equal(ds_livro[k][1], ds_nosso[k][1])
           for k in range(len(ds_livro)))
print(f"OK: {len(ds_livro)} janelas idênticas")
```

Saída:

```text
OK: 155 janelas idênticas
```

## 6. DataLoaders

`criar_dataloader` segue o `create_dataloader_v1` do livro, mas recebe os ids já tokenizados. Duas diferenças:

- `pin_memory=True` acelera a cópia dos lotes para a GPU.
- `seed`: o embaralhamento passa a ser reprodutível, o que permite **retomar um treino interrompido** do mesmo ponto (seção 8).

No treino, use `stride = max_length` (janelas sem sobreposição): cada token é visto uma vez por época, que é o orçamento da atividade.

**Código 10**

```python
def criar_dataloader(token_ids, batch_size=32, max_length=256, stride=256,
                     shuffle=True, drop_last=True, num_workers=2, seed=123):
    dataset = GPTDatasetTokens(token_ids, max_length, stride)
    gerador = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                      num_workers=num_workers, pin_memory=(device.type == "cuda"), generator=gerador)

GPT_CONFIG = {
    "vocab_size": 16000,     # tokenizador de referência
    "context_length": 256,   # contexto da configuração base da atividade
    "emb_dim": 256,
    "n_heads": 4,
    "n_layers": 4,
    "drop_rate": 0.1,
    "qkv_bias": False,
}
BATCH_SIZE = 128  # ajuste conforme a memória da GPU

train_loader = criar_dataloader(train_ids, BATCH_SIZE, GPT_CONFIG["context_length"],
                                GPT_CONFIG["context_length"], shuffle=True, drop_last=True)
val_loader = criar_dataloader(val_ids, BATCH_SIZE, GPT_CONFIG["context_length"],
                              GPT_CONFIG["context_length"], shuffle=False, drop_last=False)

tokens_por_lote = BATCH_SIZE * GPT_CONFIG["context_length"]
print(f"Treino: {len(train_loader):,} lotes por época ({len(train_loader) * tokens_por_lote:,} tokens)")
print(f"Validação: {len(val_loader):,} lotes")
```

Saída:

```text
Treino: 1,675 lotes por época (27,443,200 tokens)
Validação: 97 lotes
```

**Código 11**

```python
x, y = next(iter(train_loader))
print("Entrada:", x.shape, x.dtype, "| Alvo:", y.shape)
print("O alvo é a entrada deslocada uma posição:", torch.equal(x[0, 1:], y[0, :-1]))
print("\nInício da primeira janela do lote:\n", tokenizer.decode(x[0, :60].tolist()))
```

Saída:

```text
Entrada: torch.Size([32, 256]) torch.int64 | Alvo: torch.Size([32, 256])
O alvo é a entrada deslocada uma posição: True

Início da primeira janela do lote:
  a geometria de subvariedades em particular, de hipersuperfícies imersas em espaços Riemannianos. Neste sentido, Montiel estudou as hipersuperfícies compactas com curvatura média constante imersas em um produto warped do tipo , onde a curvatura de Ricci
```

## 7. Conferência com o `GPTModel` do livro

Antes de treinar, vale conferir três coisas:

- o modelo aceita os lotes sem erro de formato;
- a contagem de parâmetros está perto dos ~11M da configuração base;
- a **loss inicial** está perto de ln(16000) ≈ 9,68. Um modelo aleatório acerta o próximo token com probabilidade ~1/16000; uma loss muito diferente disso indica algum erro nos dados ou no modelo.

**Código 12**

```python
from llms_from_scratch.ch04 import GPTModel, generate_text_simple

model = GPTModel(GPT_CONFIG).to(device)
total = sum(p.numel() for p in model.parameters())
cabeca = model.out_head.weight.numel()
print(f"Parâmetros: {total:,} ({total - cabeca:,} se a cabeça de saída compartilhasse os pesos do embedding)")

def calc_loss_batch(input_batch, target_batch, model, device):
    # igual ao calc_loss_batch do livro (cap. 5)
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())

model.eval()
with torch.no_grad():
    loss = calc_loss_batch(x, y, model, device)
print(f"Loss inicial: {loss.item():.2f} (esperado ≈ {math.log(GPT_CONFIG['vocab_size']):.2f})")
```

Saída:

```text
Parâmetros: 11,414,016 (7,318,016 se a cabeça de saída compartilhasse os pesos do embedding)
Loss inicial: 9.86 (esperado ≈ 9.68)
```

**Código 13**

```python
# Geração com o modelo ainda não treinado: deve sair texto sem sentido
contexto = torch.tensor([tokenizer.encode("Este trabalho tem como objetivo")], device=device)
saida = generate_text_simple(model, contexto, max_new_tokens=15, context_size=GPT_CONFIG["context_length"])
print(tokenizer.decode(saida[0].tolist()))
```

Saída:

```text
Este trabalho tem como objetivo1990 francpenmpre interessante curso gengi superação mercados poderes computadores proposições línguas currí contrário
```

**Código 14**

```python
from llms_from_scratch.ch05 import train_model_simple
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Este trabalho tem como objetivo", tokenizer=tokenizer
)
```

Saída:

```text
Ep 1 (Step 000000): Train loss 9.714, Val loss 9.719
Ep 1 (Step 000005): Train loss 8.905, Val loss 8.929
Ep 1 (Step 000010): Train loss 8.372, Val loss 8.403
Ep 1 (Step 000015): Train loss 7.974, Val loss 8.032
Ep 1 (Step 000020): Train loss 7.663, Val loss 7.746
Ep 1 (Step 000025): Train loss 7.512, Val loss 7.554
Ep 1 (Step 000030): Train loss 7.349, Val loss 7.439
```

> **Erro na execução:** KeyboardInterrupt: . O traceback foi omitido; a célula foi interrompida antes de terminar.

## 8. Checkpoint e retomada

Sessões gratuitas caem. Salvem um checkpoint a cada N passos com tudo o que é preciso para continuar: modelo, otimizador, passo global, época e quantos lotes da época já foram usados. No Kaggle, salve em `/kaggle/working` (vai para o *Output* da sessão); no Colab, monte o Google Drive.

Como o embaralhamento usa uma semente por época, retomar é recriar o dataloader daquela época e **pular os lotes já vistos**.

**Código 15**

```python
from itertools import islice

def salvar_checkpoint(caminho, model, optimizer, passo, epoca, lote_na_epoca, extra=None):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "passo": passo, "epoca": epoca, "lote_na_epoca": lote_na_epoca,
                "extra": extra or {}}, caminho)

def carregar_checkpoint(caminho, model, optimizer):
    ck = torch.load(caminho, map_location=device)
    model.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optimizer"])
    return ck["passo"], ck["epoca"], ck["lote_na_epoca"], ck["extra"]

def lotes_da_epoca(epoca, pular=0):
    """Dataloader de treino da época, com semente própria, pulando os lotes já vistos."""
    loader = criar_dataloader(train_ids, BATCH_SIZE, GPT_CONFIG["context_length"],
                              GPT_CONFIG["context_length"], seed=123 + epoca)
    return islice(iter(loader), pular, None)
```

**Código 16**

```python
# Teste rápido: salva e recarrega
CKPT = os.path.join(WORK_DIR, "checkpoint_teste.pt")
optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.1)
salvar_checkpoint(CKPT, model, optimizer, passo=0, epoca=0, lote_na_epoca=0)
print(carregar_checkpoint(CKPT, model, optimizer)[:3])
os.remove(CKPT)
```

## 9. Bits por byte na validação

A perplexidade por token depende do tokenizador. **Bits por byte** (BPB) divide a loss total pelo tamanho do texto em bytes, o que permite comparar modelos com tokenizadores diferentes:

BPB = soma das losses (em nats) / (ln 2 × número de bytes)

A função abaixo usa as janelas sem sobreposição do `val_loader`. Os poucos tokens no fim que não completam uma janela ficam de fora, e os bytes são contados só para os tokens avaliados. É a mesma conta do `evaluate.py` que vocês vão entregar.

**Código 17**

```python
# Bytes de cada token do vocabulário em UTF-8 (para contar os bytes dos tokens avaliados)
bytes_por_token = torch.tensor([len(tokenizer.decode([i]).encode("utf-8")) for i in range(tokenizer.n_vocab)])
bytes_por_token[tokenizer.eot_id] = 1  # o separador conta como um caractere de quebra de linha

@torch.no_grad()
def bits_por_byte(model, loader, max_lotes=None):
    model.eval()
    soma_loss, soma_bytes = 0.0, 0
    for k, (xb, yb) in enumerate(loader):
        if max_lotes is not None and k >= max_lotes:
            break
        logits = model(xb.to(device))
        soma_loss += torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), yb.to(device).flatten(), reduction="sum").item()
        soma_bytes += bytes_por_token[yb].sum().item()
    model.train()
    return soma_loss / (math.log(2) * soma_bytes)

print(f"BPB do modelo não treinado (amostra): {bits_por_byte(model, val_loader, max_lotes=5):.2f}")
```

## 10. Próximos passos

Daqui em diante é com vocês, seguindo o enunciado:

- [ ] **Loop de treino** do capítulo 5 com as melhorias do Apêndice D: warmup linear, cosine decay, gradient clipping e AdamW com weight decay só nas matrizes.
- [ ] Orçamento de **1 época** (`len(train_loader)` passos), avaliando a loss de validação a cada ~10% do treino.
- [ ] **Checkpoint** a cada N passos com as funções da seção 8.
- [ ] Registrar loss, tokens/s e memória de pico (`torch.cuda.max_memory_allocated()`).
- [ ] Graduação: decodificação (temperatura, top-k, top-p) e a modificação do menu.
- [ ] Pós: precisão mista e FlashAttention no vanilla, depois os dois MoE e a modificação à escolha.
