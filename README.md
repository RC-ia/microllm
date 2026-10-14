# Mini IA: estatística, híbrida e LLM de 5M

Protótipo de geração autoregressiva feito em Python padrão. Não usa redes neurais, APIs, embeddings ou dependências externas. O modelo aprende contagens de palavras/pontuação, n-gramas e associações entre tokens observados próximos no mesmo documento.

O repositório tem três sistemas, do mais simples ao mais real:
1. `miniia.py` — modelo estatístico (stdlib puro, sem treino neural);
2. `responder.py` — resolvedor híbrido que recupera respostas reais da base;
3. `tinyllm.py` — LLM de verdade, Transformer decoder de ~5M parâmetros em PyTorch, treinada do zero no dataset.

## Começar

Use Python 3.10 ou superior. No Windows deste ambiente, o runtime incluído pode ser chamado pelo caminho abaixo (ou use `python` se já estiver no PATH). Treine com arquivos `.txt`, `.md`, `.json` ou diretórios:

```powershell
$PY = "C:\Users\Adoro\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $PY corrigir_dataset.py dataset.json -o dataset_texto_normal.json
& $PY miniia.py train dataset_texto_normal.json -o modelo.json
& $PY miniia.py frequency --model modelo.json --top-k 10
& $PY miniia.py chat --model modelo.json --mode frequency --tokens 60 --seed 42
& $PY miniia.py generate --model modelo.json --mode frequency --prompt "Como funciona" --tokens 60 --seed 42 --diagnostics
& $PY miniia.py window-test dataset_texto_normal.json --window 256 --max-tests 250
& $PY -m unittest -v
& $PY miniia.py evaluate dataset_texto_normal.json --seed 17
```

O leitor JSON entende registros com `messages` e usa somente cada mensagem `content`; campos como `reasoning_content` são ignorados. Diretórios são percorridos recursivamente. Treinar com mais arquivos recalcula as estatísticas conjuntamente e permite adicionar corpus sem editar código.

## Responder híbrido (recuperação + estatístico)

O `responder.py` é o sistema que responde perguntas de verdade, sem treino complexo. Ele carrega os pares `user`/`assistant` do dataset, indexa as perguntas por TF-IDF (sem dependências externas) e, quando chega uma pergunta, devolve a resposta armazenada na pergunta mais parecida — texto real do corpus, nunca inventado. Se nada for parecido o bastante, cai para o gerador estatístico do `miniia.py`. A resposta recuperada é sempre literal; a evidência (pergunta casada, similaridade e candidatos) vem junto em JSON.

```powershell
# 1. (opcional) treina o modelo estatístico usado só no fallback
& $PY miniia.py train dataset_texto_normal.json --context-window 256 -o modelo.json.gz
# 2. indexa os pares e mostra o tamanho da base
& $PY responder.py index --dataset dataset_texto_normal.json
# 3. pergunta pontual
& $PY responder.py ask "O que é uma LLM?" --dataset dataset_texto_normal.json --model modelo.json.gz
# 4. sessão interativa
& $PY responder.py chat --dataset dataset_texto_normal.json --model modelo.json.gz
# 5. valida a recuperação (leave-one-out) e sugere um limiar
& $PY responder.py eval --dataset dataset_texto_normal.json
```

Normalização: caixa baixa e remoção de acentos (`você` casa com `voce`), stopwords em português e aliases de saudação (`Oi!`, `Bom dia` → `ola`). Termos de uma única palavra não são boostados para pontuação perfeita, então "python" sozinho não finge ser igual a uma pergunta inteira. `--threshold` (padrão 0.35) define a similaridade mínima para aceitar uma recuperação; `--no-fallback` responde só pela base. No chat, uma pergunta de acompanhamento com poucos termos ("e ela como funciona?") é resolvida anexando a pergunta anterior do usuário, e os comandos `/contexto`, `/limpar` e `/sair` são os mesmos do `miniia.py`.

O `eval` faz leave-one-out (cada pergunta armazenada é consultada sem ela mesma) e reporta acurácia top-1/top-3, o score das recuperações corretas e incorretas e um limiar sugerido. No corpus deste repositório a separação não é limpa — perguntas parecidas do mesmo assunto competem entre si — então use o `eval` como referência, não como garantia. Este sistema recupera resposta conhecida; ele não raciocina nem cria conteúdo novo.

## LLM minúscula (~5M, `tinyllm.py`)

Transformer decoder de verdade, em PyTorch puro e sem baixar nada: tokenizador char-level aprendido do próprio dataset (142 símbolos: letras acentuadas, pontuação e os especiais `<user>`, `<assistant>`, `<eos>`), 4 camadas, 8 heads, embedding 256, **janela de 8192 caracteres**, head de saída com pesos amarrados — **5.293.056 parâmetros (5,29M, classe 5M)**. Treina do zero nas conversas formatadas como `<user>…<assistant>…<eos>`, então aprende a responder no formato de assistente em vez de só continuar texto.

A janela de 8192 caracteres cobre a conversa inteira do corpus (a resposta mais longa tem 7.639 caracteres). O embedding posicional custa `block_size × n_embd`, então subir a janela tira camadas para manter o total perto de 5M: 6 camadas em `--block 256` viram 4 camadas em `--block 8192`.

Custo real medido nesta máquina (GTX 1050 Ti, 4 GB, torch 2.11). O `--batch` é o batch **efetivo**; `--grad-accum` o divide em micro-lotes, então dá para manter um batch grande sem estourar a VRAM:

| `--block` | batch efetivo | grad checkpoint | s/step | pico de VRAM |
|---|---|---|---|---|
| 8192 | 8 | não | 37,7 | 4,68 GB (estoura) |
| 8192 | 8 | não | 4,4 (batch 4) | 2,34 GB |
| **8192** | **8** | **sim** | **~8** | **0,91 GB** |
| 8192 | 4 | sim | ~5,3 | 0,85 GB |

`--grad-checkpoint` é **ligado por padrão** e é a maior economia possível: recalcula as ativações no backward em vez de guardá-las. Medido: **-82% de memória** (2,34 GB → 0,53 GB) por ~19% de tempo a mais. Use `--no-grad-checkpoint` se precisar de velocidade e tiver VRAM sobrando.

Precisão mista (`fp16`/`bf16`) **não compensa nesta GPU**: ela é Pascal 6.1, sem tensor cores, então a matemática em meia precisão é emulada. Medido, `fp16` economiza memória mas fica **mais lento** que `fp32` (7,3 s contra 5,3 s por step). Se quiser testar mesmo assim: `torch.amp.autocast`.

A atenção usa `scaled_dot_product_attention` com `is_causal=True`, que aplica a máscara causal por bloco sem materializar a matriz `seq × seq` — sem isso, 8192 exigiria terabytes de memória. Além disso o lote usa comprimento dinâmico: a largura é a da maior amostra do lote, nunca `block` fixo, e o padding à direita é ignorado na loss (`ignore_index=-100`). Isso importa porque a mediana das conversas é 1.810 caracteres: fixar o bloco em 8192 desperdiçaria ~78% do compute.

Requer `torch` (`pip install torch`; neste ambiente já existe, com CUDA). Comandos:

```powershell
& $PY tinyllm.py params                                          # conta os parâmetros
& $PY tinyllm.py train --out tinyllm_8k.pt --block 8192 --batch 8 --grad-accum 2 --steps 2000
& $PY tinyllm.py generate --checkpoint tinyllm_8k.pt --prompt "O que é uma LLM?"
& $PY tinyllm.py chat --checkpoint tinyllm_8k.pt
& $PY tinyllm.py eval --checkpoint tinyllm_8k.pt --dataset dataset_texto_normal.json
```

Detalhes de treino: AdamW com decaimento cosseno e aquecimento, divisão 90/10 das conversas para validação, checkpoint salvo sempre que a perda de validação melhora (protege contra overfit — esperado num corpus de 53 conversas). Geração com temperatura e top-k, parada em `<eos>`, e `--min-new` (padrão 10) impede `<eos>` nos primeiros tokens, para checkpoint parcial não devolver resposta vazia.

Retomar um treino interrompido:

```powershell
& $PY tinyllm.py train --out tinyllm_8k.pt --block 8192 --batch 8 --grad-accum 2 --steps 2000 --tag "rodada-1"
# se parar (crash, Ctrl+C, reinício da máquina):
& $PY tinyllm.py train --out tinyllm_8k.pt --block 8192 --batch 8 --grad-accum 2 --steps 2000 --resume tinyllm_8k.pt --tag "rodada-1"
```

Não precisa de tag para retomar: o checkpoint se identifica sozinho. A tag (`--tag`) é só um rótulo gravado no arquivo para você saber qual rodada salvou. O que `--resume` restaura de verdade: o **passo** (o LR continua no lugar certo do cosseno, em vez de voltar ao topo), o **melhor `val_loss`** (para o checkpoint não ser sobrescrito por um pior), o **estado do AdamW** (os momentos do Adam) e o **RNG** (a ordem dos dados). Sem o estado do otimizador, retomar treina de um jeito diferente de não ter parado. Checkpoints antigos, salvos só com os pesos, continuam abrindo — com um aviso no terminal. Ajuste também: `--steps` é o total alvo, então para continuar basta aumentá-lo. Ajustes: `--steps`, `--batch`, `--block`, `--lr`, `--eval-every`. Teste rápido de fumaça: `--steps 300 --batch 16 --eval-every 150`.

Limites honestos: com ~280 mil chars de treino, um modelo de 5M **memoriza o estilo do corpus** em vez de generalizar como uma LLM grande. Ele gera texto fluente no formato pergunta/resposta, mas pode misturar fatos de respostas diferentes. Para respostas sempre corretas, prefira o `responder.py`; a LLM é o caminho quando você quer geração nova e um modelo neural de verdade treinado no seu dado.

O conversor `corrigir_dataset.py` remove Markdown de `messages[].content`, transforma quebras de linha reais ou escapadas em espaços, remove barras invertidas do formato de resposta e insere espaço após `!` ou `?` quando falta. O conteúdo fica em uma única linha por mensagem; o JSON original permanece intacto e a cópia é salva em `dataset_texto_normal.json`.

O comando `frequency` aceita um prompt e usa as contagens empíricas do maior sufixo observado (até `max_order - 1` tokens) para prever o próximo; se não encontrar o contexto, recua para sufixos menores e depois para a frequência global. A cada escolha, acrescenta o token ao contexto e repete. Rode `miniia.py frequency --model modelo.json --prompt "Como funciona" --tokens 5 --top-k 8`. Por padrão escolhe o token mais frequente dentro do contexto encontrado; `--sample --seed 42` faz amostragem reprodutível. Esse modo usa contagens de sequências observadas, sem as leis heurísticas do gerador `generate`.

O comando `chat` abre uma sessão interativa no terminal e mantém os tokens do usuário e das respostas anteriores como contexto para cada turno. `/contexto` mostra o histórico atual e sua contagem de tokens; `/limpar` reinicia e `/sair` encerra. Configure o limite de resposta com `--tokens`, a diversidade com `--temperature` e reprodutibilidade com `--seed`. Selecione `--mode frequency` para usar as ocorrências condicionadas ao histórico; o modo padrão é `contextual`. Como o `chat` do `miniia.py` ignora os papéis e trata tudo como continuação de texto, ali a sessão não é um assistente. Para conversas que seguem instrução, use o `responder.py`, que explora os papéis `user`/`assistant` do corpus.

`window-test` divide documentos em treino e validação e compara os dois métodos sobre as mesmas posições: `window` percorre o corpus token a token e indexa cada sufixo observado, do token anterior até todo o contexto disponível; na predição, encontra o sufixo mais longo que já ocorreu e conta os tokens que vieram depois dele. O mesmo contexto pode ter vários próximos tokens, guardados como ramificações. `global` consulta transições exatas de n-gramas até `max_order - 1`. O método global encerra a geração se não houver uma transição aprendida. O relatório inclui acurácia top-1/top-k, cross-entropy, perplexidade e exemplos. Use `--weight-mode window` ou `--weight-mode global` para executar apenas um; `--max-tests 0` avalia todas as posições elegíveis.

Para conversar e comparar os dois sistemas em tempo real, crie um modelo com contexto de 256 tokens e inicie `window-chat`:

```powershell
& $PY miniia.py train dataset_texto_normal.json --context-window 256 -o modelo_256.json.gz
& $PY miniia.py window-chat --model modelo_256.json.gz --weight-mode both --tokens 30 --seed 42
# Para amostrar outros ramos e inspecionar candidatos:
& $PY miniia.py window-chat --model modelo_256.json.gz --weight-mode window --tokens 30 --sample --diagnostics
```

Cada mensagem do usuário recebe uma resposta do peso local (`window`) e outra do peso global (`global`). Até 20 tokens iniciais usam o preenchedor do modo `frequency`, condicionado pela conversa; os tokens restantes passam ao sistema `window`/`global`. Se o preenchedor encontrar um fim de sequência aprendido antes dos 20 tokens, a resposta termina ali. Ajuste o aquecimento com `--prefill-tokens`. No restante da geração, `window` consulta o índice de todos os tokens dos documentos e usa o sufixo mais longo encontrado, sem limite fixo de tamanho; se o mesmo contexto ocorreu com continuações diferentes, todas viram candidatas proporcionais à frequência. `global` usa transições de n-gramas até `--order - 1`. O chat guarda o texto tokenizado do corpus uma vez e um índice de sufixos compacto, em vez de salvar uma cópia de cada prefixo (o que cresceria quadraticamente). Use `--context-window 256` para manter o histórico recente do chat. Os históricos ficam separados por sistema e incluem mensagens do usuário e respostas anteriores. O chat aplica por padrão penalidade de frequência de `1.5` e ativa amostragem com temperatura maior ao detectar ciclos; ajuste com `--frequency-penalty`, `--loop-chaos` e `--cycle-limit`. Durante a sessão, `/contexto` exibe os históricos, `/limpar` reinicia e `/sair` encerra.

Por padrão, a geração escolhe o candidato de maior pontuação. `--sample` amostra da distribuição para explorar ramificações; `--diagnostics` mostra os dez candidatos e suas probabilidades em cada passo.

## Arquitetura e leis

- Tokenização: palavras Unicode e sinais de pontuação; espaços são reconstruídos por regras simples, então a saída não preserva quebras de linha exatamente.
- Treinamento: frequências globais e por documento, n-gramas configuráveis (padrão 4), coocorrências em uma janela limitada (padrão 32 tokens) e um índice de sufixos do corpus. O índice é construído com todos os tokens anteriores disponíveis para cada posição, incluindo a transição seguinte, avançando um token por vez. Seu tamanho é linear em relação ao corpus, sem materializar todas as combinações de prefixos.
- Predição: parte de log-probabilidades unigramas suavizadas, soma evidência de n-gramas e compatibilidade contextual observada. O histórico inteiro permanece na lista da geração; a influência por coocorrência consulta a janela recente limitada.
- Leis heurísticas, configuráveis no objeto `ContextLaws`: frequência em documentos reduz influência; raridade aumenta peso com teto; recorrência local aumenta gradualmente com saturação; distância decai exponencialmente; compatibilidade exige coocorrência aprendida. Cada uma pode ser desligada com `--no-frequency`, `--no-rarity`, `--no-recurrence`, `--no-distance` ou `--no-compatibility`.
- Temperatura/top-k alteram a distribuição de amostragem, não as estatísticas salvas. Penalidades para tokens recentes e continuações de sequências já vistas, mais aumento limitado de temperatura após ciclos curtos, são regras independentes; `--diagnostics` reporta ativações detectadas.
- Persistência: JSON versionado, com suporte a compactação gzip quando o caminho termina em `.gz`. As associações são aproximadas e limitadas às 256 continuações mais comuns por token; n-gramas permanecem exatos para o corpus.

## Parâmetros

`--tokens`, `--temperature`, `--seed`, `--top-k`, `--repeat-penalty`, `--repeat-window`, `--cycle-limit`, `--adaptive-temperature`, `--order`, `--alpha` e `--context-window`. Ajustes de leis disponíveis: `--rarity-strength`, `--frequency-strength`, `--recurrence-strength`, `--distance-decay`.

## Avaliação e limites

`evaluate` divide documentos de modo determinístico e amostra até 1.000 posições por documento (ajuste com `--eval-tokens`), usando o contexto anterior real. Em cinco configurações, mede cross-entropy/perplexidade aproximadas e compara amostras com diversidade lexical, repetição de tokens/bigramas, ciclos detectados, tempo e tamanho serializado do modelo: baseline, frequência/raridade, distância, três leis (frequência/raridade/recorrência) e todas as leis. Com corpora pequenos, a avaliação tem alta variância; palavras nunca vistas no treino recebem uma probabilidade de fallback muito pequena. O resultado mede previsão estatística, não compreensão ou raciocínio. O custo de distribuição é O(|vocabulário| + tokens da janela + n-gramas encontrados) por token gerado; avaliação exata pode ser lenta em corpus grande.

Pesos de lei são heurísticas manuais, não parâmetros aprendidos. Texto gerado recombina padrões locais e pode perder coerência em contextos longos. O corpus `dataset.json` deste repositório tem 53 conversas e cerca de 299 mil caracteres; os exemplos/medidas nele refletem esse corpus específico.





