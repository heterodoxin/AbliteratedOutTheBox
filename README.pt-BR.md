<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="380">
  </picture>
</p>

<p align="center"><a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.de.md">Deutsch</a> · <a href="README.fr.md">Français</a> · <a href="README.es.md">Español</a> · <b>Português</b></p>

<h3 align="center">Rode um modelo de ponta no seu próprio PC gamer</h3>

<p align="center">Suporte a NVIDIA, AMD, Intel · Windows, Linux, Mac · gratuito e de código aberto</p>

<p align="center">
  <a href="https://github.com/Niko1221/Strata/releases"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?include_prereleases&label=release" alt="Latest release"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="MIT license"></a>
  <a href="https://github.com/Niko1221/Strata/stargazers"><img src="https://img.shields.io/github/stars/Niko1221/Strata?style=flat" alt="GitHub stars"></a>
</p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="A voxel pagoda garden that Strata's model wrote, running in the browser"></a><br>
<sub>Um jardim de pagodes em voxels, gerado com um único prompt, rodando numa RTX 5070 com o Strata (IQ3_S, contexto de 128K) ·
<a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">vídeo completo (49 s)</a></sub></p>

O Strata roda o **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)**, um modelo de 125 bilhões de parâmetros que
normalmente precisa de um servidor, num PC comum. Ele conversa, escreve código, lê imagens e funciona com os seus aplicativos e
agentes de programação. Suas conversas e arquivos ficam no seu PC.

**Primeira vez aqui?** Veja [o que você precisa](#o-que-você-precisa) e depois [instale o app](#opção-a-o-app-strata).

- **Roda num PC gamer.** 32 GB somando memória RAM e memória da placa de vídeo já bastam para começar.
- **Funciona com as suas ferramentas.** Ele fala as APIs da OpenAI e da Anthropic, então Claude Code, Codex CLI, OpenCode e outros agentes se conectam a ele.
- **Lê imagens.** Já vêm ativadas.
- **Gratuito.** Licença MIT, sem conta, sem nuvem.
- **Privado.** Sem telemetria. O que o Strata baixa, quando usa a internet e como confere os downloads está em [docs/SAFETY.md](docs/SAFETY.md).

Versão 0.2.3 · mecanismo 0.1.48

## O que você precisa

**Você precisa de pelo menos 32 GB somando a memória RAM e a memória da placa de vídeo (VRAM).** Num Mac ou num Ryzen AI Max, isso é a
memória unificada. Mais memória permite um modelo maior ou mais rápido.

- Um PC de 32 GB com uma placa de 12 a 16 GB roda o Coder.
- 64 GB de RAM rodam todos os tamanhos.

| Seu computador | Detalhes |
| --- | --- |
| **NVIDIA** GeForce RTX série 20, 30, 40, 50 (12 GB ou mais) | Instale o app ou o Terminal. |
| **AMD** Radeon RX 6800 e mais novas, Radeon AI PRO R9700 (12 GB ou mais) | [Placas AMD](docs/AMD_HIP.md) |
| **Intel** Arc | [Intel Arc](docs/INTEL_ARC.md), [mecanismo Intel](docs/INTEL.md) |
| **Mac com Apple Silicon** (M1 ou mais novo, macOS 15 ou mais novo) | [Mac](docs/MAC.md) |
| **AMD Ryzen AI Max (Strix Halo)** e outros sistemas de memória unificada | [Strix Halo](docs/STRIX_HALO.md) |
| Placas e processadores mais antigos | [GPUs antigas](docs/OLDER_GPUS.md), [CPUs antigas](docs/INSTALL.md#older-cpus-experimental) |

Você também precisa de:

- cerca de 80 GB de espaço livre no disco (um SSD deixa a primeira inicialização bem mais rápida)
- Windows 10 / 11, Linux ou macOS
- um driver de vídeo atualizado

Duas ou três placas da mesma marca podem dividir um modelo ([multi-GPU](docs/MULTI_GPU.md)). O Strata nunca divide um modelo entre
placas NVIDIA e AMD. Lista completa: [docs/INSTALL.md](docs/INSTALL.md#what-you-need).

## Instalação

Escolha uma opção. Todas iniciam o mesmo mecanismo e usam os mesmos modelos.

### Opção A: o app Strata

O jeito mais fácil. O app já traz o mecanismo: instale, escolha um modelo e converse. Baixe o arquivo para o seu sistema e a sua
placa de vídeo na [página de releases](https://github.com/Niko1221/Strata/releases/latest). Não sabe qual? Veja a explicação
logo abaixo da tabela.

| Sistema | Sua placa de vídeo | Arquivo para baixar |
| --- | --- | --- |
| **Windows 10 / 11** | NVIDIA RTX, driver 580 ou mais novo | [`Strata-0.2.3-windows-cuda13.exe`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-windows-cuda13.exe) |
| | NVIDIA, placa mais antiga ou driver anterior ao 580 | [`Strata-0.2.3-windows-cuda12.exe`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-windows-cuda12.exe) |
| | AMD Radeon série RX 9000 ou Radeon AI PRO R9700 | [`Strata-0.2.3-windows-hip.exe`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-windows-hip.exe) |
| | AMD Radeon série RX 7000 ou Ryzen AI Max (Strix Halo) | [`Strata-0.2.3-windows-hip-rdna3.exe`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-windows-hip-rdna3.exe) |
| **Linux** | NVIDIA RTX, driver 580 ou mais novo | [`Strata-0.2.3-linux-nvidia-cuda13.AppImage`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-nvidia-cuda13.AppImage) |
| | NVIDIA, placa mais antiga ou driver anterior ao 580 | [`Strata-0.2.3-linux-nvidia-cuda12.AppImage`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-nvidia-cuda12.AppImage) |
| | AMD Radeon série RX 9000 ou Radeon AI PRO R9700 | [`Strata-0.2.3-linux-amd-rx9070.AppImage`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-amd-rx9070.AppImage) |
| | AMD Radeon série RX 7000 | [`Strata-0.2.3-linux-amd-rx7900.AppImage`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-amd-rx7900.AppImage) |
| | AMD Ryzen AI Max (Strix Halo) | [`Strata-0.2.3-linux-amd-strixhalo.AppImage`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-amd-strixhalo.AppImage) |
| **Mac** (Apple Silicon, macOS 15 ou mais novo) | integrada | [`Strata-0.2.3-mac.dmg`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-mac.dmg) |

**Se você estiver em dúvida:** para NVIDIA, use `cuda13` se o seu driver for a versão 580 ou mais nova, e `cuda12` nos outros casos. O `cuda12`
serve para placas e drivers mais antigos. Os arquivos `hip` e `amd` são para placas AMD. Outras placas AMD (RX 6000) ainda não têm app pronto: veja [Placas AMD](docs/AMD_HIP.md).

Depois:

- **Windows:** execute o `.exe`. Ele instala só para o seu usuário e não pede direitos de administrador. Se o Windows disser "O Windows
  protegeu o computador", escolha **Mais informações** e depois **Executar assim mesmo** (o instalador ainda não tem assinatura de código).
- **Linux:** torne o arquivo executável (`chmod +x`) e inicie. É preciso ter o FUSE 2: `sudo apt install libfuse2` (Ubuntu 22.04)
  ou `libfuse2t64` (Ubuntu 24.04 e mais novos).
- **Mac:** abra o `.dmg` e arraste o Strata para Aplicativos.

Na primeira tela, o app lê o seu PC e recomenda um modelo que caiba nele. Clique em instalar. O modelo é um download grande
(cerca de 40 a 70 GB) e continua de onde parou se for interrompido. Fechar a janela encerra o Strata e o mecanismo.

### Opção B: o Terminal (uma linha)

O Terminal é o comando `strata`, para scripts, servidores e agentes de IA.

Linux e Mac:

```sh
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh | sh
```

Windows (PowerShell):

```powershell
irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

O script confere o seu PC, baixa o Terminal, verifica o SHA-256 e descompacta na sua pasta pessoal. Não precisa de direitos de
administrador. No Windows, abra uma nova janela do terminal depois. Então inicie o Strata:

```sh
strata launch
```

Ele faz uma pergunta por vez: modelo, contexto, imagens, quem pode se conectar. Depois baixa o mecanismo para a sua placa de vídeo
(assinada e testada antes do uso) e o modelo, e começa a servir. Em scripts e agentes de IA, use flags:

```sh
strata launch --model q2_0 --yes --no-tui
```

Para ler o script de instalação antes de executá-lo, baixe-o primeiro com `curl -fsSL <address> -o install.sh`, leia e então rode
`sh install.sh`. Todas as telas e flags: [docs/TERMINAL_LAUNCH.md](docs/TERMINAL_LAUNCH.md).

### Opção C: deixe a sua IA configurar

Cole isto no Claude Code, Cursor, Codex, GitHub Copilot ou outro assistente de programação:

```text
Set up Strata on this PC for me: https://github.com/Niko1221/Strata - follow docs/AI_SETUP.md in that repository.
```

Ele confere a sua placa de vídeo, a RAM e o disco, escolhe o modelo que cabe, instala e inicia, e diz como conectar os seus
aplicativos. Ferramentas de IA também podem instalar, iniciar e parar o Strata pelo [servidor MCP](docs/MCP_SERVER.md) dele.

Prefere compilar por conta própria? Veja [Compilar a partir do código-fonte](#compilar-a-partir-do-código-fonte).

## Modelos

O app e o `strata launch` recomendam um modelo para a sua memória. É um único modelo em vários tamanhos, mais ou menos comprimidos.
Os tamanhos menores são mais rápidos. Os maiores são um pouco mais inteligentes.

### Pouca memória? Experimente o Gyro

O [Gyro](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF) (da agentionai) é a menor forma de rodar este modelo.

| Tamanho | Memória necessária (RAM + VRAM) | Download | O que é |
| --- | ---: | ---: | --- |
| **Gyro-S** | 27,6 GB | 58,5 GB | Para PCs onde nenhum outro tamanho cabe. A qualidade é menor que a do IQ2_XS. |
| **Gyro-M** | 35 GB | 92,0 GB | Qualidade próxima à do IQ3_XXS, com cerca de 20% menos memória da GPU. Ele escreve as respostas 16% mais devagar que o Q2_0 numa RTX 3060 de 12 GB e 21% mais devagar numa R9700. |

O Gyro roda em placas NVIDIA e AMD (AMD é experimental) em placas Intel Arc no Linux (testado numa Arc A750) e, desde a 0.2.2, em Macs com Apple Silicon (Gyro-S num Mac mini de 32 GB: 9 tokens/s, com os especialistas lidos do SSD). Detalhes e medições:
[docs/GYRO.md](docs/GYRO.md). Inicie com `strata launch --model gyro-s`.

### Os outros tamanhos

| Sua memória (RAM + VRAM) | Escolha | Observações |
| --- | --- | --- |
| **32 GB** | **Gyro-S**, ou o **[Coder](docs/MODELS.md#coder)** para código | O Coder precisa de 32 GB de RAM e mantém metade dos experts, então é mais fraco fora de código |
| **40 GB** | **[Swift 1.5](docs/MODELS.md#swift-15)** IQ2_XS (39,2 GB), ou **Gyro-M** (35 GB) | O Swift pensa por menos tempo antes de responder. O Gyro-M chega perto do IQ3_XXS em qualidade, mas escreve as respostas 16 a 21% mais devagar que o Q2_0 |
| **48 GB** | **IQ3_XXS** (47,0 GB) | Mais lento, melhor |
| **64 GB** | **IQ3_S** (54,8 GB), o original ou o Swift 1.5 | O mais lento, o melhor: empata com o modelo completo nos testes publicados |
| **96 GB ou mais** | **UD-IQ4_XS** da Unsloth (cerca de 72 GB de RAM) até **Q8_0** (cerca de 136 GB), de 4 a 8 bits | Mais perto do modelo completo. Com menos RAM, o Strata lê parte deles do SSD, o que é bem mais lento |

Com menos RAM do que a linha indica, ainda pode funcionar: o Strata muda sozinho para o
[modo de pouca RAM](docs/MODELS.md#a-big-graphics-card-and-little-ram). Espere mais lentidão, e muito mais com uma placa de vídeo pequena.

Mais opções:

- **[Swift 1.5](docs/MODELS.md#swift-15)** é um ajuste fino que pensa por menos tempo antes de responder, com qualidade e RAM parecidas.
- **[NVFP4](docs/NVFP4.md)** é a versão de 4 bits da NVIDIA para placas NVIDIA. Precisa de cerca de 80 GB de RAM para manter todos os experts na RAM.
- **[Qwen3.6-35B-A3B](docs/QWEN36.md)** é outro modelo, menor, que lê imagens e pode atender vários pedidos ao mesmo tempo.
- **[Uncensored IQ3_XXS da OrcaRouter](docs/MODELS.md#orcarouter-uncensored-iq3_xxs)** é configurado manualmente.

Todos os tamanhos, downloads e o que cabe em cada configuração: [docs/MODELS.md](docs/MODELS.md).

### DeepSeek V4.1-Flash (no Terminal, a partir da 0.2.4)

[DeepSeek V4.1-Flash](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF) é outro modelo, bem maior. A ISTA-DASLab fez os três conjuntos abaixo. O Strata os executa em um PC com uma placa de vídeo de 12 GB e de 46 a 64 GB de RAM, e lê o resto do seu SSD. Ele roda em placas NVIDIA RTX das séries 30 a 50 e na Radeon AI PRO R9700. Usa um pacote de motor próprio, `strata-dsv41`, que o `strata launch` baixa quando você escolhe o DeepSeek. O app de desktop ainda não o oferece.

| Conjunto | Download | Qualidade em relação ao modelo original | Responde a |
| --- | ---: | --- | --- |
| **IQ1_M** (1,75 bits) | 239 GB | KL 0,217, perplexidade 3,80 (+21%); código +8,9%, tarefas de agente +10,2%; mais fraco em prosa comum em inglês. Segundo o cartão do modelo da ISTA-DASLab, mantém 98,7% da média de tarefas do modelo base (AIME25, GPQA-Diamond, LiveCodeBench v6). | ainda não medida |
| **IQ2_XXS** (2.00 bits) | 257 GB | KL 0,170, perplexidade 3,63 (+15,5%); código +8,6%, tarefas de agente +5,3% | cerca de 8 a 10 tokens/s em uma RTX 3060 (código 9,9, tarefas de agente 7,8), cerca de 5 em uma RTX 5070 |
| **IQ2_XS** (2.25 bits) | 274 GB | KL 0,107, perplexidade 3,37 (+7,3%); código +3,2%, tarefas de agente +2,2% | cerca de 11 tokens/s em uma Radeon AI PRO R9700 |
| **IQ4_NL** (tabela Engram, não é um terceiro modelo) | 111 GB, já incluídos nos três downloads acima | A tabela de consulta Engram de 4 bits que o modelo lê da RAM ou do SSD. Os três conjuntos usam este único arquivo (`DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf`). | n/a |

4x Radeon AI PRO R9700: cerca de 31 tokens/s, 55 a 69 com DSpark.

Inicie com `strata launch --model ist-das-2.00bpw` (ou `ist-das-1.75bpw`, `ist-das-2.25bpw`). Detalhes, velocidades por placa e a instalação manual: [docs/DEEPSEEK41.md](docs/DEEPSEEK41.md).

## Qual é a velocidade?

Com base numa placa de vídeo de 12 GB (RTX 5070, Ryzen 5 7600, 64 GB de RAM). Veja resultados de outras placas:
[velocidade de cada modelo](docs/MODELS.md#how-fast-is-each-size), [resultados da comunidade](docs/COMMUNITY_BENCHMARKS.md).

| Modelo | As respostas aparecem a | Lê o seu prompt a |
| --- | --- | --- |
| **Q2_0** | `████████████` 94 tokens/s | `████████████` 2.650 tokens/s |
| **IQ2_XS** | `██████████░░` 79 tokens/s | `█████████░░░` 2.090 tokens/s |
| **IQ3_XXS** | `████████░░░░` 62 tokens/s | `████████░░░░` 1.750 tokens/s |
| **IQ3_S** | `███████░░░░░` 53 tokens/s | `███████░░░░░` 1.620 tokens/s |
| **Coder** | `███████░░░░░` 55 tokens/s | `██████████░░` 2.180 tokens/s |

Um token equivale a cerca de ¾ de palavra, então 60 tokens por segundo é mais rápido do que você lê. "Lê o seu prompt" é a velocidade com que o Strata
absorve um documento longo, um código ou um histórico de conversa.

Medido numa placa AMD de 16 GB (RX 9070 XT, Ryzen 9 3900X, 47 GB de RAM):

| Modelo | As respostas aparecem a | Lê o seu prompt a |
| --- | --- | --- |
| **Q2_0** | `████████░░░░` 60 tokens/s | `█████░░░░░░░` 1.160 tokens/s |
| **IQ2_XS** | `███████░░░░░` 52 tokens/s | `█████░░░░░░░` 1.110 tokens/s |
| **Coder** | `██████░░░░░░` 44 tokens/s | `██████░░░░░░` 1.420 tokens/s |

As barras usam a mesma escala nas duas tabelas. Uma placa com mais VRAM é mais rápida: uma RTX 3090 (24 GB) deve escrever cerca de 100 a 140
tokens por segundo. A versão do mecanismo por trás de cada número está em [DETAILS.md](docs/DETAILS.md#speed-measured).

## Como usar

### No app

O app tem um **Chat**, um **Monitor** do modelo e da sua GPU, CPU e RAM, uma página de **Modelos** e as **Configurações**. O chat
pode trabalhar numa pasta do seu PC. No modo **Manual**, ele pergunta antes de alterar um arquivo ou executar um comando. Digite
`/loop 10m <o que fazer>` para repetir uma mensagem a cada 10 minutos enquanto o chat estiver aberto.

O chat pode pesquisar na web com o DuckDuckGo (ou o Bing), sem conta. Você pode adicionar uma chave de busca em **Configurações, Ferramentas**.
A busca é o único recurso que envia algo para fora do seu PC: os termos da pesquisa.

Para rodar dois modelos, ative **Executar vários modelos** e dê a cada modelo a sua própria placa da mesma marca.

<p align="center"><img src="docs/media/app-chat.png" width="900" alt="The Strata app: a chat where the agent built a small voxel pagoda garden, with the page open in the preview on the right"><br>
<sub>Um chat no app Strata: o agente escreveu um pequeno jardim de pagode em voxels em um único arquivo HTML, e a página abre na pré-visualização ao lado da conversa</sub></p>

### No Terminal

<p align="center"><img src="docs/media/terminal-monitor.png" width="900" alt="The Strata Terminal monitor overview while the model writes a reply"><br>
<sub><code>strata monitor</code> enquanto o modelo responde: velocidade, GPU, memória e as solicitações recentes</sub></p>

```sh
strata launch                                  # telas guiadas: modelo, contexto, imagens, acesso e depois servir
strata launch --model q2_0 --yes --no-tui     # sem perguntas: serve, mostra o endereço e volta
strata launch --plan --model q2_0              # mostra o que aconteceria; nada muda
strata launch --status                         # o que está sendo servido agora
strata launch --stop                           # para o serviço
strata monitor --url http://127.0.0.1:8080     # acompanha velocidade, GPU, memória e requisições
strata update --check                          # há um mecanismo novo? (strata update instala)
strata doctor                                  # confere os arquivos do mecanismo instalada
```

`strata --help` lista tudo. Iniciar e servir nunca baixam nada. Só a instalação, o `strata launch` (quando você
escolhe um modelo) e o `strata update` baixam. Para acessar o Strata do celular ou de outro PC, use
`strata launch --access lan`. Ele cria uma chave de API e não inicia numa rede sem uma. Nenhum navegador abre, a menos que
você adicione `--open`. Guia: [docs/TERMINAL_LAUNCH.md](docs/TERMINAL_LAUNCH.md). Cliente completo: [docs/TERMINAL.md](docs/TERMINAL.md).

### Com agentes de IA

Um agente instala o Strata com o comando de uma linha da Opção B e executa
`strata launch --model NAME --yes --no-tui`. Os passos, as flags e os códigos de saída estão em [docs/AI_SETUP.md](docs/AI_SETUP.md),
e as regras do repositório estão em [AGENTS.md](AGENTS.md).

### Conecte os seus outros aplicativos

| Tipo de aplicativo | URL base |
| --- | --- |
| Compatível com OpenAI (a maioria dos apps e agentes) | `http://127.0.0.1:8080/v1` (qualquer chave de API, qualquer nome de modelo) |
| API da Anthropic (Claude Code e outros) | `http://127.0.0.1:8080`, sem `/v1`. Esses aplicativos adicionam `/v1/messages` sozinhos. Claude Code: `ANTHROPIC_BASE_URL=http://127.0.0.1:8080` |
| OpenAI Responses API (Codex CLI) | `http://127.0.0.1:8080/v1/responses` ([configuração](docs/DETAILS.md#the-responses-api-and-codex-cli)) |

`strata connect` mostra as configurações exatas para OpenCode, Codex CLI, Claude Code, oh-my-pi, pi, Hermes Agent e OpenClaw.
`strata connect opencode --write` grava o arquivo depois de perguntar e mantém um backup.

### Imagens

As imagens vêm ativadas nos modelos que sabem lê-las. Anexe ou cole uma imagem no chat, ou envie pelo seu aplicativo.
O leitor de imagens roda no processador quando a sua placa de vídeo tem menos de 24 GB (alguns segundos por imagem, sem tirar nada
da memória da placa) e roda na placa de vídeo a partir de 24 GB. A regra é a mesma para NVIDIA e AMD. Desative as imagens
por modelo, ou escolha o leitor com `--vision off`, `--vision gpu` ou `--vision cpu`.

Os números por trás da regra: numa Radeon AI PRO R9700, o leitor da placa levou 57 ms para uma imagem de 260 tokens e o do
processador levou 3,1 s. Em placas de 16 GB (Tesla P100 e RTX A4000, Linux), o leitor da placa custou de 3 a 5% da velocidade do texto,
porque ocupa cerca de 1,4 GB que o modelo usaria de outra forma. No Mac, a GPU faz a leitura, seja qual for a memória (0,45 s para uma imagem de 260 tokens em um M4 de 32 GB, cerca de 1,2 GB da memória compartilhada);
o processador leva cerca de 50 s por imagem, então só lê com `--vision cpu`.

### Bom saber

- Escolha o nível de raciocínio (**off, low, medium ou high**) no menu do chat ou no "reasoning effort" do seu aplicativo. Off é o mais rápido.
- Por padrão, o Strata responde a um pedido por vez. Para vários ao mesmo tempo, use `--parallel N` no `strata launch` ou
  `"parallel": 2` na configuração ([BATCHING.md](docs/BATCHING.md)). Numa placa de 12 GB, isso deixa cada resposta mais lenta.
- O Strata lê a primeira mensagem de uma conversa por inteiro, cerca de 1 minuto a cada 30.000 tokens. As mensagens seguintes começam em segundos.
- Onde ficam as suas conversas e a API completa: [docs/INSTALL.md](docs/INSTALL.md#where-things-are-stored), [docs/DETAILS.md](docs/DETAILS.md#using-it).

## Compilar a partir do código-fonte

<details>
<summary>Windows, Linux e Mac a partir do repositório</summary>

[Baixe o Strata](https://github.com/Niko1221/Strata/archive/refs/heads/main.zip) e descompacte, ou use `git clone`.
**Windows:** dê dois cliques em `START-HERE.bat`. **Linux:** execute `./setup.sh` na pasta do Strata.

**Mac (Apple Silicon, macOS 15 ou mais novo):** instale uma vez as Command Line Tools da Apple com `xcode-select --install`
(o Xcode em si não é necessário). Abra o Terminal na pasta do Strata e execute `./setup.sh`. Ele compila o mecanismo do Mac no seu
Mac (cerca de 7 minutos), faz as mesmas perguntas, baixa o modelo e inicia. Da próxima vez, `./setup.sh` inicia
na hora. Qual tamanho cabe no seu Mac: [docs/MAC.md](docs/MAC.md).

O instalador encontra a sua placa e configura o mecanismo certo. Ele pergunta qual modelo e tamanho, quanto contexto (quanto texto
o modelo mantém em mente) e se deve ler imagens. Pressione Enter para aceitar a resposta recomendada. Depois ele baixa o
modelo (cerca de 70 GB) e inicia. Se o download parar, execute de novo e ele continua. O navegador abre a página do Strata em
`http://127.0.0.1:8080`.

> Enquanto o modelo inicia, o seu PC pode ficar lento ou parar de responder por 1 a 3 minutos (mais tempo na primeira vez). O Strata carrega
> de 35 a 55 GB na sua RAM e trava parte dela para a placa de vídeo. Espere e não feche a janela.

Da próxima vez, execute `START-HERE.bat` (ou `./setup.sh`) de novo. Ele inicia na hora e não baixa nada duas vezes. Feche a janela
dele para parar o modelo. `UPDATE.bat` (`./update.sh`) atualiza o Strata sem iniciá-lo. Atualização, Docker, várias placas,
onde ficam os arquivos e todas as opções: [docs/INSTALL.md](docs/INSTALL.md).

Para medir o seu PC, execute `BENCH.bat` (`./bench.sh`). Leva de 5 a 10 minutos e gera um relatório que você pode compartilhar
([como](docs/COMMUNITY_BENCHMARKS.md#run-the-benchmark)).

**Opcional: mais velocidade numa placa quando o modelo inteiro cabe na RAM.** Duas configurações opcionais do RubenTorney (#1904, #1905).
Elas ajudam quando você usa uma placa de vídeo e a sua RAM comporta todos os experts, por exemplo Q2_0 ou IQ3_XXS num PC de 64 GB. Abra
`strata-<model>.json` na pasta de dados do Strata ([onde fica](docs/INSTALL.md#where-things-are-stored)) e adicione:

- `"--adapt-async", "1"` à lista `"args"`. O cache de experts passa a ser atualizado numa thread auxiliar, em vez de pausar a
  resposta. Medido numa RTX 5070 (Windows, Q2_0): +4 a 5% na geração, +7 a 10% em código e documentos longos. As respostas podem diferir
  um pouco do padrão e continuam iguais de uma execução para outra.
- `"STRATA_PIN_BY_NEED": "1"` no bloco `"env"` (Windows). O Windows só consegue travar cerca de metade da RAM para a placa
  de vídeo, e isso trava primeiro os experts de que a placa precisa. Medido na RTX 5070: +4,4% na leitura do prompt.

```json
"args": [ ..., "--adapt-async", "1" ],
"env": { "STRATA_PIN_BY_NEED": "1" }
```

Reinicie o Strata depois da alteração. Para desfazer, remova as duas entradas.

</details>

## Como funciona?

Modelos como este costumam rodar em servidores com centenas de gigabytes de memória de vídeo. A sua placa de vídeo tem
12 a 24 GB. O Strata faz o modelo caber dividindo o trabalho entre o PC inteiro. Imagine uma biblioteca.

- O modelo é uma equipe de 24.576 pequenos especialistas ("experts"), como 24.576 livros de consulta. Cada palavra precisa de apenas 10 deles.
- **A sua placa de vídeo é a mesa.** Ela é pequena e rápida. Os cerca de 4.000 livros mais usados ficam abertos nela.
- **A sua RAM são as estantes da sala de leitura.** Os 24.576 livros estão lá. Quando uma palavra precisa de um livro que não está na
  mesa, o seu processador lê da estante na hora, enquanto a placa continua trabalhando.
- **O seu SSD é o arquivo.** Ele guarda uma tabela de consulta de 29 GB. O modelo lê algumas linhas dela por palavra, então ela não
  precisa ficar na RAM.

<p align="center"><img src="docs/media/how-it-works.svg" width="860" alt="The model's 24,576 experts: the busiest on the graphics card, all of them in RAM, a lookup table on the SSD"></p>

Um pequeno ajudante também adivinha as próximas palavras. O modelo grande confere todas de uma vez e fica com as certas. Você recebe
a mesma resposta, de 1,6 a 1,8 vez mais rápido. Textos longos são lidos em blocos grandes, de até 8.192 tokens por vez, a mais de 1.000 tokens
por segundo.

<p align="center"><img src="docs/media/guess-and-check.svg" width="860" alt="A small helper guesses the next words; the big model checks them all at once and keeps the right ones"></p>

A explicação mais longa: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md). Cada parte e seus números:
[os detalhes](docs/DETAILS.md#how-it-works) e o [artigo](docs/paper/Strata-Paper.pdf).

## Algo deu errado?

- **Meu PC travou na primeira vez que o Strata iniciou.** Ele carrega o modelo na RAM. Espere e não feche a janela. Ainda
  travado depois de 10 minutos? Reinicie o PC, feche outros programas e tente de novo, ou escolha um tamanho menor.
- **Parou durante o download ou a instalação.** Abra o app ou o `strata launch` de novo, ou execute `START-HERE.bat` (ou
  `./setup.sh`) outra vez. Ele continua de onde parou.
- **Está muito lento, a luz do disco não para de piscar, ou aparece "o mecanismo parou inesperadamente".** O seu PC não tem
  RAM livre suficiente. Feche outros programas (os navegadores usam muita), ou escolha um tamanho menor, como Q2_0 ou IQ2_XS.
- **Diz que não há mecanismo para este PC.** No app, abra **Configurações, Mecanismo** e clique em **Atualizar mecanismo**. No
  Terminal, execute `strata update`. O `strata doctor` mostra quais arquivos do mecanismo estão instalados.
- **Diz que a porta 8080 já está em uso.** O Strata já está rodando. Procure a janela dele.

Mais problemas e soluções: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). Ainda sem resolver? Abra uma
[issue](https://github.com/Niko1221/Strata/issues). No app, anexe o arquivo de **Configurações, Mecanismo, Diagnóstico, Ver
registros, Salvar logs**. No Terminal, cole a saída de `strata doctor`. Se estiver rodando pelo código-fonte, anexe o `strata-<model>.log` da pasta do
Strata. Para um problema de segurança, informe em particular: [SECURITY.md](SECURITY.md).

## Créditos e licença

O modelo é o [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), da equipe Qwen. Ele foi comprimido por
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), UkisAI (Swift 1.5), agentionai (Gyro)
e Unsloth. O Strata usa partes do [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). Todos os créditos:
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits). O Strata é de código aberto sob a [Licença MIT](LICENSE). Algumas partes e
todos os modelos têm licenças próprias ([quais](docs/HOW_IT_WORKS.md#license)).

## Contribuindo

Encontrou um bug, ou quer enviar uma alteração ou um benchmark? Leia primeiro o [CONTRIBUTING.md](CONTRIBUTING.md). Ele diz o que
incluir para que outras pessoas possam conferir o seu trabalho.

## Apoie o Strata

O Strata é gratuito e tem licença MIT. Eu o desenvolvo e testo em todo hardware que consigo arranjar. Se ele roda bem no seu PC, um café me ajuda
a continuar testando placas novas e corrigindo o que você relatar. Sem pressão. Usar e relatar o que encontrar ajuda tanto quanto.

<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a>

Quer patrocinar o Strata? Fale comigo pelo [GitHub](https://github.com/Niko1221).
