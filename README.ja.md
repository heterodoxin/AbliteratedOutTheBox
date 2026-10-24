<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="380">
  </picture>
</p>

<p align="center"><a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a> · <b>日本語</b> · <a href="README.de.md">Deutsch</a> · <a href="README.fr.md">Français</a> · <a href="README.es.md">Español</a> · <a href="README.pt-BR.md">Português</a></p>

<h3 align="center">最先端のモデルを、手元のゲーミングPCで動かす</h3>

<p align="center">NVIDIA、AMD、Intel に対応 · Windows、Linux、Mac · 無料のオープンソース</p>

<p align="center">
  <a href="https://github.com/Niko1221/Strata/releases"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?include_prereleases&label=release" alt="Latest release"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="MIT license"></a>
  <a href="https://github.com/Niko1221/Strata/stargazers"><img src="https://img.shields.io/github/stars/Niko1221/Strata?style=flat" alt="GitHub stars"></a>
</p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="A voxel pagoda garden that Strata's model wrote, running in the browser"></a><br>
<sub>ボクセルの塔の庭園。1回のプロンプトで生成し、RTX 5070 上の Strata で動かしています（IQ3_S、128K コンテキスト） ·
<a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">動画全体（49秒）</a></sub></p>

Strata は **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** を動かします。1250億パラメータのモデルで、
ふつうはサーバーが必要ですが、Strata なら一般的なPCで使えます。チャット、コード作成、画像の読み取りができ、
お使いのアプリやコーディングエージェントとも連携します。チャットとファイルはPCの外に出ません。

**はじめての方へ。** まず[必要な環境](#必要な環境)を確認して、[アプリをインストール](#方法a-strata-アプリ)してください。

- **ゲーミングPCで動きます。** RAMとグラフィックスメモリの合計が32 GBあれば始められます。
- **お使いのツールと連携します。** OpenAI と Anthropic の API に対応しているので、Claude Code、Codex CLI、OpenCode などのエージェントから接続できます。
- **画像を読めます。** 初期設定でオンです。
- **無料です。** MIT ライセンス。アカウントもクラウドも不要です。
- **プライバシーを守ります。** テレメトリはありません。Strata が何をダウンロードするか、いつインターネットを使うか、ダウンロードをどう検証するかは [docs/SAFETY.md](docs/SAFETY.md) に書いてあります。

バージョン 0.2.3 · エンジン 0.1.48

## 必要な環境

**RAMとグラフィックスメモリ（VRAM）の合計が、少なくとも32 GB必要です。** MacやRyzen AI Maxでは、ユニファイドメモリがその量にあたります。
メモリが多いほど、大きなモデルを使えたり、速く動いたりします。

- RAM 32 GBのPCに12〜16 GBのグラフィックスカードがあれば、Coder が動きます。
- RAM 64 GBなら、どのサイズも動きます。

| お使いのPC | 詳細 |
| --- | --- |
| **NVIDIA** GeForce RTX 20、30、40、50 シリーズ（12 GB以上） | アプリまたはターミナル版をインストールしてください。 |
| **AMD** Radeon RX 6800 以降、Radeon AI PRO R9700（12 GB以上） | [AMD のカード](docs/AMD_HIP.md) |
| **Intel** Arc | [Intel Arc](docs/INTEL_ARC.md)、[Intel エンジン](docs/INTEL.md) |
| **Apple Silicon の Mac**（M1 以降、macOS 15 以降） | [Mac](docs/MAC.md) |
| **AMD Ryzen AI Max（Strix Halo）** などのユニファイドメモリ搭載機 | [Strix Halo](docs/STRIX_HALO.md) |
| 古いカードやプロセッサ | [古いGPU](docs/OLDER_GPUS.md)、[古いCPU](docs/INSTALL.md#older-cpus-experimental) |

そのほかに必要なもの:

- 約80 GBの空きディスク容量（SSDなら初回の起動がずっと速くなります）
- Windows 10 / 11、Linux、または macOS
- 最新のグラフィックスドライバー

同じメーカーのカードを2〜3枚使って、1つのモデルを動かせます（[マルチGPU](docs/MULTI_GPU.md)）。NVIDIA と AMD のカードにまたがってモデルを分けることはありません。
一覧は [docs/INSTALL.md](docs/INSTALL.md#what-you-need) にあります。

## インストール

3つの方法から1つ選んでください。どれも同じエンジンを起動し、同じモデルを使います。

### 方法A: Strata アプリ

いちばん簡単な方法です。エンジンはアプリに入っているので、インストールしてモデルを選べばチャットを始められます。お使いのOSとグラフィックスカードに合うファイルを
[リリースページ](https://github.com/Niko1221/Strata/releases/latest)からダウンロードしてください。どれか分からない場合は、表の下の説明を見てください。

| OS | グラフィックスカード | ダウンロードするファイル |
| --- | --- | --- |
| **Windows 10 / 11** | NVIDIA RTX、ドライバー 580 以降 | [`Strata-0.2.3-windows-cuda13.exe`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-windows-cuda13.exe) |
| | NVIDIA、古いカード、またはドライバーが 580 より古い場合 | [`Strata-0.2.3-windows-cuda12.exe`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-windows-cuda12.exe) |
| | AMD Radeon RX 9000 シリーズまたは Radeon AI PRO R9700 | [`Strata-0.2.3-windows-hip.exe`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-windows-hip.exe) |
| | AMD Radeon RX 7000 シリーズまたは Ryzen AI Max (Strix Halo) | [`Strata-0.2.3-windows-hip-rdna3.exe`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-windows-hip-rdna3.exe) |
| **Linux** | NVIDIA RTX、ドライバー 580 以降 | [`Strata-0.2.3-linux-nvidia-cuda13.AppImage`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-nvidia-cuda13.AppImage) |
| | NVIDIA、古いカード、またはドライバーが 580 より古い場合 | [`Strata-0.2.3-linux-nvidia-cuda12.AppImage`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-nvidia-cuda12.AppImage) |
| | AMD Radeon RX 9000 シリーズまたは Radeon AI PRO R9700 | [`Strata-0.2.3-linux-amd-rx9070.AppImage`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-amd-rx9070.AppImage) |
| | AMD Radeon RX 7000 シリーズ | [`Strata-0.2.3-linux-amd-rx7900.AppImage`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-amd-rx7900.AppImage) |
| | AMD Ryzen AI Max (Strix Halo) | [`Strata-0.2.3-linux-amd-strixhalo.AppImage`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-amd-strixhalo.AppImage) |
| **Mac**（Apple Silicon、macOS 15 以降） | 内蔵 | [`Strata-0.2.3-mac.dmg`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-mac.dmg) |

**迷ったときは:** NVIDIA の場合、ドライバーがバージョン 580 以降なら `cuda13`、それより古ければ `cuda12` を選びます。`cuda12` は古いカードと古いドライバー向けです。
`hip` と `amd` のファイルは AMD のカード用です。それ以外の AMD カード（RX 6000）には、まだ用意されたアプリがありません。
[AMD のカード](docs/AMD_HIP.md)を見てください。

そのあとの手順:

- **Windows:** `.exe` を実行します。ユーザー単位でインストールされ、管理者権限は不要です。「Windows によってPCが保護されました」と表示されたら、
  **詳細情報**を選び、**実行**を選んでください（インストーラーにはまだコード署名がありません）。
- **Linux:** ファイルを実行可能にして（`chmod +x`）起動します。FUSE 2 が必要です。`sudo apt install libfuse2`（Ubuntu 22.04）
  または `libfuse2t64`（Ubuntu 24.04 以降）を実行してください。
- **Mac:** `.dmg` を開き、Strata を「アプリケーション」にドラッグします。

最初の画面で、アプリがPCを調べて合うモデルをすすめます。インストールを押してください。モデルは大きく（約40〜70 GB）、中断しても続きからダウンロードできます。
ウィンドウを閉じると、Strata とエンジンが止まります。

### 方法B: ターミナル（1行）

ターミナル版は `strata` コマンドです。スクリプトやサーバー、AI エージェント向けです。

Linux と Mac:

```sh
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh | sh
```

Windows（PowerShell）:

```powershell
irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

スクリプトはPCを調べ、ターミナル版をダウンロードし、SHA-256 を検証して、ホームフォルダに展開します。管理者権限は不要です。Windows では、終わったら
新しいターミナルを開いてください。そのあと Strata を起動します。

```sh
strata launch
```

質問は1つずつ表示されます。モデル、コンテキスト、画像、接続を許可する相手の順です。そのあと、お使いのグラフィックスカード用のエンジン
（署名付きで、使う前にテスト起動します）とモデルをダウンロードして、サーバーを開始します。スクリプトや AI エージェントからは、フラグを使います。

```sh
strata launch --model q2_0 --yes --no-tui
```

実行前にインストールスクリプトを読みたい場合は、先に `curl -fsSL <address> -o install.sh` でダウンロードし、内容を確認してから
`sh install.sh` を実行してください。すべての画面とフラグは [docs/TERMINAL_LAUNCH.md](docs/TERMINAL_LAUNCH.md) にあります。

### 方法C: AI にセットアップを任せる

Claude Code、Cursor、Codex、GitHub Copilot などのコーディングアシスタントに、次の文を貼り付けます。

```text
Set up Strata on this PC for me: https://github.com/Niko1221/Strata - follow docs/AI_SETUP.md in that repository.
```

AI はグラフィックスカード、RAM、ディスクを調べ、合うモデルを選んで、インストールと起動をし、アプリの接続方法を教えてくれます。
AI ツールは [MCP サーバー](docs/MCP_SERVER.md)を通じて、Strata のインストール、起動、停止もできます。

自分でビルドしたい方は[ソースからビルド](#ソースからビルド)を見てください。

## モデル

アプリと `strata launch` は、メモリ量に合うモデルをすすめます。モデルは1つで、圧縮の度合いが違う複数のサイズがあります。
小さいサイズは速く、大きいサイズは少し賢くなります。

### メモリが少ないなら Gyro

[Gyro](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF)（agentionai 製）は、このモデルをいちばん小さく動かせる方法です。

| サイズ | 必要なメモリ（RAM + VRAM） | ダウンロード | 内容 |
| --- | ---: | ---: | --- |
| **Gyro-S** | 27.6 GB | 58.5 GB | ほかのサイズが入らないPC向け。品質は IQ2_XS より下がります。 |
| **Gyro-M** | 35 GB | 92.0 GB | 品質は IQ3_XXS 程度で、GPUメモリの使用量は約20%少なくなります。回答の書き出しは Q2_0 より、12 GBの RTX 3060 で16%、R9700 で21%遅くなります。 |

Gyro は NVIDIA と AMD のカード（AMD は実験的）、および Linux 上の Intel Arc カード（Arc A750 で確認）で動きます。0.2.2 からは Apple Silicon の Mac でも動きます（32 GB の Mac mini で Gyro-S は 9 トークン/秒、エキスパートは SSD から読み込み）。詳細と測定結果は
[docs/GYRO.md](docs/GYRO.md) にあります。`strata launch --model gyro-s` で起動します。

### ほかのサイズ

| メモリ（RAM + VRAM） | 選ぶモデル | 備考 |
| --- | --- | --- |
| **32 GB** | **Gyro-S**、またはコード向けの **[Coder](docs/MODELS.md#coder)** | Coder はRAM 32 GBが必要で、エキスパートを半分だけ保持するため、コード以外は弱くなります |
| **40 GB** | **[Swift 1.5](docs/MODELS.md#swift-15)** IQ2_XS（39.2 GB）、または **Gyro-M**（35 GB） | Swift は答える前に考える時間が短くなります。Gyro-M は品質が IQ3_XXS に近いものの、回答の書き出しは Q2_0 より16〜21%遅くなります |
| **48 GB** | **IQ3_XXS**（47.0 GB） | 遅くなりますが、品質は上がります |
| **64 GB** | **IQ3_S**（54.8 GB）、オリジナルまたは Swift 1.5 | いちばん遅く、いちばん高品質です。公開されているテストでは完全版と同等です |
| **96 GB 以上** | Unsloth の **UD-IQ4_XS**（RAM 約72 GB）から **Q8_0**（約136 GB）まで、4〜8ビット | 完全版に近づきます。RAM が足りないと、Strata は一部を SSD から読むため、ずっと遅くなります |

表の行より RAM が少なくても動くことがあります。Strata は自動で[低RAMモード](docs/MODELS.md#a-big-graphics-card-and-little-ram)に切り替わります。
速度は落ち、グラフィックスカードが小さいとさらに大きく落ちます。

そのほかの選択肢:

- **[Swift 1.5](docs/MODELS.md#swift-15)** はファインチューン版で、答える前に考える時間が短くなります。品質と RAM の必要量はほぼ同じです。
- **[NVFP4](docs/NVFP4.md)** は NVIDIA カード向けの NVIDIA 製4ビット版です。すべてのエキスパートを RAM に置くには、約80 GBの RAM が必要です。
- **[Qwen3.6-35B-A3B](docs/QWEN36.md)** は、より小さい別のモデルです。画像を読め、複数のリクエストに同時に答えられます。
- **[OrcaRouter の Uncensored IQ3_XXS](docs/MODELS.md#orcarouter-uncensored-iq3_xxs)** は手動で設定します。

すべてのサイズ、ダウンロード、どの構成で何が動くかは [docs/MODELS.md](docs/MODELS.md) にあります。

### DeepSeek V4.1-Flash（ターミナル版、0.2.4 から）

[DeepSeek V4.1-Flash](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF) は、これとは別のずっと大きなモデルです。下の 3 つのセットは ISTA-DASLab が作りました。Strata は、12 GB のグラフィックスカード 1 枚と 46 から 64 GB の RAM を持つ PC でこれを動かし、残りは SSD から読みます。NVIDIA RTX 30 から 50 シリーズのカードと Radeon AI PRO R9700 で動きます。専用のエンジンパッケージ `strata-dsv41` を使い、DeepSeek を選ぶと `strata launch` がダウンロードします。デスクトップアプリはまだ対応していません。

| セット | ダウンロード | 元のモデルとの品質差 | 出力速度 |
| --- | ---: | --- | --- |
| **IQ1_M**（1.75 ビット） | 239 GB | KL 0.217、パープレキシティ 3.80（+21%）、コード +8.9%、エージェント作業 +10.2%、通常の英語の散文では弱めです。ISTA-DASLab のモデルカードによると、基本モデルのタスク平均の 98.7% を保ちます（AIME25、GPQA-Diamond、LiveCodeBench v6）。 | 未測定 |
| **IQ2_XXS** (2.00 ビット) | 257 GB | KL 0.170、パープレキシティ 3.63（+15.5%）、コード +8.6%、エージェント作業 +5.3% | RTX 3060 で毎秒約 8 から 10 トークン（コード 9.9、エージェント作業 7.8）、RTX 5070 で約 5 |
| **IQ2_XS** (2.25 ビット) | 274 GB | KL 0.107、パープレキシティ 3.37（+7.3%）、コード +3.2%、エージェント作業 +2.2% | Radeon AI PRO R9700 1 枚で毎秒約 11 トークン |
| **IQ4_NL**（Engram テーブル、3 つ目のモデルではありません） | 111 GB、上の 3 つのダウンロードに含まれています | モデルが RAM または SSD から読む 4 ビットの Engram 参照テーブルです。3 つのセットはこの 1 つのファイル（`DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf`）を共有します。 | 該当なし |

Radeon AI PRO R9700 4 枚：毎秒約 31 トークン、DSpark ありで 55 から 69。

`strata launch --model ist-das-2.00bpw`（または `ist-das-1.75bpw`、`ist-das-2.25bpw`）で起動します。詳細、カードごとの速度、手動インストールは [docs/DEEPSEEK41.md](docs/DEEPSEEK41.md) にあります。

## どのくらい速い？

12 GBのグラフィックスカード（RTX 5070、Ryzen 5 7600、RAM 64 GB）での数値です。ほかのカードの結果は
[各モデルの速度](docs/MODELS.md#how-fast-is-each-size)、[コミュニティの結果](docs/COMMUNITY_BENCHMARKS.md)を見てください。

| モデル | 回答が出る速さ | プロンプトを読む速さ |
| --- | --- | --- |
| **Q2_0** | `████████████` 94 tokens/s | `████████████` 2,650 tokens/s |
| **IQ2_XS** | `██████████░░` 79 tokens/s | `█████████░░░` 2,090 tokens/s |
| **IQ3_XXS** | `████████░░░░` 62 tokens/s | `████████░░░░` 1,750 tokens/s |
| **IQ3_S** | `███████░░░░░` 53 tokens/s | `███████░░░░░` 1,620 tokens/s |
| **Coder** | `███████░░░░░` 55 tokens/s | `██████████░░` 2,180 tokens/s |

1トークンは英単語の約¾にあたるので、毎秒60トークンは読む速さより速くなります。「プロンプトを読む速さ」は、長い文書やコード、チャット履歴を Strata が取り込む速さです。

16 GBの AMD カード（RX 9070 XT、Ryzen 9 3900X、RAM 47 GB）での測定値:

| モデル | 回答が出る速さ | プロンプトを読む速さ |
| --- | --- | --- |
| **Q2_0** | `████████░░░░` 60 tokens/s | `█████░░░░░░░` 1,160 tokens/s |
| **IQ2_XS** | `███████░░░░░` 52 tokens/s | `█████░░░░░░░` 1,110 tokens/s |
| **Coder** | `██████░░░░░░` 44 tokens/s | `██████░░░░░░` 1,420 tokens/s |

バーは2つの表で同じ目盛りです。VRAM が多いカードほど速くなり、RTX 3090（24 GB）なら毎秒100〜140トークンほどになるはずです。各数値のエンジンのバージョンは
[DETAILS.md](docs/DETAILS.md#speed-measured) にあります。

## 使い方

### アプリで

アプリには、**チャット**、モデルとGPU・CPU・RAMの状態を見る**モニター**、**モデル**ページ、**設定**があります。チャットはPC上のフォルダで作業できます。
**手動**モードでは、ファイルを変更したりコマンドを実行したりする前に確認します。`/loop 10m <やること>` と入力すると、チャットを開いている間、メッセージを10分ごとに繰り返し送ります。

チャットは DuckDuckGo（または Bing）でウェブ検索ができ、アカウントは不要です。**設定、ツール**で検索キーを追加することもできます。
検索は、PCの外に何かを送る唯一の機能で、送られるのは検索語です。

2つのモデルを動かすには、**複数のモデルを実行**をオンにして、それぞれのモデルに同じメーカーのカードを1枚ずつ割り当てます。

<p align="center"><img src="docs/media/app-chat.png" width="900" alt="The Strata app: a chat where the agent built a small voxel pagoda garden, with the page open in the preview on the right"><br>
<sub>Strata アプリでのチャット。エージェントが小さなボクセルの塔の庭を 1 つの HTML ファイルとして書き、そのページが会話の横のプレビューで開きます</sub></p>

### ターミナルで

<p align="center"><img src="docs/media/terminal-monitor.png" width="900" alt="The Strata Terminal monitor overview while the model writes a reply"><br>
<sub>モデルが返信を書いている間の <code>strata monitor</code>。速度、GPU、メモリ、最近のリクエストが表示されます</sub></p>

```sh
strata launch                                  # ガイド付きの画面: モデル、コンテキスト、画像、アクセス、そして開始
strata launch --model q2_0 --yes --no-tui     # 質問なし: 開始し、アドレスを表示して戻る
strata launch --plan --model q2_0              # 実行内容を表示するだけで、何も変更しない
strata launch --status                         # 今、何が動いているか
strata launch --stop                           # 停止する
strata monitor --url http://127.0.0.1:8080     # 速度、GPU、メモリ、リクエストを見る
strata update --check                          # 新しいエンジンがあるか確認（strata update でインストール）
strata doctor                                  # インストール済みのエンジンファイルを調べる
```

`strata --help` ですべてのコマンドを確認できます。起動とサーバーの稼働では、何もダウンロードしません。ダウンロードするのは、インストール、
`strata launch`（モデルを選んだとき）、`strata update` だけです。スマートフォンや別のPCから Strata に接続するには、
`strata launch --access lan` を使います。API キーが作られ、キーがない状態ではネットワーク上で起動しません。`--open` を付けない限り、ブラウザは開きません。
ガイド: [docs/TERMINAL_LAUNCH.md](docs/TERMINAL_LAUNCH.md)。クライアント全体: [docs/TERMINAL.md](docs/TERMINAL.md)。

### AI エージェントと使う

エージェントは、方法Bの1行コマンドで Strata をインストールし、
`strata launch --model NAME --yes --no-tui` を実行します。手順、フラグ、終了コードは [docs/AI_SETUP.md](docs/AI_SETUP.md) に、
リポジトリのルールは [AGENTS.md](AGENTS.md) にあります。

### ほかのアプリをつなぐ

| アプリの種類 | ベース URL |
| --- | --- |
| OpenAI 互換（ほとんどのアプリとエージェント） | `http://127.0.0.1:8080/v1`（API キーもモデル名も何でも可） |
| Anthropic API（Claude Code など） | `http://127.0.0.1:8080`、`/v1` は付けません。これらのアプリは `/v1/messages` を自分で付けます。Claude Code: `ANTHROPIC_BASE_URL=http://127.0.0.1:8080` |
| OpenAI Responses API（Codex CLI） | `http://127.0.0.1:8080/v1/responses`（[設定方法](docs/DETAILS.md#the-responses-api-and-codex-cli)） |

`strata connect` は、OpenCode、Codex CLI、Claude Code、oh-my-pi、pi、Hermes Agent、OpenClaw の正確な設定を表示します。
`strata connect opencode --write` は、確認したうえでファイルを書き込み、バックアップを残します。

### 画像

画像を読めるモデルでは、画像は初期設定でオンです。チャットに画像を添付または貼り付けるか、アプリから送ってください。
グラフィックスカードが24 GB未満のときは、画像リーダーがプロセッサで動きます（画像1枚につき数秒、カードのメモリは使いません）。24 GB以上のときはグラフィックスカードで動きます。
このルールは NVIDIA と AMD で同じです。画像はモデルごとにオフにでき、`--vision off`、`--vision gpu`、`--vision cpu` でリーダーを選べます。

このルールの根拠となる数値です。Radeon AI PRO R9700 では、カードのリーダーは260トークンの画像に57 ms、プロセッサのリーダーは3.1秒かかりました。
16 GBのカード（Tesla P100 と RTX A4000、Linux）では、カードのリーダーを使うとテキストの速度が3〜5%下がりました。
本来モデルが使う約1.4 GBをリーダーが占めるためです。Mac ではメモリ量にかかわらず GPU が読み取ります（32 GB の M4 で260トークンの画像に0.45秒、共有メモリを約1.2 GB使います）。プロセッサでは1枚あたり約50秒かかるため、`--vision cpu` を指定したときだけ使います。

### 知っておくと便利なこと

- 思考のレベル（**off、low、medium、high**）は、チャットのメニューか、お使いのアプリの「reasoning effort」で選びます。off がいちばん速くなります。
- Strata は初期設定では、一度に1つのリクエストに答えます。同時に複数処理するには、`strata launch` で `--parallel N` を使うか、設定に
  `"parallel": 2` を書きます（[BATCHING.md](docs/BATCHING.md)）。12 GBのカードでは、1つずつの回答が遅くなります。
- Strata はチャットの最初のメッセージを全文読みます。30,000トークンあたり約1分かかります。続きのメッセージは数秒で始まります。
- チャットの保存場所と API 全体: [docs/INSTALL.md](docs/INSTALL.md#where-things-are-stored)、[docs/DETAILS.md](docs/DETAILS.md#using-it)。

## ソースからビルド

<details>
<summary>リポジトリから Windows、Linux、Mac 用にビルドする</summary>

[Strata をダウンロード](https://github.com/Niko1221/Strata/archive/refs/heads/main.zip)して展開するか、`git clone` します。
**Windows:** `START-HERE.bat` をダブルクリックします。**Linux:** Strata フォルダで `./setup.sh` を実行します。

**Mac（Apple Silicon、macOS 15 以降）:** 最初に一度、`xcode-select --install` で Apple のコマンドラインツールをインストールします
（Xcode 本体は不要です）。Strata フォルダでターミナルを開き、`./setup.sh` を実行します。Mac 用エンジンをお使いの Mac 上でビルドし
（約7分）、同じ質問をして、モデルをダウンロードし、起動します。2回目以降は `./setup.sh` ですぐに起動します。
お使いの Mac に合うサイズは [docs/MAC.md](docs/MAC.md) を見てください。

インストーラーはカードを見つけて、合うエンジンを設定します。どのモデルとサイズにするか、コンテキストをどれだけにするか（モデルが一度に覚えておけるテキストの量）、
画像を読ませるかを聞かれます。おすすめの答えでよければ Enter を押します。そのあとモデル（約70 GB）をダウンロードして起動します。
ダウンロードが止まったら、もう一度実行すると続きから再開します。ブラウザで `http://127.0.0.1:8080` の Strata ページが開きます。

> モデルの起動中は、PCが1〜3分ほど（初回がいちばん長い）遅くなったり、応答しなくなったりすることがあります。Strata は35〜55 GBをRAMに読み込み、
> その一部をグラフィックスカード用に固定するためです。ウィンドウを閉じずに待ってください。

次回からは `START-HERE.bat`（または `./setup.sh`）をもう一度実行するだけです。すぐに起動し、同じものを2回ダウンロードすることはありません。ウィンドウを閉じるとモデルが止まります。
`UPDATE.bat`（`./update.sh`）は、起動せずに Strata を更新します。更新、Docker、複数のカード、ファイルの保存先、すべてのオプションは
[docs/INSTALL.md](docs/INSTALL.md) にあります。

PCの性能を測るには、`BENCH.bat`（`./bench.sh`）を実行します。5〜10分かかり、共有できるレポートが作られます
（[方法](docs/COMMUNITY_BENCHMARKS.md#run-the-benchmark)）。

**オプション: モデル全体がRAMに入るとき、1枚のカードをさらに速くする。** RubenTorney による、自分でオンにする2つの設定です（#1904、#1905）。
グラフィックスカードが1枚で、RAM がすべてのエキスパートを保持できるときに効果があります。たとえば64 GBのPCでの Q2_0 や IQ3_XXS です。
Strata のデータフォルダにある `strata-<model>.json`（[場所](docs/INSTALL.md#where-things-are-stored)）を開いて、次を追加します。

- `"args"` のリストに `"--adapt-async", "1"` を追加します。エキスパートキャッシュが、回答を止めずに補助スレッドで更新されます。
  RTX 5070（Windows、Q2_0）での測定: デコードが4〜5%、コードや長い文書では7〜10%速くなりました。回答は初期設定と少し違うことがありますが、実行ごとの結果は変わりません。
- `"env"` ブロックに `"STRATA_PIN_BY_NEED": "1"` を追加します（Windows）。Windows ではグラフィックスカード用に固定できる RAM が半分ほどしかありません。
  この設定は、カードが最初に必要とするエキスパートを優先して固定します。RTX 5070 での測定: プロンプトの読み取りが4.4%速くなりました。

```json
"args": [ ..., "--adapt-async", "1" ],
"env": { "STRATA_PIN_BY_NEED": "1" }
```

変更したら Strata を再起動してください。元に戻すには、2つの項目を削除します。

</details>

## 仕組み

このようなモデルは、ふつう数百GBのグラフィックスメモリを持つサーバーで動きます。お使いのグラフィックスカードは12〜24 GBです。Strata は、PC全体で作業を分担して、モデルを収めます。
図書館にたとえてみます。

- モデルは、24,576人の小さな専門家（「エキスパート」）からなるチームです。24,576冊の参考書のようなものです。1つの単語に必要なのは、そのうち10冊だけです。
- **グラフィックスカードは机です。** 小さくて速く、最もよく使う約4,000冊が開いて置いてあります。
- **RAM は閲覧室の書棚です。** 24,576冊すべてがここにあります。単語に必要な本が机になければ、プロセッサがすぐに書棚から読み、その間もカードは作業を続けます。
- **SSD は書庫です。** 29 GBの参照テーブルがしまってあります。モデルは単語ごとにその数行だけを読むので、RAM に置いておく必要はありません。

<p align="center"><img src="docs/media/how-it-works.svg" width="860" alt="The model's 24,576 experts: the busiest on the graphics card, all of them in RAM, a lookup table on the SSD"></p>

小さな助手が、次の数単語も予想します。大きなモデルがそれをまとめて確認し、正しいものだけを残します。答えは同じで、1.6〜1.8倍早く返ります。
長い文章は、一度に最大8,192トークンの大きな塊で、毎秒1,000トークンを超える速さで読みます。

<p align="center"><img src="docs/media/guess-and-check.svg" width="860" alt="A small helper guesses the next words; the big model checks them all at once and keeps the right ones"></p>

詳しい説明は [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) にあります。各部分と数値は
[詳細](docs/DETAILS.md#how-it-works)と[論文](docs/paper/Strata-Paper.pdf)を見てください。

## うまくいかないとき

- **Strata を初めて起動したときに、PCが固まりました。** モデルを RAM に読み込んでいます。ウィンドウを閉じずに待ってください。10分たっても固まったままなら、
  PCを再起動し、ほかのプログラムを閉じてからもう一度試すか、小さいサイズを選んでください。
- **ダウンロードやインストールの途中で止まりました。** アプリか `strata launch` をもう一度起動するか、`START-HERE.bat`（または
  `./setup.sh`）をもう一度実行してください。止まったところから再開します。
- **とても遅い、ディスクのランプが点滅し続ける、または「エンジンが予期せず停止しました」と表示される。** PCの空き RAM が足りません。
  ほかのプログラムを閉じるか（ブラウザはメモリを多く使います）、Q2_0 や IQ2_XS のような小さいサイズを選んでください。
- **このPCに合うエンジンがないと表示される。** アプリでは、**設定、エンジン**を開いて**エンジンを更新**を押します。ターミナルでは
  `strata update` を実行します。`strata doctor` で、どのエンジンファイルがインストールされているか分かります。
- **ポート 8080 はすでに使われていると表示される。** Strata がすでに動いています。ウィンドウを探してください。

そのほかの問題と対処法は [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) にあります。それでも解決しない場合は、
[issue](https://github.com/Niko1221/Strata/issues) を作成してください。アプリでは、**設定、エンジン、診断、ログを見る、ログを保存**で作られるファイルを添付します。
ターミナルでは、`strata doctor` の出力を貼り付けます。ソースから動かしている場合は、Strata フォルダの `strata-<model>.log` を添付してください。
セキュリティ上の問題は、非公開で報告してください: [SECURITY.md](SECURITY.md)。

## クレジットとライセンス

モデルは Qwen チームの [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) です。圧縮は
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)、UkisAI（Swift 1.5）、agentionai（Gyro）、
Unsloth が行いました。Strata は [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) の一部を使っています。すべてのクレジットは
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits) にあります。Strata は [MIT ライセンス](LICENSE)のオープンソースです。一部のコンポーネントと
すべてのモデルには、それぞれのライセンスがあります（[該当するもの](docs/HOW_IT_WORKS.md#license)）。

## コントリビューション

バグを見つけた、変更やベンチマークを送りたい、という方は、先に [CONTRIBUTING.md](CONTRIBUTING.md) を読んでください。
ほかの人があなたの成果を確認できるように、何を書くべきかが説明してあります。

## Strata を応援する

Strata は無料で、MIT ライセンスです。私が入手できるハードウェアで開発とテストをしています。お使いのPCで快適に動いたなら、コーヒー1杯分の支援は、
新しいカードのテストと、報告いただいた不具合の修正を続ける助けになります。無理のない範囲で大丈夫です。使って、気づいたことを報告してもらえるだけでも同じように助かります。

<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a>

Strata のスポンサーに関心がある方は、[GitHub](https://github.com/Niko1221) から連絡してください。
