<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="380">
  </picture>
</p>

<p align="center"><a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.de.md">Deutsch</a> · <b>Français</b> · <a href="README.es.md">Español</a> · <a href="README.pt-BR.md">Português</a></p>

<h3 align="center">Faites tourner un modèle de pointe sur votre propre PC de jeu</h3>

<p align="center">Compatible avec NVIDIA, AMD, Intel · Windows, Linux, Mac · gratuit et open source</p>

<p align="center">
  <a href="https://github.com/Niko1221/Strata/releases"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?include_prereleases&label=release" alt="Latest release"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="MIT license"></a>
  <a href="https://github.com/Niko1221/Strata/stargazers"><img src="https://img.shields.io/github/stars/Niko1221/Strata?style=flat" alt="GitHub stars"></a>
</p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="A voxel pagoda garden that Strata's model wrote, running in the browser"></a><br>
<sub>Un jardin de pagode en voxels, généré en une seule requête sur une RTX 5070 avec Strata (IQ3_S, contexte de 128K) ·
<a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">vidéo complète (49 s)</a></sub></p>

Strata fait tourner **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)**, un modèle de 125 milliards
de paramètres qui demande normalement un serveur, sur un PC ordinaire. Il discute, écrit du code, lit des images et
fonctionne avec vos applications et vos agents de code. Vos conversations et vos fichiers restent sur votre PC.

**Nouveau ici ?** Regardez [ce qu'il vous faut](#ce-quil-vous-faut), puis [installez l'application](#option-a--lapplication-strata).

- **Tourne sur un PC de jeu.** 32 Go de RAM et de mémoire graphique réunis suffisent pour commencer.
- **Fonctionne avec vos outils.** Il parle les API d'OpenAI et d'Anthropic, donc Claude Code, Codex CLI, OpenCode et d'autres agents peuvent s'y connecter.
- **Lit les images.** C'est activé par défaut.
- **Gratuit.** Licence MIT, pas de compte, pas de cloud.
- **Privé.** Aucune télémétrie. Ce que Strata télécharge, quand il utilise internet et comment il vérifie les téléchargements est expliqué dans [docs/SAFETY.md](docs/SAFETY.md).

Version 0.2.3 · moteur 0.1.48

## Ce qu'il vous faut

**Il vous faut au moins 32 Go de RAM et de mémoire graphique (VRAM) réunis.** Sur un Mac ou un Ryzen AI Max, c'est la
mémoire unifiée. Plus de mémoire permet un modèle plus gros ou plus rapide.

- Un PC de 32 Go avec une carte de 12 à 16 Go fait tourner le Coder.
- Avec 64 Go de RAM, toutes les tailles fonctionnent.

| Votre ordinateur | Détails |
| --- | --- |
| **NVIDIA** GeForce RTX série 20, 30, 40, 50 (12 Go ou plus) | Installez l'application ou le Terminal. |
| **AMD** Radeon RX 6800 et plus récent, Radeon AI PRO R9700 (12 Go ou plus) | [Cartes AMD](docs/AMD_HIP.md) |
| **Intel** Arc | [Intel Arc](docs/INTEL_ARC.md), [moteur Intel](docs/INTEL.md) |
| **Mac Apple Silicon** (M1 ou plus récent, macOS 15 ou plus récent) | [Mac](docs/MAC.md) |
| **AMD Ryzen AI Max (Strix Halo)** et autres systèmes à mémoire unifiée | [Strix Halo](docs/STRIX_HALO.md) |
| Cartes et processeurs plus anciens | [GPU anciens](docs/OLDER_GPUS.md), [CPU anciens](docs/INSTALL.md#older-cpus-experimental) |

Il faut aussi :

- environ 80 Go d'espace disque libre (un SSD accélère beaucoup le premier démarrage)
- Windows 10 / 11, Linux ou macOS
- un pilote graphique récent

Deux ou trois cartes de la même marque peuvent se partager un modèle ([multi-GPU](docs/MULTI_GPU.md)). Strata ne répartit
jamais un modèle entre des cartes NVIDIA et AMD. Liste complète : [docs/INSTALL.md](docs/INSTALL.md#what-you-need).

## Installation

Choisissez une option. Elles lancent toutes le même moteur et utilisent les mêmes modèles.

### Option A : l'application Strata

Le plus simple. L'application contient le moteur : installez-la, choisissez un modèle, discutez. Téléchargez le fichier
correspondant à votre système et à votre carte graphique depuis la [page des versions](https://github.com/Niko1221/Strata/releases/latest).
Vous ne savez pas lequel prendre ? Lisez le paragraphe sous le tableau.

| Système | Votre carte graphique | Fichier à télécharger |
| --- | --- | --- |
| **Windows 10 / 11** | NVIDIA RTX, pilote 580 ou plus récent | [`Strata-0.2.3-windows-cuda13.exe`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-windows-cuda13.exe) |
| | NVIDIA, carte plus ancienne ou pilote antérieur à 580 | [`Strata-0.2.3-windows-cuda12.exe`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-windows-cuda12.exe) |
| | AMD Radeon série RX 9000 ou Radeon AI PRO R9700 | [`Strata-0.2.3-windows-hip.exe`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-windows-hip.exe) |
| | AMD Radeon série RX 7000 ou Ryzen AI Max (Strix Halo) | [`Strata-0.2.3-windows-hip-rdna3.exe`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-windows-hip-rdna3.exe) |
| **Linux** | NVIDIA RTX, pilote 580 ou plus récent | [`Strata-0.2.3-linux-nvidia-cuda13.AppImage`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-nvidia-cuda13.AppImage) |
| | NVIDIA, carte plus ancienne ou pilote antérieur à 580 | [`Strata-0.2.3-linux-nvidia-cuda12.AppImage`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-nvidia-cuda12.AppImage) |
| | AMD Radeon série RX 9000 ou Radeon AI PRO R9700 | [`Strata-0.2.3-linux-amd-rx9070.AppImage`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-amd-rx9070.AppImage) |
| | AMD Radeon série RX 7000 | [`Strata-0.2.3-linux-amd-rx7900.AppImage`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-amd-rx7900.AppImage) |
| | AMD Ryzen AI Max (Strix Halo) | [`Strata-0.2.3-linux-amd-strixhalo.AppImage`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-linux-amd-strixhalo.AppImage) |
| **Mac** (Apple Silicon, macOS 15 ou plus récent) | intégrée | [`Strata-0.2.3-mac.dmg`](https://github.com/Niko1221/Strata/releases/latest/download/Strata-0.2.3-mac.dmg) |

**Si vous hésitez :** pour NVIDIA, prenez `cuda13` si la version de votre pilote est 580 ou plus récente, sinon `cuda12`.
`cuda12` convient aux cartes plus anciennes et aux pilotes plus anciens. Les fichiers `hip` et `amd` sont pour les cartes
AMD. Les autres cartes AMD (RX 6000) n'ont pas encore d'application prête à l'emploi : voir
[Cartes AMD](docs/AMD_HIP.md).

Ensuite :

- **Windows :** lancez le `.exe`. Il s'installe pour votre utilisateur et ne demande pas de droits d'administrateur. Si
  Windows affiche « Windows a protégé votre ordinateur », choisissez **Informations complémentaires**, puis **Exécuter
  quand même** (l'installateur n'est pas encore signé).
- **Linux :** rendez le fichier exécutable (`chmod +x`) et lancez-le. Il a besoin de FUSE 2 : `sudo apt install libfuse2`
  (Ubuntu 22.04) ou `libfuse2t64` (Ubuntu 24.04 et plus récent).
- **Mac :** ouvrez le `.dmg` et faites glisser Strata dans Applications.

Sur le premier écran, l'application analyse votre PC et recommande un modèle adapté. Appuyez sur Installer. Le modèle est un
gros téléchargement (environ 40 à 70 Go) qui reprend là où il s'est arrêté en cas d'interruption. Quand vous fermez la
fenêtre, Strata et le moteur s'arrêtent.

### Option B : le Terminal (une ligne)

Le Terminal est la commande `strata`, pensée pour les scripts, les serveurs et les agents d'IA.

Linux et Mac :

```sh
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh | sh
```

Windows (PowerShell) :

```powershell
irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

Le script analyse votre PC, télécharge le Terminal, vérifie son empreinte SHA-256 et le décompresse dans votre dossier
personnel. Il ne demande pas de droits d'administrateur. Sous Windows, ouvrez ensuite une nouvelle fenêtre de terminal.
Puis lancez Strata :

```sh
strata launch
```

Il pose une question à la fois : modèle, contexte, images, qui peut se connecter. Ensuite il télécharge le moteur pour votre
carte graphique (signé, et lancé à titre d'essai avant usage) ainsi que le modèle, puis commence à servir. Pour les scripts
et les agents d'IA, utilisez des options :

```sh
strata launch --model q2_0 --yes --no-tui
```

Pour lire le script d'installation avant de l'exécuter, téléchargez-le d'abord avec `curl -fsSL <address> -o install.sh`,
lisez-le, puis lancez `sh install.sh`. Tous les écrans et toutes les options : [docs/TERMINAL_LAUNCH.md](docs/TERMINAL_LAUNCH.md).

### Option C : laissez votre IA l'installer

Collez ceci dans Claude Code, Cursor, Codex, GitHub Copilot ou un autre assistant de programmation :

```text
Set up Strata on this PC for me: https://github.com/Niko1221/Strata - follow docs/AI_SETUP.md in that repository.
```

L'assistant vérifie votre carte graphique, votre RAM et votre disque, choisit le modèle adapté, l'installe, le démarre et
vous explique comment connecter vos applications. Les outils d'IA peuvent aussi installer, démarrer et arrêter Strata via son
[serveur MCP](docs/MCP_SERVER.md).

Vous préférez le compiler vous-même ? Voir [Compiler depuis les sources](#compiler-depuis-les-sources).

## Modèles

L'application et `strata launch` recommandent un modèle selon votre mémoire. C'est un seul modèle en plusieurs tailles,
plus ou moins compressé. Les petites tailles sont plus rapides. Les grandes sont un peu plus intelligentes.

### Peu de mémoire ? Essayez Gyro

[Gyro](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF) (par agentionai) est la façon la plus légère de faire tourner ce modèle.

| Taille | Mémoire nécessaire (RAM + VRAM) | Téléchargement | Ce que c'est |
| --- | ---: | ---: | --- |
| **Gyro-S** | 27,6 Go | 58,5 Go | Pour les PC où rien d'autre ne tient. La qualité est inférieure à celle d'IQ2_XS. |
| **Gyro-M** | 35 Go | 92,0 Go | À peu près la qualité d'IQ3_XXS avec environ 20 % de mémoire GPU en moins. Il écrit les réponses 16 % plus lentement que Q2_0 sur une RTX 3060 de 12 Go et 21 % plus lentement sur une R9700. |

Gyro fonctionne sur les cartes NVIDIA et AMD (expérimental sur AMD) sur les cartes Intel Arc sous Linux (testé sur une Arc A750) et, depuis la 0.2.2, sur les Mac Apple Silicon (Gyro-S sur un Mac mini de 32 Go : 9 tokens/s, experts lus sur le SSD). Détails et mesures :
[docs/GYRO.md](docs/GYRO.md). Lancez-le avec `strata launch --model gyro-s`.

### Les autres tailles

| Votre mémoire (RAM + VRAM) | Prenez | Remarques |
| --- | --- | --- |
| **32 Go** | **Gyro-S**, ou le **[Coder](docs/MODELS.md#coder)** pour le code | Le Coder demande 32 Go de RAM et garde la moitié des experts, il est donc moins bon hors du code |
| **40 Go** | **[Swift 1.5](docs/MODELS.md#swift-15)** IQ2_XS (39,2 Go), ou **Gyro-M** (35 Go) | Swift réfléchit moins longtemps avant de répondre. Gyro-M approche la qualité d'IQ3_XXS, mais il écrit les réponses 16 à 21 % plus lentement que Q2_0 |
| **48 Go** | **IQ3_XXS** (47,0 Go) | Plus lent, meilleur |
| **64 Go** | **IQ3_S** (54,8 Go), l'original ou Swift 1.5 | Le plus lent, le meilleur. Il égale le modèle complet dans les tests publiés |
| **96 Go ou plus** | **UD-IQ4_XS** d'Unsloth (environ 72 Go de RAM) jusqu'à **Q8_0** (environ 136 Go), 4 à 8 bits | Plus proche du modèle complet. Avec moins de RAM, Strata lit une partie depuis le SSD, ce qui est beaucoup plus lent |

Avoir moins de RAM que ce qu'indique une ligne peut encore fonctionner : Strata passe tout seul en
[mode peu de RAM](docs/MODELS.md#a-big-graphics-card-and-little-ram). Attendez-vous à plus de lenteur, et à bien plus avec
une petite carte graphique.

Autres choix :

- **[Swift 1.5](docs/MODELS.md#swift-15)** est une version affinée qui réfléchit moins longtemps avant de répondre, pour une qualité et un besoin en RAM à peu près identiques.
- **[NVFP4](docs/NVFP4.md)** est la version 4 bits de NVIDIA pour les cartes NVIDIA. Elle demande environ 80 Go de RAM pour garder tous les experts en RAM.
- **[Qwen3.6-35B-A3B](docs/QWEN36.md)** est un modèle plus petit et différent, qui lit les images et peut répondre à plusieurs requêtes à la fois.
- **[L'Uncensored IQ3_XXS d'OrcaRouter](docs/MODELS.md#orcarouter-uncensored-iq3_xxs)** se configure à la main.

Toutes les tailles, les téléchargements et ce qui tient où : [docs/MODELS.md](docs/MODELS.md).

### DeepSeek V4.1-Flash (dans le Terminal, depuis 0.2.4)

[DeepSeek V4.1-Flash](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF) est un autre modèle, beaucoup plus gros. ISTA-DASLab a produit les trois versions ci-dessous. Strata les fait tourner sur un PC avec une carte graphique de 12 Go et de 46 à 64 Go de RAM, et lit le reste sur votre SSD. Il fonctionne sur les cartes NVIDIA RTX des séries 30 à 50 et sur la Radeon AI PRO R9700. Il utilise son propre paquet de moteur, `strata-dsv41`, que `strata launch` télécharge quand vous choisissez DeepSeek. L’application de bureau ne le propose pas encore.

| Version | Téléchargement | Qualité par rapport au modèle d’origine | Écrit à |
| --- | ---: | --- | --- |
| **IQ1_M** (1,75 bit) | 239 Go | KL 0,217, perplexité 3,80 (+21 %) ; code +8,9 %, tâches d’agent +10,2 % ; plus faible sur la prose anglaise courante. D’après la fiche du modèle d’ISTA-DASLab, elle garde 98,7 % de la moyenne des tâches du modèle de base (AIME25, GPQA-Diamond, LiveCodeBench v6). | pas encore mesurée |
| **IQ2_XXS** (2.00 bits) | 257 GB | KL 0,170, perplexité 3,63 (+15,5 %) ; code +8,6 %, tâches d’agent +5,3 % | environ 8 à 10 tokens/s sur une RTX 3060 (code 9,9, tâches d’agent 7,8), environ 5 sur une RTX 5070 |
| **IQ2_XS** (2.25 bits) | 274 GB | KL 0,107, perplexité 3,37 (+7,3 %) ; code +3,2 %, tâches d’agent +2,2 % | environ 11 tokens/s sur une Radeon AI PRO R9700 |
| **IQ4_NL** (table Engram, pas un troisième modèle) | 111 Go, déjà compris dans les trois téléchargements ci-dessus | La table de consultation Engram en 4 bits que le modèle lit dans la RAM ou sur le SSD. Les trois versions utilisent ce seul fichier (`DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf`). | n/a |

4x Radeon AI PRO R9700 : environ 31 tokens/s, 55 à 69 avec DSpark.

Lancez-le avec `strata launch --model ist-das-2.00bpw` (ou `ist-das-1.75bpw`, `ist-das-2.25bpw`). Détails, vitesses par carte et installation manuelle : [docs/DEEPSEEK41.md](docs/DEEPSEEK41.md).

## Quelle est sa vitesse ?

Mesuré avec une carte graphique de 12 Go (RTX 5070, Ryzen 5 7600, 64 Go de RAM). Voir les résultats d'autres cartes :
[vitesse de chaque taille](docs/MODELS.md#how-fast-is-each-size), [résultats de la communauté](docs/COMMUNITY_BENCHMARKS.md).

| Modèle | Les réponses s'affichent à | Lit votre prompt à |
| --- | --- | --- |
| **Q2_0** | `████████████` 94 tokens/s | `████████████` 2 650 tokens/s |
| **IQ2_XS** | `██████████░░` 79 tokens/s | `█████████░░░` 2 090 tokens/s |
| **IQ3_XXS** | `████████░░░░` 62 tokens/s | `████████░░░░` 1 750 tokens/s |
| **IQ3_S** | `███████░░░░░` 53 tokens/s | `███████░░░░░` 1 620 tokens/s |
| **Coder** | `███████░░░░░` 55 tokens/s | `██████████░░` 2 180 tokens/s |

Un token vaut environ ¾ de mot, donc 60 tokens par seconde, c'est plus rapide que votre lecture. « Lit votre prompt »
indique la vitesse à laquelle Strata absorbe un long document, du code ou un historique de discussion.

Mesuré avec une carte AMD de 16 Go (RX 9070 XT, Ryzen 9 3900X, 47 Go de RAM) :

| Modèle | Les réponses s'affichent à | Lit votre prompt à |
| --- | --- | --- |
| **Q2_0** | `████████░░░░` 60 tokens/s | `█████░░░░░░░` 1 160 tokens/s |
| **IQ2_XS** | `███████░░░░░` 52 tokens/s | `█████░░░░░░░` 1 110 tokens/s |
| **Coder** | `██████░░░░░░` 44 tokens/s | `██████░░░░░░` 1 420 tokens/s |

Les barres utilisent la même échelle dans les deux tableaux. Une carte avec plus de VRAM est plus rapide. Une RTX 3090
(24 Go) devrait écrire environ 100 à 140 tokens par seconde. La version du moteur derrière chaque chiffre est indiquée dans
[DETAILS.md](docs/DETAILS.md#speed-measured).

## Utilisation

### Dans l'application

L'application comprend un **Chat**, un **Monitor** du modèle, de votre GPU, de votre CPU et de votre RAM, une page **Models** et les **Settings**. Le chat peut travailler dans un dossier de votre PC. En mode **Manual**, il demande
avant de modifier un fichier ou d'exécuter une commande. Tapez `/loop 10m <what to do>` pour répéter un message toutes les
10 minutes tant que le chat est ouvert.

Le chat peut chercher sur le web avec DuckDuckGo (ou Bing), sans compte. Vous pouvez ajouter une clé de recherche sous
**Settings, Tools**. La recherche est la seule fonction qui envoie quelque chose hors de votre PC (les mots recherchés).

Pour faire tourner deux modèles, activez **Run multiple models** (exécuter plusieurs modèles) et donnez à chacun sa propre carte de la même marque.

<p align="center"><img src="docs/media/app-chat.png" width="900" alt="The Strata app: a chat where the agent built a small voxel pagoda garden, with the page open in the preview on the right"><br>
<sub>Une discussion dans l'app Strata : l'agent a écrit un petit jardin de pagode en voxels dans un seul fichier HTML, et la page s'ouvre dans l'aperçu à côté de la conversation</sub></p>

### Dans le Terminal

<p align="center"><img src="docs/media/terminal-monitor.png" width="900" alt="The Strata Terminal monitor overview while the model writes a reply"><br>
<sub><code>strata monitor</code> pendant que le modèle répond : vitesse, GPU, mémoire et les dernières requêtes</sub></p>

```sh
strata launch                                  # guided screens: model, context, pictures, access, then serve
strata launch --model q2_0 --yes --no-tui     # no questions: serve, print the address, return
strata launch --plan --model q2_0              # show what would happen; nothing changes
strata launch --status                         # what is being served now
strata launch --stop                           # stop it
strata monitor --url http://127.0.0.1:8080     # watch speed, GPU, memory and requests
strata update --check                          # is there a new engine? (strata update installs it)
strata doctor                                  # check the installed engine files
```

`strata --help` liste tout. Démarrer et servir ne téléchargent rien. Seuls l'installation, `strata launch` (quand vous
choisissez un modèle) et `strata update` le font. Pour joindre Strata depuis votre téléphone ou un autre PC, utilisez
`strata launch --access lan`. La commande crée une clé d'API et refuse de démarrer sur un réseau sans elle. Aucun
navigateur ne s'ouvre, sauf si vous ajoutez `--open`. Guide : [docs/TERMINAL_LAUNCH.md](docs/TERMINAL_LAUNCH.md). Le client complet : [docs/TERMINAL.md](docs/TERMINAL.md).

### Avec des agents d'IA

Un agent installe Strata avec la commande en une ligne de l'option B et lance
`strata launch --model NAME --yes --no-tui`. Les étapes, les options et les codes de sortie sont dans [docs/AI_SETUP.md](docs/AI_SETUP.md),
et les règles du dépôt dans [AGENTS.md](AGENTS.md).

### Connecter vos autres applications

| Type d'application | URL de base |
| --- | --- |
| Compatible OpenAI (la plupart des applications et des agents) | `http://127.0.0.1:8080/v1` (n'importe quelle clé d'API, n'importe quel nom de modèle) |
| API Anthropic (Claude Code et d'autres) | `http://127.0.0.1:8080`, sans `/v1`. Ces applications ajoutent `/v1/messages` elles-mêmes. Claude Code : `ANTHROPIC_BASE_URL=http://127.0.0.1:8080` |
| API OpenAI Responses (Codex CLI) | `http://127.0.0.1:8080/v1/responses` ([configuration](docs/DETAILS.md#the-responses-api-and-codex-cli)) |

`strata connect` affiche les réglages exacts pour OpenCode, Codex CLI, Claude Code, oh-my-pi, pi, Hermes Agent et OpenClaw.
`strata connect opencode --write` écrit le fichier après confirmation et garde une sauvegarde.

### Images

Les images sont activées par défaut pour les modèles qui savent les lire. Joignez ou collez une image dans le chat, ou
envoyez-la depuis votre application. Le lecteur d'images tourne sur le processeur quand votre carte graphique a moins de
24 Go (quelques secondes par image, sans prendre de mémoire à la carte) et sur la carte graphique à partir de 24 Go. La
règle est la même pour NVIDIA et AMD. Vous pouvez désactiver les images par modèle, ou choisir le lecteur avec
`--vision off`, `--vision gpu` ou `--vision cpu`.

Les chiffres derrière la règle : sur une Radeon AI PRO R9700, le lecteur de la carte a pris 57 ms pour une image de
260 tokens et celui du processeur 3,1 s. Sur des cartes de 16 Go (Tesla P100 et RTX A4000, Linux), le lecteur de la carte
a coûté 3 à 5 % de vitesse sur le texte, parce qu'il occupe environ 1,4 Go que le modèle utiliserait sinon. Sur un Mac, le lecteur tourne sur le GPU quelle que soit la mémoire (0,45 s pour une image de 260 tokens sur un M4 de 32 Go, environ 1,2 Go de la
mémoire partagée) ; le processeur met environ 50 s par image, il ne lit donc qu'avec `--vision cpu`.

### Bon à savoir

- Choisissez le niveau de réflexion (**désactivé, faible, moyen ou élevé**) dans le menu du chat ou dans le réglage « reasoning effort » de votre application. Désactivé est le plus rapide.
- Strata répond à une requête à la fois par défaut. Pour en traiter plusieurs en même temps, utilisez `--parallel N` dans `strata launch` ou
  `"parallel": 2` dans la configuration ([BATCHING.md](docs/BATCHING.md)). Sur une carte de 12 Go, chaque réponse devient alors plus lente.
- Strata lit en entier le premier message d'une discussion, soit environ 1 minute pour 30 000 tokens. Les messages suivants démarrent en quelques secondes.
- Où sont stockées vos discussions et l'API complète : [docs/INSTALL.md](docs/INSTALL.md#where-things-are-stored), [docs/DETAILS.md](docs/DETAILS.md#using-it).

## Compiler depuis les sources

<details>
<summary>Windows, Linux et Mac depuis le dépôt</summary>

[Téléchargez Strata](https://github.com/Niko1221/Strata/archive/refs/heads/main.zip) et décompressez-le, ou faites un `git clone`.
**Windows :** double-cliquez sur `START-HERE.bat`. **Linux :** lancez `./setup.sh` dans le dossier Strata.

**Mac (Apple Silicon, macOS 15 ou plus récent) :** installez une fois les Command Line Tools d'Apple avec `xcode-select --install`
(Xcode lui-même n'est pas nécessaire). Ouvrez le Terminal dans le dossier Strata et lancez `./setup.sh`. Le script compile
le moteur Mac sur votre Mac (environ 7 minutes), pose les mêmes questions, télécharge le modèle et le démarre. La fois
suivante, `./setup.sh` le démarre tout de suite. Quelle taille convient à votre Mac : [docs/MAC.md](docs/MAC.md).

L'installateur détecte votre carte et prépare le bon moteur. Il demande quel modèle et quelle taille, combien de contexte
(la quantité de texte que le modèle garde en tête) et s'il doit lire les images. Appuyez sur Entrée pour la réponse
recommandée. Puis il télécharge le modèle (environ 70 Go) et le démarre. Si le téléchargement s'arrête, relancez le script
et il reprend. Votre navigateur ouvre la page de Strata à l'adresse `http://127.0.0.1:8080`.

> Pendant le démarrage du modèle, votre PC peut être lent ou ne plus répondre pendant 1 à 3 minutes (le plus longtemps la
> première fois). Strata charge 35 à 55 Go dans votre RAM et en verrouille une partie pour la carte graphique. Patientez et
> ne fermez pas la fenêtre.

La fois suivante, relancez `START-HERE.bat` (ou `./setup.sh`). Il démarre tout de suite et ne télécharge rien deux fois.
Fermez sa fenêtre pour arrêter le modèle. `UPDATE.bat` (`./update.sh`) met Strata à jour sans le démarrer. Mise à jour,
Docker, plusieurs cartes, où vont les fichiers et toutes les options : [docs/INSTALL.md](docs/INSTALL.md).

Pour mesurer votre PC, lancez `BENCH.bat` (`./bench.sh`). Cela prend 5 à 10 minutes et produit un rapport que vous pouvez
partager ([comment](docs/COMMUNITY_BENCHMARKS.md#run-the-benchmark)).

**Facultatif : plus de vitesse sur une carte quand tout le modèle tient en RAM.** Deux réglages à activer soi-même, par
RubenTorney (#1904, #1905). Ils aident quand vous utilisez une seule carte graphique et que votre RAM contient tous les
experts, par exemple Q2_0 ou IQ3_XXS sur un PC de 64 Go. Ouvrez `strata-<model>.json` dans le dossier de données de Strata ([où](docs/INSTALL.md#where-things-are-stored)) et ajoutez :

- `"--adapt-async", "1"` à la liste `"args"`. Le cache des experts se met alors à jour sur un thread auxiliaire au lieu de
  mettre la réponse en pause. Mesuré sur une RTX 5070 (Windows, Q2_0) : +4 à 5 % en décodage, +7 à 10 % sur le code et les
  longs documents. Les réponses peuvent différer légèrement de celles du réglage par défaut, et restent identiques d'une exécution à l'autre.
- `"STRATA_PIN_BY_NEED": "1"` dans le bloc `"env"` (Windows). Windows ne peut verrouiller qu'environ la moitié de la RAM
  pour la carte graphique, et ce réglage verrouille d'abord les experts dont la carte a le plus besoin. Mesuré sur la RTX 5070 : +4,4 % en lecture du prompt.

```json
"args": [ ..., "--adapt-async", "1" ],
"env": { "STRATA_PIN_BY_NEED": "1" }
```

Redémarrez Strata après la modification. Pour annuler, supprimez les deux entrées.

</details>

## Comment ça marche ?

Les modèles de ce genre tournent d'habitude sur des serveurs qui disposent de centaines de gigaoctets de mémoire graphique.
Votre carte graphique en a 12 à 24. Strata fait tenir le modèle en répartissant le travail sur tout votre PC. Imaginez une
bibliothèque.

- Le modèle est une équipe de 24 576 petits spécialistes (les « experts »), comme 24 576 ouvrages de référence. Chaque mot n'en demande que 10.
- **Votre carte graphique est le bureau.** Il est petit et rapide. Les quelque 4 000 livres les plus utilisés y sont ouverts.
- **Votre RAM, ce sont les étagères de la salle de lecture.** Les 24 576 livres s'y trouvent. Quand un mot demande un livre
  qui n'est pas sur le bureau, votre processeur va le chercher aussitôt sur l'étagère, pendant que la carte continue de travailler.
- **Votre SSD, ce sont les archives.** Il contient une table de consultation de 29 Go. Le modèle n'en lit que quelques lignes par
  mot, elle n'a donc pas besoin de tenir en RAM.

<p align="center"><img src="docs/media/how-it-works.svg" width="860" alt="The model's 24,576 experts: the busiest on the graphics card, all of them in RAM, a lookup table on the SSD"></p>

Un petit assistant devine aussi les mots suivants. Le grand modèle les vérifie tous d'un coup et garde les bons. Vous
obtenez la même réponse, 1,6 à 1,8 fois plus tôt. Les textes longs sont lus par gros morceaux, jusqu'à 8 192 tokens à la
fois, à plus de 1 000 tokens par seconde.

<p align="center"><img src="docs/media/guess-and-check.svg" width="860" alt="A small helper guesses the next words; the big model checks them all at once and keeps the right ones"></p>

L'explication plus longue : [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md). Chaque élément avec ses chiffres :
[les détails](docs/DETAILS.md#how-it-works) et l'[article](docs/paper/Strata-Paper.pdf).

## Un problème ?

- **Mon PC s'est figé au premier démarrage de Strata.** Il charge le modèle en RAM. Patientez et ne fermez pas la fenêtre.
  Toujours figé après 10 minutes ? Redémarrez le PC, fermez les autres programmes et réessayez, ou choisissez une taille plus petite.
- **Ça s'est arrêté pendant le téléchargement ou l'installation.** Relancez l'application ou `strata launch`, ou relancez
  `START-HERE.bat` (ou `./setup.sh`). Cela reprend là où ça s'était arrêté.
- **C'est très lent, le voyant du disque clignote sans arrêt, ou il affiche « the engine stopped unexpectedly ».** Votre PC
  n'a pas assez de RAM libre. Fermez d'autres programmes (les navigateurs en consomment beaucoup) ou choisissez une taille plus petite, comme Q2_0 ou IQ2_XS.
- **Il indique qu'il n'y a pas de moteur pour ce PC.** Dans l'application, ouvrez **Settings, Engine** et appuyez sur **Update engine** (mettre à jour le moteur). Dans le Terminal, lancez `strata update`. `strata doctor` montre quels fichiers du moteur sont installés.
- **Il indique que le port 8080 est déjà utilisé.** Strata tourne déjà. Cherchez sa fenêtre.

Autres problèmes et solutions : [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). Toujours bloqué ? Ouvrez une
[issue](https://github.com/Niko1221/Strata/issues). Dans l'application, joignez le fichier obtenu par **Settings, Engine, Diagnostics, View logs, Save logs**. Dans le Terminal, collez la sortie de `strata doctor`. Si
vous avez lancé depuis les sources, joignez `strata-<model>.log` du dossier Strata. Pour un problème de sécurité,
signalez-le en privé : [SECURITY.md](SECURITY.md).

## Remerciements et licence

Le modèle est [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), de l'équipe Qwen. Il a été compressé par
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), UkisAI (Swift 1.5), agentionai (Gyro)
et Unsloth. Strata utilise des parties de [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). Tous les
remerciements : [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits). Strata est open source sous [licence MIT](LICENSE).
Quelques parties et tous les modèles ont leurs propres licences ([lesquelles](docs/HOW_IT_WORKS.md#license)).

## Contribuer

Vous avez trouvé un bug, ou vous voulez proposer une modification ou un benchmark ? Lisez d'abord [CONTRIBUTING.md](CONTRIBUTING.md).
Il indique ce qu'il faut inclure pour que d'autres puissent vérifier votre travail.

## Soutenir Strata

Strata est gratuit et sous licence MIT. Je le construis et je le teste sur le matériel que je peux obtenir. S'il tourne bien sur votre PC, un café m'aide
à continuer de tester de nouvelles cartes et à corriger ce que vous signalez. Aucune obligation. L'utiliser et signaler ce que vous constatez aide tout autant.

<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a>

Vous souhaitez sponsoriser Strata ? Contactez-moi via [GitHub](https://github.com/Niko1221).
