<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.md">English</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/mcp-tool-shop-org/brand/main/logos/backpropagate/readme.png" alt="Backpropagate" width="400">
</p>

<p align="center">
  <a href="https://github.com/mcp-tool-shop-org/backpropagate/actions/workflows/ci.yml"><img src="https://github.com/mcp-tool-shop-org/backpropagate/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/backpropagate/"><img src="https://img.shields.io/pypi/v/backpropagate" alt="PyPI"></a>
  <a href="https://codecov.io/gh/mcp-tool-shop-org/backpropagate"><img src="https://img.shields.io/codecov/c/github/mcp-tool-shop-org/backpropagate" alt="Coverage"></a>
  <a href="https://scorecard.dev/viewer/?uri=github.com/mcp-tool-shop-org/backpropagate"><img src="https://api.scorecard.dev/projects/github.com/mcp-tool-shop-org/backpropagate/badge" alt="OpenSSF Scorecard"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License"></a>
  <a href="https://mcp-tool-shop-org.github.io/backpropagate/"><img src="https://img.shields.io/badge/Landing_Page-live-blue" alt="Landing Page"></a>
</p>

# 32B QLoRA या 7B एंड-टू-एंड मॉडल को एक GPU पर फाइन-ट्यून करें। इसे ओलामा में भेजें।

एकल GPU पर बड़े भाषा मॉडलों का फाइन-ट्यूनिंग बैकप्रोपेगेट करें, जो आपके पास मौजूद कार्ड के आकार के अनुसार हो। तीन पंक्तियों वाला पायथन QLoRA एक 7B–34B मॉडल को एक 32 GB उपभोक्ता कार्ड (RTX 5090) पर चलाता है; एक फ़्लैग — `--full-ft-offload` — 7B-क्लास मॉडल का पूर्ण फाइन-ट्यूनिंग करता है, जिसके लिए ऑप्टिमाइज़र स्टेट को होस्ट RAM में भेजा जाता है। एक और कमांड ओलामा में निर्यात करता है, फिर `ollama run` के साथ आपका फाइन-ट्यून चलाता है। यह आसानी से 16 GB तक स्केल हो सकता है। विंडोज पर उत्कृष्ट प्रदर्शन।

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")
trainer.train("my_data.jsonl", steps=100)
trainer.export("gguf", quantization="q4_k_m")
```

```bash
backprop export ./output/lora --format gguf --quantization q4_k_m --ollama --ollama-name my-model
ollama run my-model
```

बस इतना ही। कोई YAML कॉन्फ़िगरेशन फ़ाइल नहीं है। कोई `accelerate launch` समारोह नहीं है। कोई अलग "अब इसे GGUF में बदलें" ट्यूटोरियल नहीं है। यदि आपके पास एक CUDA GPU और आपके प्रशिक्षण डेटा वाली एक JSONL फ़ाइल है, तो आप एक कार्यात्मक फाइनट्यून से केवल तीन पंक्तियों की दूरी पर हैं।

## इंस्टॉल करें

```bash
# Recommended: isolated Python install (no conflicts with system Python or other projects)
pipx install backpropagate

# Or via uv (faster install, same isolation)
uv tool install backpropagate

# Standard pip (if you manage your own virtualenv)
pip install backpropagate
```

यदि आप वैकल्पिक सुविधाओं को चाहते हैं, तो इंस्टॉलेशन को इनमें से किसी एक से बदलें:

```bash
pipx install "backpropagate[standard]"   # adds Unsloth (2x faster training) + the web UI
pipx install "backpropagate[full]"       # adds everything: unsloth, ui, monitoring, export, etc.
```

क्या आप डॉकर का उपयोग करना पसंद करते हैं? `docker pull ghcr.io/mcp-tool-shop-org/backpropagate:latest` भी काम करेगा। दोनों `linux/amd64` और `linux/arm64` के लिए इमेज उपलब्ध हैं, इसलिए Apple सिलिकॉन और ARM लिनक्स उपयोगकर्ताओं को एक मूल इमेज मिलेगी। "कंटेनर में UI" के लिए एक मानक `compose.yaml` रिपॉजिटरी की रूट में मौजूद है - `docker compose up` वेब UI को `http://localhost:7860` पर एक स्थायी `~/.backpropagate` वॉल्यूम माउंट के साथ लाता है।

## बैकप्रोपैगेट किस क्षेत्र में काम करता है

LLM को ठीक करने के लिए कई अच्छी लाइब्रेरी उपलब्ध हैं। वे सभी अलग-अलग चीजों में उत्कृष्ट हैं:

- **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)** - यदि आप YAML कॉन्फ़िगरेशन पसंद करते हैं और आपको कॉपी करने के लिए व्यंजनों का एक समुदाय चाहिए।
- **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** - यदि आप DPO/PPO/RLHF और एक वेब GUI चाहते हैं।
- **[Unsloth](https://github.com/unslothai/unsloth)** - यदि आपको सबसे तेज़ प्रशिक्षण की आवश्यकता है और आप समर्थित मॉडल परिवार पर हैं।
- **[torchtune](https://github.com/pytorch/torchtune)** - यदि आप मेटा के पहले-पक्षीय PyTorch-नेटिव व्यंजनों को संपादित करना चाहते हैं।

बैकप्रोपैगेट वह विकल्प है जिसकी कमी थी: **एक एकल उपभोक्ता GPU पर अकेले ऑपरेटरों के लिए 3-पंक्ति पायथन API जो एक एडाप्टर को प्रशिक्षित करना और उसे भेजना चाहते हैं।** कोई YAML नहीं, कोई GUI नहीं, कोई ऑनलाइन RL (PPO/GRPO) नहीं, कोई मल्टी-नोड नहीं। बस वह लूप जिसकी वास्तव में हर किसी को आवश्यकता है और वह निर्यात चरण जो बाधा डालता है।

यदि आपने ऊपर दी गई लाइब्रेरी में से किसी एक को आज़माया और कॉन्फ़िगरेशन-फ़ाइल समारोह से निराश हो गए, या मॉडल-परिवार की कमी का सामना किया, या विंडोज-प्रथम डिफ़ॉल्ट्स चाहते थे - तो बैकप्रोपैगेट आपके लिए है।

## आप एक GPU पर क्या फाइन-ट्यून कर सकते हैं।

बैकप्रोपेगेट आपके कार्ड के अनुसार रन का आकार निर्धारित करता है। यहां 32 GB उपभोक्ता GPU (RTX 5090) और 64 GB होस्ट RAM पर व्यावहारिक सीमा दी गई है — जिस डिवाइस पर इसे ट्यून किया गया है:

| मॉडल का आकार | विधि | 32 GB कार्ड पर स्थिति |
|---|---|---|
| 7B (Qwen 2.5 7B / Llama-3.1-8B / Mistral 7B) | QLoRA | आरामदायक — ~7–8 GB। पूर्ण अनुक्रम लंबाई, बहुत अधिक जगह। |
| **14B** (Qwen2.5-14B) | QLoRA | **दैनिक उपयोग के लिए सबसे अच्छा — ~8.5 GB**, मापा गया। रैंक/अल्फा 32, पेज किया हुआ 8-बिट AdamW, 4096 ctx। |
| 24B (Mistral-Small-24B) | QLoRA | ~18 GB। 4096 ctx पर पर्याप्त जगह के साथ फिट बैठता है। |
| **32B** (Qwen2.5-32B) | QLoRA | **बस फिट बैठता है — ~26 GB**, `max_len 2048` + पेज किया हुआ 8-बिट AdamW पर। सीमा का शीर्ष। |
| ≤6B | `mode="full"` (पूर्ण फाइन-ट्यूनिंग) | शुद्ध-GPU पूर्ण FT — bf16 वजन, कोई एडाप्टर नहीं। कार्ड-जागरूक सीमा 32 GB पर 6B है। |
| **7B-क्लास** (Qwen 2.5 7B / Llama-3.1-8B / Mistral 7B) | `mode="full" --full-ft-offload` | **FSDP2 CPU-ऑफलोड के माध्यम से पूर्ण फाइन-ट्यूनिंग** — पैरामीटर + ऑप्टिमाइज़र को 64 GB होस्ट RAM में भेजता है। धीमा (बैंडविड्थ-बाउंड); लिनक्स/WSL2। |

दो चीजें जिनके लिए अधिकांश एकल-GPU लाइब्रेरी आपको कहीं और भेजती हैं — **24–34B QLoRA** और **एकल-कार्ड 7B-क्लास पूर्ण फाइन-ट्यूनिंग** — बैकप्रोपेगेट इसे एक उपभोक्ता कार्ड पर करता है, फिर परिणाम को सीधे ओलामा में निर्यात करता है।

**पूर्ण-FT सीमा कार्ड-जागरूक है।** यह आपके *पता लगाए गए* VRAM के विरुद्ध 4-एडेंड प्रशिक्षण-मेमोरी अंकगणित (वजन + ग्रेडिएंट + ऑप्टिमाइज़र + सक्रियण) से प्राप्त होती है: **16 GB → 4B, 24 GB → 5B, 32 GB → 6B** शुद्ध-GPU। `--full-ft-offload` इसे FSDP2 `fully_shard` + `CPUOffloadPolicy` के माध्यम से पैरामीटर + ऑप्टिमाइज़र स्टेट को होस्ट RAM में भेजकर **7B-क्लास** तक बढ़ाता है (धीमा, PCIe/CPU-बैंडविड्थ-बाउंड; लगभग 64 GB होस्ट RAM और एक NCCL बैकएंड की आवश्यकता होती है, यानी लिनक्स/WSL2)। `--full-ft-ceiling-billions` के साथ स्पष्ट रूप से सीमा को ओवरराइड करें। यहां तक कि ऑफलोड सीमा से आगे का मॉडल भी `RUNTIME_FULL_FT_MODEL_TOO_LARGE` के साथ बाहर निकल जाता है, जो रिकवरी (`--full-ft-offload`, या LoRA/QLoRA`) का नाम बताता है। VRAM गणित + बिडरमैन 2024 / थिंकिंग मशीन्स 2025 गुणवत्ता तुलना के लिए [पूर्ण फाइन-ट्यूनिंग हैंडबुक पृष्ठ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/full-fine-tuning/) देखें।

### 16 GB तक स्केल होता है

16 GB सीमा (RTX 4080 / 5080 / 4070 Ti Super) अभी भी उत्कृष्ट है: ~7–8 GB पर 7B QLoRA, और `mode="full"` के माध्यम से 16 GB के भीतर एक वास्तविक ~3B (SmolLM3-3B, Qwen2.5-3B, Llama-3.2-3B/1B) का सच्चा पूर्ण फाइन-ट्यूनिंग (bf16 वजन + ग्रेडिएंट चेकपॉइंटिंग + पेज किया हुआ 8-बिट AdamW)। समान कोड बैच आकार और पूर्ण-FT सीमा को चुनता है जो भी कार्ड फिट बैठता है — डिवाइसों के बीच बदलने के लिए कोई फ़्लैग नहीं।

2-बिट क्वांटाइजेशन (AQLM / QuIP#) **दायरे से बाहर** रहता है — 2-बिट बेस को पूर्ण-सटीक वजन में वापस मर्ज नहीं किया जा सकता है, जिससे मर्ज करने योग्य एडाप्टर → GGUF → ओलामा निर्यात अनुबंध टूट जाता है (पाइपलाइन का पूरा बिंदु)। बैकप्रोपेगेट के पास अन्य सुविधाएँ हैं — QLoRA, `mode="full"`, `--full-ft-offload`, और FP8 कंप्यूट पथ (`--fp8`, ब्लैकवेल/हॉपर`) — ये सभी मर्ज करने योग्य और निर्यात करने योग्य रहते हैं।

## बैकप्रोपैगेट किसके लिए नहीं है

यदि आपका उपयोग मामला नीचे दिया गया है, तो आपको किसी अन्य लाइब्रेरी के साथ बेहतर अनुभव होगा - बैकप्रोपैगेट सही विकल्प नहीं है और इसे काम करने की कोशिश करने से सही उपकरण तक पहुंचने से अधिक खर्च आएगा। शुरू करने से पहले इस अनुभाग को पढ़ने से इंस्टॉलेशन-और-वापस जाने के चक्र से बचा जा सकता है:

- **पूर्ण-पैरामीटर फाइन-ट्यूनिंग, ऑफलोड सीमा से आगे (≈13B+)** — पूर्ण-फाइन-ट्यून को अधिकतम ~6B शुद्ध-GPU और ~7B-क्लास तक `--full-ft-offload` के माध्यम से 32 GB कार्ड पर बैकप्रोपगेट करें (देखें [द एनवेलप](#what-you-can-fine-tune-on-one-gpu))। 13B+ मॉडल का *वास्तविक पूर्ण* फाइन-ट्यून इससे आगे है - इसके लिए मल्टी-GPU FSDP या एक बड़ा कार्ड चाहिए (कई GPU पर `transformers.Trainer` का उपयोग करें, या A100/H100 किराए पर लें)। उस कंप्यूटिंग शक्ति को खर्च करने से पहले: हालिया शोध ([बिडरमैन 2024](https://arxiv.org/abs/2405.09673), [थिंकिंग मशीन्स 2025](https://thinkingmachines.ai/blog/lora/)) दर्शाता है कि सही कॉन्फ़िगरेशन पर LoRA अधिकांश पोस्ट-ट्रेनिंग कार्यों (निर्देश-अनुपालन, डोमेन अनुकूलन, व्यक्तित्व/शैली) पर पूर्ण फाइन-ट्यूनिंग गुणवत्ता से मेल खाता है - लगभग 67% कंप्यूटिंग शक्ति के साथ - इसलिए QLoRA को अधिकतम 34B तक उपयोग करें, जिसे बैकप्रोपगेट एक कार्ड पर करता है, और इससे अधिकांश ऑपरेटरों द्वारा किए जाने वाले कार्यों में कोई कमी नहीं होगी।
- **ऑनलाइन RL — PPO / GRPO / RLVR** — बैकप्रोपगेट सिंगल-स्टेज SFT प्लस संदर्भ-मुक्त प्राथमिकता ट्यूनिंग (v1.5 में ORPO; v1.6 में SimPO + KTO) करता है। यह जो *नहीं* करता है, वह ऑनलाइन सुदृढीकरण सीखना है - PPO, GRPO या RLVR - जिसके लिए एक इनाम मॉडल या प्रशिक्षण चरण के शीर्ष पर पीढ़ी-और-स्कोरिंग लूप की आवश्यकता होती है। उनके लिए, सीधे TRL या LLaMA-फ़ैक्टरी का उपयोग करें। (संदर्भ-मुक्त प्राथमिकता ट्यूनिंग सिंगल-स्टेज एनवेलप में फिट बैठता है क्योंकि मेमोरी में रखने के लिए कोई अलग संदर्भ मॉडल नहीं है; [क्विक स्टार्ट](#quick-start) के तहत ORPO नोट देखें)।
- **मल्टी-नोड प्रशिक्षण** — केवल एक मशीन पर एक GPU। एक मशीन पर मल्टी-GPU काम करता है ( `accelerate launch` के माध्यम से), लेकिन आधिकारिक तौर पर समर्थित नहीं है।
- **CUDA रेल पर macOS प्रशिक्षण** — Apple सिलिकॉन में CUDA नहीं है, इसलिए CUDA पथ NVIDIA GPU वाले Linux या Windows बॉक्स पर चलता है। आप अभी भी प्रशिक्षित मॉडल को Ollama के माध्यम से Mac पर चला सकते हैं। एक **प्रायोगिक, अप्रमाणित-पूर्वावलोकन** MLX रेल (`--backend mlx`) Apple सिलिकॉन पर मूल रूप से LoRA एडाप्टर को प्रशिक्षित करता है - [Apple सिलिकॉन (MLX)](#apple-silicon-mlx--unverified-preview) देखें। यह केवल LoRA-SFT है और **वास्तविक सिलिकॉन पर डॉगफूड-प्रमाणित नहीं** है (कोई समर्थन नहीं), इसलिए LoRA SFT (ORPO, पूर्ण फाइन-ट्यून, FP8, मल्टी-रन) से परे किसी भी चीज़ के लिए आप CUDA रेल चाहते हैं।
- **परीक्षित मॉडल परिवारों के बाहर कुछ भी** — Qwen 2.5 / 3.5 (7B / 4B), Phi-4-mini-3.8B, SmolLM3-3B, Llama 3.2 (3B / 1B), Mistral 7B। अन्य मॉडल अक्सर काम करते हैं लेकिन CI में पिन नहीं किए जाते हैं।

यदि आपको इनमें से किसी भी चीज़ की आवश्यकता है, तो ऊपर सूचीबद्ध पुस्तकालयों में से किसी एक का उपयोग करें। वे इसमें बेहतर हैं।

## बैकप्रोपगेट आपको क्या प्रदान करता है

एक इंस्टॉलेशन में चार चीजें:

**1. एक वास्तविक 3-लाइन API जो बिना कॉन्फ़िगरेशन फ़ाइल के चलता है।**
इस README के शीर्ष पर दिया गया स्निपेट एंड-टू-एंड चलता है। कोई `accelerate config`, कोई YAML, कोई हाइड्रा ओवरराइड नहीं। बस `Trainer(model).train(data)` और आपके पास एक फाइन-ट्यून मॉडल है।

**2. विंडोज जो वास्तव में काम करता है।**
अधिकांश ML पुस्तकालय विंडोज को एक afterthought के रूप में मानते हैं। बैकप्रोपगेट को विंडोज + RTX 5080 पर पहले से परीक्षण किया गया है। पुस्तकालय आपके लिए रनटाइम की विशिष्टताओं को संभालता है — यह जानता है कि आपके डेटा को पहले से कैसे टोकनाइज़ करना है ताकि विंडोज मल्टीप्रोसेसिंग क्रैश न हो, यह स्वचालित रूप से RTX 40/50 कार्ड पर xformers को अक्षम कर देता है जहां यह विफल हो जाएगा, और यह डेटा लोडर सेटिंग्स चुनता है जो विफल नहीं होते हैं। आपको इनमें से किसी भी चीज़ को जानने की आवश्यकता नहीं है। यह बस चलता है।

**3. बिना निगरानी वाले रनों के लिए बनाया गया।**
प्रशिक्षण में घंटों लगते हैं। आप इसकी निगरानी नहीं करना चाहते हैं। बैकप्रोपगेट को लगातार चलने के लिए डिज़ाइन किया गया है:

- यदि आपके पास GPU मेमोरी कम हो जाती है, तो यह स्वचालित रूप से बैच आकार को आधा कर देता है और तीन बार तक पुनः प्रयास करता है। कोई मैन्युअल ट्यूनिंग नहीं।
- यदि आपका GPU बहुत गर्म हो जाता है, तो यह तब तक रुक जाता है जब तक कि चीजें शांत न हो जाएं और फिर जारी रहता है।
- प्रत्येक चेकपॉइंट को परमाणु रूप से लिखा जाता है — यदि आपका लैपटॉप मध्य-सहेजते समय क्रैश हो जाता है, तो पिछला अच्छा चेकपॉइंट अभी भी बरकरार रहता है।
- प्रत्येक प्रशिक्षण रन को एक अद्वितीय ID मिलता है जो प्रत्येक लॉग लाइन, प्रत्येक चेकपॉइंट और प्रत्येक वेट्स एंड बायसेस प्रविष्टि पर मुद्रित होता है। यदि कुछ गलत हो जाता है, तो एक ID एक रखरखावकर्ता को सब कुछ सहसंबंधित करने की अनुमति देता है।
- त्रुटियां स्थिर कोड के साथ आती हैं (`RUNTIME_GPU_OOM`, `DEP_OLLAMA_REGISTRATION_FAILED`, आदि) ताकि आप अपने लॉग और [समस्या निवारण मार्गदर्शिका](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) में समाधान खोज सकें। CUDA-विशिष्ट विफलताओं के लिए एक समर्पित [CUDA समस्या निवारण पृष्ठ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/) है।

**4. प्रशिक्षित एडाप्टर से `ollama run` तक एक कमांड।**
कई पुस्तकालय एक मॉडल को प्रशिक्षित करते हैं। उनमें से कुछ ही आपको वास्तव में इसका उपयोग करने पर रास्ते से हटते हैं। बैकप्रोपगेट GGUF (Ollama द्वारा उपयोग किया जाने वाला प्रारूप) में निर्यात करता है और एक कमांड में एक Ollama मॉडल को पंजीकृत करता है। आप "प्रशिक्षण पूरा" से "मैं अपने फाइन-ट्यून मॉडल के साथ चैट कर सकता हूं" लगभग 30 सेकंड में जा सकते हैं।

## त्वरित शुरुआत

रिपो एक छोटा सा उदाहरण डेटासेट भेजता है ताकि इस README के शीर्ष पर दिया गया स्निपेट एक स्वच्छ इंस्टॉलेशन पर चले:

```bash
pipx install "backpropagate[standard]"

python -c "
from backpropagate import Trainer
trainer = Trainer('Qwen/Qwen2.5-7B-Instruct')
trainer.train('examples/quickstart.jsonl', steps=10)
trainer.export('gguf', quantization='q4_k_m')
"
```

यह 5 छोटे ShareGPT-प्रारूप वार्तालापों पर Qwen 2.5 7B एडाप्टर को प्रशिक्षित करता है, फिर परिणाम को GGUF में निर्यात करता है। अपने स्वयं के डेटा के लिए, अपने JSONL को एक पंक्ति में एक उदाहरण के रूप में स्वरूपित करें:

```jsonl
{"conversations": [{"from": "human", "value": "What is Python?"}, {"from": "gpt", "value": "A programming language."}]}
{"conversations": [{"from": "human", "value": "Explain recursion."}, {"from": "gpt", "value": "A function that calls itself."}]}
```

अल्पाका (`instruction` / `output`), OpenAI चैट (`messages`), और कच्चे पाठ प्रारूप भी काम करते हैं — बैकप्रोपगेट प्रारूप का स्वचालित रूप से पता लगाता है।

### प्राथमिकता ट्यूनिंग (ORPO, SimPO, KTO)

v1.5 में नया: सादे प्रदर्शनों के बजाय प्राथमिकताओं पर प्रशिक्षित करें। ORPO संदर्भ-मुक्त और सिंगल-स्टेज है — यह प्राथमिकता संकेत को SFT चरण में जोड़ता है, इसलिए कोई अलग इनाम या संदर्भ मॉडल नहीं है और 3-लाइन आकार अपरिवर्तित रहता है। `--method orpo` (CLI) या `method="orpo"` (Python) पास करें और `{prompt, chosen, rejected}` (या केवल `{chosen, rejected}`) पंक्तियों के एक डेटासेट को फीड करें:

```jsonl
{"prompt": "What is Python?", "chosen": "A high-level programming language known for readability.", "rejected": "idk look it up"}
{"prompt": "Explain recursion.", "chosen": "A function that calls itself with a smaller input until a base case.", "rejected": "when something repeats"}
```

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct", method="orpo")
trainer.train("preferences.jsonl", steps=100)
trainer.export("gguf", quantization="q4_k_m")
```

```bash
backprop train --data preferences.jsonl --method orpo --steps 100
```

डिफ़ॉल्ट शिक्षण दर स्वचालित रूप से ORPO के लिए `8e-6` तक कम हो जाती है (हानि साधारण SFT की तुलना में अधिक तीव्र होती है); `--orpo-beta` को ट्यून करें (डिफ़ॉल्ट `0.1`) ताकि ऑड्स-अनुपात दंड को भार दिया जा सके। ORPO केवल `mode="lora"` है।

**v1.6 में नया — SimPO और KTO।** `--method simpo` ([मेंग एट अल. 2024](https://arxiv.org/abs/2405.14734)) संदर्भ-मुक्त है, जिसमें लंबाई-मानकीकृत इनाम होता है और यह ORPO के समान युग्मित `{prompt, chosen, rejected}` डेटा लेता है (`--simpo-beta`, `--simpo-gamma`)। `--method kto` ([एथयाराज एट अल. 2024](https://arxiv.org/abs/2402.01306)) **युग्मित नहीं** `{prompt, completion, label}` डेटा लेता है - प्रति-उदाहरण थम्स-अप/डाउन - प्रतिक्रिया के उस बड़े वर्ग के लिए जो क्यूरेटेड A/B जोड़े नहीं हैं; यह आपके लेबल गणना से वांछनीय/अवांछनीय हानि भार को स्वचालित रूप से संतुलित करता है। दोनों केवल `mode="lora"` हैं और सिंगल-GPU SFT एनवेलप में रहते हैं (कोई अलग संदर्भ मॉडल नहीं)। [प्राथमिकता-ट्यूनिंग हैंडबुक](https://mcp-tool-shop-org.github.io/backpropagate/handbook/preference-tuning/) देखें कि किसका उपयोग करना है। ऑनलाइन RL (PPO/GRPO) के लिए, [बैकप्रोपगेट क्या नहीं है](#what-backpropagate-is-not-for) देखें।

### तर्क-आधारित SFT (R1 डिस्टिलेशन)

v1.5 में नया: तर्क मॉडल को आसानी से डिस्टिल करें। `--reasoning-trace` (CLI) या `Trainer(..., reasoning_trace=True)` (Python) पास करें और ऐसे ट्रेस प्रदान करें जो सहायक के टर्न के अंदर `<think>...</think>` श्रृंखला-आधारित विचार को बनाए रखें — यह [DeepSeek-R1](https://arxiv.org/abs/2501.12948) डिस्टिलेशन का शुद्ध-SFT आधा हिस्सा है, RL की आवश्यकता नहीं है। बैकप्रोपैगेट प्रशिक्षण लक्ष्य में `<think>` को बनाए रखता है, खाली/अत्यधिक लंबे ट्रेस को हटा देता है (ट्रेस-लंबाई फ़िल्टरिंग), और लंबे CoT के लिए डिफ़ॉल्ट `max_seq_length` को 8192 तक बढ़ाता है। महत्वपूर्ण रूप से, `<think>` **सादा पाठ** रहता है — कोई विशेष टोकन नहीं, कोई एम्बेडिंग आकार परिवर्तन नहीं — इसलिए मर्ज किया गया GGUF अभी भी किसी अन्य फाइन-ट्यून की तरह Ollama में निर्यात होता है। केवल SFT। डेटासेट आकार और समायोज्य टोकन बैंड के लिए [reasoning-trace recipe](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/#reasoning-trace-sft-r1-distillation) देखें।

### Apple सिलिकॉन (MLX) — अप्रमाणित पूर्वावलोकन

> ⚠️ **अप्रमाणित पूर्वावलोकन - समर्थित सुविधा सेट का हिस्सा नहीं।** MLX रेल को बनाया और यूनिट-परीक्षण किया गया है, लेकिन इसे वास्तविक Apple सिलिकॉन पर डॉगफूड-प्रमाणित नहीं किया गया है (`mlx-lm` केवल Apple के लिए है और बैकप्रोपगेट जिस पर विकसित किया गया है, NVIDIA रिग्स पर नहीं चल सकता)। नीचे दी गई हर चीज को प्रयोगात्मक मानें, अपने जोखिम पर उपयोग करें, और यदि आप इसे M-श्रृंखला Mac पर चलाते हैं तो [विसंगतियों की रिपोर्ट करें](#reporting-bugs)।

v1.5 में नया: **एक API, दो रेल।** CUDA डिफ़ॉल्ट, सत्यापित बैकएंड बना रहता है; MLX एक दूसरी रेल है जो Apple के [`mlx_lm.lora`](https://github.com/ml-explore/mlx-lm) टूलचेन के माध्यम से M-श्रृंखला Mac पर प्रशिक्षित होती है (एकीकृत मेमोरी, कोई CUDA नहीं)। समान 3-पंक्ति आकार हार्डवेयर द्वारा रेल का चयन करता है — `backend='auto'` (डिफ़ॉल्ट) NVIDIA पर CUDA और Apple सिलिकॉन पर MLX पर रूट करता है, इसलिए मौजूदा CUDA सेटअप बाइट-दर-बाइट समान हैं:

```python
from backpropagate import Trainer

# On an M-series Mac with `pip install 'backpropagate[mlx]'`:
trainer = Trainer("mlx-community/Qwen2.5-0.5B-Instruct-4bit", backend="mlx")
trainer.train("examples/quickstart.jsonl", steps=100)
```

```bash
backprop train --data my_data.jsonl --backend mlx --steps 100
```

v1.5 में MLX रेल **केवल LoRA SFT है** — कोई ORPO नहीं, कोई FP8 नहीं, कोई `mode='full'` नहीं, अभी तक MLX पर कोई मल्टी-रन नहीं (प्रत्येक को `CONFIG_INVALID_SETTING` के साथ अस्वीकार कर दिया जाता है; उन कार्यों के लिए NVIDIA बॉक्स पर `backend='cuda'`/`'auto'` का उपयोग करें)। परिणामी एडाप्टर सादा सेफ़टेंसर है और उसी पथ के माध्यम से CUDA रेल की तरह Ollama में निर्यात होता है।

> गैर-Apple होस्ट पर `--backend mlx` को जबरदस्ती लागू करने से `CONFIG_INVALID_SETTING` त्रुटि आती है; Mac पर लापता `mlx_lm` टूलचेन से `DEP_MLX_UNAVAILABLE` उत्पन्न होता है।

अधिक एंड-टू-एंड वर्कफ़्लो (फाइन-ट्यून-एंड-पुश-टू-HF-हब, OOM के बाद फिर से शुरू करें, लंबे अभियान में मल्टी-रन SLAO, आदि) के लिए [हैंडबुक रेसिपी पेज](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/) देखें।

### वेब UI (वैकल्पिक)

यदि आप Python में टाइप करने के बजाय क्लिक करना पसंद करते हैं, तो UI एक्स्ट्रा स्थापित करें और लॉन्च करें:

```bash
pipx install "backpropagate[ui]"
backprop ui --port 7862
```

एक स्थानीय वेब इंटरफ़ेस `http://localhost:7862` पर खुलता है, जिसका उपयोग डेटासेट ब्राउज़ करने, प्रारूपों को मान्य करने और प्रशिक्षण कॉन्फ़िगरेशन को दृश्य रूप से इकट्ठा करने के लिए किया जा सकता है। प्रशिक्षण स्वयं `backprop train` के माध्यम से चलता है (UI-संचालित प्रशिक्षण रोडमैप पर है — स्टार्ट बटन वर्तमान में उस नोट को प्रदर्शित करता है)। UI डिफ़ॉल्ट रूप से केवल स्थानीय है। इसे अन्य उपकरणों तक विस्तारित करने के लिए, [Web UI](#web-ui) नीचे `--share` + `--auth` सुरक्षा अनुबंध देखें।

## मल्टी-रन प्रशिक्षण

यदि आप कई डेटासेट में वृद्धिशील रूप से फाइन-ट्यून करना चाहते हैं — मान लीजिए कि आपको हर हफ्ते नया प्रशिक्षण डेटा मिलता है और आप इसे पहले से सीखे गए ज्ञान को भूले बिना जोड़ना चाहते हैं — तो बैकप्रोपैगेट का `multi_run` मोड आपके लिए है:

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")

result = trainer.multi_run(
    dataset="HuggingFaceH4/ultrachat_200k",
    num_runs=5,
    steps_per_run=100,
    samples_per_run=1000,
)
```

यह पांच प्रशिक्षण पास चलाता है, जो रन के बीच एडाप्टर को इस तरह से मर्ज करता है कि पहले के ज्ञान को संरक्षित किया जा सके, जबकि नए उदाहरणों को शामिल किया जा सके। यह तकनीक हालिया निरंतर-शिक्षण अनुसंधान पर आधारित है — इस README के नीचे [References](#references) देखें।

CLI संस्करण:

```bash
backprop multi-run --data my_data.jsonl --runs 5 --steps 100 --samples 1000
```

## चेकपॉइंट से फिर से शुरू करें

5-रन प्रशिक्षण जो रन 4 पर क्रैश हो जाता है, उसे ठीक किया जा सकता है। प्रत्येक मल्टी-रन सत्र अपने रन आईडी को ऑन-डिस्क इतिहास और चेकपॉइंट मेनिफेस्ट में लिखता है, इसलिए जहां आपने छोड़ा था, वहां से उठाना एक कमांड है:

```bash
backprop resume <run-id>
backprop multi-run --data ... --resume <run-id>
backprop train --data ... --resume <run-id>     # single-run resume
```

`backprop multi-run` (कोई `--resume` नहीं) का डिफ़ॉल्ट व्यवहार उसी आउटपुट निर्देशिका में एक प्रगति पर प्रविष्टि का स्वचालित रूप से पता लगाता है और उसे जारी रखता है। एक स्वच्छ शुरुआत के लिए, एक ताज़ा आउटपुट निर्देशिका पर इंगित करें।

## प्रशिक्षण इतिहास

प्रत्येक `backprop train` और `backprop multi-run` आह्वान `<output>/run_history.json` में एक पंक्ति रिकॉर्ड करता है — उपयोग किया गया मॉडल, डेटासेट, हाइपरपैरामीटर, स्थिति, अंतिम हानि, हानि इतिहास। आप पिछले रन को सूचीबद्ध और निरीक्षण कर सकते हैं:

```bash
backprop list-runs                         # last 20 runs
backprop list-runs --status failed         # filter by status
backprop list-runs --json --limit 100      # machine-readable
backprop show-run abcd1234                 # detail view (partial ID is fine)
```

## प्रयोग ट्रैकिंग

बैकप्रोपैगेट स्थापित प्रयोग ट्रैकर्स (वेट्स एंड बायसेस, टेंसरबोर्ड, MLflow) का स्वचालित रूप से पता लगाता है और उन्हें जोड़ता है। यदि `wandb` स्थापित है और आप लॉग इन हैं, तो प्रत्येक रन स्वचालित रूप से W&B में एक रन नाम के साथ लॉग करता है जो ऑन-डिस्क रन आईडी से मेल खाता है — इसलिए आप W&B, अपने लॉग और `run_history.json` में एक पहचानकर्ता का उपयोग करके खोज सकते हैं।

```bash
pip install backpropagate[monitoring]   # installs wandb + psutil
wandb login                             # one-time setup
backprop train --data my_data.jsonl
```

`Trainer(report_to=["wandb"])`, `Trainer(report_to=["tensorboard"])`, या `Trainer(report_to="none")` के साथ ओवरराइड करें ताकि बाहर निकल सकें।

## वेब UI

रिफ्लेक्स वेब इंटरफ़ेस वैकल्पिक है — `pipx install "backpropagate[ui]"` के साथ स्थापित करें और लॉन्च करें:

```bash
backprop ui --port 7862
```

UI स्थानीय रूप से `http://localhost:7862` पर चलता है। आज यह वर्कफ़्लो के **ब्राउज़ / मान्य / कॉन्फ़िगर** आधे हिस्से को कवर करता है — इसे एक डेटासेट पर इंगित करें, स्वचालित रूप से पता लगाए गए प्रारूप और आँकड़ों की जांच करें, एक मॉडल चुनें और एक रन कॉन्फ़िगरेशन को इकट्ठा करें। **रन लॉन्च CLI से किया जाता है** (`backprop train` / `backprop multi-run`); UI में स्टार्ट बटन उस पर इंगित करने वाला एक नोट प्रदर्शित करता है। UI-संचालित प्रशिक्षण एक नियोजित अनुवर्ती है — तब तक UI ऑन-रैंप है और CLI ट्रिगर है।

इसे अन्य उपकरणों (आपके नेटवर्क पर अन्य लोग, एक सार्वजनिक URL, आदि) के साथ साझा करने के लिए, आपको `--share` (या `--host`) को `--auth` के साथ जोड़ना होगा:

```bash
backprop ui --share --auth alice:hunter2
```

`backprop ui --share` बिना `--auth` के त्रुटि के साथ समाप्त हो जाता है। कारण: `--share` एक URL प्रकाशित करता है जिस तक इंटरनेट पर कोई भी पहुंच सकता है, और प्रमाणीकरण के बिना, इसका मतलब है कि कोई भी आपके प्रशिक्षण पाइपलाइन को चला सकता है और आपके HuggingFace टोकन को पढ़ सकता है। इसके लिए कोई विकल्प नहीं है - यदि आप क्रेडेंशियल सेट नहीं करना चाहते हैं, तो इसके बजाय SSH पोर्ट-फ़ॉरवर्डिंग का उपयोग करें:

```bash
# On the client:
ssh -L 7860:localhost:7860 <your-training-host>
# On the server:
backprop ui                             # no --share
# Then open http://localhost:7860 in your local browser
```

पूर्ण खतरे के मॉडल के लिए [handbook/security.md](https://mcp-tool-shop-org.github.io/backpropagate/handbook/security/) देखें।

UI से फ़ाइल सिस्टम में लिखी जाने वाली फ़ाइलें एक ही निर्देशिका में सीमित हैं:

- डिफ़ॉल्ट: `~/.backpropagate/ui-outputs`
- ओवरराइड: `BACKPROPAGATE_UI__OUTPUT_DIR=/path/you/own` सेट करें
- ओवरराइड को अस्वीकृति सूची के साथ मान्य किया जाता है - सिस्टम या क्रेडेंशियल पथ (`/etc`, `~/.ssh`, `~/.aws`, `C:\Windows\System32`, आदि) अस्वीकार किए जाते हैं।

## प्लेटफ़ॉर्म नोट्स

**आवश्यकताएं:** Python 3.10+ · CUDA GPU (8GB+ VRAM) · PyTorch 2.0+

Python 3.10 को कम से कम v1.6 तक समर्थित किया गया है; यह अक्टूबर 2026 में अपस्ट्रीम जीवन के अंत तक पहुंचता है और इसके बाद पहले रिलीज में इसे हटाने की योजना बनाई गई है। नए इंस्टॉलेशन के लिए, Python 3.11 या 3.12 को प्राथमिकता दें - 3.11 सबसे अधिक परीक्षण किया गया न्यूनतम संस्करण है।

Backpropagate विभिन्न प्लेटफ़ॉर्म पर प्रशिक्षण के दौरान आने वाली रनटाइम संबंधी विशिष्टताओं को संभालता है, लेकिन यह इंस्टॉलेशन के समय आने वाली समस्याओं को ठीक नहीं कर सकता है। दो सबसे आम समस्याएं हैं:

- **गलत CUDA व्हील।** PyTorch प्रत्येक CUDA संस्करण के लिए एक बाइनरी के रूप में प्रकाशित किया जाता है। यदि आप गलत संस्करण चुनते हैं, तो आपको चुपचाप केवल CPU-आधारित PyTorch मिलेगा और प्रशिक्षण असंभव रूप से धीमा होगा। अपने ड्राइवर के लिए <https://pytorch.org/get-started/locally/> पर व्हील पिकर का उपयोग करें। अपना ड्राइवर / CUDA संस्करण देखने के लिए `nvidia-smi` चलाएं।
- **Windows + GGUF निर्यात।** `[export]` एक्स्ट्रा स्रोत से `llama-cpp-python` बनाता है, जिसके लिए विज़ुअल स्टूडियो बिल्ड टूल्स (C++ घटक) और CMake की आवश्यकता होती है।

**macOS:** CUDA समर्थित नहीं है (कोई CUDA नहीं) - CUDA-आधारित `trainer.train()` `DEP_GPU_NOT_AVAILABLE` त्रुटि उत्पन्न करेगा, और आप प्रशिक्षित एडाप्टर को Ollama के माध्यम से Mac पर चला सकते हैं। **v1.5 में नया:** एक प्रायोगिक MLX रेल (`--backend mlx`, `pip install 'backpropagate[mlx]'`) `mlx_lm.lora` के माध्यम से Apple Silicon पर मूल रूप से एक LoRA एडाप्टर को प्रशिक्षित करता है - केवल LoRA SFT, और निर्मित + यूनिट-परीक्षण किया गया है लेकिन अभी तक वास्तविक सिलिकॉन पर सत्यापित नहीं किया गया है (देखें [Apple Silicon (MLX)](#apple-silicon-mlx--experimental-v15))। CUDA पथ के लिए, या ORPO / पूर्ण फाइन-ट्यून / FP8 / मल्टी-रन के लिए, एक CUDA Linux या Windows मशीन का उपयोग करें।

[समस्या निवारण हैंडबुक पृष्ठ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) पर लंबे प्रारूप में इंस्टॉलेशन फिक्स-इट गाइड देखें, और समर्पित [CUDA समस्या निवारण पृष्ठ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/) पर ड्राइवर / VRAM / xformers / bf16-vs-fp16 मुद्दों के बारे में जानकारी देखें।

## CLI

प्रत्येक Python API का एक CLI मिरर है:

```bash
backprop train --data my_data.jsonl --model Qwen/Qwen2.5-7B-Instruct --steps 100
backprop multi-run --data my_data.jsonl --runs 5 --steps 100
backprop export ./output/lora --format gguf --quantization q4_k_m --ollama --ollama-name my-model
backprop ui --port 7862
backprop info                          # environment + version snapshot
backprop list-runs                     # past training runs
backprop show-run <run-id>             # detail view
backprop resume <run-id>               # resume a crashed run
backprop push ./output/lora --repo me/my-model    # push adapter to HuggingFace Hub
backprop diff-runs <run-a> <run-b>     # diff two runs side by side
backprop replay <run-id>               # re-run with same config / dataset
backprop export-runs --format jsonl    # bulk export run history
```

पूर्ण संदर्भ [CLI हैंडबुक पृष्ठ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/cli-reference/) पर या `backprop <subcommand> --help`।

## कॉन्फ़िगरेशन

प्रत्येक सेटिंग को `BACKPROPAGATE_` उपसर्ग का उपयोग करके एक पर्यावरण चर के साथ ओवरराइड किया जा सकता है:

| चर | डिफ़ॉल्ट | टिप्पणियाँ |
|---|---|---|
| `BACKPROPAGATE_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `BACKPROPAGATE_LOG_JSON` | ऑटो | JSON या कंसोल लॉग को बाध्य करें |
| `BACKPROPAGATE_MODEL__NAME` | `Qwen/Qwen2.5-7B-Instruct` | डिफ़ॉल्ट मॉडल |
| `BACKPROPAGATE_TRAINING__LEARNING_RATE` | `2e-4` | सीखने की दर |
| `BACKPROPAGATE_LORA__R` | `256` | LoRA रैंक (v1.3 डिफ़ॉल्ट; v1.2.x के 16 के डिफ़ॉल्ट के लिए `--lora-preset=fast` पास करें) |
| `BACKPROPAGATE_UI__OUTPUT_DIR` | `~/.backpropagate/ui-outputs` | UI फ़ाइल सिस्टम सैंडबॉक्स |

नेस्टेड कुंजियों में डबल अंडरस्कोर (`MODEL__NAME`, `MODEL_NAME` नहीं) का उपयोग किया जाता है। पूर्ण संदर्भ [पर्यावरण-चर हैंडबुक पृष्ठ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/env-vars/) पर है।

## मॉडल प्रीसेट

| प्रीसेट | VRAM | लाइसेंस | टिप्पणियाँ |
|---|---|---|---|
| Qwen-3.5-4B | ~8GB | Apache 2.0 | 5B से कम के लिए अनुशंसित डिफ़ॉल्ट। इस आकार पर सर्वोत्तम गुणवत्ता। |
| Phi-4-mini-3.8B | ~8GB | MIT | तर्क / गणित / कोड पर मजबूत। सख्त लाइसेंस-स्वच्छ। |
| SmolLM3-3B | ~6GB | Apache 2.0 | पूरी तरह से खुला नुस्खा। मूल 64K संदर्भ। |
| Qwen 2.5 7B | ~12GB | Apache 2.0 | मौजूदा डिफ़ॉल्ट। पुराने 7B प्रीसेट की सर्वोत्तम गुणवत्ता। |
| Qwen 2.5 3B | ~8GB | Qwen-Research | ⚠ अनुसंधान लाइसेंस - वाणिज्यिक उपयोग से पहले Qwen लाइसेंस शर्तों को देखें। |
| Llama 3.2 3B | ~8GB | Llama Community | अनुमति देने वाली शर्तों के साथ Qwen 3B का एक ठोस विकल्प। |
| Llama 3.2 1B | ~6GB | Llama Community | छोटे कार्ड पर त्वरित प्रयोगों के लिए। |
| Mistral 7B | ~12GB | Apache 2.0 | Qwen 7B के समान, अलग चैट टेम्पलेट। |
| Llama-3.1-8B | ~7-8GB (QLoRA) | Llama-3.1-Community | 8B QLoRA, 128K मूल संदर्भ ( >700M-MAU खंड के लिए एक अलग मेटा लाइसेंस की आवश्यकता है)। |
| **Qwen2.5-14B** | ~8.5GB (QLoRA) | Apache 2.0 | **32 GB दैनिक-ड्राइवर स्वीट स्पॉट** — रैंक/अल्फा 32, पेज्ड 8-बिट AdamW, 4096 ctx। |
| Mistral-Small-24B | ~18GB (QLoRA) | Apache 2.0 | 4096-ctx हेडरूम के साथ 32 GB कार्ड पर 24B QLoRA। |
| **Qwen2.5-32B** | ~26GB (QLoRA) | Apache 2.0 | **32 GB एनवेलप का शीर्ष** — यह `max_len 2048` + पेज्ड 8-बिट AdamW पर बस फिट बैठता है। |

अन्य मॉडल अक्सर काम करते हैं; ऊपर दी गई पंक्तियाँ क्यूरेटेड प्रीसेट हैं - 14B–32B टियर को 32 GB कार्ड के लिए QLoRA-ट्यून किया गया है (मापा एनवेलप)। Biderman 2024 + Thinking Machines 2025 के अनुसार रैंक-256 / सभी-रैखिक लक्ष्यों के लिए `--lora-preset=quality` (डिफ़ॉल्ट) पास करें, या यदि आपको v1.2.x फ़ुटप्रिंट की आवश्यकता है तो विरासत रैंक-16 / q+v लक्ष्य के लिए `--lora-preset=fast`।

## समस्या निवारण

सबसे आम पहली-रन विफलताओं का एक संक्षिप्त सूचकांक। पूर्ण रिवर्स इंडेक्स [समस्या निवारण हैंडबुक पृष्ठ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) पर है। ड्राइवर / VRAM / मिश्रित-सटीक गहन-विश्लेषण के लिए [CUDA समस्या निवारण पृष्ठ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/) देखें।

| लक्षण | त्रुटि कोड | समाधान |
|---|---|---|
| प्रशिक्षण के दौरान GPU की मेमोरी समाप्त हो जाती है | `RUNTIME_GPU_OOM` | स्वचालित — बैकप्रोपैगेट बैच आकार को आधा कर देता है और 3 बार तक पुनः प्रयास करता है। इसे बंद करने के लिए: `Trainer(oom_recovery=False)`। आकार को कम करने के लिए: `--batch-size 1`। |
| हगिंगफेस 401 / "मॉडल नहीं मिला" लौटाता है। | `DEP_MODEL_LOAD_FAILED` | `huggingface-cli login` और पुनः प्रयास करें। टाइपो के लिए, <https://huggingface.co/models> से सटीक आईडी कॉपी करें। |
| `register_with_ollama` कनेक्शन अस्वीकृत। | `DEP_OLLAMA_REGISTRATION_FAILED` | डेमॉन शुरू करें: `ollama serve`। <https://ollama.com> से स्थापित करें। पुनः प्रयास किया जा सकता है। |
| चेकपॉइंट सहेजते समय डिस्क भर गई। | `STATE_CHECKPOINT_INVALID` | एटॉमिक लेखन क्रैश होने पर `.partial` नामक एक निर्देशिका छोड़ देता है — इसे सुरक्षित रूप से हटाया जा सकता है। पिछला अच्छा चेकपॉइंट बरकरार है। |
| जीपीयू ज़्यादा गरम होने पर प्रशिक्षण रोक दिया गया। | `RUNTIME_GPU_TEMPERATURE_CRITICAL` | स्वचालित — बैकप्रोपैगेट तापमान सीमा पर रुक जाता है और जैसे ही जीपीयू ठंडा होता है, फिर से शुरू हो जाता है। यदि यह लगातार होता रहता है, तो वायु प्रवाह में सुधार करें। |
| `backprop ui --share` अस्वीकृत। | `RUNTIME_UI_AUTH_NOT_ENFORCED` | `--auth user:password` पास करें, या इसके बजाय SSH पोर्ट-फ़ॉरवर्डिंग का उपयोग करें (देखें [वेब यूआई](#web-ui))। |
| पहला प्रयास करने पर जीजीयूएफ निर्यात विफल। | `RUNTIME_GGUF_EXPORT_FAILED` | `pip install backpropagate[export]`; विंडोज पर आपको विज़ुअल सी++ बिल्ड टूल्स + सीएमएके की भी आवश्यकता होगी। |

## बग की रिपोर्ट करना।

जब कुछ विफल हो जाता है, तो बैकप्रोपैगेट स्टार्टअप पर एक पंक्ति प्रिंट करता है, जैसे `run_started run_id=<uuid>` और प्रत्येक लॉग पंक्ति, प्रत्येक चेकपॉइंट और प्रत्येक वेट्स और बायसेस प्रविष्टि से समान आईडी को जोड़ता है। **किसी भी बग रिपोर्ट में `run_id` शामिल करें** — इससे एक रखरखावकर्ता उस विशिष्ट रन के लिए सब कुछ सहसंबंधित कर सकता है।

एक अच्छी बग रिपोर्ट में शामिल हैं:

1. **`run_id`** — स्टार्टअप पर मुद्रित UUID। एक UUID एक रखरखावकर्ता को उस विशिष्ट रन के लिए प्रत्येक लॉग पंक्ति, प्रत्येक चेकपॉइंट और प्रत्येक वेट्स और बायसेस प्रविष्टि को सहसंबंधित करने देता है।
2. **त्रुटि कोड** — stderr में `[CODE_NAME]: message` पंक्ति। स्थिर कोड की सूची के लिए [त्रुटि कोड](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/) देखें।
3. **संपादित ट्रेसबैक।** गैर-विस्तृत मोड में stderr स्वचालित रूप से संपादित किया जाता है (बेयरर टोकन, `sk-*`, `hf_*`, AWS कुंजियाँ, `password=` / `token=` / `api_key=` जोड़े हटा दिए जाते हैं) — इसे सुरक्षित रूप से पेस्ट किया जा सकता है। पूर्ण, बिना संपादित ट्रेसबैक के लिए, `BACKPROPAGATE_DEBUG=1` (या `--verbose`) के साथ पुनः चलाएं; पोस्ट करने से पहले समीक्षा करें।
4. **`backprop info` आउटपुट।** एक कमांड पायथन / पायटॉर्च / CUDA / GPU मॉडल / VRAM / OS / स्थापित अतिरिक्त — सब कुछ प्रिंट करता है जिसकी रखरखावकर्ता को प्लेटफ़ॉर्म-विशिष्ट प्रतिगमन को अलग करने के लिए आवश्यकता होती है।

[बग रिपोर्ट टेम्पलेट](https://github.com/mcp-tool-shop-org/backpropagate/issues/new?template=bug_report.yml) इनमें से प्रत्येक के लिए स्पष्ट रूप से संकेत देता है ताकि जांच तेजी से हो सके। प्रश्न, विचार, या "क्या यह अपेक्षित है?" थ्रेड [गिटहब डिस्कशन](https://github.com/mcp-tool-shop-org/backpropagate/discussions) में होने चाहिए। सुरक्षा संबंधी मुद्दों की निजी तौर पर [गिटहब सुरक्षा सलाहकार](https://github.com/mcp-tool-shop-org/backpropagate/security/advisories/new) फॉर्म के माध्यम से रिपोर्ट की जानी चाहिए — नीति और प्रतिक्रिया समयसीमा के लिए [SECURITY.md](SECURITY.md) देखें।

## गोपनीयता

सभी प्रशिक्षण आपके जीपीयू पर स्थानीय रूप से होते हैं। बैकप्रोपैगेट हगिंगफेस से मॉडल डाउनलोड करने के अलावा कोई नेटवर्क अनुरोध नहीं करता है (जिसे आप शुरू करते हैं)। कोई टेलीमेट्री नहीं, कोई क्लाउड निर्भरता नहीं।

## संदर्भ

बैकप्रोपैगेट की डिफ़ॉल्ट सेटिंग्स और मल्टी-रन प्रशिक्षण मोड हाल के शोध पर आधारित हैं। यदि आप अंतर्निहित तकनीकों में रुचि रखते हैं:

- **हु एट अल. 2021।** *लोरा: बड़े भाषा मॉडल का कम-रैंक अनुकूलन।* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) — यह मूलभूत पेपर है जो लोरा का परिचय देता है, जिससे बैकप्रोपैगेट कुशलतापूर्वक एडेप्टर को प्रशिक्षित करता है।
- **बिडरमैन एट अल. 2024।** *लोरा कम सीखता है और कम भूलता है।* [arXiv:2405.09673](https://arxiv.org/abs/2405.09673) — अनुभवजन्य प्रमाण कि रैंक 256 पर सभी-रैखिक लक्ष्यों के साथ लोरा, अधिकांश पोस्ट-प्रशिक्षण कार्यों पर 67% कंप्यूट पर पूर्ण फाइन-ट्यूनिंग गुणवत्ता से मेल खाता है। बैकप्रोपैगेट के v1.3 डिफ़ॉल्ट लोरा कॉन्फ़िगरेशन को चलाता है।
- **थिंकिंग मशीन 2025।** *लोरा बिना पछतावे के।* [thinkingmachines.ai/blog/lora](https://thinkingmachines.ai/blog/lora/) — व्यावहारिक अनुवर्ती जो उच्च लोरा रैंक पर आवश्यक 10 गुना शिक्षण-दर-बनाम-पूर्ण-एफटी सुधार की पहचान करता है।
- **किर्कपैट्रिक एट अल. 2017।** *तंत्रिका नेटवर्क में विनाशकारी विस्मरण को दूर करना।* [arXiv:1612.00796](https://arxiv.org/abs/1612.00796) — तंत्रिका नेटवर्क के "भूलने" के मूल लक्षण वर्णन जब आप नए डेटा पर फाइन-ट्यून करते हैं (ईडब्ल्यूसी — इलास्टिक वेट कंसॉलिडेशन)।
- **वांग एट अल. 2023।** *भाषा मॉडल निरंतर सीखने के लिए ऑर्थोगोनल सबस्पेस लर्निंग।* [arXiv:2310.14152](https://arxiv.org/abs/2310.14152) — ओ-लोरा, निरंतर सीखने के लिए लोरा का उपयोग करने का एक प्रारंभिक दृष्टिकोण, नए एडेप्टर को ऑर्थोगोनल सबस्पेस तक सीमित करके।
- **यादव एट अल. 2023।** *टीआईएस-मर्जिंग: मॉडल को मर्ज करते समय हस्तक्षेप को हल करना।* [arXiv:2306.01708](https://arxiv.org/abs/2306.01708) — कई फाइन-ट्यून किए गए मॉडल को हस्तक्षेप के बिना मर्ज करने के लिए एक मूलभूत तकनीक।
- **कियाओ और महदावी 2025।** *भूलने से पहले मर्ज करें: निरंतर मर्जिंग के माध्यम से एक एकल लोरा निरंतर सीखना।* [arXiv:2512.23017](https://arxiv.org/abs/2512.23017) — विशिष्ट एल्गोरिदम जिसे बैकप्रोपैगेट का मल्टी-रन मर्जर लागू करता है। दिसंबर 2025 का एक प्रीप्रिंट; बैकप्रोपैगेट पेपर का पहला ज्ञात डाउनस्ट्रीम अपनाने वाला है।

## लाइसेंस

एमआईटी — [लाइसेंस](LICENSE) देखें।

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>
