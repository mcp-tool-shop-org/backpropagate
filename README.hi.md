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

# एक एडाप्टर को प्रशिक्षित करें। इसे ओलामा पर भेजें। आगे बढ़ें।

बैकप्रोपैगेट एक पायथन लाइब्रेरी है जो बड़े भाषा मॉडल को एक सिंगल जीपीयू पर फाइन-ट्यून करने के लिए उपयोग की जाती है। तीन पंक्तियों के कोड से 7 बिलियन पैरामीटर वाला मॉडल 16 जीबी के कार्ड पर प्रशिक्षित किया जा सकता है। एक और कमांड इसे ओलामा पर एक्सपोर्ट कर देता है ताकि आप `ollama run` कमांड का उपयोग करके अपने फाइन-ट्यून मॉडल को चला सकें। यह विंडोज पर भी आसानी से काम करता है।

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

बस इतना ही। इसमें कोई YAML कॉन्फ़िगरेशन फ़ाइल नहीं है। कोई `accelerate launch` प्रक्रिया नहीं है। कोई अलग "अब इसे GGUF में बदलें" ट्यूटोरियल नहीं है। यदि आपके पास एक CUDA जीपीयू है और आपके प्रशिक्षण डेटा के साथ एक JSONL फ़ाइल है, तो आप केवल तीन पंक्तियों से एक काम करने वाला फाइन-ट्यून मॉडल प्राप्त कर सकते हैं।

## इंस्टॉल करें

```bash
# Recommended: isolated Python install (no conflicts with system Python or other projects)
pipx install backpropagate

# Or via uv (faster install, same isolation)
uv tool install backpropagate

# Standard pip (if you manage your own virtualenv)
pip install backpropagate
```

यदि आप वैकल्पिक सुविधाओं का उपयोग करना चाहते हैं, तो इंस्टॉलेशन को इनमें से किसी एक से बदलें:

```bash
pipx install "backpropagate[standard]"   # adds Unsloth (2x faster training) + the web UI
pipx install "backpropagate[full]"       # adds everything: unsloth, ui, monitoring, export, etc.
```

क्या आप डॉकर का उपयोग करना पसंद करते हैं? `docker pull ghcr.io/mcp-tool-shop-org/backpropagate:latest` भी काम करता है। इमेज `linux/amd64` और `linux/arm64` दोनों के लिए उपलब्ध हैं, इसलिए Apple Silicon और ARM Linux उपयोगकर्ताओं को एक देशी इमेज मिलती है। "UI एक कंटेनर में" के लिए एक मानक `compose.yaml` फ़ाइल रिपॉजिटरी के रूट पर मौजूद है - `docker compose up` कमांड वेब UI को `http://localhost:7860` पर चलाता है, जिसमें एक स्थायी `~/.backpropagate` वॉल्यूम माउंट भी होता है।

## बैकप्रोपैगेट कहाँ फिट बैठता है

बड़े भाषा मॉडल (LLMs) को फाइन-ट्यून करने के लिए कई अच्छी लाइब्रेरी उपलब्ध हैं। प्रत्येक लाइब्रेरी अलग-अलग कार्यों में उत्कृष्ट है:

- **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)** — यदि आपको YAML कॉन्फ़िगरेशन पसंद हैं और आप कॉपी करने के लिए व्यंजनों का एक समुदाय चाहते हैं।
- **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** — यदि आप एक वेब GUI और DPO/PPO/RLHF के लिए अंतर्निहित समर्थन चाहते हैं।
- **[Unsloth](https://github.com/unslothai/unsloth)** — यदि आपको सबसे तेज़ प्रशिक्षण की आवश्यकता है और आप एक समर्थित मॉडल परिवार का उपयोग कर रहे हैं।
- **[torchtune](https://github.com/pytorch/torchtune)** — यदि आप मेटा के पहले-पार्टी PyTorch-देशी व्यंजनों को संपादित करना चाहते हैं।

बैकप्रोपैगेट एक लापता विकल्प है: **एक सिंगल उपभोक्ता जीपीयू पर काम करने वाले ऑपरेटरों के लिए 3-पंक्ति पायथन एपीआई, जो एक एडाप्टर को प्रशिक्षित करना और उसे भेजना चाहते हैं।** इसमें कोई YAML, कोई GUI, कोई DPO/PPO, और कोई मल्टी-नोड सिस्टम नहीं है। इसमें केवल वह लूप है जिसकी वास्तव में सभी को आवश्यकता होती है, और वह एक्सपोर्ट स्टेप जो बाधा उत्पन्न करता है।

यदि आपने ऊपर दी गई लाइब्रेरी में से किसी एक को आज़माया है और कॉन्फ़िगरेशन फ़ाइल प्रक्रिया से निराश हो गए हैं, या किसी मॉडल परिवार की सीमा का सामना किया है, या विंडोज-फर्स्ट डिफ़ॉल्ट सेटिंग्स चाहते हैं - तो बैकप्रोपैगेट आपके लिए है।

## 16 जीबी के उपभोक्ता जीपीयू पर आप क्या फाइन-ट्यून कर सकते हैं

यहां 16 जीबी के कार्ड (RTX 4080 / 5080 / 4070 Ti Super) पर व्यावहारिक सीमाएं दी गई हैं:

| मॉडल | विधि | स्थिति |
|---|---|---|
| Qwen-3.5-4B / Phi-4-mini-3.8B / SmolLM3-3B | LoRA / QLoRA / DoRA | आरामदायक। पूर्ण सीक्वेंस लंबाई, अतिरिक्त जगह। |
| Phi-4-mini-3.8B / Qwen-3.5-4B / SmolLM3-3B (≤3B पैरामीटर की सीमा) | `mode="full"` (पूर्ण फाइन-ट्यूनिंग) | v1.4 — `backprop train` पर `--mode=full` या `Trainer(..., mode="full")` का उपयोग करें। ग्रेडिएंट चेकपॉइंटिंग + पेज्ड 8-बिट एडम, एक्टिवेशन मेमोरी को sqrt(L) पर बनाए रखता है। |
| Qwen-2.5-7B / Llama-3.1-8B / Mistral-7B | QLoRA | मानक। लगभग 7-8 जीबी। बैकप्रोपैगेट की डिफ़ॉल्ट सेटिंग्स। |
| Llama-3 13B | QLoRA + सैंपल पैकिंग | कठिन लेकिन काम करता है। छोटे सीक्वेंस का उपयोग करें। |
| Mixtral 8x7B (कुल 47 बिलियन पैरामीटर) | AQLM 2-बिट + LoRA | v1.5 के लिए योजनाबद्ध — जब पोस्ट किया जाए तो V1_5_BRIEF देखें। |

AQLM 2-बिट क्वांटाइजेशन (`quant_method="aqlm"`), जो Mixtral-8x7B के लिए 16GB पर एक प्रायोगिक विकल्प है, v1.4 के लिए निर्धारित था और अब v1.5 के लिए योजनाबद्ध है। `aqlm` लाइब्रेरी परिपक्व है; v1.4 में, पूर्ण फाइन-ट्यूनिंग के समर्थन को प्राथमिकता दी गई (≤3B मॉडलों के लिए `mode="full"`), एक नए क्वांटाइजेशन बैकएंड को जोड़ने की तुलना में। v1.5 के कार्यान्वयन योजना के लिए V1_5_BRIEF देखें जब वह पोस्ट किया जाए।

3B और उससे छोटे मॉडलों के लिए, 16GB पर पूर्ण फाइन-ट्यूनिंग (सिर्फ LoRA नहीं) संभव है और अब v1.4 में `mode="full"` के रूप में उपलब्ध है। इसे सक्षम करने के लिए `Trainer(..., mode="full")` या `backprop train --mode=full --model phi-4-mini-3.8b` का उपयोग करें। 3B से बड़े मॉडलों के लिए, यह मोड `RUNTIME_FULL_FT_MODEL_TOO_LARGE` त्रुटि के साथ अस्वीकार कर दिया जाता है, और LoRA और 3B से छोटे मॉडलों को रिकवरी विकल्प के रूप में सूचीबद्ध किया जाता है। कॉन्फ़िगरेशन गणित और Biderman 2024 / Thinking Machines 2025 द्वारा किए गए गुणवत्ता तुलना के लिए [पूर्ण फाइन-ट्यूनिंग हैंडबुक पृष्ठ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/full-fine-tuning/) देखें। 7B+ मॉडलों के लिए, पूर्ण फाइन-ट्यूनिंग के लिए 24GB+ GPU की आवश्यकता होती है — एक A100 क्लाउड रेंटल पर विचार करें, या LoRA का उपयोग करें, जो हाल के शोध से पता चलता है कि अधिकांश पोस्ट-ट्रेनिंग कार्यों पर पूर्ण फाइन-ट्यूनिंग की गुणवत्ता से मेल खाता है (संदर्भों के लिए [एंटी-पिच अनुभाग](#what-backpropagate-is-not-for) देखें)।

## बैकप्रोपैगेट किसके लिए नहीं है

यदि आपका उपयोग-मामला नीचे दिए गए में से है, तो आपको किसी अन्य लाइब्रेरी के साथ बेहतर अनुभव होगा — Backpropagate सही विकल्प नहीं है, और इसे काम करने के लिए मजबूर करने से सही उपकरण का उपयोग करने की तुलना में अधिक लागत आएगी। इस अनुभाग को शुरू करने से पहले पढ़ना, इंस्टॉलेशन और परीक्षण चक्र को बचाता है:

- **7B+ मॉडलों का पूर्ण-पैरामीटर फाइन-ट्यूनिंग** — Backpropagate LoRA / QLoRA का उपयोग करता है, जो प्रत्येक वजन को अपडेट करने के बजाय एक छोटे एडाप्टर को प्रशिक्षित करता है। 7B और उससे बड़े मॉडलों के लिए, पूर्ण फाइन-ट्यूनिंग के लिए 24GB+ GPU मेमोरी की आवश्यकता होती है और यह 16GB के उपभोक्ता कार्ड पर फिट नहीं होता है। 3B और उससे छोटे मॉडलों के लिए, पूर्ण फाइन-ट्यूनिंग 16GB पर संभव है और v1.4 में `mode="full"` के रूप में उपलब्ध है ( `Trainer(..., mode="full")` या `--mode=full` का उपयोग करके CLI पर; 3B से बड़े मॉडलों के लिए `RUNTIME_FULL_FT_MODEL_TOO_LARGE` त्रुटि उत्पन्न होती है और LoRA और 3B से छोटे मॉडल को रिकवरी विकल्प के रूप में सूचीबद्ध किया जाता है)। बड़ी तस्वीर: हाल के शोध ([Biderman 2024](https://arxiv.org/abs/2405.09673), [Thinking Machines 2025](https://thinkingmachines.ai/blog/lora/)) से पता चलता है कि LoRA, सही कॉन्फ़िगरेशन के साथ, अधिकांश पोस्ट-ट्रेनिंग कार्यों (निर्देश-अनुसरण, डोमेन अनुकूलन, व्यक्तित्व/शैली) पर पूर्ण फाइन-ट्यूनिंग की गुणवत्ता से मेल खाता है, और यह 67% कम कंप्यूट संसाधनों का उपयोग करता है — इसलिए, अधिकांश ऑपरेटरों द्वारा किए जाने वाले कार्यों के लिए, LoRA का उपयोग करने से आपको कुछ भी नहीं खोना पड़ता। `mode="full"` उन मामलों के लिए मौजूद है जहां आपने गुणवत्ता में अंतर को मापा है और अतिरिक्त कंप्यूट संसाधनों का उपयोग करने का निर्णय लिया है। यदि आपको वास्तव में 7B+ मॉडल का पूर्ण फाइन-ट्यूनिंग करने की आवश्यकता है, तो HuggingFace `transformers.Trainer` का उपयोग सीधे 24GB+ कार्ड पर करें।
- **DPO / PPO / GRPO / प्राथमिकता ट्यूनिंग** — Backpropagate केवल सिंगल-स्टेज सुपरवाइज्ड फाइन-ट्यूनिंग करता है। प्राथमिकता सीखने के लिए, सीधे TRL या LLaMA-Factory का उपयोग करें।
- **मल्टी-नोड प्रशिक्षण** — केवल एक मशीन पर सिंगल GPU। एक मशीन पर मल्टी-GPU काम करता है ( `accelerate launch` के माध्यम से), लेकिन आधिकारिक तौर पर समर्थित नहीं है।
- **macOS प्रशिक्षण** — Apple Silicon में CUDA नहीं है, इसलिए प्रशिक्षण को Linux या Windows बॉक्स पर NVIDIA GPU के साथ चलाना होगा। आप प्रशिक्षित मॉडल को Ollama के माध्यम से Mac पर चला सकते हैं।
- **परीक्षण किए गए मॉडल परिवारों के बाहर की कोई भी चीज़** — Qwen 2.5 / 3.5 (7B / 4B), Phi-4-mini-3.8B, SmolLM3-3B, Llama 3.2 (3B / 1B), Mistral 7B। अन्य मॉडल अक्सर काम करते हैं लेकिन CI में पिन नहीं किए गए हैं।

यदि आपको इनमें से किसी भी चीज़ की आवश्यकता है, तो ऊपर सूचीबद्ध पुस्तकालयों में से किसी एक का उपयोग करें। वे इसमें बेहतर हैं।

## बैकप्रोपगेट आपको क्या देता है

एक इंस्टॉलेशन में, चार चीजें:

**1. एक वास्तविक 3-पंक्ति API जो किसी कॉन्फ़िगरेशन फ़ाइल के बिना चलता है।**
इस README के शीर्ष पर दिया गया स्निपेट एंड-टू-एंड चलता है। कोई `accelerate config` नहीं, कोई YAML नहीं, कोई Hydra ओवरराइड नहीं। बस `Trainer(model).train(data)` और आपके पास फाइन-ट्यूनिंग है।

**2. विंडोज जो वास्तव में काम करता है।**
अधिकांश ML लाइब्रेरी विंडोज को एक afterthought के रूप में मानते हैं। बैकप्रोपगेट का परीक्षण Windows + RTX 5080 पर पहले दर्जे का किया गया है। लाइब्रेरी आपके लिए रनटाइम की विचित्रताओं को संभालती है - यह जानता है कि आपके डेटा को कैसे प्री-टोकनाइज़ करना है ताकि विंडोज मल्टीप्रोसेसिंग क्रैश न हो, यह स्वचालित रूप से RTX 40/50 कार्ड पर xformers को अक्षम कर देता है जहां यह खराब हो जाता है, और यह डेटालोडर सेटिंग्स चुनता है जो खराब नहीं होती हैं। आपको इनमें से कुछ भी जानने की आवश्यकता नहीं है। यह बस चलता है।

**3. बिना पर्यवेक्षण के चलने के लिए बनाया गया।**
प्रशिक्षण में घंटों लगते हैं। आप इसका ध्यान नहीं रखना चाहते। बैकप्रोपगेट को चलने के लिए डिज़ाइन किया गया है:

- यदि आपके पास GPU मेमोरी खत्म हो जाती है, तो यह स्वचालित रूप से बैच आकार को आधा कर देता है और पुनः प्रयास करता है - अधिकतम तीन बार। कोई मैनुअल ट्यूनिंग नहीं।
- यदि आपका GPU बहुत गर्म हो जाता है, तो यह तब तक रुक जाता है जब तक कि चीजें ठंडी न हो जाएं और फिर जारी रहता है।
- प्रत्येक चेकपॉइंट को परमाणु रूप से लिखा जाता है - यदि आपका लैपटॉप बचत के दौरान क्रैश हो जाता है, तो पिछला अच्छा चेकपॉइंट अभी भी बरकरार रहता है।
- प्रत्येक प्रशिक्षण रन को एक अद्वितीय ID मिलता है जो प्रत्येक लॉग लाइन, प्रत्येक चेकपॉइंट और प्रत्येक Weights & Biases प्रविष्टि पर अंकित होता है। यदि कुछ गलत होता है, तो एक ID एक रखरखावकर्ता को सब कुछ सहसंबंधित करने की अनुमति देता है।
- त्रुटियां स्थिर कोड के साथ आती हैं (`RUNTIME_GPU_OOM`, `DEP_OLLAMA_REGISTRATION_FAILED`, आदि) ताकि आप अपने लॉग को खोज सकें और [समस्या निवारण गाइड](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) में समाधान ढूंढ सकें। CUDA-विशिष्ट विफलताओं के लिए एक समर्पित [CUDA समस्या निवारण पृष्ठ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/) है।

**4. प्रशिक्षित एडाप्टर से `ollama run` तक एक कमांड।**
कई लाइब्रेरी एक मॉडल को प्रशिक्षित करती हैं। उनमें से बहुत कम ही आपको तब भी रास्ता देते हैं जब आप वास्तव में इसका उपयोग करना चाहते हैं। बैकप्रोपगेट GGUF (वह प्रारूप जिसका उपयोग Ollama करता है) में निर्यात करता है और एक कमांड में एक Ollama मॉडल को पंजीकृत करता है। आप "प्रशिक्षण पूरा हुआ" से "मैं अपने फाइन-ट्यून मॉडल के साथ चैट कर सकता हूं" तक लगभग 30 सेकंड में पहुँच जाते हैं।

## शुरुआत कैसे करें।

यह रिपॉजिटरी एक छोटा उदाहरण डेटासेट प्रदान करता है, ताकि इस README फ़ाइल के शीर्ष पर दिया गया कोड एक नए इंस्टॉलेशन पर भी चल सके:

```bash
pipx install "backpropagate[standard]"

python -c "
from backpropagate import Trainer
trainer = Trainer('Qwen/Qwen2.5-7B-Instruct')
trainer.train('examples/quickstart.jsonl', steps=10)
trainer.export('gguf', quantization='q4_k_m')
"
```

यह 5 छोटे ShareGPT-प्रारूप वाले वार्तालापों पर Qwen 2.5 7B एडेप्टर को प्रशिक्षित करता है, और फिर परिणाम को GGUF प्रारूप में निर्यात करता है। अपने स्वयं के डेटा के लिए, अपने JSONL फ़ाइल को एक उदाहरण प्रति पंक्ति के प्रारूप में व्यवस्थित करें:

```jsonl
{"conversations": [{"from": "human", "value": "What is Python?"}, {"from": "gpt", "value": "A programming language."}]}
{"conversations": [{"from": "human", "value": "Explain recursion."}, {"from": "gpt", "value": "A function that calls itself."}]}
```

Alpaca (`instruction` / `output`), OpenAI चैट (`messages`), और कच्चे टेक्स्ट प्रारूप भी काम करते हैं - Backpropagate स्वचालित रूप से प्रारूप का पता लगाता है।

अधिक एंड-टू-एंड वर्कफ़्लो (जैसे, फाइन-ट्यूनिंग और Hugging Face हब पर अपलोड करना, OOM होने पर पुनः आरंभ करना, एक लंबे अभियान में कई बार चलाना, आदि) के लिए, [हैंडबुक रेसिपीज़ पेज](https://mcp-tool-shop-org.github.io/backpropagate/handbook/recipes/) देखें।

### वेब UI (वैकल्पिक)

यदि आप Python कोड लिखने के बजाय क्लिक करना पसंद करते हैं, तो UI एक्सट्रा स्थापित करें और लॉन्च करें:

```bash
pipx install "backpropagate[ui]"
backprop ui --port 7862
```

एक स्थानीय वेब इंटरफ़ेस `http://localhost:7862` पर खुलता है, जहाँ आप एक डेटासेट का चयन कर सकते हैं, एक मॉडल चुन सकते हैं, प्रशिक्षण कर सकते हैं और परिणाम निर्यात कर सकते हैं। डिफ़ॉल्ट रूप से, UI केवल स्थानीय रूप से उपलब्ध होता है। इसे अन्य उपकरणों पर उपलब्ध कराने के लिए, `--share` + `--auth` सुरक्षा अनुबंध के लिए [वेब UI](#web-ui) अनुभाग देखें।

## कई बार प्रशिक्षण

यदि आप कई डेटासेट पर क्रमिक रूप से फाइन-ट्यूनिंग करना चाहते हैं - उदाहरण के लिए, यदि आपको हर हफ्ते नया प्रशिक्षण डेटा मिलता है और आप इसे जोड़ना चाहते हैं, लेकिन पहले सीखी गई जानकारी को भूलना नहीं चाहते हैं - तो Backpropagate का `multi_run` मोड आपके लिए है:

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

यह पांच प्रशिक्षण चक्र चलाता है, और प्रत्येक चक्र के बीच एडेप्टर को इस तरह मर्ज करता है कि पिछली जानकारी बनी रहे और नए उदाहरण शामिल हों। यह तकनीक हाल के निरंतर-सीखने अनुसंधान पर आधारित है - [संदर्भ](#references) अनुभाग में अधिक जानकारी प्राप्त करें, जो इस README फ़ाइल के अंत में दिया गया है।

CLI (कमांड लाइन इंटरफ़ेस) संस्करण:

```bash
backprop multi-run --data my_data.jsonl --runs 5 --steps 100 --samples 1000
```

## चेकपॉइंट से पुनः आरंभ करें

एक 5-चक्र प्रशिक्षण जो चौथे चक्र में क्रैश हो जाता है, उसे पुनः आरंभ किया जा सकता है। प्रत्येक `multi_run` सत्र अपनी रन आईडी को ऑन-डिस्क इतिहास और चेकपॉइंट मैनिफेस्ट में लिखता है, इसलिए जहां से आपने छोड़ा था, वहीं से शुरू करने के लिए केवल एक कमांड की आवश्यकता होती है:

```bash
backprop resume <run-id>
backprop multi-run --data ... --resume <run-id>
backprop train --data ... --resume <run-id>     # single-run resume
```

`backprop multi-run` का डिफ़ॉल्ट व्यवहार (बिना `--resume` विकल्प के) समान आउटपुट निर्देशिका में चल रहे सत्र का पता लगाता है और उसे जारी रखता है। यदि आप एक नए सत्र से शुरुआत करना चाहते हैं, तो एक नई आउटपुट निर्देशिका का उपयोग करें।

## प्रशिक्षण इतिहास

प्रत्येक `backprop train` और `backprop multi-run` कमांड `<output>/run_history.json` फ़ाइल में एक पंक्ति रिकॉर्ड करता है - उपयोग किया गया मॉडल, डेटासेट, हाइपरपैरामीटर, स्थिति, अंतिम हानि, हानि इतिहास। आप पिछले रनों को सूचीबद्ध और जांच सकते हैं:

```bash
backprop list-runs                         # last 20 runs
backprop list-runs --status failed         # filter by status
backprop list-runs --json --limit 100      # machine-readable
backprop show-run abcd1234                 # detail view (partial ID is fine)
```

## प्रयोग ट्रैकिंग

Backpropagate स्थापित प्रयोग ट्रैकर (Weights & Biases, TensorBoard, MLflow) का स्वचालित रूप से पता लगाता है और उन्हें एकीकृत करता है। यदि `wandb` स्थापित है और आप लॉग इन हैं, तो प्रत्येक रन स्वचालित रूप से W&B पर लॉग इन हो जाता है, और रन का नाम ऑन-डिस्क रन आईडी से मेल खाता है - जिससे आप W&B, अपने लॉग और `run_history.json` फ़ाइल में एक ही पहचानकर्ता का उपयोग करके खोज कर सकते हैं।

```bash
pip install backpropagate[monitoring]   # installs wandb + psutil
wandb login                             # one-time setup
backprop train --data my_data.jsonl
```

`Trainer(report_to=["wandb"])`, `Trainer(report_to=["tensorboard"])`, या `Trainer(report_to="none")` का उपयोग करके इसे ओवरराइड किया जा सकता है।

## वेब UI

Reflex वेब इंटरफ़ेस वैकल्पिक है - इसे `pipx install "backpropagate[ui]"` के साथ स्थापित करें और लॉन्च करें:

```bash
backprop ui --port 7862
```

UI स्थानीय रूप से `http://localhost:7862` पर चलता है। इसे अन्य उपकरणों पर उपलब्ध कराने के लिए (आपके नेटवर्क पर अन्य लोग, एक सार्वजनिक URL, आदि), आपको `--share` (या `--host`) को `--auth` के साथ जोड़ना होगा:

```bash
backprop ui --share --auth alice:hunter2
```

`backprop ui --share` बिना `--auth` के एक त्रुटि के साथ समाप्त हो जाता है। इसका कारण यह है कि `--share` एक URL प्रकाशित करता है जिसे इंटरनेट पर कोई भी व्यक्ति एक्सेस कर सकता है, और बिना प्रमाणीकरण के, इसका मतलब है कि कोई भी आपके प्रशिक्षण पाइपलाइन को चला सकता है और आपके HuggingFace टोकन को पढ़ सकता है। इसके लिए कोई विकल्प उपलब्ध नहीं है - यदि आप क्रेडेंशियल सेट नहीं करना चाहते हैं, तो SSH पोर्ट-फॉरवर्डिंग का उपयोग करें:

```bash
# On the client:
ssh -L 7860:localhost:7860 <your-training-host>
# On the server:
backprop ui                             # no --share
# Then open http://localhost:7860 in your local browser
```

पूर्ण खतरे के मॉडल के लिए [handbook/security.md](https://mcp-tool-shop-org.github.io/backpropagate/handbook/security/) देखें।

UI से किए गए फ़ाइल सिस्टम लेखन को एक ही डायरेक्टरी तक सीमित कर दिया गया है:

- डिफ़ॉल्ट: `~/.backpropagate/ui-outputs`
- बदलने के लिए: `BACKPROPAGATE_UI__OUTPUT_DIR=/path/you/own` सेट करें
- इस बदलाव को 'ब्लैकलिस्ट' से जांचा जाता है - सिस्टम या क्रेडेंशियल पाथ (`/etc`, `~/.ssh`, `~/.aws`, `C:\Windows\System32`, आदि) को अनुमति नहीं है।

## प्लेटफ़ॉर्म संबंधी जानकारी

**आवश्यकताएं:** पायथन 3.10 या उससे ऊपर का संस्करण, CUDA जीपीयू (8 जीबी या उससे अधिक वीआरएएम), पायटॉर्च 2.0 या उससे ऊपर का संस्करण।

Python 3.10 का जीवनकाल अक्टूबर 2026 में समाप्त हो रहा है, और Backpropagate v1.4 में 3.10 को हटा देगा। नए इंस्टॉलेशन के लिए, Python 3.11 या 3.12 को प्राथमिकता दें - 3.11 सबसे अधिक परीक्षण किया गया संस्करण है।

Backpropagate विभिन्न प्लेटफ़ॉर्म पर प्रशिक्षण के दौरान आने वाली समस्याओं को संभालता है, लेकिन यह इंस्टॉलेशन के समय होने वाली समस्याओं को ठीक नहीं कर सकता। दो सबसे आम समस्याएं हैं:

- **गलत CUDA ड्राइवर।** PyTorch प्रत्येक CUDA संस्करण के लिए एक बाइनरी जारी करता है। यदि आप गलत बाइनरी चुनते हैं, तो आपको केवल CPU-आधारित PyTorch मिलेगा और प्रशिक्षण बहुत धीमा होगा। अपने ड्राइवर के लिए <https://pytorch.org/get-started/locally/> पर दिए गए ड्राइवर चयनकर्ता का उपयोग करें। अपने ड्राइवर/CUDA संस्करण को देखने के लिए `nvidia-smi` कमांड चलाएं।
- **Windows + GGUF एक्सपोर्ट।** `[export]` एक्सट्रा `llama-cpp-python` को स्रोत कोड से बनाता है, जिसके लिए Visual Studio Build Tools (C++ घटक) और CMake की आवश्यकता होती है।

**macOS:** GPU प्रशिक्षण समर्थित नहीं है (CUDA उपलब्ध नहीं है)। आप प्रशिक्षित एडेप्टर को Ollama के माध्यम से Mac पर चला सकते हैं, लेकिन `trainer.train()` में `DEP_GPU_NOT_AVAILABLE` त्रुटि उत्पन्न होगी। प्रशिक्षण के लिए CUDA Linux या Windows मशीन का उपयोग करें।

स्थापना संबंधी समस्याओं को ठीक करने के लिए विस्तृत गाइड के लिए [समस्या निवारण मार्गदर्शिका पृष्ठ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) देखें, और ड्राइवर/VRAM/xformers/bf16-vs-fp16 संबंधी समस्याओं के लिए समर्पित [CUDA समस्या निवारण पृष्ठ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/) देखें।

## CLI

प्रत्येक Python API के लिए एक CLI (कमांड लाइन इंटरफेस) विकल्प उपलब्ध है:

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

पूरा संदर्भ [CLI मार्गदर्शिका पृष्ठ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/cli-reference/) पर उपलब्ध है, या `backprop <उप-कमांड> --help` कमांड का उपयोग करें।

## कॉन्फ़िगरेशन

प्रत्येक सेटिंग को `BACKPROPAGATE_` उपसर्ग का उपयोग करके एक पर्यावरण चर के माध्यम से बदला जा सकता है:

| चर | डिफ़ॉल्ट | टिप्पणियाँ |
|---|---|---|
| `BACKPROPAGATE_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `BACKPROPAGATE_LOG_JSON` | ऑटो | JSON या कंसोल लॉग को बाध्यकारी करें। |
| `BACKPROPAGATE_MODEL__NAME` | `Qwen/Qwen2.5-7B-Instruct` | डिफ़ॉल्ट मॉडल |
| `BACKPROPAGATE_TRAINING__LEARNING_RATE` | `2e-4` | लर्निंग रेट |
| `BACKPROPAGATE_LORA__R` | `256` | LoRA रैंक (v1.3 डिफ़ॉल्ट; v1.2.x के डिफ़ॉल्ट 16 के लिए `--lora-preset=fast` का उपयोग करें)। |
| `BACKPROPAGATE_UI__OUTPUT_DIR` | `~/.backpropagate/ui-outputs` | UI फ़ाइल सिस्टम सैंडबॉक्स। |

नेस्टेड कुंजियों के लिए डबल अंडरस्कोर (`_`) का उपयोग करें (`MODEL__NAME`, `MODEL_NAME` नहीं)। पूरा संदर्भ [पर्यावरण चर मार्गदर्शिका पृष्ठ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/env-vars/) पर उपलब्ध है।

## मॉडल प्रीसेट

| प्रीसेट | VRAM | लाइसेंस | टिप्पणियाँ |
|---|---|---|---|
| Qwen-3.5-4B | ~8GB | Apache 2.0 | 5B से कम आकार के लिए अनुशंसित डिफ़ॉल्ट। इस आकार पर सर्वोत्तम गुणवत्ता। |
| Phi-4-mini-3.8B | ~8GB | MIT | तर्क/गणित/कोड में मजबूत। सख्त लाइसेंस-अनुकूल। |
| SmolLM3-3B | ~6GB | Apache 2.0 | पूरी तरह से खुला स्रोत। 64K का मूल संदर्भ। |
| Qwen 2.5 7B | ~12GB | Apache 2.0 | मौजूदा डिफ़ॉल्ट। पुराने 7B प्रीसेट में सर्वोत्तम गुणवत्ता। |
| Qwen 2.5 3B | ~8GB | Qwen-Research | ⚠ अनुसंधान लाइसेंस - वाणिज्यिक उपयोग से पहले Qwen के लाइसेंस नियमों को देखें। |
| Llama 3.2 3B | ~8GB | Llama Community | Qwen 3B का एक अच्छा विकल्प, कुछ शर्तों के साथ। |
| Llama 3.2 1B | ~6GB | Llama Community | छोटे कार्ड पर त्वरित प्रयोग के लिए। |
| Mistral 7B | ~12GB | Apache 2.0 | Qwen 7B के समान, अलग चैट टेम्पलेट। |

अन्य मॉडल भी काम कर सकते हैं, लेकिन केवल ये आठ मॉडल CI (निरंतर एकीकरण) में उपयोग किए जाते हैं। रैंक-256 / सभी-लीनियर लक्ष्यों के लिए Biderman 2024 + Thinking Machines 2025 के अनुसार `--lora-preset=quality` (डिफ़ॉल्ट) या यदि आपको v1.2.x का आकार चाहिए तो रैंक-16 / q+v लक्ष्य के लिए `--lora-preset=fast` का उपयोग करें।

## समस्या निवारण

सबसे आम शुरुआती विफलताओं का संक्षिप्त विवरण। पूरा रिवर्स इंडेक्स [समस्या निवारण मार्गदर्शिका पृष्ठ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) पर उपलब्ध है। ड्राइवर/VRAM/मिश्रित परिशुद्धता के बारे में अधिक जानकारी के लिए, [CUDA समस्या निवारण पृष्ठ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting-cuda/) देखें।

| लक्षण | त्रुटि कोड | समाधान |
|---|---|---|
| प्रशिक्षण के दौरान GPU की मेमोरी समाप्त हो जाती है। | `RUNTIME_GPU_OOM` | स्वचालित — बैकप्रोपगेट बैच के आकार को आधा कर देता है और 3 बार तक पुनः प्रयास करता है। इसे निष्क्रिय करने के लिए: `Trainer(oom_recovery=False)`. बैच के आकार को कम करने के लिए: `--batch-size 1`. |
| हगिंगफेस 401 / "मॉडल नहीं मिला" त्रुटि देता है। | `DEP_MODEL_LOAD_FAILED` | `huggingface-cli login` चलाएं और फिर से प्रयास करें। टाइपिंग की गलतियों के लिए, <https://huggingface.co/models> से सटीक आईडी कॉपी करें। |
| `register_with_ollama` कनेक्शन अस्वीकृत। | `DEP_OLLAMA_REGISTRATION_FAILED` | डेमॉन शुरू करें: `ollama serve`। <https://ollama.com> से इंस्टॉल करें। पुनः प्रयास करने योग्य। |
| चेकपॉइंट सहेजते समय डिस्क भर गई। | `STATE_CHECKPOINT_INVALID` | क्रैश होने पर एटॉमिक राइट्स `.partial` नामक एक डायरेक्टरी बनाते हैं - इसे हटाना सुरक्षित है। पिछला अच्छा चेकपॉइंट बरकरार है। |
| जीपीयू के अत्यधिक गर्म होने के कारण प्रशिक्षण रुका हुआ है। | `RUNTIME_GPU_TEMPERATURE_CRITICAL` | स्वचालित — बैकप्रोपगेट तापमान सीमा पर रुक जाता है और जीपीयू के ठंडा होने पर फिर से शुरू हो जाता है। यदि यह बार-बार होता रहता है, तो वायु प्रवाह में सुधार करें। |
| `backprop ui --share` अस्वीकृत। | `INPUT_AUTH_REQUIRED` | `--auth user:password` का उपयोग करें, या इसके बजाय एसएसएच पोर्ट-फॉरवर्डिंग का उपयोग करें (देखें [वेब यूआई](#web-ui))। |
| GGUF एक्सपोर्ट पहली बार में विफल रहा। | `RUNTIME_GGUF_EXPORT_FAILED` | `pip install backpropagate[export]`। विंडोज पर, आपको Visual C++ Build Tools + CMake की भी आवश्यकता है। |

## बग की रिपोर्ट करना

जब कुछ विफल होता है, तो बैकप्रोपगेट स्टार्टअप पर एक पंक्ति प्रिंट करता है जैसे `run_started run_id=<uuid>` और हर लॉग पंक्ति, हर चेकपॉइंट और हर वेट्स एंड बायसेस प्रविष्टि के साथ समान आईडी को जोड़ता है। **किसी भी बग रिपोर्ट में `run_id` शामिल करें** — यह एक रखरखावकर्ता को उस विशिष्ट रन के लिए सब कुछ सहसंबंधित करने की अनुमति देता है।

एक अच्छी बग रिपोर्ट में शामिल हैं:

1. **`run_id`**: यह UUID (यूनिवर्सल यूनिक आइडेंटिफायर) है जो स्टार्टअप के समय प्रदर्शित होता है। एक UUID एक रखरखावकर्ता को उस विशेष रन के लिए प्रत्येक लॉग लाइन, प्रत्येक चेकपॉइंट और प्रत्येक वेट्स एंड बायसेस प्रविष्टि को जोड़ने में मदद करता है।
2. **त्रुटि कोड**: यह `stderr` में `[कोड_नाम]: संदेश` के रूप में दिखाई देने वाली लाइन है। स्थिर कोडों की सूची के लिए, [त्रुटि कोड](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/) देखें।
3. **संशोधित ट्रेसबैक (Traceback)।** गैर-विस्तृत मोड में `stderr` स्वचालित रूप से संशोधित किया जाता है (Bearer टोकन, `sk-*`, `hf_*`, AWS कुंजियाँ, `password=` / `token=` / `api_key=` जोड़े हटा दिए जाते हैं) - इसे सुरक्षित रूप से पेस्ट किया जा सकता है। पूर्ण, बिना संशोधित ट्रेसबैक के लिए, `BACKPROPAGATE_DEBUG=1` (या `--verbose`) के साथ पुनः चलाएं; पोस्ट करने से पहले इसकी समीक्षा करें।
4. **`backprop info` आउटपुट।** एक कमांड Python / PyTorch / CUDA / GPU मॉडल / VRAM / ऑपरेटिंग सिस्टम / स्थापित अतिरिक्त सुविधाओं के बारे में जानकारी प्रदर्शित करता है - यह सब कुछ एक रखरखावकर्ता को किसी प्लेटफ़ॉर्म-विशिष्ट समस्या का पता लगाने के लिए आवश्यक होता है।

[बग रिपोर्ट टेम्पलेट](https://github.com/mcp-tool-shop-org/backpropagate/issues/new?template=bug_report.yml) इन सभी चीजों के लिए स्पष्ट रूप से जानकारी मांगता है, इसलिए समस्या निवारण प्रक्रिया तेजी से होती है। प्रश्न, विचार, या "क्या यह अपेक्षित है?" जैसे विषय [GitHub Discussions](https://github.com/mcp-tool-shop-org/backpropagate/discussions) पर पोस्ट किए जाने चाहिए। सुरक्षा संबंधी मुद्दों की रिपोर्ट [GitHub Security Advisory](https://github.com/mcp-tool-shop-org/backpropagate/security/advisories/new) फॉर्म के माध्यम से निजी तौर पर की जानी चाहिए - नीति और प्रतिक्रिया समय-सीमा के लिए [SECURITY.md](SECURITY.md) देखें।

## गोपनीयता

सभी प्रशिक्षण आपके GPU पर स्थानीय रूप से होता है। Backpropagate केवल HuggingFace से मॉडल डाउनलोड करने के लिए नेटवर्क अनुरोध करता है (जो आप शुरू करते हैं)। कोई टेलीमेट्री नहीं, कोई क्लाउड निर्भरता नहीं।

## संदर्भ

बैकप्रोपगेट की डिफ़ॉल्ट सेटिंग्स और मल्टी-रन प्रशिक्षण मोड हाल के शोध पर आधारित हैं। यदि आप अंतर्निहित तकनीकों में रुचि रखते हैं:

- **हु एट अल। 2021।** *लोरा: बड़े भाषा मॉडल का निम्न-रैंक अनुकूलन।* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) — लोरा पेश करने वाला मौलिक पेपर, जिसका उपयोग बैकप्रोपगेट कुशलतापूर्वक एडेप्टर को प्रशिक्षित करने के लिए करता है।
- **बिडरमैन एट अल। 2024।** *लोरा सीखता है कम और भूलता है कम।* [arXiv:2405.09673](https://arxiv.org/abs/2405.09673) — अनुभवजन्य प्रमाण कि 256 की रैंक पर सभी-रैखिक लक्ष्यों के साथ लोरा, अधिकांश पोस्ट-प्रशिक्षण कार्यों पर पूर्ण फाइन-ट्यूनिंग की गुणवत्ता से मेल खाता है, जो 67% कंप्यूट है। यह बैकप्रोपगेट के v1.3 डिफ़ॉल्ट लोरा कॉन्फ़िगरेशन को चलाता है।
- **थिंकिंग मशीन 2025।** *लोरा बिना पछतावे के।* [thinkingmachines.ai/blog/lora](https://thinkingmachines.ai/blog/lora/) — व्यावहारिक अनुवर्ती जो उच्च लोरा रैंक पर आवश्यक 10 गुना लर्निंग-रेट-vs-फुल-एफटी सुधार की पहचान करता है।
- **किर्कपैट्रिक एट अल। 2017।** *तंत्रिका नेटवर्क में विनाशकारी भूल को दूर करना।* [arXiv:1612.00796](https://arxiv.org/abs/1612.00796) — यह मूल विशेषता है कि तंत्रिका नेटवर्क "भूल" जाते हैं जब आप नए डेटा पर फाइन-ट्यून करते हैं (ईडब्ल्यूसी — इलास्टिक वेट कंसोलिडेशन)।
- **वांग एट अल। 2023।** *भाषा मॉडल निरंतर सीखने के लिए ऑर्थोगोनल सबस्पेस लर्निंग।* [arXiv:2310.14152](https://arxiv.org/abs/2310.14152) — ओ-लोरा, एक प्रारंभिक दृष्टिकोण जो निरंतर सीखने के लिए लोरा का उपयोग करता है, जो नए एडेप्टर को ऑर्थोगोनल सबस्पेस तक सीमित करता है।
- **यादव एट अल। 2023।** *टीईएस-मर्जिंग: मॉडलों को मर्ज करते समय हस्तक्षेप को हल करना।* [arXiv:2306.01708](https://arxiv.org/abs/2306.01708) — कई फाइन-ट्यून किए गए मॉडलों को हस्तक्षेप के बिना मर्ज करने के लिए एक मौलिक तकनीक।
- **कियाओ और महदावी 2025।** *मर्ज बिफोर फॉरगेट: एक सिंगल लोरा निरंतर लर्निंग वाया कंटीन्यूअल मर्जिंग।* [arXiv:2512.23017](https://arxiv.org/abs/2512.23017) — विशिष्ट एल्गोरिथ्म जिसे बैकप्रोपगेट का मल्टी-रन मर्जर लागू करता है। दिसंबर 2025 का एक प्रीप्रिंट; बैकप्रोपगेट इस पेपर का पहला ज्ञात डाउनस्ट्रीम एडॉप्टर है।

## लाइसेंस

एमआईटी — [लाइसेंस](LICENSE) देखें।

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>
