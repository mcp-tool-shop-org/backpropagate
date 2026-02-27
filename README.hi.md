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
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License"></a>
  <a href="https://mcp-tool-shop-org.github.io/backpropagate/"><img src="https://img.shields.io/badge/Landing_Page-live-blue" alt="Landing Page"></a>
</p>

"हेडलेस एलएलएम (LLM) का फाइन-ट्यूनिंग, केवल तीन लाइनों में। इसमें स्मार्ट डिफ़ॉल्ट सेटिंग्स, वीआरएएम (VRAM) को ध्यान में रखते हुए बैच का आकार निर्धारण, मल्टी-रन एसएलएओ (SLAO), और ओलामा (Ollama) के लिए एक-क्लिक जीजीयूएफ (GGUF) एक्सपोर्ट की सुविधा है।"

*सिर्फ तीन पंक्तियों के कोड का उपयोग करके बड़े भाषा मॉडल (LLMs) को प्रशिक्षित करें। फिर, इसे केवल एक और पंक्ति के कोड से ओलामा (Ollama) में एक्सपोर्ट करें।*

## शुरुआत कैसे करें।

```bash
pip install backpropagate[standard]
```

```python
from backpropagate import Trainer

trainer = Trainer("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
trainer.train("my_data.jsonl", steps=100)
trainer.export("gguf", quantization="q4_k_m")  # Ready for Ollama
```

## बैकप्रोपगेशन क्यों किया जाता है?

| समस्या। | समाधान। |
|---------|----------|
| बारीक समायोजन एक जटिल प्रक्रिया है। | 3 पंक्तियाँ: लोड, ट्रेन, सेव। |
| विंडोज एक दुःस्वप्न जैसा है। | विंडोज के लिए उत्कृष्ट समर्थन। |
| वीआरएएम (VRAM) का प्रबंधन एक जटिल कार्य है। | स्वचालित बैच आकार निर्धारण, जीपीयू (ग्राफिक्स प्रोसेसिंग यूनिट) की निगरानी। |
| मॉडल निर्यात करना एक जटिल प्रक्रिया है। | एक क्लिक में GGUF और Ollama का पंजीकरण। |
| लंबे समय तक लगातार काम करने से भूलने की समस्या हो सकती है। | बहु-बार एसएलएओ प्रशिक्षण। |

## मुख्य विशेषताएं।

- **डिज़ाइन द्वारा हेडलेस:** सीआई/सीडी पाइपलाइनों, स्वचालित वर्कफ़्लो और प्रोग्रामेटिक निष्पादन के लिए निर्मित।
- **स्मार्ट डिफ़ॉल्ट सेटिंग्स:** आपके हार्डवेयर और डेटासेट के आधार पर, स्वचालित रूप से इष्टतम हाइपरपैरामीटर कॉन्फ़िगर करता है।
- **मल्टी-रन एसएलएओ प्रशिक्षण:** लंबे समय तक चलने वाले प्रशिक्षण के दौरान विनाशकारी भूलने से बचाने के लिए उन्नत प्रशिक्षण रणनीतियाँ।
- **विंडोज के लिए उत्कृष्ट समर्थन:** विंडोज वातावरण में परीक्षण और अनुकूलित, सामान्य पायटॉर्च/CUDA समस्याओं से बचा जाता है।
- **आसान निर्यात:** एक क्लिक में जीजीयूएफ प्रारूप में निर्यात करें और ओलामा के साथ स्वचालित रूप से पंजीकृत करें।
- **मॉड्यूलर आर्किटेक्चर:** केवल उन निर्भरताओं को स्थापित करें जिनकी आपको आवश्यकता है (उदाहरण के लिए, `[unsloth]`, `[ui]`, `[export]`)।

## स्थापना।

```bash
pip install backpropagate              # Core only (minimal)
pip install backpropagate[unsloth]     # + Unsloth 2x faster training
pip install backpropagate[ui]          # + Gradio web UI
pip install backpropagate[standard]    # unsloth + ui (recommended)
pip install backpropagate[full]        # Everything
```

| अतिरिक्त। | विवरण। | निर्भरताएँ। |
|-------|-------------|--------------|
| `unsloth` | 2 गुना तेज़ प्रशिक्षण, 50% कम वीआरएएम (VRAM) की आवश्यकता। | अनस्लोथ। |
| `ui` | ग्रैडियो वेब इंटरफेस। | gradio >= 5.6.0 |
| `validation` | पायडैंटिक का कॉन्फ़िगरेशन सत्यापन। | pydantic, pydantic-सेटिंग्स |
| `export` | ओलामा के लिए जीजीयूएफ प्रारूप में निर्यात। | llama-cpp-python |
| `monitoring` | वैंडबी (WandB) और सिस्टम मॉनिटरिंग। | wandb, psutil |

**आवश्यकताएं:** पायथन 3.10 या उससे ऊपर का संस्करण, CUDA जीपीयू (8 जीबी या उससे अधिक वीआरएएम), पायटॉर्च 2.0 या उससे ऊपर का संस्करण।

## उपयोग

### बुनियादी प्रशिक्षण।

```python
from backpropagate import Trainer

trainer = Trainer("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
trainer.train("my_data.jsonl", steps=100)
trainer.save("./my-model")
trainer.export("gguf", quantization="q4_k_m")
```

### बहु-चरण एसएलएओ प्रशिक्षण।

```python
from backpropagate import Trainer

trainer = Trainer("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")

result = trainer.multi_run(
    dataset="HuggingFaceH4/ultrachat_200k",
    num_runs=5,
    steps_per_run=100,
    samples_per_run=1000,
    merge_mode="slao",  # Smart LoRA merging
)
```

### ओलामा में निर्यात करें।

```python
trainer.export(
    format="gguf",
    quantization="q4_k_m",
    register_ollama=True,
    model_name="my-finetuned-model",
)
# ollama run my-finetuned-model
```

### सीएलआई (CLI)

```bash
backprop train --data my_data.jsonl --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit --steps 100
backprop multi-run --data my_data.jsonl --runs 5 --steps 100
backprop export ./output/lora --format gguf --quantization q4_k_m --ollama --ollama-name my-model
backpropagate --ui --port 7862
```

## विंडोज सहायता।

"बैकप्रोपगेट" को शुरुआत से ही विंडोज ऑपरेटिंग सिस्टम पर सुचारू रूप से काम करने के लिए डिज़ाइन किया गया है।

- मल्टीप्रोसेसिंग के दौरान होने वाली त्रुटियों से बचने के लिए, पहले टोकनाइजेशन किया गया है।
- RTX 40/50 सीरीज के लिए, xformers स्वचालित रूप से निष्क्रिय कर दिए जाते हैं।
- सुरक्षित डेटा लोडर सेटिंग्स।
- RTX 5080 (16GB VRAM) पर परीक्षण किया गया।

## मॉडल प्रीसेट।

| पूर्व-निर्धारित। | वीआरएएम (VRAM) - वीडियो रैंडम एक्सेस मेमोरी. | गति। | गुणवत्ता। |
|--------|------|-------|---------|
| क्वेन 2.5, 7 बिलियन (पैरामीटर)। | लगभग 12 जीबी। | माध्यम। | सर्वश्रेष्ठ। |
| क्वेन 2.5, 3 बिलियन पैरामीटर वाला मॉडल। | लगभग 8 जीबी। | तेज़। | अच्छा। |
| लामा 3.2, 3 बिलियन पैरामीटर वाला मॉडल। | लगभग 8 जीबी। | तेज़। | अच्छा। |
| लामा 3.2, 1 बिलियन पैरामीटर वाला मॉडल। | लगभग 6 जीबी। | सबसे तेज़। | बुनियादी। |
| मिस्ट्रल 7बी। | लगभग 12 जीबी। | माध्यम। | अच्छा। |

## आर्किटेक्चर।

```
backpropagate/
├── trainer.py           # Core Trainer class
├── multi_run.py         # Multi-run SLAO training
├── slao.py              # SLAO LoRA merging algorithm
├── datasets.py          # Dataset loading & filtering
├── export.py            # GGUF/Ollama export
├── config.py            # Pydantic settings
├── gpu_safety.py        # GPU monitoring & safety
└── ui.py                # Gradio interface
```

## गोपनीयता।

सभी प्रशिक्षण आपके जीपीयू (GPU) पर ही स्थानीय रूप से होता है। बैकप्रोपगेशन (Backpropagation) किसी भी नेटवर्क से अनुरोध नहीं करता है, सिवाय हगिंगफेस (Hugging Face) से मॉडल डाउनलोड करने के (जिसे आप स्वयं शुरू करते हैं)। इसमें कोई टेलीमेट्री (telemetry) नहीं है, और न ही यह किसी क्लाउड पर निर्भर करता है।

## स्कोरकार्ड।

| श्रेणी। | स्कोर। | टिप्पणियाँ। |
|----------|-------|-------|
| अ. सुरक्षा। | 10/10 | सुरक्षा.md, CI में बैंडिट, सेमग्रेप, ट्रिवी और ट्रफलहॉग का उपयोग, पाथ ट्रैवर्सल सुरक्षा। |
| बी. त्रुटि प्रबंधन। | 8/10 | संरचित त्रुटियाँ, जीपीयू सुरक्षा सीमाएँ, चेकपॉइंट पुनर्प्राप्ति। |
| सी. ऑपरेटर दस्तावेज़। | 9/10 | रीडमी (README), बदलावों का विवरण (CHANGELOG), मॉड्यूलर इंस्टॉलेशन गाइड, कमांड-लाइन इंटरफेस (CLI) सहायता। |
| डी. शिपिंग स्वच्छता। | 9/10 | सीआई (CI) परीक्षण (33 फाइलें), पाइपीआई (PyPI) पर प्रकाशन, कोडकव (Codecov) कवरेज। |
| ई. पहचान। | 10/10 | लोगो, अनुवाद, लैंडिंग पृष्ठ, PyPI लिस्टिंग। |
| **Total** | **46/50** | |

## लाइसेंस

MIT - विवरण के लिए [LICENSE](LICENSE) देखें।

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>
