<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
</p>

<p align="center">
  <img src="assets/logo.png" alt="Backpropagate" width="400">
</p>

<p align="center">
  <a href="https://github.com/mcp-tool-shop-org/backpropagate/actions/workflows/ci.yml"><img src="https://github.com/mcp-tool-shop-org/backpropagate/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/backpropagate/"><img src="https://img.shields.io/pypi/v/backpropagate" alt="PyPI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License"></a>
  <a href="https://mcp-tool-shop-org.github.io/backpropagate/"><img src="https://img.shields.io/badge/Landing_Page-live-blue" alt="Landing Page"></a>
</p>

**3 पंक्तियों में हेडलेस एलएलएम फाइन-ट्यूनिंग। स्मार्ट डिफ़ॉल्ट सेटिंग्स, वीआरएएम-जागरूक बैच आकार, मल्टी-रन एसएलएओ, और ओलामा के लिए वन-क्लिक जीजीयूएफ एक्सपोर्ट।**

*3 पंक्तियों के कोड में एलएलएम को प्रशिक्षित करें। इसे एक और पंक्ति में ओलामा में एक्सपोर्ट करें।*

## शुरुआत कैसे करें

```bash
pip install backpropagate[standard]
```

```python
from backpropagate import Trainer

trainer = Trainer("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
trainer.train("my_data.jsonl", steps=100)
trainer.export("gguf", quantization="q4_k_m")  # Ready for Ollama
```

## बैकप्रोपैगेशन क्यों?

| समस्या | समाधान |
| --------- | ---------- |
| फाइन-ट्यूनिंग जटिल है | 3 पंक्तियाँ: लोड करें, प्रशिक्षित करें, सहेजें |
| विंडोज एक दुःस्वप्न है | विंडोज के लिए उत्कृष्ट समर्थन |
| वीआरएएम प्रबंधन मुश्किल है | स्वचालित बैच आकार, जीपीयू मॉनिटरिंग |
| मॉडल एक्सपोर्ट भ्रमित करने वाला है | वन-क्लिक जीजीयूएफ + ओलामा पंजीकरण |
| लंबे समय तक चलने वाले प्रशिक्षण से भूलने की समस्या होती है | मल्टी-रन एसएलएओ प्रशिक्षण |

## मुख्य विशेषताएं

- **डिजाइन द्वारा हेडलेस**: सीआई/सीडी पाइपलाइनों, स्वचालित वर्कफ़्लो और प्रोग्रामेटिक निष्पादन के लिए बनाया गया।
- **स्मार्ट डिफ़ॉल्ट**: आपके हार्डवेयर और डेटासेट के आधार पर इष्टतम हाइपरपैरामीटर स्वचालित रूप से कॉन्फ़िगर करता है।
- **मल्टी-रन एसएलएओ प्रशिक्षण**: लंबे समय तक चलने वाले प्रशिक्षण के दौरान विनाशकारी भूलने से रोकने के लिए उन्नत प्रशिक्षण रणनीतियाँ।
- **विंडोज के लिए उत्कृष्ट समर्थन**: विंडोज वातावरण के लिए परीक्षण किया गया और अनुकूलित, सामान्य पायटॉर्च/सीयूडीए समस्याओं से बचा जाता है।
- **आसान एक्सपोर्ट**: जीजीयूएफ प्रारूप में वन-क्लिक एक्सपोर्ट और ओलामा के साथ स्वचालित पंजीकरण।
- **मॉड्यूलर आर्किटेक्चर**: केवल आवश्यक निर्भरताएँ स्थापित करें (जैसे, `[unsloth]`, `[ui]`, `[export]`)।

## स्थापना

```bash
pip install backpropagate              # Core only (minimal)
pip install backpropagate[unsloth]     # + Unsloth 2x faster training
pip install backpropagate[ui]          # + Gradio web UI
pip install backpropagate[standard]    # unsloth + ui (recommended)
pip install backpropagate[full]        # Everything
```

| Extra | विवरण | निर्भरताएँ |
| ------- | ------------- | -------------- |
| `unsloth` | प्रशिक्षण 2 गुना तेज, वीआरएएम 50% कम | अनस्लोथ |
| `ui` | ग्राडियो वेब इंटरफ़ेस | gradio>=5.6.0 |
| `validation` | पायडैंटिक कॉन्फ़िगरेशन सत्यापन | pydantic, pydantic-settings |
| `export` | ओलामा के लिए जीजीयूएफ एक्सपोर्ट | llama-cpp-python |
| `monitoring` | वैंडबी + सिस्टम मॉनिटरिंग | wandb, psutil |

**आवश्यकताएँ:** पायथन 3.10+, सीयूडीए जीपीयू (8GB+ वीआरएएम), पायटॉर्च 2.0+

## उपयोग

### बुनियादी प्रशिक्षण

```python
from backpropagate import Trainer

trainer = Trainer("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
trainer.train("my_data.jsonl", steps=100)
trainer.save("./my-model")
trainer.export("gguf", quantization="q4_k_m")
```

### मल्टी-रन एसएलएओ प्रशिक्षण

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

### ओलामा में एक्सपोर्ट करें

```python
trainer.export(
    format="gguf",
    quantization="q4_k_m",
    register_ollama=True,
    model_name="my-finetuned-model",
)
# ollama run my-finetuned-model
```

### सीएलआई

```bash
backprop train --data my_data.jsonl --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit --steps 100
backprop multi-run --data my_data.jsonl --runs 5 --steps 100
backprop export ./output/lora --format gguf --quantization q4_k_m --ollama --ollama-name my-model
backpropagate --ui --port 7862
```

## विंडोज समर्थन

बैकप्रोपैगेशन को बिना किसी समस्या के विंडोज पर काम करने के लिए डिज़ाइन किया गया है:

- मल्टीप्रोसेसिंग क्रैश से बचने के लिए प्री-टोकनाइजेशन
- RTX 40/50 श्रृंखला के लिए स्वचालित एक्सफॉर्मर्स डिसेबल
- सुरक्षित डेटालोडर सेटिंग्स
- RTX 5080 (16GB VRAM) पर परीक्षण किया गया

## मॉडल प्रीसेट

| प्रीसेट | VRAM | Speed | गुणवत्ता |
| -------- | ------ | ------- | --------- |
| Qwen 2.5 7B | ~12GB | मध्यम | Best |
| Qwen 2.5 3B | ~8GB | Fast | Good |
| Llama 3.2 3B | ~8GB | Fast | Good |
| Llama 3.2 1B | ~6GB | सबसे तेज़ | Basic |
| Mistral 7B | ~12GB | मध्यम | Good |

## आर्किटेक्चर

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

## संबंधित परियोजनाएँ

[**MCP Tool Shop**](https://mcp-tool-shop.github.io/) का हिस्सा:

- [Tool Compass](https://github.com/mcp-tool-shop-org/tool-compass) — सिमेंटिक एमसीपी टूल खोज
- [File Compass](https://github.com/mcp-tool-shop-org/file-compass) — सिमेंटिक फ़ाइल खोज
- [Comfy Headless](https://github.com/mcp-tool-shop-org/comfy-headless) — जटिलता के बिना ComfyUI

## लाइसेंस

एमआईटी — विवरण के लिए [LICENSE](LICENSE) देखें।
