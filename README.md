# Nue ASR
![rinna-icon](./assets/rinna.png)

[[Paper]](https://arxiv.org/abs/2312.03668)
[[Model card]](https://huggingface.co/rinna/nue-asr)

This repository includes codes for an end-to-end speech recognition model, `Nue ASR`, which integrates pre-trained speech and language models.

The name `Nue` comes from the Japanese word ([`鵺/ぬえ/Nue`](https://en.wikipedia.org/wiki/Nue)), one of the Japanese legendary creatures ([`妖怪/ようかい/Yōkai`](https://en.wikipedia.org/wiki/Y%C5%8Dkai)).

This model is capable of performing highly accurate Japanese speech recognition.
By utilizing a GPU, it can recognize speech at speeds exceeding real-time.

Benchmark score including our models can be seen at https://rinnakk.github.io/research/benchmarks/asr/

## Usage

First, install the code for inference this model.

```bash
pip install git+https://github.com/rinnakk/nue-asr.git
```

Command-line interface and python interface are available.

## Command-line usage
The following command will transcribe the audio file via the command line interface.
Audio files will be automatically downsampled to 16kHz.
```bash
nue-asr audio1.wav
```
You can specify multiple audio files.
```bash
nue-asr audio1.wav audio2.flac audio3.mp3
```

We can use DeepSpeed-Inference to accelerate the inference speed of GPT-NeoX module.
If you use DeepSpeed-Inference, you need to install DeepSpeed.
```bash
pip install deepspeed
```

Then, you can use DeepSpeed-Inference as follows:
```bash
nue-asr --use-deepspeed audio1.wav
```

Run `nue-asr --help` for more information.

## Python usage
The example of python interface is as follows:
```python
import nue_asr

model = nue_asr.load_model("rinna/nue-asr")
tokenizer = nue_asr.load_tokenizer("rinna/nue-asr")

result = nue_asr.transcribe(model, tokenizer, "path_to_audio.wav")
print(result.text)
```
`nue_asr.transcribe` function can accept audio data as either a `numpy.array` or a `torch.Tensor`, in addition to traditional audio waveform file paths.

Accelerating the inference speed of models using DeepSpeed-Inference is also available through the python interface.
```python
import nue_asr

model = nue_asr.load_model("rinna/nue-asr", use_deepspeed=True)
tokenizer = nue_asr.load_tokenizer("rinna/nue-asr")

result = nue_asr.transcribe(model, tokenizer, "path_to_audio.wav")
print(result.text)
```


## How to cite
```bibtex
@article{hono2023integration,
    title={An Integration of Pre-Trained Speech and Language Models for End-to-End Speech Recognition},
    author={Hono, Yukiya and Mitsuda, Koh and Zhao, Tianyu and Mitsui, Kentaro and Wakatsuki, Toshiaki and Sawada, Kei},
    journal={arXiv preprint arXiv:2312.03668},
    year={2023}
}

@misc{rinna-nue-asr,
    title={rinna/nue-asr},
    author={Hono, Yukiya and Mitsuda, Koh and Zhao, Tianyu and Mitsui, Kentaro and Wakatsuki, Toshiaki and Sawada, Kei},
    url={https://huggingface.co/rinna/nue-asr}
}
```


## License
[The Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0)
