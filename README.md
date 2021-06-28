# ✨ VggishAudioEncoder

**VggishAudioEncoder** is a class that wraps the [VGGISH][https://github.com/tensorflow/models/tree/master/research/audioset/vggish] model for generating embeddings for audio data. 

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [🌱 Prerequisites](#-prerequisites)
- [🚀 Usages](#-usages)
- [🎉️ Example](#%EF%B8%8F-example)
- [🔍️ Reference](#%EF%B8%8F-reference)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## 🌱 Prerequisites

Run the provided bash script `download_model.sh` to download the pretrained model.

## 🚀 Usages

### 🚚 Via JinaHub

#### using docker images
Use the prebuilt images from JinaHub in your python codes, 

```python
from jina import Flow
	
f = Flow().add(uses='jinahub+docker://VGGishAudioEncoder')
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://VGGishAudioEncoder'
```

#### using source codes
Use the source codes from JinaHub in your python codes,

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://VGGishAudioEncoder')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub://VGGishAudioEncoder'
```


### 📦️ Via Pypi

1. Install the `jinahub-VGGishAudioEncoder` package.

	```bash
	pip install git+https://github.com/jina-ai/executor-audio-VGGishEncoder.git
	```

1. Use `jinahub-MY-DUMMY-EXECUTOR` in your code

	```python
	from jina import Flow
	from jinahub.SUB_PACKAGE_NAME.MODULE_NAME import VggishAudioEncoder
	
	f = Flow().add(uses=VggishAudioEncoder)
	```


### 🐳 Via Docker

1. Clone the repo and build the docker image

	```shell
	git clone https://github.com/jina-ai/executor-audio-VggishAudioEncoder.git
	cd executor-audio-VGGishEncoder
	docker build -t executor-audio-VGGishEncoder-image .
	```

1. Use `my-dummy-executor-image` in your codes

	```python
	from jina import Flow
	
	f = Flow().add(uses='docker://executor-audio-VGGishEncoder-image:latest')
	```

## 🎉️ Example 


```python
from jina import Flow, Document

f = Flow().add(uses='jinahub+docker://VggishAudioEncoder')

with f:
    resp = f.post(on='foo', inputs=Document(), return_resutls=True)
	print(f'{resp}')
```

### Inputs 

`Document` with `blob` of containing loaded audio.

### Returns

`Document` with `embedding` fields filled with an `ndarray` of the shape `embedding_dim` with `dtype=nfloat32`.


## 🔍️ Reference
- [VGGISH paper][https://research.google/pubs/pub45611/]
- [VGGISH code][https://github.com/tensorflow/models/tree/master/research/audioset/vggish]

