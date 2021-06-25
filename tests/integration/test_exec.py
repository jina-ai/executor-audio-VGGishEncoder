__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
import librosa

from jina import Flow, Document, DocumentArray
from vggish import vggish_input

try:
    from vggish_audio_encoder import VggishAudioEncoder
except:
    from jinahub.encoders.audio.vggish_audio_encoder import VggishAudioEncoder

cur_dir = os.path.dirname(os.path.abspath(__file__))


def test_flow_from_yml():

    doc = DocumentArray([Document()])
    print('\n\n\nBEFORE FLOW\n\n\n')
    with Flow.load_config(os.path.join(cur_dir, 'flow.yml')) as f:
        print('\n\n\nHERE\n\n\n')
        resp = f.post(on='test', inputs=doc, return_results=True)

    assert resp is not None


def test_embedding():

    x_audio, sample_rate = librosa.load(os.path.join(cur_dir, '../data/sample.mp3'))
    log_mel_examples = vggish_input.waveform_to_examples(x_audio, sample_rate)
    doc = DocumentArray([Document(blob=log_mel_examples)])

    print('\n\n\nBEFORE FLOW EMBEDDING\n\n\n')
    with Flow.load_config(os.path.join(cur_dir, 'flow.yml')) as f:
        print('\n\n\nINSIDE FLOW EMBEDDING\n\n\n')
        responses = f.post(on='test', inputs=doc, return_results=True)

    assert responses[0].docs[0].embedding is not None
