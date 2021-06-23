import os

from jina import Executor, Document, DocumentArray

try:
    from vggish_audio_encoder import VggishEncoder
except:
    from jinahub.encoders.audio.vggish_audio_encoder import VggishEncoder



cur_dir = os.path.dirname(os.path.abspath(__file__))

def test_load():
    encoder = Executor.load_config(os.path.join(cur_dir, '../../config.yml'))
    assert encoder.path_encoder.endswith('tfidf_vectorizer.pickle')



