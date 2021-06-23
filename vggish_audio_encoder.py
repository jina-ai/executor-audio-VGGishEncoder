__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Any

from jina.executors.decorators import batching, single
from jina.executors.encoders.frameworks import BaseTFEncoder
from vggish.vggish_params import *
from vggish.vggish_postprocess import *
from vggish.vggish_slim import *


class VggishEncoder(BaseTFEncoder):
    def __init__(self, model_path: str, pca_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_path = model_path
        self.pca_path = pca_path

    def post_init(self):
        self.to_device()
        import tensorflow as tf
        tf.compat.v1.disable_eager_execution()
        self.sess = tf.compat.v1.Session()
        define_vggish_slim()
        load_vggish_slim_checkpoint(self.sess, self.model_path)
        self.feature_tensor = self.sess.graph.get_tensor_by_name(
            INPUT_TENSOR_NAME)
        self.embedding_tensor = self.sess.graph.get_tensor_by_name(
            OUTPUT_TENSOR_NAME)
        self.post_processor = Postprocessor(self.pca_path)

    @batching
    def encode(self, content: Any, *args, **kwargs) -> Any:
        [embedding_batch] = self.sess.run([self.embedding_tensor],
                                          feed_dict={self.feature_tensor: content})
        result = self.post_processor.postprocess(embedding_batch)
        return (np.float32(result) - 128.) / 128.


