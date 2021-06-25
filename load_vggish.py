import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from vggish.vggish_postprocess import *
from vggish.vggish_slim import *
define_vggish_slim()

model_path = 'models/vggish_model.ckpt'
sess = tf.compat.v1.Session()
load_vggish_slim_checkpoint(sess, model_path)

