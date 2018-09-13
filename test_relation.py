import numpy as np

import tensorflow as tf
from tensorflow.python.keras import testing_utils, backend
from tensorflow.python.keras.utils import get_custom_objects
from tensorflow.python.platform import test

from relation import Relation


class RelationTest(test.TestCase):

    @staticmethod
    def test_relation_layer():
        backend.set_session(None)
        input_data = np.array([[[3, 2, 4],
                                [1, 5, 2]],

                               [[30, 20, 40],
                                [10, 50, 20]]], dtype=np.float32)
        weights = np.array([[1, 0],
                            [5, 6],
                            [7, 8]], dtype=np.float32)

        bias = np.array([4, 7], dtype=np.float32)

        expected_output = np.array([[[6926, 8642],
                                     [6845, 8822]],

                                    [[663440, 807500],
                                     [655340, 825500]]], dtype=np.float32)
        tf.reset_default_graph()
        get_custom_objects()['Relation'] = Relation
        kwargs = {'relations': 2,
                  'kernel_initializer': tf.constant_initializer(weights),
                  'bias_initializer': tf.constant_initializer(bias)
                  }
        a = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        output = testing_utils.layer_test(Relation,
                                          kwargs=kwargs,
                                          input_data=input_data,
                                          expected_output=expected_output)
        if not np.array_equal(output, expected_output):
            raise AssertionError('The output is not equal to our expected output')

    @staticmethod
    def test_my_case():
        backend.set_session(None)
        input_data = np.array([[[15, 0, 10],
                                [13, 1, 10],
                                [13, 5, 19],
                                [19, 19, 4]],

                               [[5, 14, 10],
                                [9, 11, 12],
                                [4, 7, 7],
                                [1, 9, 0]],

                               [[14, 17, 1],
                                [1, 9, 16],
                                [7, 6, 9],
                                [17, 7, 3]]], dtype=np.float32)
        w1 = np.array([[3, 8, 6, 7, 8],
                       [3, 1, 0, 8, 7],
                       [4, 9, 8, 1, 9]], dtype=np.float32)
        w2 = np.array([[3, 0, 0, 2, 3],
                       [3, 8, 9, 5, 7],
                       [3, 9, 7, 0, 7]], dtype=np.float32)
        g = np.array([[7, 9, 4, 9, 0],
                      [1, 1, 5, 4, 0],
                      [6, 1, 7, 1, 3]], dtype=np.float32)

        bias = np.array([0, 0, 0, 0, 0], dtype=np.float32)
        expected_output = np.array([[[128560, 219497, 209334, 128295,  48435],
                                     [126451, 210377, 201342, 124197,  47274],
                                     [160195, 262057, 249294, 152200,  61335],
                                     [160195, 217673, 193350, 247137,  62754]],

                                    [[61893,   76272, 131794,  81197,  34404],
                                     [65721,   87599, 151154,  83381,  36927],
                                     [49365,   66150, 117274,  57173,  27096],
                                     [38577,   45665,  81458,  52805,  20745]],

                                    [[86352,  109637, 130294, 167782,  34419],
                                     [84894,  119274, 153562,  92742,  32244],
                                     [75660,  111732, 142482,  98638,  29112],
                                     [80034,  123045, 149130, 137230,  31983]]], dtype=np.float32)
        kwargs = {'relations': 5}
        layer_cls = Relation(**kwargs)
        my_i = tf.keras.layers.Input(input_data.shape[1:], dtype=tf.float32)
        my_layer = layer_cls(my_i)
        model = tf.keras.Model(my_i, my_layer)
        weights = [w1, w2, g, bias]
        layer_cls.set_weights(weights)
        output = model.predict(input_data)
        if not np.array_equal(output, expected_output):
            raise AssertionError("The output is not equal with the expected output")


if __name__ == '__main__':
    test.main()
