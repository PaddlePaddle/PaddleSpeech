from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
import ctc_beam_search_decoder as tested_decoder


def test_beam_search_decoder():
    max_time_steps = 6
    beam_size = 20
    num_results_per_sample = 20

    input_prob_matrix_0 = np.asarray(
        [
            [0.30999, 0.309938, 0.0679938, 0.0673362, 0.0708352, 0.173908],
            [0.215136, 0.439699, 0.0370931, 0.0393967, 0.0381581, 0.230517],
            [0.199959, 0.489485, 0.0233221, 0.0251417, 0.0233289, 0.238763],
            [0.279611, 0.452966, 0.0204795, 0.0209126, 0.0194803, 0.20655],
            [0.51286, 0.288951, 0.0243026, 0.0220788, 0.0219297, 0.129878],
            # Random entry added in at time=5
            [0.155251, 0.164444, 0.173517, 0.176138, 0.169979, 0.160671]
        ],
        dtype=np.float32)

    # Add arbitrary offset - this is fine
    input_log_prob_matrix_0 = np.log(input_prob_matrix_0)  #+ 2.0

    # len max_time_steps array of batch_size x depth matrices
    inputs = ([
        input_log_prob_matrix_0[t, :][np.newaxis, :]
        for t in range(max_time_steps)
    ])

    inputs_t = [ops.convert_to_tensor(x) for x in inputs]
    inputs_t = array_ops.stack(inputs_t)

    # run CTC beam search decoder in tensorflow
    with tf.Session() as sess:
        decoded, log_probabilities = tf.nn.ctc_beam_search_decoder(
            inputs_t, [max_time_steps],
            beam_width=beam_size,
            top_paths=num_results_per_sample,
            merge_repeated=False)
        tf_decoded = sess.run(decoded)
        tf_log_probs = sess.run(log_probabilities)

    # run tested CTC beam search decoder     
    beam_result = tested_decoder.ctc_beam_search_decoder(
        input_probs_matrix=input_prob_matrix_0,
        beam_size=beam_size,
        blank_id=5,  # default blank_id in tensorflow decoder is (num classes-1)
        space_id=4,  # doesn't matter
        max_time_steps=max_time_steps,
        num_results_per_sample=num_results_per_sample)

    # compare decoding result
    print(
        "{tf_decoder log probs} \t {tested_decoder log probs}:  {tf_decoder result}  {tested_decoder result}"
    )
    for index in range(len(beam_result)):
        print(('%6f\t%6f: ') % (tf_log_probs[0][index], beam_result[index][0]),
              tf_decoded[index].values, '  ', beam_result[index][1])


if __name__ == '__main__':
    test_beam_search_decoder()
