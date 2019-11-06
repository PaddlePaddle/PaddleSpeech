import json


class Words(object):

    def __init__(self, prob_split, metadata, alphabet, frame_to_sec=.03):

        self.raw_output = ''
        self.extended_output = []

        word = ''
        start_step, confidence, num_char = 0, 1.0, 0
        metadata_size = metadata.tokens.size()

        for i in range(metadata_size):
            token = metadata.tokens[i]
            letter = alphabet.string_from_label(token)
            time_step = metadata.timesteps[i]

            # prepare raw output
            self.raw_output += letter

            # prepare extended output
            if token != alphabet.blank_token:
                word.append(letter)
                confidence *= prob_split[time_step][token]
                num_char += 1
                if len(word) == 1:
                    start_step = time_step

            if token == alphabet.blank_token or i == metadata_size-1:
                duration_step = time_step - start_step

                if duration_step < 0:
                    duration_step = 0

                self.extended_output.append({"word": word,
                                             "start_time": frame_to_sec * start_step,
                                             "duration": frame_to_sec * duration_step,
                                             "confidence": confidence**(1.0/num_char)})
                # reset
                word = ''
                start_step, confidence, num_char = 0, 1.0, 0

    def to_json(self):
        return json.dumps({"raw_output": self.raw_output,
                          "extended_output": self.extended_output})

    def save_json(self, file_path):
        with open(file_path, 'w') as outfile:
            json.dump({"raw_output": self.raw_output,
                       "extended_output": self.extended_output},
                      outfile)
