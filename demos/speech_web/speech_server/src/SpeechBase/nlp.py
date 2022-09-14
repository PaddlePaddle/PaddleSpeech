from paddlenlp import Taskflow


class NLP:
    def __init__(self, ie_model_path=None):
        schema = ["时间", "出发地", "目的地", "费用"]
        if ie_model_path:
            self.ie_model = Taskflow(
                "information_extraction",
                schema=schema,
                task_path=ie_model_path)
        else:
            self.ie_model = Taskflow("information_extraction", schema=schema)

        self.dialogue_model = Taskflow("dialogue")

    def chat(self, text):
        result = self.dialogue_model([text])
        return result[0]

    def ie(self, text):
        result = self.ie_model(text)
        return result
