from paddlenlp import Taskflow

class NLP:
    def __init__(self, ie_model_path=None):
        schema = ["时间", "出发地", "目的地", "费用"]
        if ie_model_path:
            self.ie_model = Taskflow("information_extraction",
                                    schema=schema, task_path=ie_model_path)
        else:
            self.ie_model = Taskflow("information_extraction",
                                    schema=schema)
            
        self.dialogue_model = Taskflow("dialogue")
    
    def chat(self, text):
        result = self.dialogue_model([text])
        return result[0]
    
    def ie(self, text):
        result = self.ie_model(text)
        return result

if __name__ == '__main__':
    ie_model_path = "../../source/model/"
    nlp = NLP(ie_model_path=ie_model_path)
    text = "今天早上我从大牛坊去百度科技园花了七百块钱"
    print(nlp.ie(text))
    