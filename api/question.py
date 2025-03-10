class Question:
    def __init__(self, q_type, context, question, answer, source, page):
        self.type = q_type
        self.context = context
        self.question = question
        self.answer = answer
        self.source = source
        self.page = page
    def __str__(self):
            return (f"Question(type={self.type}\ncontext={self.context[:50]}...\n"
                    f"question={self.question}\nanswer={self.answer}\n"
                    f"source={self.source}\npage={self.page}\n)")
                   