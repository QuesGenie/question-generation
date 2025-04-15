class Question:

    def __init__(self, q_type, context, question, answer, source, loc):
        self.type = q_type
        self.question = question
        self.answer = answer
        self.source = source
        self.context = context
        self.contextLocation = loc

    def __str__(self):
        return (
            f"Question(type={self.type}, "
            f"question={self.question}, "
            f"answer={self.answer}, "
            f"source={self.source}, "
            f"context={self.context}, "
            f"contextLocation={self.contextLocation})"
        )
