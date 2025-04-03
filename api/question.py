class Question:
    def __init__(self, q_type, context, question, answer, source, page=None, timestamp=None):
        self.type = q_type
        self.context = context
        self.question = question
        self.answer = answer
        self.source = source
        self.page = page
        self.timestamp = timestamp

    def __str__(self):
        details = [
            f"Question(type={self.type}",
            f"context={self.context[:50]}...",
            f"question={self.question}",
            f"answer={self.answer}",
            f"source={self.source}",
        ]
        if self.page is not None:
            details.append(f"page={self.page}")
        if self.timestamp is not None:
            details.append(f"timestamp={self.timestamp}")
        return "\n".join(details) + ")"