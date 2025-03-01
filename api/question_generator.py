import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from input_preprocessing.documents.utils.core import Chunk
from typing import List

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
                    
class QuestionGenerator:
    def __init__(self, q_type):
        self.q_type=q_type
        if q_type=='base':
            self.model_name = "fares7elsadek/t5-base-finetuned-question-generation"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        
    def _generate_question(self, context, answer="[MASK]", max_length=64):
        """
        Generates a question and answer pair from the provided context.

        Args:
            context (str): The context passage.
            answer (str): The answer text. Use "[MASK]" to prompt the model to predict the answer.
            max_length (int): Maximum length of the generated sequence.

        Returns:
            str: The generated question and answer pair.
        """
        input_text = f"context: {context} answer: {answer} </s>"
        inputs = self.tokenizer([input_text], return_tensors="pt", truncation=True, padding=True)
        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    @staticmethod
    def _parse_question(raw_question):
        if raw_question.strip():
            try:
                question_part, answer_part = raw_question.split('answer:', 1)
                question = question_part.replace('question:', '').strip()
                answer = answer_part.replace('answer:', '').strip()
                return (question,answer)
            except ValueError:
                print(f"Skipping malformed question: {raw_question}")

    def _chunk_to_questions(self, chunk: Chunk):
        """
        Generates a question and answer pair from a given chunk.

        Args:
            context (str): The context passage.
            chunks (Chunk): The chunk to generate from

        Returns:
            str: The generated question and answer pair.
        """
        question = Question(q_type=self.q_type,
        context=chunk.text,
        source=chunk.source,
        page=chunk.page,
        question=None,
        answer=None)

        raw_question = self._generate_question(chunk.text)
        try:
            question.question, question.answer = self._parse_question(raw_question)
        except:
            question = None
        return question

    def generate_questions(self, chunks):
        questions = []
        for chunk in chunks:
            question= self._chunk_to_questions(chunk)
            if question:
                questions.append(question)
        return questions
