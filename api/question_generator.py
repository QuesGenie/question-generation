import json
from tkinter import Image
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from api.question import Question
from input_preprocessing.documents.utils.core import Chunk
from typing import List
from tqdm import tqdm

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
        for chunk in tqdm(chunks):
            question= self._chunk_to_questions(chunk)
            if question:
                questions.append(question)
        return questions

# class VisualQuestionGenerator:
#     def __init__(self, model, processor, device):
#         self.model =  PaliGemmaForConditionalGeneration.from_pretrained("ahmed-masry/chartgemma")
#         self.processor = AutoProcessor.from_pretrained("ahmed-masry/chartgemma")
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#     def process_inputs(self, image_path, input_text):
#         image = Image.open(image_path).convert('RGB')
#         inputs = self.processor(text=input_text, images=image, return_tensors="pt")
#         inputs = {k: v.to(self.device) for k, v in inputs.items()}
#         return inputs

#     def generate_text(self, inputs, num_beams=4, max_new_tokens=512):
#         prompt_length = inputs['input_ids'].shape[1]
#         generate_ids = self.model.generate(**inputs, num_beams=num_beams, max_new_tokens=max_new_tokens)
#         output_text = self.processor.batch_decode(
#             generate_ids[:, prompt_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False
#         )[0]
#         return output_text

#     def generate_description(self, image_path):
#         input_text = "<image> generate a full description about this diagram"
#         inputs = self.process_inputs(image_path, input_text)
#         return self.generate_text(inputs, num_beams=10)

#     def generate_question(self, image_path):
#         input_text = "<image> Generate a question on this image"
#         inputs = self.process_inputs(image_path, input_text)
#         return self.generate_text(inputs)
    
#     def generate_answer(self, image_path,question):
#         input_text = f"<image> {question}"
#         inputs = self.process_inputs(image_path, input_text)
#         return self.generate_text(inputs)
