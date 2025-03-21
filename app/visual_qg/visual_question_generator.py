from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import torch
from api.question import Question

class VisualQuestionGenerator:
    def __init__(self):
        self.model =  PaliGemmaForConditionalGeneration.from_pretrained("ahmed-masry/chartgemma")
        self.processor = AutoProcessor.from_pretrained("ahmed-masry/chartgemma")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def process_inputs(self, image_path, input_text):
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(text=input_text, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs

    def generate_text(self, inputs, num_beams=4, max_new_tokens=512):
        prompt_length = inputs['input_ids'].shape[1]
        generate_ids = self.model.generate(**inputs, num_beams=num_beams, max_new_tokens=max_new_tokens)
        output_text = self.processor.batch_decode(
            generate_ids[:, prompt_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output_text

    def generate_description(self, image_path):
        input_text = "<image> generate a full description about this diagram"
        inputs = self.process_inputs(image_path, input_text)
        return self.generate_text(inputs, num_beams=10)

    def generate_question(self, image_path):
        print("here==============")
        input_text = "<image> Generate a question on this chart"
        print("here==============1")

        inputs = self.process_inputs(image_path, input_text)
        print("here==============2")

        question =self.generate_text(inputs)
        print(f"question=========== {question}")
        return question
    
    def generate_answer(self, image_path,question):
        input_text = question
        inputs = self.process_inputs(image_path, input_text)
        return self.generate_text(inputs)
