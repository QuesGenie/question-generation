import json

from api.question import Question

class JSONImageParser:
    def __init__(self, json_file):
        self.json_file = json_file
        self.image_paths = self.load_json()
        self.page=None
    def load_json(self):
        with open(self.json_file, 'r') as file:
            data = json.load(file)
        
        image_paths = []
        for slide in data.get("slides", []):
            self.page = slide["page_number"]
            for content in slide.get("content", []):
                if content.get("type") == "image" and "image_path" in content:
                    image_paths.append(content["image_path"])
        return image_paths

    def process_images(self, vqg):
        for image_path in self.image_paths:
            description = vqg.generate_description(image_path)
            question = vqg.generate_question(image_path)
            answer = vqg.generate_answer(question)
            
        return Question(q_type=self.q_type,
        context=image_path,
        source=self.json_file,
        page=self.page,
        question=question,
        answer=answer)