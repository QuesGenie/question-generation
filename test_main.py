from app.visual_qg.visual_question_generator import VisualQuestionGenerator 
image_path = "chart1.jpg"

vqg =VisualQuestionGenerator()
vqg.process_inputs(image_path,"")
question = vqg.generate_question(image_path)
print(question)