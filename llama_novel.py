# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import time
from utils import inference_one_at_a_time, inference_conversation, conversation

# hyperparameters
generation_config = GenerationConfig(
    max_new_tokens=4096, # max token length that model generates
    num_beams=3,
    do_sample=False,
)
verbose = False
system_prompt = """
You are a famous creative novel writer now. Please follow the script of the novel privided by the user. Always generate novel with interesting plot twist and vivid characters.
"""

user_input = """
Character biography:

Man: Chu Qing
Chu Qing, tall, is the main hall of the dragon of the summer, the identity of the honorable, in order to hide the identity with Shen Yuning for five years, silently pay behind Shen Yuning, because of the lowly status often be looked down upon, rejected by people, Shen Yuning chose to divorce him, Chu Qing this only to see the true face of Shen Yuning, this time he decided not to hide the identity, he wants to make everyone regret.

Man 2: Xiao Shize
Xiao Shize, the big son of Xiao family, all the year round with glasses pretending to be deep, because like Shen Yuning, all kinds of difficulties to Chu Qing, mocking Chu Qing, also impostor Chu Qing's credit, was Chu Qing crazy hit the face, kneeling for mercy, let could have listed Xiao family bankruptcy overnight, the end is miserable.

Female protagonist: Su Zhifei

Su Zhifei, dressed in cheongsam, elegant temperament, is the big miss Su home, when young life miserable, after growing up was taken back to Su home, in the status of Su home, because do not like marriage objects, think of the marriage contract when young, so came to southern province to find Chu Qing, want to cooperate with him.

Female 2: Shen Yuning:
Shen Yuning, president of Shen's group, is Chu Qing's ex-wife, Shen's bankruptcy get Chu Qing help in the early stage, also thought it was their own ability, Shen's increasingly brilliant, she also gradually looked down on Chu Qing, so with Chu Qing divorce, she was indifferent when Chu Qing was mocked, and finally because of greed led to Shen's bankruptcy.

Script outline:
Shen Jia bankruptcy, in order to help Shen Yuning tide over the difficulties, Chu Qing choose to hide the identity secretly help Shen Yuning, but also at home to do five years of family husband, Shen Jia increasingly brilliant, did not expect in Shen Yuning got two million orders determined to divorce and Chu Qing, Chu Qing dont understand dont five years of companionship than the class identity, The hidden identity of Chu Qing was threatened and despised by the major families, but did not think he was the real big man, at the same time Chu Qing father gave him set fiancee also came to the south province want to marry him, Chu Qing in the face of her father agreed to her request, agreed to go to Beijing with her.


Below is the outline of the novel.

Chapter 1: The Unexpected Encounter

Chu Qing, dressed in a simple yet elegant cheongsam, walked into the crowded streets of the southern province. He had been hiding his true identity for five years, working tirelessly to help Shen Yuning, his ex-wife, overcome her financial difficulties. Little did he know, his secret was about to be exposed, and his life would never be the same again.

Chapter 2: The Truth Revealed

As Chu Qing navigated through the bustling streets, he was suddenly confronted by Xiao Shize, the son of a wealthy family. Xiao Shize was known for his cunning and arrogance, and he had a personal vendetta against Chu Qing. With a sly smile, he revealed Chu Qing's true identity to the public, and the news quickly spread like wildfire.

Chapter 3: The Fallout

The revelation of Chu Qing's true identity caused a stir in the community. Many people who had once looked up to him now turned their backs on him, and he found himself ostracized and ridiculed. Shen Yuning, his ex-wife, was particularly cold towards him, and he realized that she had never truly appreciated him.

Chapter 4: The Betrayal

As Chu Qing struggled to come to terms with his newfound isolation, he received a surprise visit from Su Zhifei, a young woman from a wealthy family. She had come to the southern province in search of Chu Qing, hoping to collaborate with him on a business venture. However, Chu Qing soon discovered that Su Zhifei was not what she seemed, and he found himself caught in a web of deceit and betrayal.

Chapter 5: The Unexpected Ally

As Chu Qing navigated through the treacherous waters of deception and betrayal, he received an unexpected ally in the form of Su Zhifei's younger sister, Su Qian. Su Qian was a kind and innocent soul, and she saw something in Chu Qing that no one else did - a kindred spirit. Together, they hatched a plan to expose the truth and bring justice to those who had wronged Chu Qing.

Chapter 6: The Showdown

As the truth began to unravel, Chu Qing found himself face to face with Xiao Shize and Shen Yuning. The three of them engaged in a fierce battle of wits, with Chu Qing determined to prove his innocence and expose the true culprits. In the end, Chu Qing emerged victorious, and justice was served.

Chapter 7: The New Beginning

With the truth finally revealed, Chu Qing was able to start anew. He and Su Qian formed a partnership, and together they built a successful business empire. Chu Qing also reconnected with his father, who had always been proud of him, and he finally found the happiness and fulfillment he had been searching for.

Chapter 8: The Legacy

As Chu Qing looked back on his journey, he realized that his experiences had taught him valuable lessons about loyalty, perseverance, and the true meaning of success. He knew that he had found his true calling, and he was determined to leave a lasting legacy that would inspire future generations.

Epilogue: The Final Chapter

Years later, Chu Qing sat in his office, surrounded by his loved ones. He looked back on his life with pride, knowing that he had overcome every obstacle and achieved his dreams. As he reflected on his journey, he realized that the true victory was not in the success he had achieved, but in the people he had touched along the way. And with that, the story of Chu Qing came to a close, leaving behind a legacy that would be remembered for generations to come.

Please elaborate the Epilogue: The Final Chapter.
""" # user prompt

messages = [
    {"role":"system","content":system_prompt},
    {"role":"user","content":user_input}
]

# model init
model_init_start_time = time.time()

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", padding_side="left")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map="auto", torch_dtype=torch.float16)
model = torch.compile(model)
tokenizer.pad_token = tokenizer.eos_token # Most LLMs don't have a pad token by default

model_init_end_time = time.time()

print(f'model init time cost: {model_init_end_time - model_init_start_time} s')
# print(tokenizer.chat_template)

if __name__ == "__main__":
    # conversation(model, tokenizer, messages, generation_config, verbose)
    inference_one_at_a_time(model, tokenizer, system_prompt, user_input, generation_config, use_system_instruction=True)