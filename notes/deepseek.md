# Deepseek

Accuracy rewards (whether response is correct) and format rewards (put thinking between <think> </think>)

RL empowers DeepSeek-R1-Zero to attain robust reasoning capabilities without the need for any supervised
fine-tuning data.

DeepSeek-R1-Zero naturally acquires the
ability to solve increasingly complex reasoning tasks by leveraging extended test-time computation. This computation ranges from generating hundreds to thousands of reasoning tokens,
allowing the model to explore and refine its thought processes in greater depth.

It underscores the power and beauty of reinforcement learning: rather
than explicitly teaching the model on how to solve a problem, we simply provide it with the
right incentives, and it autonomously develops advanced problem-solving strategies.

## Training Pipeline

1. we collect thousands of cold-start data to fine-tune the DeepSeek-V3-Base as
   the starting point for RL.

- A key limitation of DeepSeek-R1-Zero is that its content is often not suitable
  for reading. Responses may mix multiple languages or lack markdown formatting to
  highlight answers for users. In contrast, when creating cold-start data for DeepSeek-R1,
  we design a readable pattern that includes a summary at the end of each response and
  filters out responses that are not reader-friendly.

- e focuses on enhancing the model’s reasoning capabilities, particularly in reasoning-intensive tasks such as coding, mathematics, science, and logic reasoning, which involve well-defined problems with
  clear solutions.

2. collect SFT (Supervised Fine-Tuning) data for the subsequent round. Unlike the initial cold-start data, which primarily focuses on reasoning, this stage incorporates data from other domains to enhance the model’s capabilities in writing, role-playing, and other general-purpose tasks.
