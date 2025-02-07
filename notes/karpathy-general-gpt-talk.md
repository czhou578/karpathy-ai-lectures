Always prefer 1d token sequences

Want to distribute computation of tokens across the answer.

If you want the model to respond to a math problem like "The answer is 3", that's bad since you are cramming all computation into one single token. Every token has finite computation being done to it.

Intermediary results / calculations not that expensive. Spread out reasoning across tokens (mini problems in each token). Store that in working memory and get to result easier!

Use tools whenever possible (like code)

Ask for way too much in a single token

Model is good at copy pasting:
Model cannot count and spell.

Base model: internet document simulator

Reinforcement Learning:

- prompts to practice, trial and error until you reach correct answer

Can launch parallel attempts at solving a prompt.
Generate 15 solutions, only 4 were right. Take the top solution
and train on it. Rinse and repeat.

Uses advanced reasoning: uses Reinforcement Learning

Distillation - training on chains of thought

Learning in unverifiable domains

thousands of updates -> thousands of prompts -> thousands of generations

Reinforcement learning from Human Feedback
