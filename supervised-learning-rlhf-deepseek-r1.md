# Supervised Learning, RLHF, and DeepSeek R1

Author: Xiang Shi
Published: 2025-02-11T20:38:06-05:00
Language: en
Tags: machine learning
Canonical: https://xshi19.github.io/supervised-learning-rlhf-deepseek-r1.html
Source: https://x.com/xxiangshi/status/1889489122063999257

> Image note: the original text’s positional references are retained; article images are collected at the end because their positions are unknown.

![Original cover — Supervised Learning, RLHF, and DeepSeek R1](/images/x-articles/supervised-learning-rlhf-deepseek-r1/cover.jpg)

Imagine you are solving a math problem, say $2^x + x = 5$. Most people would first work on scratch paper, trying different approaches (analogous to AI's “chain of thought,” or CoT), before writing out the complete solution and answer.

## Supervised learning (traditional large language models)

No scratch work is allowed. The model must give the complete solution steps and answer directly.

As shown at the top of the image, the LLM can generate only one word (token) at a time. That word is then compared with the “standard answer” in the training data.

A rigid language teacher grades your math problem as if it were a poetry dictation exercise, deducting points for every wrong character. (This is a simplified version that leaves out probabilities.)

These penalties add up to a total score (or negative loss), and the training objective is to maximize that score.

The model's score is calculated using all the generated tokens.

## Reinforcement learning from human feedback (RLHF)

No scratch work is allowed. The model must give the complete solution steps and answer directly.

Humans evaluate how good the answer is and give a reward.

But a human teacher's grading may be influenced by many factors, such as the answer's format, the clarity of its expression, or even personal preferences (whether the mathematical notation is neat, whether the language is polite, and so on). So even two answers with the same content may receive different scores.

The middle of the image shows two approaches: one says to use the Newton–Raphson method; the other says to make a guess first. Even if the final answer is the same, the teacher will surely prefer the first student.

This method also calculates the score using all the generated tokens. The difference is that each token may carry a different weight in the score, entirely determined by human subjective judgment.

## DeepSeek R1 reinforcement learning

The model is allowed to do scratch work, as much as it likes. In other words, it can freely reason through a “chain of thought” (CoT).

The final score depends only on whether the final answer is correct. It has nothing to do with what was written in the intermediate scratch work.

## Additional notes

The supervised learning example is simplified. A better analogy would be to imagine ten thousand parallel universes, with a version of you solving a problem in each one. In the end, one thousand of you, or 10%, get the right answer. The final score is based on that 10% (taking its logarithm).

DeepSeek R1's reward mechanism also checks the answer's format, such as whether the LLM's CoT is enclosed in `<think>...</think>`. In other words, what you write in your scratch work does not matter, but you have to put it on scratch paper, separate from the exam paper. I left this detail out above for simplicity.

The GRPO algorithm in the DeepSeek R1 paper is important, but [this article](https://hkust-nlp.notion.site/simplerl-reason) shows that ordinary PPO also works.

Chain of thought (CoT) is not a direct product of reinforcement learning. [This paper](https://arxiv.org/pdf/2501.19393) demonstrates that a model can generate very long CoT even without reinforcement learning.

The key is how you score the LLM's answers. A student may already know how to reason and solve problems, but the way they show it will be shaped by how the teacher grades.

## Images from the original

The images below follow the order supplied with the source; their positions in the original text are unknown.

![Original image 1](/images/x-articles/supervised-learning-rlhf-deepseek-r1/image-1.png)



This article was translated from the Chinese original posted on [X](https://x.com/xxiangshi/status/1889489122063999257).

I also wrote a separate [English article on X](https://x.com/xxiangshi/status/1889177092476015036).
