# 别了， Yann LeCun

Author: Xiang Shi
Published: 2024-11-12T23:15:21-05:00
Language: zh
Tags: machine learning, economics
Canonical: https://xshi19.github.io/zh/blog/farewell-yann-lecun/
Source: https://x.com/xxiangshi/status/1856551399854637207

> 图片说明：原文中的方位描述予以保留；正文图片集中列于文末，位置未作推断。

![原文封面 — 别了， Yann LeCun](/images/x-articles/farewell-yann-lecun/cover.jpg)

这是我subscribe X后的第一篇文章，但刚好也赶上了伟大的AI科学家，图灵奖得主Yann LeCun的离开。在持续了数个月跟Elon Musk的嘴炮后（大多数Yann自己的单方面行为），他大概还是破防了。

在一开始我还是支持Yann的，毕竟Elon在卷积神经网络（CNN）上翻了大车——毫无疑问特斯拉基于视觉的自动驾驶系统是离不开卷积神经网络的，模型的架构可以千变万化，但依旧离不开卷积对图像的处理。

但随着时间的流逝，Yann LeCun（或许是因为政治立场）开始不断在其他领域攻击马斯克。很遗憾的是，在那些他不了解的领域中，他展示了惊人的无知。黑天鹅的作者Nassim Taleb有一个很好的词汇来形容这样的人： Intellectual Yet Idiot (IYI，知识分子型傻瓜)。

IYI有点类似于我们常说的小镇做题家，喜欢把各种指标当成是目的而不是手段，比如考试成绩，以及Yann LeCun最喜欢的论文发表和引用。

IYI很喜欢使用非常长的逻辑线，这样可以让他们的论述看上去更严谨。但Yann LeCun大师的第一环就很离谱：科学研究为了确保正确和可复现性，必须通过发表论文和同行评议来完成。这就是典型的把IQ分数等同于智力，把四六级成绩等同于英语水平。论文，同行评议和复现只是确保科学严谨的一种方式，但从来不等同于科学真理。科学从来是应用和解决问题的方式——如果特斯拉成功制造出了智能机器人，你不需要发表论文展示技术细节让别人来评判你的成果是否“科学”。

Yann LeCun是幸运的，在他的领域，复现他人的结果是非常容易的事情。必须承认他提倡的开源思想是人工智能能发展如此迅速的关键。大部分的机器学习论文都有伴随开源代码，而且大概率这些代码都是基于诸如PyTorch这样的开源架构。在有硬件支持的情况下，一个优秀的研究员只需要几天就能复现他人的模型，并根据自己的use case进行调整。在这种情况下一个优秀的模型（比如transformer架构）能很快在研究人员中传播开来。

但其他的领域就并非如此，尤其是在诸如社会科学，行为科学，经济学这类的“软科学”内。一个典型的例子就是行为科学家Francesca Gino，她在2012那篇著名的关于签名位置能显示人是否诚实的论文在9年后才被发现无法复现结果。她在2023年才被哈佛大学解雇（推荐这一期的Freak Economics Podcast https://freakonomics.com/podcast/why-is-there-so-much-fraud-in-academia/）。

目前为止，她的那篇争议论文在Google Scholar上还有着500多的引用。她本人也没有受到任何实质性的处罚，她的畅销书依旧在亚马逊上热卖。她在Google Scholar上的h-index高达92。按照Yann LeCun大师的逻辑，她对于“科学”的贡献想必是比没有论文的Elon Musk强太多了。

当IYI把手段当成是结果，指标当成是事实的时候，他们就很容易被愚弄。学生可以通过各种专门应对考试的“技巧”来提高自己成绩。在学术圈，相互刷引用并不鲜见。另一个常见现象统计的滥用——如果人们都用p-value来作为论文能否发表的统一标准，那么p-hacking就不可避免。在所有人都应用错误的统计方法时，能否“复现”结果就根本没有意义。

Yann LeCun显然无法理解这些，所以当他在讨论经济和政治的时候，也表现出了和在机器学习领域一样的自信。

Yann LeCun对于银行体系的理解可能还停留在我们中学通识教育的水平，他似乎认为通过官方解释和维基就能了解美联储运作的真正方式。喜欢参考教科书上的“标准答案”和官方权威解释也是IYI的特征之一。

当讨论到经济指标时，他又犯了把指标当事实的毛病。

在他引用的那个图中，CPI指数和Average hourly earnings都是一个均值。它们可以被进一步分为不同的维度，比如当你根据不同收入群体来分解hourly wage的时候就能看到一个完全不同的故事：中产不升反降，而他们是这次大选投票的主力。低收入人群虽然有点增长，但他们对通胀极为敏感，对于live paycheck by paycheck的人来说，只要工资稍有波动他们很可能陷入困境。

关于数据，Jeff Bezos在访谈中就曾说到

When the data and the anecdotes disagree, the anecdotes are usually right. And it doesn't  mean you just slavishly go follow the anecdotes then. It means you go  examine the data. 'Cause the data, and it's usually not that the data is being mis-collected, it's usually that you're not measuring the right thing.

当数据和轶事不一致时，通常是轶事是对的。这并不是说你就应该盲目地去追随这些轶事，而是应该重新审视数据。因为问题往往不是数据的采集错误，而是你测量的内容本身不对。

Jeff Bezos这样的企业家和IYI的最大区别在于有无skin in the game。如果Bezos盲目相信某些数据和指标，他的公司就会失去客户，从而被市场淘汰。以上这些都是每一个亚马逊数据科学家入职培训的一部分。

而Yann LeCun这样的IYI，也包括许多经济学家，都热衷于把自己的模型和数据当成是物理定律，当大众都在抱怨经济和通胀时，这些高高在上精英们就会拿出他们在象牙塔内琢磨出的“事实”，来教育那些没有“科学”素养，只会被阴谋论洗脑的“民粹”。他们热衷于嘲笑有他人的失败，比如Trump之前的破产，他们喜欢说：“你看我早就说过了。”因为他们没有skin in the game，所以他们从来没有失败过。

有skin in the game的人也会犯错，比如买下Washington Post，并在过去几年一直对Trump口诛笔伐的Bezos。但最终他依旧能直面问题https://www.washingtonpost.com/opinions/2024/10/28/jeff-bezos-washington-post-trust/。而一个在没有skin in the game的环境下呆久的人，就像一只被人类养大的动物，永远失去了在野外生存的能力。所以Yann LeCun最终选择成为一只鸵鸟，回到了自己的舒适区。

## 原文图片

以下图片按素材中的顺序列出；原文中的具体位置未知。

![原文图片 1](/images/x-articles/farewell-yann-lecun/image-1.jpg)

![原文图片 2](/images/x-articles/farewell-yann-lecun/image-2.jpg)

![原文图片 3](/images/x-articles/farewell-yann-lecun/image-3.jpg)

![原文图片 4](/images/x-articles/farewell-yann-lecun/image-4.jpg)

[原文图片 5（外链）](https://pbs.twimg.com/media/GcJxsZsXwAAjd-r.jpg)

![原文图片 6](/images/x-articles/farewell-yann-lecun/image-6.jpg)



[中文原文发表于 X](https://x.com/xxiangshi/status/1856551399854637207)。
