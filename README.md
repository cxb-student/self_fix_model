# self_fix_model
try1 是 初步的想法，目的是通过logits判断内容自信程度并生成合适的prompt

try2 为了保证训练方便更改了promt的生成为带有梯度的fussion来拼接输入，其中trian_decoder是主模型的训练方式

虽然最后结果没有那么好，但可以为诸位提供一种思路

想法摘要：

      在transformer基础架构上，取transformer最后的logits与hidden_states，两者拼接形成修正矩阵，将修正矩阵与output放入fusion融合，再将其输出调入transformer主架构。

可行性来源：

       logits与hidden_states含有model对于这个token自信与否的信息，用decoder提取信息便可以提出置信度，同时也可以提取出修正函数，结合output来对下一次输入做调整。

实验过程：

2.1~5号初版模型：

         ffn代替了现在的fussion，结果不理想，在喂了多方，多领域数据后，每一个的表现都有所下降。

          其基线模型的正确率在zero—shot的情况下依旧与改进后的模型相差不大。

2.7 号 中间模型：

          generate取代直接生成，正确率有明显提升（5%），但在接触了更多领域数据后，极其容易在以前一些简单的任务上有重大错误。

2.8号  最后的模型：

           用decode代替拼接，后接参数量来到了0.3b，但最后采用了常规的前向传播，因为这样更可控了一点，在算力平台上训练加测试了一天，容易过拟合，我感觉是因为grad在中间transformer冻结了的后果

          最终，以高于基线2%的成绩完成coding任务，并以低于基线5%的成绩完成数学推理。
2.11号 再次更改：

           用qwen1.5b代替原模型，训练了7个epoch，稍微有些过拟合
           
           于是我降低learning_rate到0.0001，训练了14轮，差不多达到了效果（相邻两轮loss差很小了），最后结果均比ds的蒸馏模型差，但比基线高了7%

结论：

       用loggits取置信度是正确的，但很大程度上解决不了问题，最后相当于对输入做一个残差的prompt生成保证不会说瞎话

       这是我的本意，当然我也同时希望一些魔法发生，最后其实并没有。

       对于我这个初学者而言，主要还是一个熟悉流程的实验。
