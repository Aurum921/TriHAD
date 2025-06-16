# TriHAD
先去huggingface下喜欢的模型，把模型文件夹直接放到代码相同路径下（记得改代码里model_path的名称），然后依次运行stage1.py、stage2.py、stage3.py，最后与运行full_eval_multiclass.py即可

create_fake_sample.py这个文件是生成伪样本的，已经生成好放到data里面了，不需要再单独运行

test2.py和eval.py是多任务协同学习的，拿来跟三阶段那个对比的，写的更是超级粗糙，想跑的话依次运行一下这俩就行

有时间或者未来要深入研究这个领域的话会把代码结构重新优化一下的，现在还是有点乱

