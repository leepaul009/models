# Deep Speech 2 on PaddlePaddle

## Firstly, plz check the commit number shown as follow
```
commit f9801433701abe54c7cbc442bd509698f44b6f0c
Merge: 950f451 10ee066
Author: Cao Ying <lcy.seso@gmail.com>
Date:   Thu Nov 16 18:06:21 2017 +0800
```

## Secondly, plz apply the changes shown below
### use manifest.test for inference
In the infer.py, replace the `infer_manifest` with `manifest.test`,
```
add_arg('infer_manifest', str, 'data/librispeech/manifest.test', "Filepath of manifest to infer.")
```

### calculate the elapsed time of inference
In the `infer.py`, apply the following change
```
ds2_model = DeepSpeech2Model(...)
# record the elapsed time of inference, and we can calculate 10 times average for profiling
time_infer = [] 
result_transcripts = ds2_model.infer_batch(..., iter=10, time_infer=time_infer)
print("batchsize=%d, latency=%g(ms), standard deviation=%g(ms)"
          % (args.num_samples, time_infer[0]*1000.0, time_infer[1]*1000.0))
```
In the `model_utils/model.py`, add the parameter `iters` and `time_infer` to the method `infer_batch` of class `DeepSpeech2Model`
```
def infer_batch(self, ..., iters, time_infer):
```
Then, to calculate the elapesd time, apply the following changes to the method `infer_batch`
```
tms = []
tm_avg = 0
tm_sd = 0
# get average elapsed time of inference
for _ in xrange(iters):
          tm_beg = time.time()
          infer_results = self._inferer.infer(
                    input=infer_data, feeding=feeding_dict)
          tm = time.time() - tm_beg
          tms.append(tm)
          tm_avg = tm_avg + tm
tm_avg = float(tm_avg)/float(iters)
for it in tms:
          tm_sd = tm_sd + (it - tm_avg)*(it - tm_avg)
tm_sd = math.sqrt(tm_sd / iters)
# time_infer[0] record the average elapsed time, and time_infer[1] record the standard deviation
time_infer.append(tm_avg)
time_infer.append(tm_sd)
```


### additional argments for deepSpeech2
In the `infer.py`, add the argments of `rnn_use_batch` and `use_mkldnn` and let users to decide which rnn mode is used and if mkldnn is used
```
add_arg('rnn_use_batch', bool, True, "rnn_use_batch")
add_arg('use_mkldnn', bool, True, "use_mkldnn")
```
add the argments of `rnn_use_batch` and `use_mkldnn` to `paddle.init`
```
in the paddle.init(use_gpu=args.use_gpu,
                trainer_count=args.trainer_count,
                rnn_use_batch=args.rnn_use_batch,
                use_mkldnn=args.use_mkldnn)
```


########
model_utils/model.py
_create_network(...fuse_bn=False)
deep_speech_v2_network(...fuse_bn=fuse_bn)


model_utils/network.py
deep_speech_v2_network(...fuse_bn=False)


cnn/rnn bias_attr=True


conv_group(...fuse_bn=fuse_bn)
conv_bn_layer(...fuse_bn=fuse_bn)


rnn_group(...fuse_bn=fuse_bn)
bidirectional_gru_bn_layer(...fuse_bn=fuse_bn)
bidirectional_simple_rnn_bn_layer(...fuse_bn=fuse_bn)











